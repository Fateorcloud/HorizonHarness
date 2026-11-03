"""High-level start, inspect, and safe cross-process resume entry point."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence, cast

from horizon_harness.agents import AgentSession
from horizon_harness.context_view import effective_token_budget
from horizon_harness.core import run
from horizon_harness.hooks import HookMap
from horizon_harness.memory import (
    IndexedMemory,
    Memory,
    MemoryContext,
    MemoryEvidenceSource,
    MemoryExtractionResult,
    MemoryExtractor,
    TaskMemoryEvidence,
)
from horizon_harness.llm import LLMRequest, LLMResponse
from horizon_harness.protocols import Event, TurnEndEvent, TurnStartEvent
from horizon_harness.spec import AgentSpec
from horizon_harness.state import State
from horizon_harness.tools import AbortFlag, AgentTool
from horizon_harness.trace.run_trace import event_record

from .context import WorkspaceContextBuilder, render_workspace
from .codec import encode_message
from .models import Artifact, Checkpoint, VerifierResult, WorkspaceRecord
from .store import UnsafeCheckpointError, WorkspaceStore
from .execution import (
    make_durable_edit_tool,
    reconcile_edit,
)
from .drift import adopt_current, drifted_step_ids, drift_resume_message
from .verification import (
    AcceptanceContract,
    VerificationOutcome,
    VerificationReceipt,
    run_verifier,
)
from .snapshot import capture_tree
from .replay import fork_from_checkpoint
from .tools import (
    make_history_read_tool,
    make_workspace_update_tool,
    make_workspace_read_tool,
)

WORKSPACE_INSTRUCTIONS = """
Maintain the Task Workspace after meaningful progress with workspace_update.
Use its current revision, keep stable step ids, and attach concrete evidence.
Only mark status completed when the requested work is complete. A verifier runs
separately, so never claim that a workspace update proves tests passed. Use
history_read only when an archived observation is needed.
""".strip()

SUMMARY_INSTRUCTIONS = """
Continue from the supplied bounded summary and recent complete turns. Use
history_read only when an archived observation is needed. The summary is a
continuation aid, not verifier evidence or permission to change the task.
""".strip()


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    run_id: str
    status: str
    record: WorkspaceRecord
    state: State
    events: tuple[Event, ...]
    memory_entry_ids: tuple[str, ...] = ()
    memory_extraction: MemoryExtractionResult | None = None
    memory_extraction_error: str = ""


class TaskSession:
    """One synchronous writer around the ordinary single-agent loop."""

    def __init__(
        self,
        provider,
        *,
        code_dir: str | Path,
        store_dir: str | Path,
        spec: AgentSpec | None = None,
        context_budget: int = 12_000,
        output_reserve: int = 2_048,
        safety_buffer: int = 1_024,
        request_extra=None,
        memory: Memory | None = None,
        memory_scope: str = "",
        memory_extractor: MemoryExtractor | None = None,
        bash_exec_prefix: tuple[str, ...] = (),
        complete_fn: Callable[[LLMRequest], LLMResponse] | None = None,
        context_representation: str = "workspace",
        continuation_summary: str = "",
        summary_revision: int | None = None,
        durable: bool = True,
    ) -> None:
        provider_max = provider.default_max_tokens
        self.provider = replace(
            provider,
            default_max_tokens=(
                output_reserve
                if provider_max is None
                else min(provider_max, output_reserve)
            ),
        )
        self.code_dir = Path(code_dir).expanduser().resolve()
        self.store = WorkspaceStore(store_dir)
        self.durable = durable
        store_root = self.store.root
        if store_root == self.code_dir or store_root.is_relative_to(self.code_dir):
            raise ValueError("store_dir must be outside code_dir")
        self.spec = spec or AgentSpec(
            name="workspace-agent",
            capabilities=("read", "edit"),
            workflow="single",
        )
        if self.spec.workflow != "single":
            raise ValueError("TaskSession supports only workflow='single'")
        if "task" in self.spec.capabilities or "skills" in self.spec.capabilities:
            raise ValueError("TaskSession does not enable task delegation or skills")
        if self.durable:
            unsupported = set(self.spec.capabilities) - {
                "read",
                "edit",
                "workspace",
                "memory",
                "history",
            }
            if unsupported:
                raise ValueError(
                    "durable TaskSession does not support capabilities: "
                    + ", ".join(sorted(unsupported))
                )
        if context_budget <= 0:
            raise ValueError("context_budget must be greater than zero")
        if output_reserve <= 0:
            raise ValueError("output_reserve must be greater than zero")
        if safety_buffer < 0:
            raise ValueError("safety_buffer must not be negative")
        window_budget = (
            effective_token_budget(
                self.provider.context_window,
                output_reserve=output_reserve,
                safety_buffer=safety_buffer,
            )
            if self.provider.context_window is not None
            else context_budget
        )
        self.context_budget = min(context_budget, window_budget)
        if self.context_budget <= 0:
            raise ValueError(
                "model context window leaves no input budget after output reserve "
                "and safety buffer"
            )
        self.output_reserve = output_reserve
        self.safety_buffer = safety_buffer
        self.request_extra = request_extra
        self.memory = memory
        self.memory_scope = memory_scope
        if memory_extractor is not None and not isinstance(memory, IndexedMemory):
            raise ValueError("memory_extractor requires an IndexedMemory backend")
        if memory_extractor is not None and not memory_scope.strip():
            raise ValueError("memory_extractor requires a non-empty memory_scope")
        self.memory_extractor = memory_extractor
        self.bash_exec_prefix = bash_exec_prefix
        self.complete_fn = complete_fn
        if context_representation not in ("workspace", "summary", "combined"):
            raise ValueError("context_representation must be workspace or summary")
        if context_representation == "summary" and not continuation_summary.strip():
            raise ValueError("summary context requires continuation_summary")
        self.context_representation = context_representation
        self.continuation_summary = continuation_summary
        self.summary_revision = summary_revision
        if context_representation == "combined" and (
            not continuation_summary.strip() or summary_revision is None
        ):
            raise ValueError("combined context requires summary and source revision")

    def start(
        self,
        task: str,
        *,
        constraints: Sequence[str] = (),
        acceptance: Sequence[str] = (),
        acceptance_contract: AcceptanceContract | None = None,
        task_id: str | None = None,
        max_turns: int = 20,
        pause_after_turns: int | None = None,
    ) -> TaskRunResult:
        record = self.store.create(
            task,
            constraints=constraints,
            acceptance=acceptance,
            acceptance_contract_id=(
                acceptance_contract.contract_id if acceptance_contract else ""
            ),
            acceptance_contract_version=(
                acceptance_contract.version if acceptance_contract else 0
            ),
            acceptance_commands=(
                acceptance_contract.required_commands if acceptance_contract else ()
            ),
            task_id=task_id,
        )
        return self._drive(
            record,
            max_turns=max_turns,
            pause_after_turns=pause_after_turns,
            resumed=False,
        )

    def resume(
        self,
        task_id: str,
        *,
        max_turns: int = 20,
        pause_after_turns: int | None = None,
        abort: AbortFlag = lambda: False,
    ) -> TaskRunResult:
        record = self.store.assert_resumable(task_id)
        return self._drive(
            record,
            max_turns=max_turns,
            pause_after_turns=pause_after_turns,
            resumed=True,
            abort=abort,
        )

    def inspect(self, task_id: str) -> WorkspaceRecord:
        return self.store.load(task_id)

    def adopt_current(
        self,
        task_id: str,
        *,
        step_id: str,
        expected_sha: str,
        note: str = "",
    ) -> dict[str, object]:
        """Adopt observed external bytes without executing any workspace tool."""
        with self.store.writer(task_id):
            record = self.store.load(task_id)
            self._require_code_identity(record)
            result = adopt_current(
                self.store.journal(task_id),
                root=self.code_dir,
                step_id=step_id,
                expected_sha=expected_sha,
                note=note,
            )
            return {"task_id": task_id, **result}

    def fork_task(
        self,
        task_id: str,
        *,
        child_task_id: str,
        child_code_dir: str | Path,
        child_store_dir: str | Path,
    ) -> WorkspaceRecord:
        """Fork one complete checkpoint into an independent resumable task."""
        with self.store.writer(task_id):
            parent = self.store.load(task_id)
            journal = self.store.journal(task_id)
            parent_seq = journal.applied_seq()
            if self.store.has_uncommitted(task_id) or journal.recoverable_batches():
                raise UnsafeCheckpointError("fork requires a complete task boundary")
            if not parent.checkpoint.snapshot_path:
                raise UnsafeCheckpointError(
                    "fork requires an immutable checkpoint snapshot"
                )
            messages = tuple(self.store.load_messages(parent))
            fork_from_checkpoint(
                self.store.task_dir(task_id),
                parent.checkpoint.snapshot_path,
                child_code_dir,
            )
            child_store = WorkspaceStore(child_store_dir)
            child = child_store.create(
                parent.task,
                constraints=parent.constraints,
                acceptance=parent.acceptance,
                acceptance_contract_id=parent.acceptance_contract_id,
                acceptance_contract_version=parent.acceptance_contract_version,
                acceptance_commands=parent.acceptance_commands,
                task_id=child_task_id,
            )
            child_session = TaskSession(
                self.provider,
                code_dir=child_code_dir,
                store_dir=child_store_dir,
                spec=self.spec,
                complete_fn=self.complete_fn,
            )
            child = replace(
                child,
                steps=parent.steps,
                decisions=parent.decisions,
                artifacts=parent.artifacts,
                blockers=(*parent.blockers, "fork requires independent verification"),
                next_action=parent.next_action,
                status="paused",
                parent_task_id=parent.task_id,
                parent_revision=parent.revision,
                parent_snapshot_sha256=parent.checkpoint.workspace_tree_sha256,
                checkpoint=Checkpoint(
                    run_id=f"fork-{uuid.uuid4().hex}",
                    completed_turn=parent.checkpoint.completed_turn,
                    resumable=True,
                    provider_fingerprint=parent.checkpoint.provider_fingerprint,
                    code_identity=child_session._code_identity(),
                    phase="ready",
                    workspace_tree_sha256=parent.checkpoint.workspace_tree_sha256,
                ),
            )
            child_store.save(child, event_kind="ForkCreated")
            child_journal = child_store.journal(child_task_id)
            for message in messages:
                child_journal.append(
                    "MessageEvent", {"message": encode_message(message)}
                )
            child = child_store.commit_checkpoint(
                child,
                messages,
                (),
                code_dir=child_code_dir,
            )
            if (
                journal.applied_seq() != parent_seq
                or self.store.load(task_id) != parent
            ):
                raise RuntimeError("parent changed while creating fork")
            return child

    def rebuild_state(self, task_id: str) -> State:
        """Rebuild an in-memory State from canonical Journal messages."""
        journal = self.store.journal(task_id)
        record = self.store.load(task_id)
        state = State(task=record.task)
        for message in journal.rebuild_messages():
            state.record(message)
        for batch in journal.recoverable_batches():
            journal.batch_assistant_message(batch["batch_id"])
        return state

    def record_verifier(
        self,
        task_id: str,
        *,
        command: str,
        exit_code: int,
        evidence_ref: str,
    ) -> WorkspaceRecord:
        with self.store.writer(task_id):
            record = self.store.load(task_id)
            result = VerifierResult(
                command=command,
                passed=exit_code == 0,
                exit_code=exit_code,
                evidence_ref=evidence_ref,
            )
            revised = replace(
                record,
                revision=record.revision + 1,
                verifier_results=(*record.verifier_results, result),
            )
            self.store.save(revised)
            return revised

    def verify(
        self,
        task_id: str,
        contract: AcceptanceContract,
        *,
        verifier_version: str = "local-v1",
    ) -> VerificationReceipt:
        """Run a deterministic verifier and persist its current-tree receipt."""
        with self.store.writer(task_id):
            record = self.store.load(task_id)
            self._require_frozen_contract(record, contract)
            receipt = run_verifier(
                task_id,
                self.code_dir,
                contract,
                verifier_version=verifier_version,
                task_revision=record.revision + 1,
            )
            result = VerifierResult(
                command=receipt.command,
                passed=receipt.exit_code == 0,
                exit_code=receipt.exit_code,
                evidence_ref=receipt.output_digest,
                tree_sha256=receipt.tree_sha256,
                contract_id=receipt.contract_id,
                contract_version=receipt.contract_version,
                verifier_version=receipt.verifier_version,
                task_revision=receipt.task_revision,
                outcome=receipt.outcome,
            )
            revised = replace(
                record,
                revision=record.revision + 1,
                verifier_results=(*record.verifier_results, result),
            )
            self.store.save(revised, event_kind="VerificationRecorded")
            return receipt

    def complete_if_verified(
        self,
        task_id: str,
        contract: AcceptanceContract,
        *,
        verifier_version: str = "local-v1",
    ) -> WorkspaceRecord:
        """Commit completed only when a current successful receipt exists."""
        with self.store.writer(task_id):
            record = self.store.load(task_id)
            self._require_frozen_contract(record, contract)
            if record.status == "failed":
                return record
            if self.store.has_uncommitted(task_id):
                raise UnsafeCheckpointError("completion rejected: uncommitted turn")
            pending = self.store.journal(task_id).unsettled_step_ids()
            if pending:
                raise UnsafeCheckpointError(
                    "completion rejected: unsettled steps: " + ", ".join(pending)
                )
            if not record.verifier_results:
                raise UnsafeCheckpointError(
                    "completion rejected: no verification evidence"
                )
            latest = record.verifier_results[-1]
            from .verification import receipt_is_current

            receipt = VerificationReceipt(
                task_id=task_id,
                task_revision=latest.task_revision,
                tree_sha256=latest.tree_sha256,
                contract_id=latest.contract_id,
                contract_version=latest.contract_version,
                verifier_version=latest.verifier_version,
                command=latest.command,
                exit_code=latest.exit_code,
                outcome=cast(VerificationOutcome, latest.outcome),
                output_digest=latest.evidence_ref,
            )
            evidence_revision = (
                record.revision - 1 if record.status == "completed" else record.revision
            )
            if not receipt_is_current(
                receipt,
                task_id=task_id,
                root=self.code_dir,
                contract=contract,
                verifier_version=verifier_version,
                task_revision=evidence_revision,
            ):
                raise UnsafeCheckpointError(
                    "completion rejected: verification evidence is stale or failed"
                )
            if record.status == "completed":
                return record
            revised = replace(
                record,
                revision=record.revision + 1,
                status="completed",
                checkpoint=replace(record.checkpoint, resumable=False),
            )
            self.store.save(revised, event_kind="CompletionAccepted")
            return revised

    def finalize_memory(
        self, task_id: str, *, outcome: str
    ) -> tuple[tuple[str, ...], MemoryExtractionResult]:
        """Finalize a verified task and run the configured sourced extraction."""

        if outcome not in ("completed", "failed"):
            raise ValueError("outcome must be completed or failed")
        if not isinstance(self.memory, IndexedMemory) or self.memory_extractor is None:
            raise ValueError("finalize_memory requires indexed memory and an extractor")
        with self.store.writer(task_id):
            record = self.store.load(task_id)
            if record.status != outcome:
                raise ValueError(
                    "finalize_memory cannot change task status; completion must "
                    "pass the trusted verification gate"
                )
            messages = self.store.load_messages(record)
        state = State(task=record.task)
        for message in messages:
            state.record(message)
        return self.memory.extract_task(
            self._memory_evidence(record, state, record.checkpoint.run_id),
            self.memory_extractor,
        )

    def _drive(
        self,
        record: WorkspaceRecord,
        *,
        max_turns: int,
        pause_after_turns: int | None,
        resumed: bool,
        abort: AbortFlag = lambda: False,
    ) -> TaskRunResult:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if pause_after_turns is not None and pause_after_turns <= 0:
            raise ValueError("pause_after_turns must be greater than zero")
        run_id = uuid.uuid4().hex
        holder = [record]
        completed_turns = record.checkpoint.completed_turn
        turns_started_in_run = 0
        turn_opened_in_run = False
        pause_requested = False
        captured: list[Event] = []

        def current() -> WorkspaceRecord:
            return holder[0]

        def update(value: WorkspaceRecord) -> None:
            holder[0] = value

        def should_abort() -> bool:
            return pause_requested or abort()

        update_tool = make_workspace_update_tool(
            current=current, update=update, code_dir=self.code_dir
        )
        history_tool = make_history_read_tool(self.store, record.task_id)
        workspace_tools = (
            (update_tool, history_tool)
            if self.context_representation != "summary"
            else (history_tool,)
        )
        memory_tools = ()
        if self.context_representation == "combined":
            workspace_tools = (
                *workspace_tools,
                make_workspace_read_tool(self.store, record.task_id),
            )
        memory_hooks: HookMap = {}
        if self.memory is not None:
            binding = self.memory.bind(
                MemoryContext(
                    agent=self.spec.name,
                    task=record.task,
                    session_id=record.task_id,
                    run_id=run_id,
                    data={"scope": self.memory_scope},
                )
            )
            memory_tools = binding.tools
            memory_hooks = binding.hooks
        system_prompt = self.spec.system_prompt or self.spec.role or ""
        continuation_instructions = (
            WORKSPACE_INSTRUCTIONS
            if self.context_representation != "summary"
            else SUMMARY_INSTRUCTIONS
        )
        system_prompt = "\n\n".join(
            part for part in (system_prompt, continuation_instructions) if part
        )
        session_spec = replace(self.spec, system_prompt=system_prompt)
        base_builder = WorkspaceContextBuilder(
            self.store,
            record.task_id,
            max_input_tokens=self.context_budget,
            run_id=run_id,
            model_context_window=self.provider.context_window,
            output_reserve=self.output_reserve,
            safety_buffer=self.safety_buffer,
            representation=self.context_representation,
            summary=self.continuation_summary,
            summary_revision=self.summary_revision,
        )
        bound_tools: list[tuple[AgentTool, ...]] = [()]

        def context_builder(agent_name, state, policy):
            tools = bound_tools[0]
            if not tools:
                raise RuntimeError(
                    "workspace context requested before tools were bound"
                )
            return replace(
                base_builder,
                fixed_overhead_tokens=_estimate_fixed_overhead(system_prompt, tools),
                fixed_overhead_tools=tuple(tool.name for tool in tools),
            )(agent_name, state, policy)

        with self.store.writer(record.task_id):
            if resumed:
                # Re-check after acquiring the writer lock; another process may
                # have advanced the task between the optimistic read and now.
                locked = self.store.assert_resumable(record.task_id)
                if locked.revision != record.revision:
                    raise UnsafeCheckpointError(
                        "workspace changed before resume acquired the writer lock"
                    )
                self._require_code_identity(locked)
                expected_provider = locked.checkpoint.provider_fingerprint
                current_provider = self._provider_fingerprint()
                if expected_provider and expected_provider != current_provider:
                    raise UnsafeCheckpointError(
                        "provider configuration does not match the checkpoint fingerprint"
                    )
                journal = self.store.journal(record.task_id)
                if self.durable:
                    self._validate_pending_batches(journal)
                holder[0] = self._reconcile_artifacts(locked)
                record = holder[0]
            with AgentSession(
                self.provider,
                spec=session_spec,
                cwd=self.code_dir,
                tools=(*workspace_tools, *memory_tools),
                hooks=memory_hooks,
                context_builder=context_builder,
                request_extra=self.request_extra,
                max_turns=max_turns,
                bash_exec_prefix=self.bash_exec_prefix,
                complete_fn=self.complete_fn,
            ) as agent_session:
                bound_tools[0] = agent_session.agent.tools
                agent_session.agent.journal = self.store.journal(record.task_id)
                if self.durable:
                    agent_session.agent.recover_durable_batches = True

                    def prepare_durable_recovery() -> None:
                        drifted = drifted_step_ids(self.store.journal(record.task_id))
                        if drifted:
                            raise UnsafeCheckpointError(
                                drift_resume_message(record.task_id, drifted[0])
                            )
                        actions = reconcile_edit(
                            self.store.journal(record.task_id),
                            root=self.code_dir,
                            retry_before=False,
                        )
                        ambiguous = next(
                            (
                                action
                                for action in actions
                                if action.startswith("ambiguous:")
                            ),
                            None,
                        )
                        if ambiguous is not None:
                            raise UnsafeCheckpointError(
                                drift_resume_message(
                                    record.task_id,
                                    ambiguous.removeprefix("ambiguous:").removesuffix(
                                        ":workspace drift"
                                    ),
                                )
                            )

                    agent_session.agent.prepare_durable_recovery = (
                        prepare_durable_recovery
                    )
                    agent_session.agent.tools = tuple(
                        make_durable_edit_tool(
                            root=self.code_dir, journal=agent_session.agent.journal
                        )
                        if tool.name == "edit"
                        else tool
                        for tool in agent_session.agent.tools
                    )
                if resumed:
                    state = State(task=record.task)
                    journal = self.store.journal(record.task_id)
                    messages = journal.rebuild_messages()
                    if not messages:
                        legacy_pending = journal.recoverable_batches()
                        if legacy_pending:
                            journal.batch_assistant_message(
                                legacy_pending[0]["batch_id"]
                            )
                        messages = tuple(self.store.load_messages(record))
                    for message in messages:
                        state.record(message)
                    event_iter: Iterator[Event] = run(
                        agent_session.agent,
                        state,
                        max_turns=max_turns,
                        abort=should_abort,
                    )
                else:
                    state, event_iter = agent_session.run(
                        record.task, max_turns=max_turns, abort=should_abort
                    )
                for event in event_iter:
                    captured.append(event)
                    if isinstance(event, TurnStartEvent):
                        turn_opened_in_run = True
                        self.store.begin_turn(
                            record.task_id, run_id, completed_turns + 1
                        )
                    if isinstance(event, TurnEndEvent):
                        completed_turns += 1
                        if turn_opened_in_run:
                            turns_started_in_run += 1
                            turn_opened_in_run = False
                        if (
                            pause_after_turns is not None
                            and turns_started_in_run >= pause_after_turns
                        ):
                            pause_requested = True
                        status = holder[0].status
                        resumable = status not in ("completed", "failed")
                        if pause_requested and status == "active":
                            status = "paused"
                        checkpoint = Checkpoint(
                            run_id=run_id,
                            completed_turn=completed_turns,
                            resumable=resumable,
                            recent_message_refs=tuple(
                                f"{run_id}:{index}"
                                for index in range(
                                    max(0, len(state.messages) - 4), len(state.messages)
                                )
                            ),
                            message_refs=holder[0].checkpoint.message_refs,
                            provider_fingerprint=self._provider_fingerprint(),
                            code_identity=self._code_identity(),
                            phase="ready",
                            next_tool_index=0,
                            workspace_tree_sha256=capture_tree(
                                self.code_dir
                            ).tree_sha256,
                        )
                        holder[0] = replace(
                            holder[0], status=status, checkpoint=checkpoint
                        )
                        holder[0] = self.store.commit_checkpoint(
                            holder[0],
                            state.messages,
                            [event_record(item) for item in state.events],
                            code_dir=self.code_dir,
                        )
                # A final response without an explicit completion update is a
                # safe pause, not proof that the task itself is complete.
                if holder[0].status == "active":
                    holder[0] = replace(holder[0], status="paused")
                    self.store.save(holder[0])
                journal = self.store.journal(record.task_id)
                durable_pause = bool(journal.recoverable_batches()) or (
                    journal.cursor().get("phase") == "seal_pending"
                )
                if self.store.has_uncommitted(record.task_id) and not durable_pause:
                    raise UnsafeCheckpointError(
                        "run ended without a complete turn; automatic resume is unsafe"
                    )
        memory_entry_ids: tuple[str, ...] = ()
        memory_extraction = None
        memory_extraction_error = ""
        if (
            self.memory_extractor is not None
            and isinstance(self.memory, IndexedMemory)
            and holder[0].status in ("completed", "failed")
        ):
            try:
                memory_entry_ids, memory_extraction = self.memory.extract_task(
                    self._memory_evidence(holder[0], state, run_id),
                    self.memory_extractor,
                )
            except Exception as exc:
                # Task completion remains valid when optional extraction fails,
                # but the failure is returned explicitly for the caller/report.
                memory_extraction_error = f"{type(exc).__name__}: {exc}"
        return TaskRunResult(
            task_id=record.task_id,
            run_id=run_id,
            status=holder[0].status,
            record=holder[0],
            state=state,
            events=tuple(captured),
            memory_entry_ids=memory_entry_ids,
            memory_extraction=memory_extraction,
            memory_extraction_error=memory_extraction_error,
        )

    def _memory_evidence(
        self, record: WorkspaceRecord, state: State, run_id: str
    ) -> TaskMemoryEvidence:
        refs = record.checkpoint.message_refs
        if len(refs) != len(state.messages):
            raise ValueError("workspace message refs do not match the final transcript")
        sources = [
            MemoryEvidenceSource(
                ref=ref,
                run_id=ref.split(":", 1)[0],
                kind="message",
                content=json.dumps(
                    encode_message(message), ensure_ascii=False, sort_keys=True
                ),
            )
            for ref, message in zip(refs, state.messages)
        ]
        sources.append(
            MemoryEvidenceSource(
                ref=f"workspace:{record.revision}",
                run_id=run_id,
                kind="workspace",
                content=render_workspace(record),
            )
        )
        sources.extend(
            MemoryEvidenceSource(
                ref=f"artifact:{artifact.path}",
                run_id=run_id,
                kind="artifact",
                content=json.dumps(
                    {
                        "path": artifact.path,
                        "sha256": artifact.sha256,
                        "verification_refs": artifact.verification_refs,
                    },
                    sort_keys=True,
                ),
            )
            for artifact in record.artifacts
        )
        sources.extend(
            MemoryEvidenceSource(
                ref=f"verifier:{index}",
                run_id=run_id,
                kind="verifier",
                content=json.dumps(asdict(result), sort_keys=True),
            )
            for index, result in enumerate(record.verifier_results)
        )
        return TaskMemoryEvidence(
            task_id=record.task_id,
            run_id=run_id,
            source_revision=record.revision,
            outcome=record.status,
            scope=self.memory_scope,
            task=record.task,
            observed_at=datetime.now(timezone.utc).isoformat(),
            sources=tuple(sources),
        )

    def _reconcile_artifacts(self, record: WorkspaceRecord) -> WorkspaceRecord:
        changed: list[str] = []
        artifacts: list[Artifact] = []
        for artifact in record.artifacts:
            path = (self.code_dir / artifact.path).resolve()
            actual = self.store.sha256(path) if path.is_file() else "missing"
            if actual != artifact.sha256:
                changed.append(
                    f"artifact changed and requires verification: {artifact.path}"
                )
                artifacts.append(replace(artifact, sha256=actual, verification_refs=()))
            else:
                artifacts.append(artifact)
        if not changed:
            return record
        revised = replace(
            record,
            revision=record.revision + 1,
            artifacts=tuple(artifacts),
            blockers=(*record.blockers, *changed),
            status="paused",
        )
        self.store.save(revised)
        return revised

    @staticmethod
    def _validate_pending_batches(journal) -> None:
        """Fail closed on incomplete history before recovery can change files."""
        for batch in journal.recoverable_batches():
            assistant = journal.batch_assistant_message(batch["batch_id"])
            actual = tuple(
                (call.id, call.name, dict(call.arguments))
                for call in assistant.tool_calls
            )
            planned = tuple(
                (call.get("id"), call.get("name"), call.get("arguments"))
                for call in batch["calls"]
            )
            if actual != planned:
                raise UnsafeCheckpointError(
                    "durable batch mismatch during resume validation"
                )

    @staticmethod
    def _require_frozen_contract(
        record: WorkspaceRecord, contract: AcceptanceContract
    ) -> None:
        frozen = (
            record.acceptance_contract_id,
            record.acceptance_contract_version,
            record.acceptance_commands,
        )
        supplied = (
            contract.contract_id,
            contract.version,
            contract.required_commands,
        )
        if not frozen[0] or frozen != supplied:
            raise UnsafeCheckpointError(
                "verification contract does not match the frozen task acceptance contract"
            )

    def _require_code_identity(self, record: WorkspaceRecord) -> None:
        expected = record.checkpoint.code_identity
        actual = self._code_identity()
        if expected and expected != actual:
            raise UnsafeCheckpointError(
                f"code directory identity mismatch: checkpoint={expected!r}, current={actual!r}"
            )

    def _code_identity(self) -> str:
        head = subprocess.run(
            ["git", "-C", str(self.code_dir), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        return f"{self.code_dir}|{head or 'no-git-head'}"

    def _provider_fingerprint(self) -> str:
        safe = {
            "id": self.provider.id,
            "api": self.provider.api,
            "model": self.provider.model,
            "base_url": self.provider.base_url,
            "reasoning": self.provider.default_reasoning,
            "default_max_tokens": self.provider.default_max_tokens,
            "context_window": self.provider.context_window,
        }
        return hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()


def _estimate_fixed_overhead(system_prompt: str, tools) -> int:
    chars = len(system_prompt) + len(
        json.dumps(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ],
            sort_keys=True,
        )
    )
    return (chars + 2) // 3 + 32
