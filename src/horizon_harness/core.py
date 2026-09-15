"""Balanced message runtime for HorizonHarness.

The core model is small::

    Agent + Message + State + build_context_view() + run()

`run()` drives one agent as a generator: each turn, it builds a context view
of the visible messages, calls `agent.generate(...)`, records request/response
trace events, and dispatches any returned tool calls. The loop stops on the
first turn whose output is the agent's `final` message, or when `max_turns`
is exhausted (truncated runs surface as `agent_end(reason="max_turns")` so
traces can distinguish "agent decided to stop" from "ran out of budget").

`Agent.run(task)` is a convenience wrapper that initializes the state with a
task message and calls `run(self, state, ...)`, returning `(state, events)` so
the caller can stream events and still inspect the populated state.

Multi-agent flows are expressed as a parent agent that delegates through
`tools.task_tool([b, c, d])`: the parent picks one sub-agent via the
`subagent_type` enum and the chosen sub-agent's final message comes back as
the tool result.
"""

from __future__ import annotations

import dataclasses
import uuid

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generator, Iterator

if TYPE_CHECKING:
    from .workspace.journal import ExecutionJournal

from .compression import maybe_compress_context
from .context_view import (
    ContextPolicy,
    ContextView,
    build_context_view,
)
from .hooks import HookContext, HookMap, HookPoint, fire_hooks
from .llm import llm_message, messages_to_llm_messages
from .messages import (
    AssistantMessage,
    ContentInput,
    Message,
    ToolCallBlock,
    ToolResultBlock,
    message_tool_calls,
    tool_results_message,
)
from .protocols import (
    AgentEndEvent,
    AgentEndReason,
    AgentStartEvent,
    Event,
    ModelRequestEvent,
    ModelResponseEvent,
    MessageEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .state import State
from .tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result
from .tools import tool_arguments_hash

if TYPE_CHECKING:
    from .llm.provider import Provider


# An Agent's `generate` produces the next message from the visible context.
# It takes no other arguments: name/role/tools/etc. are closed over at the
# point where the function is built (see `llm_agent.py` and the test fakes).
GenerateFn = Callable[[list[Message]], Message]
DurableRecoveryPreparationFn = Callable[[], None]
# A state initializer builds the initial `State` for `Agent.run` from a task.
# The default (see `Agent._default_init_state`) records a single task message; a
# higher layer can supply one that records extra context up front — e.g. the
# skills layer records a skills menu and any mentioned skill bodies before the
# task. Kept a plain callable taking the agent explicitly so `core` never
# imports a higher layer: the dependency always points inward (skills imports
# core, not the reverse).
StateInitFn = Callable[["Agent", ContentInput], State]
TraceStateFn = Callable[["Agent", State], State | None]
ContextBuilderFn = Callable[[str, State, ContextPolicy], ContextView]


@dataclass(frozen=True)
class ToolDispatchOutcome:
    completed: bool
    terminated: bool = False


@dataclass
class Agent:
    name: str
    generate: GenerateFn
    role: str = ""
    # `tools` is a tuple to make explicit that tools are bound at construction:
    # `run()` snapshots them once per call, so mutating this attribute mid-run
    # has no effect on the in-flight loop.
    tools: tuple[AgentTool, ...] = ()
    context_policy: ContextPolicy | None = None
    # Optional owner of the complete model-visible projection. When present it
    # replaces the default compression + visibility path for this request.
    context_builder: ContextBuilderFn | None = None
    # Lifecycle hooks consulted during `run()` — observe, block a PRE_TOOL_USE
    # call, or emit messages (append-only; never edit). A bare
    # `{HookPoint: [hook, ...]}` dict, like `tools` is a bare tuple. Sub-agents
    # carry their own, so `task_tool` delegation inherits hooks for free.
    hooks: HookMap = field(default_factory=dict)
    # Mirror of the system prompt `generate` closes over (see `llm_agent.py`),
    # so `run()` can record it in the request trace. Empty means none was sent.
    system_prompt: str = ""
    # Set only by `make_llm_agent`. Programmatic facades leave it unset so their
    # deterministic `generate` calls do not look like model calls in traces.
    llm_provider: Provider | None = None
    # Optional durable event sink. The core loop remains the sole state
    # transition owner; the sink only records committed event metadata.
    journal: object | None = None
    # TaskSession enables durable startup phases after validating the persisted
    # plan and binding tool-specific adapters. A plain event sink need not
    # implement them.
    recover_durable_batches: bool = False
    # Tool-specific reconciliation stays behind this adapter; core alone
    # decides when the recovery phase may advance it.
    prepare_durable_recovery: DurableRecoveryPreparationFn | None = None
    # How `run` builds the initial `State`. `None` uses `_default_init_state`
    # (just the task message); the skills layer installs one that records the
    # skills menu first, so a bare `agent.run(task)` is skills-aware.
    init_state: StateInitFn | None = None
    # Optional final trace projection. A workflow facade runs many sub-agents
    # behind one outer `generate`; this lets it export a richer final trace
    # without changing the message loop. None means "trace the run state as-is".
    compose_trace_state: TraceStateFn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            self.tools = tuple(self.tools)

    def run(
        self,
        task: ContentInput,
        *,
        max_turns: int = 10,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Drive this agent on `task` until it emits a final message.

        `task` is `str` or a sequence of content blocks (`ContentInput`), so a
        multimodal task (text + `ImageBlock`) is seeded the same way as plain
        text — the message layer normalizes both to content blocks.

        The initial `State` comes from `self.init_state` when set, else
        `_default_init_state`. A custom initializer is how a layer like skills
        makes a *bare* agent skills-aware: `agent.run(task)` advertises the menu
        and injects bodies because the initializer recorded them — no separate
        run path.

        Returns `(state, events)`. Caller iterates `events` to advance the
        loop and inspects `state` for the message/event history. Callers
        that need to prepend extra messages (e.g. a sub-agent context
        prelude) can `state.record(...)` them after this call and before
        starting to iterate `events`.
        """
        state = (
            self.init_state(self, task)
            if self.init_state is not None
            else self._default_init_state(task)
        )
        events = run(
            self,
            state,
            max_turns=max_turns,
            abort=abort,
        )
        return state, events

    def trace_state(self, state: State) -> State:
        """Return the state that should be used for final trace export."""

        if self.compose_trace_state is None:
            return state
        composed = self.compose_trace_state(self, state)
        return composed if composed is not None else state

    def continue_session(
        self,
        state: State,
        followup: ContentInput,
        *,
        max_turns: int = 10,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Continue an existing `state` with a follow-up user message.

        Unlike `run`, which seeds a fresh `State`, this appends `followup`
        to the conversation already on `state` and drives the loop over it.
        Both the session (the prior messages stay visible) and the trace
        (events keep accumulating on the same `state.events`) carry over — so
        a `run_trace_from_state(state)` after this spans both turns.

        This is the seam for swapping the agent between turns while keeping
        continuity — e.g. rebuild with a different `reasoning` effort and
        continue the same `state`. Build the new agent with the SAME `name`,
        since context visibility/routing is keyed on the agent name; a
        mismatched name would change which prior messages are visible.

        This method does not persist checkpoints and cannot recover an
        interrupted process. It only continues an in-memory state owned by the
        caller.

        Returns `(state, events)` like `run`, so callers can treat the two
        interchangeably (the returned `state` is the one passed in).
        """
        state.send("task", "user", self.name, followup)
        events = run(
            self,
            state,
            max_turns=max_turns,
            abort=abort,
        )
        return state, events

    def _default_init_state(self, task: ContentInput) -> State:
        """Initialize a fresh `State` with just the task message."""

        state = State(task=task)
        state.send("task", "user", self.name, task)
        return state


def run(
    agent: Agent,
    state: State,
    *,
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
) -> Iterator[Event]:
    """Run one agent as a generator until it emits `final` or hits `max_turns`.

    Each yielded `Event` is also recorded in `state`.
    """
    name = agent.name
    journal = getattr(agent, "journal", None)

    journal_events = ()
    if journal is not None:
        journal_events = journal.events()
    if journal is not None and not any(
        kind == "MessageEvent" for _, kind, _ in journal_events
    ):
        # TaskSession seeds the initial task Message before attaching the
        # durable sink. Import that already-created canonical fact exactly
        # once; resumed runs already have Journal history and skip this path.
        for existing in state.events:
            journal.append(type(existing).__name__, dataclasses.asdict(existing))

    def record(event: Event) -> Event:
        recorded = state.prepare_event(event)
        if journal is not None:
            journal.append(
                type(recorded).__name__,
                dataclasses.asdict(recorded),
            )
        return state.apply_prepared_event(recorded)

    def record_batch(events: tuple[Event, ...]) -> tuple[Event, ...]:
        prepared = tuple(
            dataclasses.replace(
                state.prepare_event(event), index=len(state.events) + offset
            )
            for offset, event in enumerate(events)
        )
        if journal is not None:
            journal.append_many(
                (type(event).__name__, dataclasses.asdict(event)) for event in prepared
            )
        for event in prepared:
            state.apply_prepared_event(event)
        return prepared

    tool_by_name = {tool.name: tool for tool in agent.tools}
    hooks = agent.hooks
    api = agent.llm_provider.api if agent.llm_provider is not None else ""

    def session_hook(point: HookPoint) -> Iterator[Event]:
        _, hook_events = fire_hooks(
            hooks, HookContext(point=point, agent=name, state=state), state
        )
        yield from hook_events

    yield record(AgentStartEvent(agent=name, system_prompt=agent.system_prompt))
    yield from session_hook(HookPoint.SESSION_START)
    final_emitted = False
    # Default outcome; overridden when the loop breaks on `final` or terminate.
    end_reason: AgentEndReason = "max_turns"
    stop_before_model = False
    if journal is not None and agent.recover_durable_batches:
        pending = journal.recoverable_batches()
        cursor = journal.cursor()
        if len(pending) > 1:
            raise RuntimeError("multiple pending tool batches cannot be recovered")
        if pending:
            batch = pending[0]
            batch_id = batch["batch_id"]
            if cursor.get("batch_id") != batch_id:
                raise RuntimeError("execution cursor does not identify pending batch")
            assistant = journal.batch_assistant_message(batch_id)
            if abort():
                end_reason = "abort"
                stop_before_model = True
            else:
                if agent.prepare_durable_recovery is not None:
                    agent.prepare_durable_recovery()
                recovered = yield from dispatch_tool_calls(
                    assistant,
                    tool_by_name,
                    state,
                    abort=abort,
                    hooks=hooks,
                    journal=journal,
                    batch_id=batch_id,
                )
                if not recovered.completed:
                    end_reason = "abort"
                    stop_before_model = True
                else:
                    journal.set_cursor(turn_no=batch["turn_no"], phase="ready")
                    yield record(
                        TurnEndEvent(agent=name, terminated=recovered.terminated)
                    )
                    if recovered.terminated:
                        end_reason = "tool_terminate"
                        stop_before_model = True
        elif cursor.get("phase") == "tools_pending":
            raise RuntimeError("execution cursor references a missing pending batch")
        elif cursor.get("phase") == "seal_pending":
            terminated = journal.turn_has_terminating_batch(cursor["turn_no"])
            journal.set_cursor(turn_no=cursor["turn_no"], phase="ready")
            yield record(TurnEndEvent(agent=name, terminated=terminated))
            if terminated:
                end_reason = "tool_terminate"
                stop_before_model = True

    for _ in range(0 if stop_before_model else max_turns):
        if abort():
            end_reason = "abort"
            break
        if journal is not None:
            journal.set_cursor(
                turn_no=len([e for e in state.events if isinstance(e, TurnStartEvent)])
                + 1,
                phase="model_pending",
            )
        yield record(TurnStartEvent(agent=name))

        policy = agent.context_policy or ContextPolicy()
        if agent.context_builder is None:
            for compression_event in maybe_compress_context(agent, state, policy):
                yield compression_event
            context = build_context_view(
                name,
                state.active_context_messages(),
                policy=policy,
            )
        else:
            context = agent.context_builder(name, state, policy)
        visible = list(context.messages)
        # Match make_llm_agent / provider wire shape (no routing headers).
        # Prepend the agent's system prompt as a leading system message so the
        # recorded request mirrors what actually crosses the wire (adapters
        # send it as the first system entry); without this the trace would
        # silently drop the system prompt that `generate` passes via
        # `LLMRequest.system_prompt`. `visible_count` stays the conversation
        # count; `llm_message_count` reflects the full payload including it.
        llm_payload = messages_to_llm_messages(visible, with_header=False)
        if agent.system_prompt:
            llm_payload = [llm_message("system", agent.system_prompt), *llm_payload]

        yield record(
            ModelRequestEvent(
                agent=name,
                api=api,
                visible_count=len(visible),
                llm_message_count=len(llm_payload),
                context_view=context.as_dict(),
                tools=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tool_by_name.values()
                ],
                llm_payload=llm_payload,
            )
        )

        output = agent.generate(visible)
        output_tool_calls = message_tool_calls(output)
        turn_no = len([e for e in state.events if isinstance(e, TurnStartEvent)])
        decision_events = (
            ModelResponseEvent(
                agent=name,
                api=api,
                output_kind=output.kind,
                target=output.target,
                tool_call_count=len(output_tool_calls),
                usage=output.usage if isinstance(output, AssistantMessage) else None,
                model=output.model if isinstance(output, AssistantMessage) else "",
            ),
            MessageEvent(message=output),
        )
        if journal is not None:
            batch_id = str(uuid.uuid4()) if output_tool_calls else None
            calls = [
                {
                    "id": call.id,
                    "execution_step_id": f"{batch_id}:{index}",
                    "name": call.name,
                    "arguments": dict(call.arguments),
                    "args_hash": tool_arguments_hash(dict(call.arguments)),
                }
                for index, call in enumerate(output_tool_calls)
            ]
            prepared = tuple(
                dataclasses.replace(
                    state.prepare_event(event), index=len(state.events) + offset
                )
                for offset, event in enumerate(decision_events)
            )
            journal.commit_model_decision(
                events=[
                    (type(event).__name__, dataclasses.asdict(event))
                    for event in prepared
                ],
                batch_id=batch_id,
                turn_no=turn_no,
                calls=calls,
                assistant_message=dataclasses.asdict(output),
            )
            for event in prepared:
                yield state.apply_prepared_event(event)
        else:
            yield from record_batch(decision_events)

        if output.sender == name and output.kind == "final":
            final_emitted = True

        if tool_by_name and output_tool_calls:
            dispatched = yield from dispatch_tool_calls(
                output,
                tool_by_name,
                state,
                abort=abort,
                hooks=hooks,
                journal=journal,
                batch_id=batch_id if journal is not None else None,
            )
            if not dispatched.completed:
                end_reason = "abort"
                break
            if dispatched.terminated:
                yield record(TurnEndEvent(agent=name, terminated=True))
                end_reason = "tool_terminate"
                break

        if journal is not None:
            journal.set_cursor(
                turn_no=len([e for e in state.events if isinstance(e, TurnStartEvent)]),
                phase="ready",
            )
        yield record(TurnEndEvent(agent=name))

        if final_emitted:
            end_reason = "done"
            break

    # Single exit: SESSION_END then agent_end, whatever stopped the loop.
    yield from session_hook(HookPoint.SESSION_END)
    yield record(AgentEndEvent(reason=end_reason))


def dispatch_tool_calls(
    assistant_msg: Message,
    tools: dict[str, AgentTool],
    state: State,
    *,
    abort: AbortFlag = lambda: False,
    max_concurrency: int = 8,
    hooks: HookMap | None = None,
    journal: ExecutionJournal | None = None,
    batch_id: str | None = None,
) -> Generator[Event, None, ToolDispatchOutcome]:
    """Run assistant tool calls and append deterministic tool-result messages.

    Before any call reaches the thread pool, each is run through the
    `PRE_TOOL_USE` hook gate — synchronously, in this generator, never in a
    worker thread (hooks emit events and `state.record_event` is not
    thread-safe; the pool runs only `tool.execute`). A hook can **block** a
    call: it never executes, and a synthesized error result is added so the
    model can self-correct next turn, exactly like a tool that raised.
    """
    tool_calls = message_tool_calls(assistant_msg)
    if not tool_calls:
        return ToolDispatchOutcome(completed=True)

    hooks = hooks or {}
    target = assistant_msg.sender or "agent"
    batch_id = batch_id or str(uuid.uuid4())
    if journal is not None and journal.batch(batch_id) is None:
        journal.plan_batch(
            batch_id=batch_id,
            turn_no=journal.cursor()["turn_no"],
            calls=[
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in tool_calls
            ],
            assistant_message=dataclasses.asdict(assistant_msg),
        )

    def record_tool(event: Event) -> Event:
        stamped = state.prepare_event(event)
        if journal is not None:
            journal.append(type(stamped).__name__, dataclasses.asdict(stamped))
        return state.apply_prepared_event(stamped)

    if journal is not None:
        journal.materialize_terminal_batch_results(batch_id)
        results = dict(journal.decoded_batch_results(batch_id))
        terminated = any(result.terminate for result in results.values())
        durable_batch = journal.batch(batch_id)
        if durable_batch is None:
            raise ValueError(f"missing durable batch: {batch_id}")
        planned_by_id = {call["id"]: call for call in durable_batch["calls"]}

        for call_index, tool_call in enumerate(tool_calls):
            if tool_call.id in results:
                continue
            if abort():
                return ToolDispatchOutcome(completed=False, terminated=terminated)
            yield record_tool(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                )
            )
            decision, hook_events = fire_hooks(
                hooks,
                HookContext(
                    point=HookPoint.PRE_TOOL_USE,
                    agent=target,
                    state=state,
                    tool_call=tool_call,
                ),
                state,
            )
            yield from hook_events
            if decision.block_reason:
                result = text_result(decision.block_reason, is_error=True)
            else:
                tool = tools.get(tool_call.name)
                if tool is None:
                    raise ValueError(f"unsupported durable tool: {tool_call.name}")
                planned = planned_by_id.get(tool_call.id)
                if planned is None:
                    raise ValueError(
                        f"durable batch has no planned call: {tool_call.id}"
                    )
                execution_id = (
                    planned.get("execution_step_id", tool_call.id)
                    if tool_call.name == "edit"
                    else tool_call.id
                )
                updates: list[ToolResult] = []

                def on_update(partial: ToolResult) -> None:
                    updates.append(partial)

                try:
                    result = tool.execute(
                        execution_id,
                        dict(tool_call.arguments),
                        abort,
                        on_update,
                    )
                except Exception as exc:
                    result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)
                for partial in updates:
                    yield record_tool(
                        ToolExecutionUpdateEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            partial=partial,
                        )
                    )
            results[tool_call.id] = result
            terminated = terminated or result.terminate
            yield record_tool(
                ToolExecutionEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    is_error=result.is_error,
                    terminate=result.terminate,
                )
            )
            journal.record_batch_tool_result(batch_id, call_index, tool_call.id, result)
            journal.set_cursor(
                turn_no=journal.cursor()["turn_no"],
                phase="tools_pending",
                next_tool_index=call_index + 1,
                batch_id=batch_id,
            )

        bundle = _tool_results_bundle(tool_calls, results, target=target)
        bundle_event = state.prepare_event(MessageEvent(message=bundle))
        journal.commit_batch_bundle(batch_id, dataclasses.asdict(bundle_event))
        yield state.apply_prepared_event(bundle_event)
        yield from _post_tool_use_events(tool_calls, hooks, target, state)
        return ToolDispatchOutcome(completed=True, terminated=terminated)

    for tool_call in tool_calls:
        yield record_tool(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            )
        )

    # PRE_TOOL_USE gate (sequential, before the pool). A hook can only block:
    # a blocked call skips the pool and seeds `results` with its error directly
    # (the model self-corrects next turn); the rest go to `effective`.
    effective: list[ToolCallBlock] = []
    results: dict[str, ToolResult] = {}
    for tool_call in tool_calls:
        decision, hook_events = fire_hooks(
            hooks,
            HookContext(
                point=HookPoint.PRE_TOOL_USE,
                agent=target,
                state=state,
                tool_call=tool_call,
            ),
            state,
        )
        yield from hook_events
        if decision.block_reason:
            results[tool_call.id] = text_result(decision.block_reason, is_error=True)
            yield record_tool(
                ToolExecutionEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    is_error=True,
                    terminate=False,
                )
            )
            continue
        effective.append(tool_call)

    sequential = any(
        (tool := tools.get(call.name)) is not None
        and tool.execution_mode == "sequential"
        for call in effective
    )
    workers = 1 if sequential else min(max_concurrency, len(effective))

    update_buffers: dict[str, list[ToolResult]] = {call.id: [] for call in effective}

    def make_on_update(call_id: str) -> ToolUpdateFn:
        def on_update(partial: ToolResult) -> None:
            update_buffers[call_id].append(partial)

        return on_update

    if effective and journal is None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_tool_call = {
                pool.submit(
                    _execute_one,
                    call,
                    tools,
                    abort,
                    make_on_update(call.id),
                ): call
                for call in effective
            }

            for future in as_completed(future_to_tool_call):
                call = future_to_tool_call[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = text_result(f"{type(exc).__name__}: {exc}", is_error=True)
                results[call.id] = result

                for partial in update_buffers[call.id]:
                    yield record_tool(
                        ToolExecutionUpdateEvent(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            partial=partial,
                        )
                    )
                yield record_tool(
                    ToolExecutionEndEvent(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=result.is_error,
                        terminate=result.terminate,
                    )
                )

    # `results` now holds every call by its original id (blocked ones seeded in
    # the gate, executed ones filled by the pool). The bundle is built over the
    # original `tool_calls` so each result lands under the id and name the model
    # emitted, whether or not a hook blocked or rewrote the call.
    bundle = _tool_results_bundle(tool_calls, results, target=target)
    bundle_event = state.prepare_event(MessageEvent(message=bundle))
    if journal is not None:
        journal.commit_batch_bundle(batch_id, dataclasses.asdict(bundle_event))
    yield state.apply_prepared_event(bundle_event)

    # POST_TOOL_USE fires only after the provider-required tool-result bundle is
    # in the transcript. Hooks may append reminder/context messages here without
    # orphaning the assistant tool_call -> user tool_result pair.
    yield from _post_tool_use_events(tool_calls, hooks, target, state)
    return ToolDispatchOutcome(
        completed=True,
        terminated=any(result.terminate for result in results.values()),
    )


def _tool_results_bundle(
    tool_calls: tuple[ToolCallBlock, ...],
    results: dict[str, ToolResult],
    *,
    target: str,
) -> Message:
    return tool_results_message(
        [
            ToolResultBlock(
                tool_call_id=call.id,
                tool_name=call.name,
                content=tuple(results[call.id].content),
                is_error=results[call.id].is_error,
            )
            for call in tool_calls
        ],
        target=target,
        sidecar={
            "details": {call.id: results[call.id].details for call in tool_calls},
        },
    )


def _post_tool_use_events(
    tool_calls: tuple[ToolCallBlock, ...],
    hooks: HookMap,
    target: str,
    state: State,
) -> Iterator[Event]:
    for tool_call in tool_calls:
        _, hook_events = fire_hooks(
            hooks,
            HookContext(
                point=HookPoint.POST_TOOL_USE,
                agent=target,
                state=state,
                tool_call=tool_call,
            ),
            state,
        )
        yield from hook_events


def _execute_one(
    tool_call: ToolCallBlock,
    tools: dict[str, AgentTool],
    abort: AbortFlag,
    on_update: ToolUpdateFn | None,
) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return text_result(f"Tool {tool_call.name!r} not found", is_error=True)

    def run_tool() -> ToolResult:
        try:
            return tool.execute(
                tool_call.id, dict(tool_call.arguments), abort, on_update
            )
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    if tool.timeout_seconds is None:
        return run_tool()

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(run_tool)
        try:
            return future.result(timeout=tool.timeout_seconds)
        except FuturesTimeoutError:
            return text_result(
                f"Tool {tool_call.name!r} timed out after {tool.timeout_seconds}s",
                is_error=True,
            )
    finally:
        pool.shutdown(wait=False)
