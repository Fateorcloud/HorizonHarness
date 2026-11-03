"""Project-owned values persisted by the task workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .snapshot import FileEntry, FileTreeSnapshot


StepStatus = Literal["pending", "in_progress", "done", "blocked"]
TaskStatus = Literal["active", "paused", "completed", "blocked", "failed"]


@dataclass(frozen=True)
class TaskStep:
    id: str
    description: str
    status: StepStatus = "pending"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.description.strip():
            raise ValueError("workspace step id and description must not be empty")
        if self.status not in ("pending", "in_progress", "done", "blocked"):
            raise ValueError(f"invalid workspace step status {self.status!r}")


@dataclass(frozen=True)
class Decision:
    content: str
    rationale: str
    step_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    step_ids: tuple[str, ...] = ()
    verification_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierResult:
    command: str
    passed: bool
    exit_code: int
    evidence_ref: str
    tree_sha256: str = ""
    contract_id: str = ""
    contract_version: int = 0
    verifier_version: str = ""
    task_revision: int = 0
    outcome: str = ""
    acceptance_tree_sha256: str = ""
    verification_exclude_dirs: tuple[str, ...] | None = None
    workspace_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verification_exclude_dirs is not None:
            object.__setattr__(
                self, "verification_exclude_dirs", tuple(self.verification_exclude_dirs)
            )
        object.__setattr__(self, "workspace_changes", tuple(self.workspace_changes))


@dataclass(frozen=True)
class Checkpoint:
    run_id: str = ""
    completed_turn: int = 0
    resumable: bool = False
    messages_path: str = ""
    trace_path: str = ""
    recent_message_refs: tuple[str, ...] = ()
    archive_refs: tuple[str, ...] = ()
    message_refs: tuple[str, ...] = ()
    provider_fingerprint: str = ""
    code_identity: str = ""
    phase: str = "ready"
    next_tool_index: int = 0
    workspace_tree_sha256: str = ""
    snapshot_path: str = ""
    context_manifest_path: str = ""


@dataclass(frozen=True)
class WorkspaceRecord:
    schema_version: int
    task_id: str
    revision: int
    task: str
    constraints: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()
    acceptance_contract_id: str = ""
    acceptance_contract_version: int = 0
    acceptance_commands: tuple[str, ...] = ()
    steps: tuple[TaskStep, ...] = ()
    decisions: tuple[Decision, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    verifier_results: tuple[VerifierResult, ...] = ()
    blockers: tuple[str, ...] = ()
    next_action: str = ""
    status: TaskStatus = "active"
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    parent_task_id: str = ""
    parent_revision: int = -1
    parent_snapshot_sha256: str = ""
    acceptance_dir: str = ""
    acceptance_tree_sha256: str = ""
    acceptance_snapshot: FileTreeSnapshot | None = None
    verification_exclude_dirs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported workspace schema_version {self.schema_version!r}; expected 1"
            )
        if not self.task_id.strip() or not self.task.strip():
            raise ValueError("workspace task_id and task must not be empty")
        if self.revision < 0:
            raise ValueError("workspace revision must not be negative")
        if self.status not in ("active", "paused", "completed", "blocked", "failed"):
            raise ValueError(f"invalid workspace task status {self.status!r}")
        step_ids = {step.id for step in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("workspace step ids must be unique")
        for references in (
            *(decision.step_ids for decision in self.decisions),
            *(artifact.step_ids for artifact in self.artifacts),
        ):
            unknown = next((item for item in references if item not in step_ids), None)
            if unknown is not None:
                raise ValueError(f"unknown workspace step reference {unknown!r}")

    @property
    def progress(self) -> tuple[int, int]:
        return sum(step.status == "done" for step in self.steps), len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_from_dict(value: dict[str, Any]) -> WorkspaceRecord:
    version = value.get("schema_version")
    if version != 1:
        raise ValueError(
            f"unsupported workspace schema_version {version!r}; expected 1"
        )
    checkpoint = value.get("checkpoint", {})
    acceptance_snapshot = value.get("acceptance_snapshot")
    return WorkspaceRecord(
        schema_version=1,
        task_id=str(value["task_id"]),
        revision=int(value["revision"]),
        task=str(value["task"]),
        constraints=tuple(map(str, value.get("constraints", ()))),
        acceptance=tuple(map(str, value.get("acceptance", ()))),
        acceptance_contract_id=str(value.get("acceptance_contract_id", "")),
        acceptance_contract_version=int(value.get("acceptance_contract_version", 0)),
        acceptance_commands=tuple(map(str, value.get("acceptance_commands", ()))),
        steps=tuple(
            TaskStep(
                id=row["id"],
                description=row["description"],
                status=row.get("status", "pending"),
                evidence=tuple(row.get("evidence", ())),
            )
            for row in value.get("steps", ())
        ),
        decisions=tuple(
            Decision(
                content=row["content"],
                rationale=row["rationale"],
                step_ids=tuple(row.get("step_ids", ())),
                evidence=tuple(row.get("evidence", ())),
            )
            for row in value.get("decisions", ())
        ),
        artifacts=tuple(
            Artifact(
                path=row["path"],
                sha256=row["sha256"],
                step_ids=tuple(row.get("step_ids", ())),
                verification_refs=tuple(row.get("verification_refs", ())),
            )
            for row in value.get("artifacts", ())
        ),
        verifier_results=tuple(
            VerifierResult(**row) for row in value.get("verifier_results", ())
        ),
        blockers=tuple(map(str, value.get("blockers", ()))),
        next_action=str(value.get("next_action", "")),
        status=value.get("status", "active"),
        checkpoint=Checkpoint(
            run_id=str(checkpoint.get("run_id", "")),
            completed_turn=int(checkpoint.get("completed_turn", 0)),
            resumable=bool(checkpoint.get("resumable", False)),
            messages_path=str(checkpoint.get("messages_path", "")),
            trace_path=str(checkpoint.get("trace_path", "")),
            recent_message_refs=tuple(checkpoint.get("recent_message_refs", ())),
            archive_refs=tuple(checkpoint.get("archive_refs", ())),
            message_refs=tuple(checkpoint.get("message_refs", ())),
            provider_fingerprint=str(checkpoint.get("provider_fingerprint", "")),
            code_identity=str(checkpoint.get("code_identity", "")),
            phase=str(checkpoint.get("phase", "ready")),
            next_tool_index=int(checkpoint.get("next_tool_index", 0)),
            workspace_tree_sha256=str(checkpoint.get("workspace_tree_sha256", "")),
            snapshot_path=str(checkpoint.get("snapshot_path", "")),
            context_manifest_path=str(checkpoint.get("context_manifest_path", "")),
        ),
        parent_task_id=str(value.get("parent_task_id", "")),
        parent_revision=int(value.get("parent_revision", -1)),
        parent_snapshot_sha256=str(value.get("parent_snapshot_sha256", "")),
        acceptance_dir=str(value.get("acceptance_dir", "")),
        acceptance_tree_sha256=str(value.get("acceptance_tree_sha256", "")),
        acceptance_snapshot=(
            FileTreeSnapshot(
                root=acceptance_snapshot["root"],
                entries=tuple(
                    FileEntry(**row) for row in acceptance_snapshot["entries"]
                ),
                tree_sha256=acceptance_snapshot["tree_sha256"],
            )
            if acceptance_snapshot is not None
            else None
        ),
        verification_exclude_dirs=(
            tuple(value["verification_exclude_dirs"])
            if value.get("verification_exclude_dirs") is not None
            else None
        ),
    )
