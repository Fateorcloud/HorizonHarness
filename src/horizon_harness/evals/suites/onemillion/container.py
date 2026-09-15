"""OneMillion-Bench container half: the functions a suite supplies.

OneMillion-Bench is a *rubric-graded Q&A* benchmark, not an agent-in-a-repo
benchmark, so its mapping onto the generic framework (`SwebenchSuite` is the
reference) is:

- **Generation** uses the ``AgentSpec`` workflow axis. ``single`` (default) is
  one tool-free model turn; recipe workflows (``reflection`` / ``parallel`` /
  …) use the same loop through ``AgentSession``. Because ``extract_result``
  cannot see the agent's messages (only the workspace), the final answer is persisted to
  ``model_response.txt`` — the analog of SWE-bench writing a ``git diff`` to the
  filesystem.
- **Scoring** is the in-environment ``evaluate`` hook: the host stages the
  case's weighted rubrics via ``eval_inputs`` (gold the agent must not see), and
  this hook calls a *judge* model to grade the response against them,
  reproducing the upstream ``omb`` rubric scoring (see ``grading``). The verdict
  is merged into ``result.json``.

The judge is configured from the environment so the same code grades in-process
(`LocalProcessBackend`) or in a container: ``HORIZON_JUDGE_MODEL`` /
``HORIZON_JUDGE_AUTH_TOKEN`` / ``HORIZON_JUDGE_BASE_URL`` /
``HORIZON_JUDGE_API_KIND``, each falling back to the generator's
``OPENAI_*`` value. Secrets stay in env (never in the staged ``eval.json``).

It imports only the standard library and the installed wheel (the ``llm`` layer,
``llm_agent``, ``messages``, ``evals.protocols``, and the sibling ``grading``
module), so it runs in any eval environment with no copied files.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from horizon_harness.llm import (
    LLMRequest,
    Provider,
    complete_with_retry,
    llm_message,
)
from horizon_harness.llm.env import JUDGE_ENV, OPENAI_ENV, provider_from_env
from horizon_harness.messages import text_of
from horizon_harness.spec import AgentSpec
from horizon_harness.state import State

from .grading import (
    build_grading_prompt,
    convert_scores,
    parse_grading_response,
    score_summary,
)

# Where the tool-free generator persists its answer so ``extract_result`` (which
# only sees the workspace, never the agent state) can collect it.
RESPONSE_FILENAME = "model_response.txt"
AGENT_NAME = "onemillion_agent"
AGENT_ROLE = "Answer the professional question directly and completely."
# Mirrors the upstream omb generation system prompt: direct, complete answers,
# never asking the user for more information.
AGENT_SYSTEM_PROMPT = (
    "你是一个专业的问题解答助手。你必须直接、完整地回答问题，"
    "禁止向用户提问或要求用户提供更多信息。请基于问题中已有的信息，"
    "给出准确、详细、结构清晰的答案。"
)

# The judge provider's env contract (`JUDGE_*`, falling back to `OPENAI_*`) and
# its builder live in `horizon_harness.llm.env`; `judge_provider_from_env` below
# is a thin wrapper.
JUDGE_TIMEOUT_S = 600.0


# --------------------------------------------------------------------------- #
# Generation (tool-free agent + answer capture)
# --------------------------------------------------------------------------- #
def agent_spec() -> AgentSpec:
    """Tool-free generation identity; the generic runner owns Recipes."""

    return AgentSpec(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        capabilities=(),
    )


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """The model-visible task: the case's (optional system) prompt + question."""

    del workdir  # the generator answers in chat; no filesystem context needed
    prompt = _first_str(instance, "prompt", "Prompt", "question")
    system_prompt = _first_str(instance, "system_prompt", "System_Prompt")
    parts: list[str] = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
        parts.append("")
    parts.append(prompt.strip())
    return "\n".join(parts).strip()


def extract_result(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    state: State | None = None,
) -> dict[str, Any]:
    """Collect the generated answer from the completed runtime State."""

    del context
    response = _state_response(state) or _read_response(Path(workspace))
    if response:
        (Path(workspace) / RESPONSE_FILENAME).write_text(response, encoding="utf-8")
    return {
        "model_response": response,
        "case_id": instance.get("case_id"),
        "instance_id": instance.get("instance_id"),
    }


# --------------------------------------------------------------------------- #
# In-environment scoring (the rubric judge)
# --------------------------------------------------------------------------- #
def evaluate(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade the answer against the host-staged rubrics with a judge model.

    The host stages ``{prompt, rubrics, human_scores}`` under EVAL_KEY; the
    generic runner threads it in as ``context["eval"]`` and only calls this hook
    when that gold is present. Reproduces the upstream ``omb`` rubric scoring and
    returns a self-contained verdict merged into ``result.json``.
    """

    gold = dict((context or {}).get("eval") or {})
    rubrics = list(gold.get("rubrics") or [])
    if not rubrics:
        return {"scored": False, "status": "no_rubrics"}

    prompt = str(gold.get("prompt") or _first_str(instance, "prompt", "Prompt"))
    human_scores = dict(gold.get("human_scores") or {})
    model_response = _read_response(Path(workspace))

    judge = judge_provider_from_env()
    grading_prompt = build_grading_prompt(prompt, model_response, rubrics, human_scores)
    judge_text = _judge_complete(judge, grading_prompt)

    raw_results = parse_grading_response(judge_text, rubrics)
    final_scores = convert_scores(raw_results, rubrics)
    summary = score_summary(final_scores, rubrics)

    return {
        "scored": True,
        "judge_model": judge.model,
        "model_response_chars": len(model_response),
        "total_score": summary["total_score"],
        "max_score": summary["max_score"],
        "min_score": summary["min_score"],
        "score": summary["accuracy"],
        "rubric_scores": {str(k): v for k, v in final_scores.items()},
        "judge_verdicts": {str(k): v for k, v in raw_results.items()},
        # The judge model's full input/output, so a verdict is auditable, not
        # just the parsed scores.
        "judge_prompt": grading_prompt,
        "judge_raw_response": judge_text,
    }


def judge_provider_from_env(env: Mapping[str, str] | None = None) -> Provider:
    """Build the judge `Provider` from JUDGE_* env, falling back to OPENAI_*.

    Thin wrapper over `horizon_harness.llm.env.provider_from_env`: the judge
    reads `JUDGE_*`, falls back to `OPENAI_*`, and grades at ``temperature=0.0``.
    """
    return provider_from_env(
        JUDGE_ENV,
        fallback=OPENAI_ENV,
        env=env,
        default_temperature=0.0,
        label="the OneMillion-Bench judge",
    )


def _judge_complete(provider: Provider, grading_prompt: str) -> str:
    """One blocking judge call (no tools); transient throttling is retried."""

    request = LLMRequest(
        provider=provider,
        messages=[llm_message("user", grading_prompt)],
        tools=[],
        system_prompt=None,
        temperature=0.0,
        timeout_seconds=JUDGE_TIMEOUT_S,
    )
    return complete_with_retry(request).text


def _read_response(workspace: Path) -> str:
    try:
        return (workspace / RESPONSE_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _state_response(state: State | None) -> str:
    if state is None:
        return ""
    for message in reversed(state.messages):
        if message.sender == AGENT_NAME and message.kind == "final":
            return text_of(message.content)
    return ""


def _first_str(instance: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = instance.get(key)
        if value:
            return str(value)
    return ""
