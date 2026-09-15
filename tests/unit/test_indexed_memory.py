from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from horizon_harness.core import Agent
from horizon_harness.memory import (
    MemoryEvidenceSource,
    ModelMemoryExtractor,
    TaskMemoryEvidence,
    IndexedMemory,
    MemoryCandidate,
    MemoryContext,
    tokenize,
)
from horizon_harness.llm import FAKE_PROVIDER, LLMResponse, TextBlock, TokenUsage
from horizon_harness.messages import (
    ToolCallBlock,
    assistant_message,
    text_of,
    tool_results_of,
)
from horizon_harness.tools.bash import make_bash_tool


class FakeEmbedder:
    model_id = "fake"
    revision = "r1"
    dimension = 2

    def encode(self, texts):
        result = []
        for text in texts:
            lowered = text.lower()
            result.append([1.0, 0.0] if "alpha" in lowered else [0.0, 1.0])
        return result


def candidate(**changes):
    values = {
        "type": "fact",
        "content": "alpha uses run_ci",
        "scope": "repo",
        "task_id": "d",
        "run_id": "run-d",
        "source_ref": "run-d:3",
        "observed_at": "2026-09-11T00:00:00Z",
        "source_revision": 2,
    }
    values.update(changes)
    return MemoryCandidate(**values)


class IndexedMemoryTests(unittest.TestCase):
    def make_memory(self, tmp, **kwargs):
        memory = IndexedMemory(Path(tmp) / "memory.sqlite3", **kwargs)
        memory.register_sources("d", "run-d", ("run-d:3", "artifact:test"))
        return memory

    def test_source_validation_idempotent_finish_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            with self.assertRaisesRegex(ValueError, "source reference"):
                memory.finish_task(
                    task_id="d",
                    source_revision=2,
                    outcome="completed",
                    candidates=(candidate(source_ref="wrong"),),
                )
            ids = memory.finish_task(
                task_id="d",
                source_revision=2,
                outcome="completed",
                candidates=(candidate(),),
            )
            again = memory.finish_task(
                task_id="d", source_revision=2, outcome="completed", candidates=()
            )
            self.assertEqual(ids, again)
            reopened = IndexedMemory(Path(tmp) / "memory.sqlite3")
            self.assertEqual(reopened.search("runCI", "repo")[0].entry.id, ids[0])

    def test_experience_requires_condition_method_and_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            with self.assertRaisesRegex(ValueError, "condition"):
                memory.finish_task(
                    task_id="d",
                    source_revision=3,
                    outcome="failed",
                    candidates=(candidate(type="experience"),),
                )

    def test_candidate_rejects_credential_shaped_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            with self.assertRaisesRegex(ValueError, "credential"):
                memory.finish_task(
                    task_id="d",
                    source_revision=4,
                    outcome="failed",
                    candidates=(candidate(content="api_key=sk-secretsecret123"),),
                )

    def test_scope_supersession_and_conflict_marking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            first = memory.finish_task(
                task_id="d",
                source_revision=1,
                outcome="completed",
                candidates=(candidate(),),
            )[0]
            second = memory.finish_task(
                task_id="d",
                source_revision=2,
                outcome="completed",
                candidates=(candidate(content="alpha new", supersedes=first),),
            )[0]
            hits = memory.search("alpha", "repo")
            self.assertEqual([hit.entry.id for hit in hits], [second])
            self.assertEqual(memory.search("alpha", "other"), ())
            with self.assertRaisesRegex(ValueError, "another scope"):
                memory.finish_task(
                    task_id="d",
                    source_revision=3,
                    outcome="completed",
                    candidates=(candidate(scope="other", supersedes=second),),
                )

    def test_unresolved_same_scope_conflict_is_returned_as_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            memory.finish_task(
                task_id="d",
                source_revision=9,
                outcome="completed",
                candidates=(
                    candidate(content="alpha chooses one", conflict_group="choice"),
                    candidate(
                        content="alpha chooses two",
                        source_ref="artifact:test",
                        conflict_group="choice",
                    ),
                ),
            )
            self.assertTrue(all(hit.conflict for hit in memory.search("alpha", "repo")))

    def test_hybrid_rrf_is_deterministic_and_revision_mismatch_is_explicit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp, mode="hybrid", embedder=FakeEmbedder())
            memory.finish_task(
                task_id="d",
                source_revision=2,
                outcome="completed",
                candidates=(
                    candidate(),
                    candidate(content="beta only", source_ref="artifact:test"),
                ),
            )
            hits = memory.search("alpha", "repo")
            self.assertEqual(hits[0].entry.content, "alpha uses run_ci")
            mismatch = FakeEmbedder()
            mismatch.revision = "r2"
            reopened = IndexedMemory(
                Path(tmp) / "memory.sqlite3", mode="hybrid", embedder=mismatch
            )
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                reopened.search("alpha", "repo")

    def test_vector_rebuild_reuses_matching_content_hash_and_revision(self) -> None:
        class CountingEmbedder(FakeEmbedder):
            def __init__(self):
                self.calls = 0

            def encode(self, texts):
                self.calls += 1
                return super().encode(texts)

        with tempfile.TemporaryDirectory() as tmp:
            embedder = CountingEmbedder()
            memory = self.make_memory(tmp, mode="hybrid", embedder=embedder)
            memory.finish_task(
                task_id="d",
                source_revision=2,
                outcome="completed",
                candidates=(candidate(),),
            )
            calls = embedder.calls
            self.assertEqual(memory.rebuild_vectors(), 0)
            self.assertEqual(embedder.calls, calls)

    def test_fake_agent_searches_reads_and_uses_memory_in_executable_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self.make_memory(tmp)
            memory.finish_task(
                task_id="d",
                source_revision=2,
                outcome="completed",
                candidates=(candidate(),),
            )
            tools = [
                *memory.tools(
                    MemoryContext(agent="a", task="", data={"scope": "repo"})
                ),
                make_bash_tool(cwd=tmp),
            ]
            turn = 0

            def generate(messages):
                nonlocal turn
                turn += 1
                if turn == 1:
                    return assistant_message(
                        (ToolCallBlock("s", "memory_search", {"query": "runCI"}),),
                        sender="a",
                        kind="step",
                    )
                if turn == 2:
                    result = tool_results_of(messages[-1].content)[0]
                    entry_id = json.loads(text_of(result.content))[0]["id"]
                    return assistant_message(
                        (ToolCallBlock("r", "memory_read", {"ids": [entry_id]}),),
                        sender="a",
                        kind="step",
                    )
                if turn == 3:
                    result = tool_results_of(messages[-1].content)[0]
                    content = json.loads(text_of(result.content))[0]["content"]
                    return assistant_message(
                        (
                            ToolCallBlock(
                                "b",
                                "bash",
                                {
                                    "command": f"printf %s {json.dumps(content)} > used.txt"
                                },
                            ),
                        ),
                        sender="a",
                        kind="step",
                    )
                return assistant_message("done", sender="a", kind="final")

            state, events = Agent("a", generate, tools=tuple(tools)).run(
                "reuse prior fact"
            )
            tuple(events)
            self.assertEqual((Path(tmp) / "used.txt").read_text(), "alpha uses run_ci")
            self.assertEqual(state.messages[-1].kind, "final")

    def test_tokenizer_handles_camel_snake_and_chinese_bigrams(self) -> None:
        self.assertEqual(
            tokenize("runCI foo_bar 长期记忆"), "run ci foo bar 长期 期记 记忆"
        )

    def test_model_extraction_maps_task_evidence_and_validates_source(self) -> None:
        evidence = TaskMemoryEvidence(
            task_id="d",
            run_id="run-d",
            source_revision=4,
            outcome="completed",
            scope="repo",
            task="fix it",
            observed_at="2026-09-11T00:00:00Z",
            sources=(
                MemoryEvidenceSource(
                    ref="run-d:3",
                    run_id="run-d",
                    kind="message",
                    content="focused test passed",
                ),
            ),
        )
        response = LLMResponse(
            content=(
                TextBlock(
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "type": "fact",
                                    "content": "focused test covers the fix",
                                    "source_ref": "run-d:3",
                                }
                            ]
                        }
                    )
                ),
            ),
            usage=TokenUsage(input_tokens=12, output_tokens=8),
            model="fake-extractor",
        )
        with tempfile.TemporaryDirectory() as tmp:
            memory = IndexedMemory(Path(tmp) / "memory.sqlite3")
            extractor = ModelMemoryExtractor(
                FAKE_PROVIDER, complete_fn=lambda request: response
            )
            ids, result = memory.extract_task(evidence, extractor)

            self.assertEqual(len(ids), 1)
            self.assertEqual(result.usage.context_tokens, 20)
            self.assertEqual(memory.search("focused test", "repo")[0].entry.id, ids[0])

            bad_response = LLMResponse(
                content=(
                    TextBlock(
                        json.dumps(
                            {
                                "candidates": [
                                    {
                                        "type": "fact",
                                        "content": "invented",
                                        "source_ref": "missing",
                                    }
                                ]
                            }
                        )
                    ),
                )
            )
            bad = ModelMemoryExtractor(
                FAKE_PROVIDER, complete_fn=lambda request: bad_response
            )
            with self.assertRaisesRegex(ValueError, "source reference"):
                memory.extract_task(
                    replace(evidence, source_revision=5),
                    bad,
                )


if __name__ == "__main__":
    unittest.main()
