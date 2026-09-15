import json
import unittest

from horizon_harness.messages import assistant_message, ToolCallBlock, ThinkingBlock
from horizon_harness.llm.bridge import messages_to_llm_messages
from horizon_harness.workspace.codec import encode_message, decode_message


class WorkspaceCodecTests(unittest.TestCase):
    def test_round_trip_preserves_protocol_metadata_and_order(self):
        original = assistant_message(
            (
                ThinkingBlock(text="reason", signature="signature"),
                ToolCallBlock("A", "read", {"path": "a"}),
                ToolCallBlock("B", "edit", {"path": "b"}),
            ),
            sender="agent",
            target="user",
            kind="step",
            sidecar={
                "extra": {
                    "reasoning_items": [{"id": "r", "encrypted_content": "opaque"}]
                }
            },
        )
        encoded = encode_message(original)
        restored = decode_message(json.loads(json.dumps(encoded)))
        self.assertEqual(restored, original)
        self.assertEqual(
            messages_to_llm_messages([restored]), messages_to_llm_messages([original])
        )
        encoded["sidecar"]["extra"].clear()
        self.assertTrue(original.sidecar["extra"])
        self.assertTrue(restored.sidecar["extra"])

    def test_unknown_version_and_unserializable_metadata_fail(self):
        with self.assertRaisesRegex(ValueError, "version"):
            decode_message({"message_version": 99})
        with self.assertRaises(TypeError):
            encode_message(assistant_message("x", sidecar={"extra": {"bad": object()}}))
