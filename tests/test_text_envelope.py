from __future__ import annotations

import unittest

from text_envelope import unwrap_text_envelope


class TextEnvelopeTestCase(unittest.TestCase):
    def test_legacy_values_are_unchanged(self):
        for value in (
            None,
            [],
            "text",
            {},
            {"text": 12},
            {"text": '{"action":"evidence","text":"正文"}'},
            {"text": "ordinary", "action": "evidence"},
        ):
            with self.subTest(value=value):
                self.assertIs(unwrap_text_envelope(value), value)

    def test_both_actions_preserve_original_and_unwrap_once(self):
        originals = (
            '  中文😀\r\n"quoted"\t\n  ',
            '{"action":"evidence","text":"正文"}',
            "AIDT:1:detect\nprivate original",
            "AIDT:99:unknown\nprivate original",
            "测" * 20000,
        )
        for action in ("detect", "evidence"):
            for original in originals:
                with self.subTest(action=action, length=len(original)):
                    self.assertEqual(
                        unwrap_text_envelope({"text": f"AIDT:1:{action}\n{original}"}),
                        {"text": original, "action": action},
                    )

    def test_reserved_malformed_envelopes_are_rejected_without_original(self):
        for text in (
            "AIDT:",
            "AIDT:1:evidence",
            "AIDT:2:evidence\nprivate",
            "AIDT:01:evidence\nprivate",
            "AIDT:1:unknown\nprivate",
            "AIDT:1:EVIDENCE\nprivate",
            "AIDT:1:detect:extra\nprivate",
            "AIDT:1:detect\r\nprivate",
            "AIDT:1:evidence\nprivate\ud800",
        ):
            with self.subTest(header=text.partition("\n")[0]):
                with self.assertRaisesRegex(ValueError, "^Invalid text envelope\\.$"):
                    unwrap_text_envelope({"text": text})

    def test_envelope_rejects_all_outer_extra_fields(self):
        for extra in ({"action": "detect"}, {"action": "evidence"}, {"other": 1}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    unwrap_text_envelope({"text": "AIDT:1:evidence\nprivate", **extra})


if __name__ == "__main__":
    unittest.main()
