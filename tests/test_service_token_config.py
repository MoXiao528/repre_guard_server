from __future__ import annotations

import unittest
from unittest.mock import patch

from config import RepreGuardConfig


class ServiceTokenConfigTestCase(unittest.TestCase):
    def test_valid_service_token_is_returned(self) -> None:
        token = "detector-test-token-that-is-at-least-32-characters"

        self.assertEqual(RepreGuardConfig(service_token=token).require_service_token(), token)

    def test_missing_short_or_non_header_safe_token_is_rejected(self) -> None:
        for token in ("", "too-short", "x" * 31, "测" * 32, "x" * 31 + " "):
            with self.subTest(token=token):
                with self.assertRaisesRegex(RuntimeError, "REPRE_GUARD_SERVICE_TOKEN"):
                    RepreGuardConfig(service_token=token).require_service_token()

    def test_startup_rejects_missing_token_before_loading_model(self) -> None:
        import server

        with (
            patch.object(server, "settings", RepreGuardConfig(service_token="")),
            patch.object(server, "get_detector") as get_detector,
        ):
            with self.assertRaisesRegex(RuntimeError, "REPRE_GUARD_SERVICE_TOKEN"):
                server.load_detector()

        get_detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
