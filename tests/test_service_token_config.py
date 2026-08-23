from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import RepreGuardConfig


class ServiceTokenConfigTestCase(unittest.TestCase):
    def _run_isolated_config(
        self,
        dotenv_token: str,
        process_token: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            module_dir = temp_root / "module"
            working_dir = temp_root / "working"
            module_dir.mkdir()
            working_dir.mkdir()
            shutil.copy2(project_root / "config.py", module_dir / "config.py")
            (module_dir / ".env").write_text(
                f"REPRE_GUARD_SERVICE_TOKEN={dotenv_token}\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(module_dir)
            if process_token is None:
                env.pop("REPRE_GUARD_SERVICE_TOKEN", None)
            else:
                env["REPRE_GUARD_SERVICE_TOKEN"] = process_token

            return subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from config import settings; settings.require_service_token(); print(settings.service_token)",
                ],
                cwd=working_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_project_dotenv_is_loaded_from_config_directory(self) -> None:
        token = "dotenv-token-that-is-at-least-32-characters"

        result = self._run_isolated_config(token)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), token)

    def test_process_environment_overrides_project_dotenv(self) -> None:
        dotenv_token = "dotenv-token-that-is-at-least-32-characters"
        process_token = "process-token-that-is-at-least-32-characters"

        result = self._run_isolated_config(dotenv_token, process_token)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), process_token)

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

    def test_admission_settings_are_bounded(self) -> None:
        for max_pending in (-1, 33):
            with self.subTest(max_pending=max_pending):
                with self.assertRaisesRegex(RuntimeError, "REPRE_GUARD_MAX_PENDING_REQUESTS"):
                    RepreGuardConfig(max_pending_requests=max_pending).require_valid_admission_settings()

        for queue_timeout in (0.0, -1.0, 60.1, float("nan"), float("inf")):
            with self.subTest(queue_timeout=queue_timeout):
                with self.assertRaisesRegex(RuntimeError, "REPRE_GUARD_QUEUE_TIMEOUT_SECONDS"):
                    RepreGuardConfig(queue_timeout_seconds=queue_timeout).require_valid_admission_settings()

        RepreGuardConfig(
            max_pending_requests=3,
            queue_timeout_seconds=15.0,
        ).require_valid_admission_settings()


if __name__ == "__main__":
    unittest.main()
