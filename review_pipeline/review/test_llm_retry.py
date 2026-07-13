from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from review import llm_retry


class LlmRetryTests(unittest.TestCase):
    def test_non_retryable_quota_error_stops_after_first_attempt(self) -> None:
        attempts = 0

        def fake_run(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "status": "error",
                            "error": "PermissionDeniedError: insufficient_user_quota",
                        }
                    ),
                    "stderr": "",
                },
            )()

        with TemporaryDirectory() as tmpdir, patch.object(llm_retry.subprocess, "run", side_effect=fake_run):
            result = llm_retry.call_llm_json_with_retry(
                api_key="key",
                base_url="https://example.invalid/v1",
                model_name="model",
                system_prompt="system",
                user_content="user",
                response_model_schema={"type": "object"},
                timeout_seconds=1,
                max_retries=3,
                helper_path=Path("worker.py"),
                label="quota",
                debug_dir=Path(tmpdir),
            )

        self.assertIsNone(result)
        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()
