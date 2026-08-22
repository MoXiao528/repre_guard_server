from __future__ import annotations

import unittest

from loadtest_detector_api import detect_once


SERVICE_TOKEN = "detector-test-token-that-is-at-least-32-characters"


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    content = b"{}"
    text = "{}"

    @staticmethod
    def json() -> dict:
        return {
            "score": 0.1,
            "threshold": 0.2,
            "label": "HUMAN",
            "model_name": "test-model",
            "score_type": "probability",
        }


class RecordingClient:
    def __init__(self) -> None:
        self.headers: dict[str, str] | None = None

    async def post(self, _url: str, *, headers: dict[str, str], json: dict[str, str]) -> FakeResponse:
        self.headers = headers
        return FakeResponse()


class LoadTestClientTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_detector_request_carries_service_token(self) -> None:
        client = RecordingClient()

        result = await detect_once(
            client=client,
            url="http://127.0.0.1:9000/detect",
            text="sample",
            request_id=1,
            round_index=0,
            service_token=SERVICE_TOKEN,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(client.headers["X-RepreGuard-Token"], SERVICE_TOKEN)


if __name__ == "__main__":
    unittest.main()
