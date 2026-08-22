from __future__ import annotations

import json
import unittest
from collections.abc import Awaitable, Callable
from typing import Any

from ingress import DETECT_BODY_LIMIT_BYTES, DetectorIngressMiddleware


SERVICE_TOKEN = "detector-test-token-that-is-at-least-32-characters"
TOKEN_HEADER = (b"x-repreguard-token", SERVICE_TOKEN.encode("ascii"))


class RecordingApp:
    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[bytes] = []

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self.calls += 1
        body = b""
        if scope["path"] == "/detect":
            while True:
                message = await receive()
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
        self.bodies.append(body)
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [(b"content-length", b"0")],
            }
        )
        await send({"type": "http.response.body", "body": b""})


class IngressTestCase(unittest.IsolatedAsyncioTestCase):
    async def invoke(
        self,
        *,
        path: str,
        headers: list[tuple[bytes, bytes]] | None = None,
        chunks: list[bytes] | None = None,
    ) -> tuple[RecordingApp, list[dict[str, Any]], int]:
        downstream = RecordingApp()
        middleware = DetectorIngressMiddleware(downstream, SERVICE_TOKEN)
        body_chunks = chunks if chunks is not None else [b""]
        messages = [
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index < len(body_chunks) - 1,
            }
            for index, chunk in enumerate(body_chunks)
        ]
        receive_calls = 0
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal receive_calls
            receive_calls += 1
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST" if path == "/detect" else "GET",
                "path": path,
                "headers": headers or [],
            },
            receive,
            send,
        )
        return downstream, sent, receive_calls

    @staticmethod
    def status(sent: list[dict[str, Any]]) -> int:
        return next(message["status"] for message in sent if message["type"] == "http.response.start")

    async def test_missing_token_is_rejected_without_reading_body(self) -> None:
        downstream, sent, receive_calls = await self.invoke(path="/detect", chunks=[b"secret text"])

        self.assertEqual(self.status(sent), 401)
        self.assertEqual(downstream.calls, 0)
        self.assertEqual(receive_calls, 0)

    async def test_wrong_or_duplicate_token_is_rejected_without_reading_body(self) -> None:
        cases = [
            [(b"x-repreguard-token", b"wrong")],
            [TOKEN_HEADER, TOKEN_HEADER],
        ]
        for headers in cases:
            with self.subTest(headers=headers):
                downstream, sent, receive_calls = await self.invoke(
                    path="/detect",
                    headers=headers,
                    chunks=[b"secret text"],
                )
                self.assertEqual(self.status(sent), 401)
                self.assertEqual(downstream.calls, 0)
                self.assertEqual(receive_calls, 0)

    async def test_health_requires_token_but_middleware_does_not_read_body(self) -> None:
        downstream, sent, receive_calls = await self.invoke(
            path="/health",
            headers=[TOKEN_HEADER],
            chunks=[b"ignored"],
        )

        self.assertEqual(self.status(sent), 204)
        self.assertEqual(downstream.calls, 1)
        self.assertEqual(receive_calls, 0)

    async def test_declared_oversize_is_rejected_without_reading_body(self) -> None:
        downstream, sent, receive_calls = await self.invoke(
            path="/detect",
            headers=[TOKEN_HEADER, (b"content-length", str(DETECT_BODY_LIMIT_BYTES + 1).encode("ascii"))],
            chunks=[b"unread"],
        )

        self.assertEqual(self.status(sent), 413)
        self.assertEqual(downstream.calls, 0)
        self.assertEqual(receive_calls, 0)

    async def test_invalid_or_multiple_content_length_fails_closed(self) -> None:
        cases = [
            [(b"content-length", b"abc")],
            [(b"content-length", b"-1")],
            [(b"content-length", b"1"), (b"content-length", b"1")],
        ]
        for length_headers in cases:
            with self.subTest(length_headers=length_headers):
                downstream, sent, receive_calls = await self.invoke(
                    path="/detect",
                    headers=[TOKEN_HEADER, *length_headers],
                    chunks=[b"x"],
                )
                self.assertEqual(self.status(sent), 400)
                self.assertEqual(downstream.calls, 0)
                self.assertEqual(receive_calls, 0)

    async def test_streamed_body_without_length_cannot_bypass_limit(self) -> None:
        downstream, sent, receive_calls = await self.invoke(
            path="/detect",
            headers=[TOKEN_HEADER],
            chunks=[b"a" * 65_536, b"b" * 65_537],
        )

        self.assertEqual(self.status(sent), 413)
        self.assertEqual(downstream.calls, 0)
        self.assertEqual(receive_calls, 2)

    async def test_forged_small_length_cannot_bypass_actual_stream_limit(self) -> None:
        downstream, sent, _ = await self.invoke(
            path="/detect",
            headers=[TOKEN_HEADER, (b"content-length", b"1")],
            chunks=[b"a" * DETECT_BODY_LIMIT_BYTES, b"b"],
        )

        self.assertEqual(self.status(sent), 413)
        self.assertEqual(downstream.calls, 0)

    async def test_escaped_and_unknown_field_oversize_bodies_fail_before_downstream(self) -> None:
        cases = [
            json.dumps({"text": "😀" * 20_000}).encode("utf-8"),
            json.dumps({"text": "ok", "unknown": "x" * DETECT_BODY_LIMIT_BYTES}).encode("utf-8"),
        ]
        for body in cases:
            with self.subTest(body_size=len(body)):
                downstream, sent, _ = await self.invoke(
                    path="/detect",
                    headers=[TOKEN_HEADER],
                    chunks=[body[:65_536], body[65_536:]],
                )
                self.assertEqual(self.status(sent), 413)
                self.assertEqual(downstream.calls, 0)

    async def test_exact_limit_is_replayed_to_downstream(self) -> None:
        body = b"x" * DETECT_BODY_LIMIT_BYTES
        downstream, sent, _ = await self.invoke(
            path="/detect",
            headers=[TOKEN_HEADER, (b"content-length", str(len(body)).encode("ascii"))],
            chunks=[body[:65_536], body[65_536:]],
        )

        self.assertEqual(self.status(sent), 204)
        self.assertEqual(downstream.bodies, [body])

    async def test_twenty_thousand_chinese_characters_fit_the_ingress_limit(self) -> None:
        body = json.dumps({"text": "测" * 20_000}, ensure_ascii=False).encode("utf-8")
        self.assertLess(len(body), DETECT_BODY_LIMIT_BYTES)

        downstream, sent, _ = await self.invoke(
            path="/detect",
            headers=[TOKEN_HEADER, (b"content-length", str(len(body)).encode("ascii"))],
            chunks=[body],
        )

        self.assertEqual(self.status(sent), 204)
        self.assertEqual(downstream.bodies, [body])

    async def test_unrelated_paths_are_not_claimed_by_internal_protocol(self) -> None:
        downstream, sent, receive_calls = await self.invoke(path="/openapi.json")

        self.assertEqual(self.status(sent), 204)
        self.assertEqual(downstream.calls, 1)
        self.assertEqual(receive_calls, 0)


if __name__ == "__main__":
    unittest.main()
