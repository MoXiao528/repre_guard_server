from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from config import is_valid_service_token


DETECT_BODY_LIMIT_BYTES = 128 * 1024
SERVICE_TOKEN_HEADER = b"x-repreguard-token"
PROTECTED_PATHS = frozenset({"/detect", "/health"})

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


async def _send_error(send: Send, status: int, code: str, message: str) -> None:
    body = json.dumps(
        {"detail": {"code": code, "message": message}},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _header_values(scope: dict[str, Any], name: bytes) -> list[bytes]:
    return [value for key, value in scope.get("headers", []) if key.lower() == name]


def _declared_content_length(scope: dict[str, Any]) -> int | None:
    values = _header_values(scope, b"content-length")
    if not values:
        return None
    if len(values) != 1 or not values[0] or not values[0].isdigit():
        raise ValueError("invalid Content-Length")
    return int(values[0])


class DetectorIngressMiddleware:
    def __init__(self, app: ASGIApp, service_token: str) -> None:
        self.app = app
        try:
            token_bytes = service_token.encode("ascii")
        except UnicodeEncodeError:
            token_bytes = b""
        self._expected_token = token_bytes if is_valid_service_token(service_token) else b""

    def _is_authorized(self, scope: dict[str, Any]) -> bool:
        values = _header_values(scope, SERVICE_TOKEN_HEADER)
        return bool(
            self._expected_token
            and len(values) == 1
            and secrets.compare_digest(values[0], self._expected_token)
        )

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in PROTECTED_PATHS:
            await self.app(scope, receive, send)
            return

        if not self._is_authorized(scope):
            await _send_error(
                send,
                401,
                "DETECT_SERVICE_UNAUTHORIZED",
                "Detector service authentication failed.",
            )
            return

        if scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return

        try:
            declared_length = _declared_content_length(scope)
        except ValueError:
            await _send_error(send, 400, "INVALID_CONTENT_LENGTH", "Content-Length is invalid.")
            return

        if declared_length is not None and declared_length > DETECT_BODY_LIMIT_BYTES:
            await _send_error(
                send,
                413,
                "REQUEST_BODY_TOO_LARGE",
                f"Request body exceeds {DETECT_BODY_LIMIT_BYTES} bytes.",
            )
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                await _send_error(send, 400, "INVALID_REQUEST_BODY", "Request body stream is invalid.")
                return

            chunk = message.get("body", b"")
            if len(body) + len(chunk) > DETECT_BODY_LIMIT_BYTES:
                await _send_error(
                    send,
                    413,
                    "REQUEST_BODY_TOO_LARGE",
                    f"Request body exceeds {DETECT_BODY_LIMIT_BYTES} bytes.",
                )
                return

            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> ASGIMessage:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)
