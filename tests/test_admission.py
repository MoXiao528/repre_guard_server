from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import server
from repreGuard_service import DetectResult, DetectServiceError


def result_for(text: str) -> DetectResult:
    return DetectResult(
        text=text,
        score=0.1,
        threshold=0.2,
        label="HUMAN",
        model="test-model",
        score_type="probability",
    )


class FakeRequest:
    def __init__(self, *, disconnected: bool = False) -> None:
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected

    def disconnect(self) -> None:
        self._disconnected = True


class BlockingPostAcquireRequest(FakeRequest):
    def __init__(self) -> None:
        super().__init__()
        self.checks = 0
        self.post_acquire_check = asyncio.Event()
        self.release_check = asyncio.Event()

    async def is_disconnected(self) -> bool:
        self.checks += 1
        if self.checks == 1:
            return False
        self.post_acquire_check.set()
        await self.release_check.wait()
        return False


class AdmissionTestCase(unittest.IsolatedAsyncioTestCase):
    async def wait_until(self, predicate, *, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("condition was not reached before timeout")
            await asyncio.sleep(0.001)

    def assert_busy(self, exc: HTTPException, code: str) -> None:
        self.assertEqual(exc.status_code, 503)
        self.assertEqual(exc.detail["code"], code)
        self.assertIsNotNone(exc.headers)
        self.assertTrue(exc.headers["Retry-After"].isdigit())

    async def test_capacity_is_hard_bounded_and_extra_request_never_runs(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=3, queue_timeout_seconds=15)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking_detect(text: str) -> DetectResult:
            calls.append(text)
            started.set()
            if not release.wait(5):
                raise RuntimeError("test inference was not released")
            return result_for(text)

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(started.is_set)
            queued = [
                asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text=f"queued-{index}")))
                for index in range(3)
            ]
            await self.wait_until(lambda: admission.pending_count == 3)

            with self.assertRaises(HTTPException) as exc_info:
                await asyncio.wait_for(
                    server.detect(FakeRequest(), server.DetectRequest(text="rejected")),
                    timeout=0.2,
                )

            self.assert_busy(exc_info.exception, "DETECT_QUEUE_FULL")
            self.assertEqual(admission.admitted_count, 4)
            self.assertEqual(calls, ["active"])

            for task in queued[1:]:
                task.cancel()
            await asyncio.gather(*queued[1:], return_exceptions=True)
            self.assertEqual(admission.pending_count, 1)
            release.set()
            await active
            queued_response = await queued[0]
            self.assertEqual(queued_response.label, "HUMAN")

        self.assertEqual(admission.admitted_count, 0)
        self.assertEqual(admission.active_count, 0)
        self.assertEqual(calls, ["active", "queued-0"])

    async def test_queue_timeout_releases_pending_and_does_not_run_later(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=1, queue_timeout_seconds=0.05)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking_detect(text: str) -> DetectResult:
            calls.append(text)
            started.set()
            release.wait(5)
            return result_for(text)

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(started.is_set)

            with self.assertRaises(HTTPException) as exc_info:
                await server.detect(FakeRequest(), server.DetectRequest(text="timed-out"))

            self.assert_busy(exc_info.exception, "DETECT_QUEUE_TIMEOUT")
            self.assertEqual(admission.pending_count, 0)
            self.assertEqual(calls, ["active"])
            release.set()
            await active
            await asyncio.sleep(0.02)

        self.assertEqual(calls, ["active"])
        self.assertEqual(admission.admitted_count, 0)

    async def test_disconnect_and_task_cancel_remove_pending_work(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=2, queue_timeout_seconds=15)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking_detect(text: str) -> DetectResult:
            calls.append(text)
            started.set()
            release.wait(5)
            return result_for(text)

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(started.is_set)

            disconnected_request = FakeRequest()
            disconnected = asyncio.create_task(
                server.detect(disconnected_request, server.DetectRequest(text="disconnected"))
            )
            cancelled = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="cancelled")))
            await self.wait_until(lambda: admission.pending_count == 2)

            disconnected_request.disconnect()
            with self.assertRaises(HTTPException) as exc_info:
                await disconnected
            self.assertEqual(exc_info.exception.status_code, 499)

            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled

            self.assertEqual(admission.pending_count, 0)
            self.assertEqual(calls, ["active"])
            release.set()
            await active

        self.assertEqual(admission.admitted_count, 0)
        self.assertEqual(calls, ["active"])

    async def test_disconnect_wins_when_slot_release_and_disconnect_share_a_loop_turn(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=1, queue_timeout_seconds=1)
        started = threading.Event()
        release = threading.Event()
        request = FakeRequest()
        calls: list[str] = []

        def blocking_detect(text: str) -> DetectResult:
            calls.append(text)
            started.set()
            release.wait(5)
            return result_for(text)

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(started.is_set)
            pending = asyncio.create_task(server.detect(request, server.DetectRequest(text="disconnected")))
            await self.wait_until(lambda: admission.pending_count == 1)

            release.set()
            request.disconnect()
            with self.assertRaises(HTTPException) as exc_info:
                await pending
            await active

        self.assertEqual(exc_info.exception.status_code, 499)
        self.assertEqual(calls, ["active"])
        self.assertEqual(admission.admitted_count, 0)
        self.assertEqual(admission.active_count, 0)

    async def test_cancel_after_slot_acquire_does_not_leak_semaphore(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=0, queue_timeout_seconds=1)
        request = BlockingPostAcquireRequest()

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", side_effect=lambda text: result_for(text)) as fake_detect,
        ):
            cancelled = asyncio.create_task(server.detect(request, server.DetectRequest(text="cancelled")))
            await request.post_acquire_check.wait()
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled

            self.assertEqual(admission.admitted_count, 0)
            self.assertEqual(admission.active_count, 0)
            response = await server.detect(FakeRequest(), server.DetectRequest(text="next"))

        self.assertEqual(response.label, "HUMAN")
        fake_detect.assert_called_once_with("next")
        self.assertEqual(admission.admitted_count, 0)

    async def test_full_asgi_chain_removes_disconnected_pending_request(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=1, queue_timeout_seconds=1)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking_detect(text: str) -> DetectResult:
            calls.append(text)
            started.set()
            release.wait(5)
            return result_for(text)

        body = b'{"text":"disconnected through ASGI"}'
        source: asyncio.Queue[dict] = asyncio.Queue()
        source.put_nowait({"type": "http.request", "body": body, "more_body": False})
        sent: list[dict] = []

        async def receive() -> dict:
            return await source.get()

        async def send(message: dict) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/detect",
            "raw_path": b"/detect",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-repreguard-token", server.settings.service_token.encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 9000),
        }

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(started.is_set)
            disconnected = asyncio.create_task(server.app(scope, receive, send))
            await self.wait_until(lambda: admission.pending_count == 1)
            source.put_nowait({"type": "http.disconnect"})
            await disconnected
            release.set()
            await active

        response_status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        self.assertEqual(response_status, 499)
        self.assertEqual(calls, ["active"])
        self.assertEqual(admission.admitted_count, 0)

    async def test_active_cancellation_keeps_gpu_slot_until_thread_finishes(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=0, queue_timeout_seconds=1)
        release = threading.Event()
        calls: list[str] = []
        running = 0
        max_running = 0
        state_lock = threading.Lock()

        def blocking_detect(text: str) -> DetectResult:
            nonlocal running, max_running
            calls.append(text)
            with state_lock:
                running += 1
                max_running = max(max_running, running)
            try:
                release.wait(5)
                return result_for(text)
            finally:
                with state_lock:
                    running -= 1

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", blocking_detect),
        ):
            active = asyncio.create_task(server.detect(FakeRequest(), server.DetectRequest(text="active")))
            await self.wait_until(lambda: admission.active_count == 1)
            await self.wait_until(lambda: bool(calls))

            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await active

            self.assertEqual(admission.active_count, 1)
            self.assertEqual(admission.admitted_count, 1)
            with self.assertRaises(HTTPException) as exc_info:
                await server.detect(FakeRequest(), server.DetectRequest(text="overlap"))
            self.assert_busy(exc_info.exception, "DETECT_QUEUE_FULL")

            release.set()
            await admission.drain()

        self.assertEqual(max_running, 1)
        self.assertEqual(calls, ["active"])
        self.assertEqual(admission.admitted_count, 0)
        self.assertEqual(admission.active_count, 0)

    async def test_inference_errors_release_capacity_for_the_next_request(self) -> None:
        admission = server.InferenceAdmission(max_pending_requests=0, queue_timeout_seconds=1)

        def service_failure(_text: str) -> DetectResult:
            raise DetectServiceError("expected failure")

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", service_failure),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await server.detect(FakeRequest(), server.DetectRequest(text="service failure"))
            self.assertEqual(exc_info.exception.detail["code"], "DETECT_INTERNAL_ERROR")

        self.assertEqual(admission.admitted_count, 0)

        def unexpected_failure(_text: str) -> DetectResult:
            raise ValueError("secret internal detail")

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", unexpected_failure),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await server.detect(FakeRequest(), server.DetectRequest(text="unexpected failure"))
            self.assertEqual(exc_info.exception.status_code, 500)
            self.assertEqual(exc_info.exception.detail["code"], "DETECT_INTERNAL_ERROR")

        self.assertEqual(admission.admitted_count, 0)

        with (
            patch.object(server, "INFERENCE_ADMISSION", admission),
            patch.object(server, "detect_text", lambda text: result_for(text)),
        ):
            response = await server.detect(FakeRequest(), server.DetectRequest(text="recovered"))

        self.assertEqual(response.label, "HUMAN")
        self.assertEqual(admission.admitted_count, 0)


if __name__ == "__main__":
    unittest.main()
