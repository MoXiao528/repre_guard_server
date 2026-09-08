from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI

from ingress import DetectorIngressMiddleware
from repreGuard_service import DetectResult
import server


TOKEN = "evidence-test-service-token-longer-than-32-characters"
SHA = "a" * 64


def prediction(status="predicted", reason=None):
    return {
        "status": status,
        "reason": reason,
        "routerArtifactSha256": SHA,
        "route": {
            "language": "en",
            "domain": "academic",
            "confidence": {"language": 0.01, "domain": 0.02},
        }
        if status == "predicted"
        else None,
    }


class EvidenceEndpointTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.payload = prediction()

        def predict(text):
            self.calls.append(text)
            return self.payload

        self.router = SimpleNamespace(
            status="ready", artifact_sha256=SHA, predict_route=predict
        )
        self.admission = server.InferenceAdmission(
            max_pending_requests=0, queue_timeout_seconds=1
        )
        self.main_admission = server.InferenceAdmission(
            max_pending_requests=0, queue_timeout_seconds=1
        )
        for name, value in (
            ("EVIDENCE_ROUTER", self.router),
            ("EVIDENCE_ADMISSION", self.admission),
            ("INFERENCE_ADMISSION", self.main_admission),
            ("EVIDENCE_TIMEOUT_SECONDS", 1.0),
        ):
            context = patch.object(server, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.addAsyncCleanup(server.drain_inference)
        # Exercise the actual handlers and ingress without a socket, lifespan or model startup.
        self.app = FastAPI()
        self.app.router.routes = server.app.router.routes
        self.app.exception_handlers.update(server.app.exception_handlers)
        self.app.add_middleware(DetectorIngressMiddleware, service_token=TOKEN)

    async def invoke(
        self, payload=None, *, body=None, path="/evidence/route", token=TOKEN
    ):
        if body is None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        source = asyncio.Queue()
        source.put_nowait({"type": "http.request", "body": body, "more_body": False})
        sent = []

        async def receive():
            return await source.get()

        async def send(message):
            sent.append(message)

        await self.app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET" if path == "/health" else "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-repreguard-token", token.encode()),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 9000),
            },
            receive,
            send,
        )
        code = next(m["status"] for m in sent if m["type"] == "http.response.start")
        value = json.loads(b"".join(m.get("body", b"") for m in sent))
        return code, value

    async def wait_for(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), timeout=2)

    async def test_fulltext_once_and_exact_d1_response_with_low_confidence(self):
        text = "  Ａ\r\nB\u00a0C  "
        code, value = await self.invoke({"text": text})
        self.assertEqual(code, 200)
        self.assertEqual(
            value, {"schemaVersion": 1, **prediction(), "status": "routed"}
        )
        self.assertEqual(self.calls, [text])
        self.assertEqual(self.admission.admitted_count, 0)
        code, _ = await self.invoke({"text": "测" * 20000})
        self.assertEqual(code, 200)
        self.assertEqual(len(self.calls[-1]), 20000)

    async def test_request_validation_and_auth_do_not_expose_input_or_run_model(self):
        for payload in (
            {},
            {"text": ""},
            {"text": " \n"},
            {"text": 12},
            {"text": None},
            {"text": "x" * 20001},
            {"text": "private", "score": 0.9},
            ["private"],
        ):
            with self.subTest(kind=type(payload).__name__):
                code, value = await self.invoke(payload)
                self.assertEqual(code, 422)
                self.assertEqual(value["detail"]["code"], "INVALID_EVIDENCE_REQUEST")
                self.assertNotIn("private", json.dumps(value))
        code, value = await self.invoke(body=b'{"text":"private",')
        self.assertEqual(code, 422)
        self.assertNotIn("private", json.dumps(value))
        code, _ = await self.invoke({"text": "private"}, token="wrong")
        self.assertEqual(code, 401)
        code, _ = await self.invoke({"text": "private", "extra": "x" * 131072})
        self.assertEqual(code, 413)
        self.assertEqual(self.calls, [])

    async def test_unsupported_and_failures_have_null_route_and_cached_sha(self):
        for status, reason in (
            ("unsupported", "unsupported_language"),
            ("failed", "language_undetermined"),
            ("failed", "model_failure"),
            ("failed", "model_unavailable"),
        ):
            self.payload = prediction(status, reason)
            code, value = await self.invoke({"text": "text"})
            self.assertEqual(code, 200)
            self.assertEqual(value, {"schemaVersion": 1, **self.payload})
        self.router.status = "failed"
        count = len(self.calls)
        code, value = await self.invoke({"text": "text"})
        self.assertEqual(code, 200)
        self.assertEqual(
            value, server._evidence_failure("model_unavailable").model_dump()
        )
        self.assertEqual(len(self.calls), count)

    async def test_bad_internal_results_are_private_and_next_request_recovers(self):
        for payload in (
            {**prediction(), "routerArtifactSha256": "b" * 64},
            {**prediction(), "route": {"text": "private"}},
            prediction("failed", "private exception"),
            None,
        ):
            self.payload = payload
            code, value = await self.invoke({"text": "private"})
            self.assertEqual(code, 200)
            self.assertEqual(
                value, server._evidence_failure("model_failure", SHA).model_dump()
            )
            self.assertNotIn("private", json.dumps(value))
        with patch.object(
            self.router, "predict_route", side_effect=RuntimeError("private path")
        ):
            _, value = await self.invoke({"text": "private"})
            self.assertEqual(value["reason"], "model_failure")
        self.payload = prediction()
        _, value = await self.invoke({"text": "recovered"})
        self.assertEqual(value["status"], "routed")
        self.assertEqual(self.admission.admitted_count, 0)

    async def test_timeout_and_busy_keep_slot_until_worker_exits_and_main_is_independent(
        self,
    ):
        started, release = threading.Event(), threading.Event()

        def blocking(text):
            self.calls.append(text)
            started.set()
            if not release.wait(5):
                raise RuntimeError("test worker was not released")
            return prediction()

        result = DetectResult(
            text="main",
            score=0.1,
            threshold=0.2,
            label="HUMAN",
            model="main",
            score_type="probability",
        )
        with (
            patch.object(self.router, "predict_route", blocking),
            patch.object(server, "EVIDENCE_TIMEOUT_SECONDS", 0.1),
            patch.object(server, "detect_text", return_value=result) as main,
        ):
            active = asyncio.create_task(self.invoke({"text": "first"}))
            try:
                await self.wait_for(started.is_set)
                _, busy = await self.invoke({"text": "second"})
                self.assertEqual(busy["reason"], "busy")
                _, main_value = await self.invoke({"text": "main"}, path="/detect")
                self.assertEqual(
                    main_value,
                    {
                        "score": 0.1,
                        "threshold": 0.2,
                        "label": "HUMAN",
                        "model_name": "main",
                        "score_type": "probability",
                    },
                )
                main.assert_called_once_with("main")
                _, health = await self.invoke(path="/health")
                self.assertEqual(health, {"status": "ok"})
                _, timed_out = await active
                self.assertEqual(timed_out["reason"], "timeout")
                self.assertEqual(self.admission.active_count, 1)
                self.assertEqual(self.admission.pending_count, 0)
                _, busy = await self.invoke({"text": "third"})
                self.assertEqual(busy["reason"], "busy")
                self.assertEqual(self.calls, ["first"])
            finally:
                release.set()
                await asyncio.gather(active, return_exceptions=True)
                await self.admission.drain()
        _, recovered = await self.invoke({"text": "next"})
        self.assertEqual(recovered["status"], "routed")

    async def test_active_cancellation_and_prestart_disconnect_do_not_leak_capacity(
        self,
    ):
        started, release = threading.Event(), threading.Event()

        def blocking(text):
            started.set()
            release.wait(5)
            return prediction()

        with patch.object(self.router, "predict_route", blocking):
            active = asyncio.create_task(self.invoke({"text": "first"}))
            try:
                await self.wait_for(started.is_set)
                active.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await active
                self.assertEqual(self.admission.active_count, 1)
                _, busy = await self.invoke({"text": "second"})
                self.assertEqual(busy["reason"], "busy")
            finally:
                release.set()
                await self.admission.drain()

        class Disconnected:
            async def is_disconnected(self):
                return True

        with self.assertRaises(server.HTTPException) as error:
            await server.route_evidence(
                Disconnected(), server.EvidenceRequest(text="never run")
            )
        self.assertEqual(error.exception.status_code, 499)
        self.assertEqual(self.admission.admitted_count, 0)
        self.assertEqual(self.calls, [])

    async def test_optional_startup_errors_do_not_stop_main_and_off_does_not_load(self):
        for enabled, timeout in (
            ("false", "bad"),
            ("true", "nan"),
            ("true", "inf"),
            ("true", "0"),
            ("true", "61"),
            ("true", "bad"),
        ):
            config = replace(
                server.settings,
                service_token=TOKEN,
                evidence_enabled=enabled,
                evidence_timeout_seconds=timeout,
            )
            with (
                patch.object(server, "settings", config),
                patch.object(server, "get_detector") as main,
                patch.object(server, "EvidenceRouter") as load,
            ):
                server.load_detector()
                server.load_evidence_router()
                main.assert_called_once()
                load.assert_not_called()
                self.assertIsNone(server.EVIDENCE_ROUTER)
        config = replace(
            server.settings, evidence_enabled="true", evidence_timeout_seconds="2.5"
        )
        with (
            patch.object(server, "settings", config),
            patch.object(server, "EvidenceRouter", side_effect=RuntimeError("private")),
        ):
            server.load_evidence_router()
            _, value = await self.invoke({"text": "text"})
            self.assertEqual(value["reason"], "model_unavailable")
        with (
            patch.object(server, "settings", config),
            patch.object(server, "EvidenceRouter", return_value=self.router) as load,
        ):
            server.load_evidence_router()
            await self.invoke({"text": "one"})
            await self.invoke({"text": "two"})
            load.assert_called_once_with(
                enabled="true",
                model_path=config.evidence_model_path,
                artifact_sha256=config.evidence_artifact_sha256,
            )
            self.assertEqual(server.EVIDENCE_TIMEOUT_SECONDS, 2.5)


if __name__ == "__main__":
    unittest.main()
