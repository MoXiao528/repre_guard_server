import asyncio
import math
from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings
from evidence_router import EvidenceRouter
from ingress import DetectorIngressMiddleware
from repreGuard_service import (
    DetectResult,
    DetectServiceError,
    detect_text,
    get_detector,
)


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str
    score_type: Literal["probability"] = "probability"


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Text cannot be blank.")
        return text


class EvidenceConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    language: float = Field(ge=0, le=1, allow_inf_nan=False)
    domain: float = Field(ge=0, le=1, allow_inf_nan=False)


class EvidenceRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    language: Literal["ar", "de", "en", "es", "fr", "pt", "ru", "zh"]
    domain: Literal["academic", "news", "novel", "seo", "webtext", "wiki"]
    confidence: EvidenceConfidence


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schemaVersion: Literal[1] = 1
    status: Literal["routed", "unsupported", "failed"]
    routerArtifactSha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    route: EvidenceRoute | None = None
    reason: (
        Literal[
            "unsupported_language",
            "language_undetermined",
            "model_unavailable",
            "model_failure",
            "busy",
            "timeout",
        ]
        | None
    ) = None


class InferenceQueueFull(Exception):
    pass


class InferenceQueueTimeout(Exception):
    pass


class PendingRequestDisconnected(Exception):
    pass


class InferenceAdmission:
    DISCONNECT_POLL_SECONDS = 0.05

    def __init__(
        self, *, max_pending_requests: int, queue_timeout_seconds: float
    ) -> None:
        self.max_pending_requests = max_pending_requests
        self.queue_timeout_seconds = queue_timeout_seconds
        self.retry_after_seconds = max(
            1,
            math.ceil(queue_timeout_seconds / max(1, max_pending_requests)),
        )
        self._max_admitted = 1 + max_pending_requests
        self._gpu_semaphore = asyncio.Semaphore(1)
        self._admitted = 0
        self._active = 0
        self._workers: set[asyncio.Task] = set()

    @property
    def admitted_count(self) -> int:
        return self._admitted

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def pending_count(self) -> int:
        return self._admitted - self._active

    def _admit(self) -> None:
        # No await between the check and increment: this is atomic on the app event loop.
        if self._admitted >= self._max_admitted:
            raise InferenceQueueFull
        self._admitted += 1

    def _release_admission(self) -> None:
        if self._admitted <= 0:
            raise RuntimeError("inference admission counter underflow")
        self._admitted -= 1

    async def _acquire_gpu_or_abort(self, request: Request) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.queue_timeout_seconds

        while True:
            if await request.is_disconnected():
                raise PendingRequestDisconnected

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise InferenceQueueTimeout

            try:
                await asyncio.wait_for(
                    self._gpu_semaphore.acquire(),
                    timeout=min(self.DISCONNECT_POLL_SECONDS, remaining),
                )
                return
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    raise InferenceQueueTimeout from None

    async def _execute_and_release(self, text: str, handler: Callable):
        try:
            return await asyncio.to_thread(handler, text)
        finally:
            self._active -= 1
            self._gpu_semaphore.release()
            self._release_admission()

    def _worker_done(self, task: asyncio.Task) -> None:
        self._workers.discard(task)
        if not task.cancelled():
            task.exception()

    async def run(
        self, request: Request, text: str, *, handler: Callable | None = None
    ):
        self._admit()
        owns_admission = True
        owns_gpu_slot = False

        try:
            await self._acquire_gpu_or_abort(request)
            owns_gpu_slot = True
            await asyncio.sleep(0)
            if await request.is_disconnected():
                raise PendingRequestDisconnected

            self._active += 1
            try:
                worker = asyncio.create_task(
                    self._execute_and_release(
                        text, detect_text if handler is None else handler
                    )
                )
            except BaseException:
                self._active -= 1
                raise

            self._workers.add(worker)
            worker.add_done_callback(self._worker_done)
            owns_admission = False
            owns_gpu_slot = False
            return await asyncio.shield(worker)
        finally:
            if owns_gpu_slot:
                self._gpu_semaphore.release()
            if owns_admission:
                self._release_admission()

    async def drain(self) -> None:
        while self._workers:
            workers = tuple(self._workers)
            await asyncio.gather(
                *(asyncio.shield(worker) for worker in workers), return_exceptions=True
            )


settings.require_valid_admission_settings()
INFERENCE_ADMISSION = InferenceAdmission(
    max_pending_requests=settings.max_pending_requests,
    queue_timeout_seconds=settings.queue_timeout_seconds,
)
# ponytail: separate capacity, shared process CPU/RSS; use a worker process if
# deployment measurements require hard resource or hung-thread isolation.
EVIDENCE_ADMISSION = InferenceAdmission(
    max_pending_requests=0, queue_timeout_seconds=10
)
EVIDENCE_ROUTER: EvidenceRouter | None = None
EVIDENCE_TIMEOUT_SECONDS = 10.0
app = FastAPI(title="RepreGuard Detect Service")
app.add_middleware(DetectorIngressMiddleware, service_token=settings.service_token)


@app.on_event("startup")
def load_detector() -> None:
    """Initialize the detector pipeline once at process startup."""
    settings.require_service_token()
    get_detector()


@app.on_event("startup")
def load_evidence_router() -> None:
    """Optional loading must never prevent the main detector from starting."""
    global EVIDENCE_ROUTER, EVIDENCE_TIMEOUT_SECONDS
    EVIDENCE_ROUTER = None
    try:
        if settings.evidence_enabled.strip().lower() in {"", "false", "0", "no", "off"}:
            return
        timeout = float(settings.evidence_timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            return
        EVIDENCE_ROUTER = EvidenceRouter(
            enabled=settings.evidence_enabled,
            model_path=settings.evidence_model_path,
            artifact_sha256=settings.evidence_artifact_sha256,
        )
        EVIDENCE_TIMEOUT_SECONDS = timeout
    except Exception:
        EVIDENCE_ROUTER = None


@app.on_event("shutdown")
async def drain_inference() -> None:
    await asyncio.gather(INFERENCE_ADMISSION.drain(), EVIDENCE_ADMISSION.drain())


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == "/evidence/route":
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "INVALID_EVIDENCE_REQUEST",
                    "message": "Invalid Evidence request.",
                }
            },
            headers={"Cache-Control": "no-store"},
        )
    return await request_validation_exception_handler(request, exc)


def _evidence_failure(reason: str, sha: str | None = None) -> EvidenceResponse:
    return EvidenceResponse(status="failed", routerArtifactSha256=sha, reason=reason)


@app.post("/evidence/route", response_model=EvidenceResponse)
async def route_evidence(request: Request, req: EvidenceRequest) -> EvidenceResponse:
    router = EVIDENCE_ROUTER
    if router is None or router.status != "ready":
        return _evidence_failure("model_unavailable")
    sha = router.artifact_sha256
    try:
        result = await asyncio.wait_for(
            EVIDENCE_ADMISSION.run(request, req.text, handler=router.predict_route),
            timeout=EVIDENCE_TIMEOUT_SECONDS,
        )
        if result["routerArtifactSha256"] != sha:
            return _evidence_failure("model_failure", sha)
        if result["status"] == "predicted" and result["reason"] is None:
            return EvidenceResponse(
                status="routed",
                routerArtifactSha256=sha,
                route=EvidenceRoute(**result["route"]),
            )
        if result["route"] is None:
            if (
                result["status"] == "unsupported"
                and result["reason"] == "unsupported_language"
            ):
                return EvidenceResponse(
                    status="unsupported",
                    routerArtifactSha256=sha,
                    reason="unsupported_language",
                )
            if result["status"] == "failed" and result["reason"] in {
                "model_failure",
                "model_unavailable",
                "language_undetermined",
            }:
                return _evidence_failure(result["reason"], sha)
        return _evidence_failure("model_failure", sha)
    except InferenceQueueFull:
        return _evidence_failure("busy", sha)
    except (asyncio.TimeoutError, InferenceQueueTimeout):
        # Admission shields its worker: the slot remains occupied until the CPU
        # thread actually finishes, even though this request has timed out.
        return _evidence_failure("timeout", sha)
    except PendingRequestDisconnected as exc:
        raise HTTPException(
            status_code=499, detail={"code": "CLIENT_DISCONNECTED"}
        ) from exc
    except Exception:
        return _evidence_failure("model_failure", sha)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request, req: DetectRequest) -> DetectResponse:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "TEXT_EMPTY", "message": "Text cannot be empty."},
        )

    try:
        result: DetectResult = await INFERENCE_ADMISSION.run(request, text)
    except InferenceQueueFull as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "DETECT_QUEUE_FULL", "message": "Detect service is busy."},
            headers={"Retry-After": str(INFERENCE_ADMISSION.retry_after_seconds)},
        ) from exc
    except InferenceQueueTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DETECT_QUEUE_TIMEOUT",
                "message": "Detect request waited too long for capacity.",
            },
            headers={"Retry-After": str(INFERENCE_ADMISSION.retry_after_seconds)},
        ) from exc
    except PendingRequestDisconnected as exc:
        raise HTTPException(
            status_code=499,
            detail={
                "code": "CLIENT_DISCONNECTED",
                "message": "Client disconnected while waiting for capacity.",
            },
        ) from exc
    except DetectServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "DETECT_INTERNAL_ERROR",
                "message": "Detect service failed unexpectedly.",
            },
        ) from exc

    return DetectResponse(
        score=result.score,
        threshold=result.threshold,
        label=result.label,
        model_name=result.model,
        score_type=result.score_type,
    )
