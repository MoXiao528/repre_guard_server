import asyncio
import math
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from config import settings
from ingress import DetectorIngressMiddleware
from repreGuard_service import DetectResult, DetectServiceError, detect_text, get_detector


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str
    score_type: Literal["probability"] = "probability"


class InferenceQueueFull(Exception):
    pass


class InferenceQueueTimeout(Exception):
    pass


class PendingRequestDisconnected(Exception):
    pass


class InferenceAdmission:
    DISCONNECT_POLL_SECONDS = 0.05

    def __init__(self, *, max_pending_requests: int, queue_timeout_seconds: float) -> None:
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
        self._workers: set[asyncio.Task[DetectResult]] = set()

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

    async def _execute_and_release(self, text: str) -> DetectResult:
        try:
            return await asyncio.to_thread(detect_text, text)
        finally:
            self._active -= 1
            self._gpu_semaphore.release()
            self._release_admission()

    def _worker_done(self, task: asyncio.Task[DetectResult]) -> None:
        self._workers.discard(task)
        if not task.cancelled():
            task.exception()

    async def run(self, request: Request, text: str) -> DetectResult:
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
                worker = asyncio.create_task(self._execute_and_release(text))
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
            await asyncio.gather(*(asyncio.shield(worker) for worker in workers), return_exceptions=True)


settings.require_valid_admission_settings()
INFERENCE_ADMISSION = InferenceAdmission(
    max_pending_requests=settings.max_pending_requests,
    queue_timeout_seconds=settings.queue_timeout_seconds,
)
app = FastAPI(title="RepreGuard Detect Service")
app.add_middleware(DetectorIngressMiddleware, service_token=settings.service_token)


@app.on_event("startup")
def load_detector() -> None:
    """Initialize the detector pipeline once at process startup."""
    settings.require_service_token()
    get_detector()


@app.on_event("shutdown")
async def drain_inference() -> None:
    await INFERENCE_ADMISSION.drain()


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
            detail={"code": "DETECT_QUEUE_TIMEOUT", "message": "Detect request waited too long for capacity."},
            headers={"Retry-After": str(INFERENCE_ADMISSION.retry_after_seconds)},
        ) from exc
    except PendingRequestDisconnected as exc:
        raise HTTPException(
            status_code=499,
            detail={"code": "CLIENT_DISCONNECTED", "message": "Client disconnected while waiting for capacity."},
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
