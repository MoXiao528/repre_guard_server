from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import is_valid_service_token


DEFAULT_URL = "http://127.0.0.1:9000/detect"
DEFAULT_USERS = 100
DEFAULT_ROUNDS = 1
DEFAULT_CHARS = 500
DEFAULT_TIMEOUT = 300.0
RESULTS_DIR = Path(__file__).resolve().parent / "loadtest_results"

SENTENCES = [
    "This load test sample describes a normal product review workflow with clear context and ordinary business language. ",
    "The team checks the service response, compares latency between requests, and records failed samples for later debugging. ",
    "The text intentionally avoids unusual symbols, repeated random tokens, and adversarial wording so the detector receives realistic input. ",
    "Each request exercises the internal detection endpoint and returns the model score, threshold, label, and score type. ",
    "The report keeps per-request timing, status codes, response sizes, and a small set of failure details for quick inspection. ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test the detector API directly.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--chars", type=int, default=DEFAULT_CHARS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--verify-tls", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be > 0")


def build_text(target_chars: int, request_id: int) -> str:
    parts = [f"Load test sample {request_id + 1}. "]
    cursor = request_id % len(SENTENCES)
    while len("".join(parts)) < target_chars:
        parts.append(SENTENCES[cursor % len(SENTENCES)])
        cursor += 1
    return "".join(parts)[:target_chars]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    index = math.ceil((p / 100) * len(ordered)) - 1
    return round(ordered[max(0, min(index, len(ordered) - 1))], 2)


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "avg": round(statistics.fmean(values), 2),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": round(max(values), 2),
    }


def validate_detector_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "response JSON is not an object"
    for key in ("score", "threshold", "label", "model_name", "score_type"):
        if key not in payload:
            return f"response missing {key}"
    if payload.get("score_type") != "probability":
        return f"unexpected score_type={payload.get('score_type')!r}"
    if payload.get("label") not in {"AI", "HUMAN"}:
        return f"unexpected label={payload.get('label')!r}"
    return None


async def detect_once(
    *,
    client: httpx.AsyncClient,
    url: str,
    text: str,
    request_id: int,
    round_index: int,
    service_token: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        response = await client.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-RepreGuard-Token": service_token,
            },
            json={"text": text},
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        failure_detail = None
        parsed: Any = None
        validation_error = None

        if response.headers.get("content-type", "").startswith("application/json"):
            parsed = response.json()
            validation_error = validate_detector_payload(parsed) if response.status_code == 200 else None

        if response.status_code != 200:
            failure_detail = response.text[:500]
        elif validation_error:
            failure_detail = validation_error

        return {
            "request_id": request_id,
            "round": round_index + 1,
            "status_code": response.status_code,
            "ok": response.status_code == 200 and validation_error is None,
            "elapsed_ms": elapsed_ms,
            "response_size_bytes": len(response.content),
            "score": parsed.get("score") if isinstance(parsed, dict) else None,
            "label": parsed.get("label") if isinstance(parsed, dict) else None,
            "score_type": parsed.get("score_type") if isinstance(parsed, dict) else None,
            "failure_detail": failure_detail,
            "error": None,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "request_id": request_id,
            "round": round_index + 1,
            "status_code": None,
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "response_size_bytes": 0,
            "score": None,
            "label": None,
            "score_type": None,
            "failure_detail": None,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }


async def run_loadtest(args: argparse.Namespace, *, service_token: str) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    timeout = httpx.Timeout(args.timeout)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=timeout,
        verify=args.verify_tls,
        limits=limits,
        follow_redirects=False,
    ) as client:
        for round_index in range(args.rounds):
            batch = []
            for request_index in range(args.users):
                sequence = round_index * args.users + request_index
                batch.append(
                    detect_once(
                        client=client,
                        url=args.url,
                        text=build_text(args.chars, sequence),
                        request_id=sequence + 1,
                        round_index=round_index,
                        service_token=service_token,
                    )
                )
            results.extend(await asyncio.gather(*batch))

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    latencies = [float(item["elapsed_ms"]) for item in results]
    status_counts = Counter(
        str(item["status_code"]) if item["status_code"] is not None else "EXCEPTION"
        for item in results
    )
    label_counts = Counter(str(item["label"]) for item in results if item["label"])
    failures = [item for item in results if not item["ok"]]

    return {
        "url": args.url,
        "users": args.users,
        "rounds": args.rounds,
        "chars_per_request": args.chars,
        "total_requests": len(results),
        "duration_ms": duration_ms,
        "rps": round((len(results) / duration_ms) * 1000, 2) if duration_ms > 0 else 0.0,
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "success_count": len(results) - len(failures),
        "failure_count": len(failures),
        "latency": summarize_latencies(latencies),
        "failed_samples": failures[:10],
        "request_results": results,
    }


def write_report(report: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"detector-api-{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def async_main() -> None:
    args = parse_args()
    require_positive("--users", args.users)
    require_positive("--rounds", args.rounds)
    require_positive("--chars", args.chars)
    service_token = os.getenv("REPRE_GUARD_SERVICE_TOKEN", "")
    if not is_valid_service_token(service_token):
        raise SystemExit(
            "REPRE_GUARD_SERVICE_TOKEN must contain at least 32 printable ASCII characters without whitespace."
        )

    print(
        f"target={args.url} users={args.users} rounds={args.rounds} "
        f"requests={args.users * args.rounds} chars≈{args.users * args.rounds * args.chars}"
    )
    report = await run_loadtest(args, service_token=service_token)
    report_path = write_report(report)
    print(
        f"done: success={report['success_count']} failure={report['failure_count']} "
        f"rps={report['rps']} latency={report['latency']} report={report_path}"
    )
    if report["failure_count"]:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
