"""Opt-in D1 real-model acceptance. No installs, downloads or production changes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parent
SAMPLES = 30
SUPPORTED = {
    "en": "The library will open every Sunday so students can read and study together.",
    "fr": "La bibliothèque ouvrira le dimanche pour permettre aux étudiants de travailler ensemble.",
    "de": "Die Bibliothek ist auch am Sonntag geöffnet, damit die Studierenden gemeinsam lernen können.",
    "es": "La biblioteca abrirá los domingos para que los estudiantes puedan estudiar juntos.",
    "pt": "A biblioteca estará aberta aos domingos para que os estudantes possam estudar juntos.",
    "ru": "Библиотека будет открыта по воскресеньям, чтобы студенты могли заниматься вместе.",
    "ar": "ستفتح المكتبة أبوابها يوم الأحد حتى يتمكن الطلاب من القراءة والدراسة معًا.",
    "zh": "市图书馆下个月开始在星期日开放，学生可以预约自习室，一起阅读和学习。",
}
ITALIAN = "La biblioteca sarà aperta anche la domenica e gli studenti potranno studiare insieme."
PACKAGES = (
    "torch",
    "transformers",
    "safetensors",
    "tokenizers",
    "sentencepiece",
    "numpy",
    "py3langid",
    "psutil",
    "httpx",
    "uvicorn",
    "fastapi",
    "pydantic",
)


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main_samples():
    from loadtest_detector_api import build_text

    return [build_text(500, i) for i in range(SAMPLES)]


def same_detection(expected, actual):
    return (
        set(expected) == set(actual)
        and all(expected[k] == actual[k] for k in expected if k != "score")
        and math.isfinite(actual["score"])
        and abs(expected["score"] - actual["score"]) <= 1e-6
    )


def worker(mode, port, output):
    # Instrument only the temporary process, never replace model predictions.
    import uvicorn
    import server
    from repreGuard_service import get_detector

    obs = {"mode": mode, "calls": 0, "completions": 0, "forwards": 0}

    @server.app.on_event("startup")
    def observe():
        import torch

        detector = get_detector()
        obs.update(
            main_device=str(detector.device),
            main_threshold=detector.threshold,
            main_revision=server.settings.model_revision,
            loaded_rss_bytes=psutil.Process().memory_info().rss,
            packages=versions(),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            py3langid_imported="py3langid" in sys.modules,
            main_max_sample_tokens=max(
                detector.count_tokens(t) for t in main_samples()
            ),
        )
        require(
            obs["main_max_sample_tokens"] <= server.settings.max_input_tokens,
            "main_sample_too_long",
        )
        router = server.EVIDENCE_ROUTER
        obs["router_status"] = None if router is None else router.status
        if router is not None and router.status == "ready":
            obs.update(
                router_sha=router.artifact_sha256,
                router_device=str(next(router._model.parameters()).device),
                router_dtype=str(next(router._model.parameters()).dtype),
                router_training=router._model.training,
            )
            original = router.predict_route

            def observed(text):
                obs["calls"] += 1
                try:
                    return original(text)
                finally:
                    obs["completions"] += 1

            def forward_started(*_):
                obs["forwards"] += 1

            router.predict_route = observed
            router._model.register_forward_pre_hook(forward_started)
        write_json(output, obs)

    instance = uvicorn.Server(
        uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            workers=1,
            access_log=False,
            log_level="warning",
        )
    )

    def stop():
        sys.stdin.readline()
        instance.should_exit = True

    threading.Thread(target=stop, daemon=True).start()
    try:
        instance.run()
    finally:
        import torch

        obs.update(
            final_admitted=server.EVIDENCE_ADMISSION.admitted_count,
            final_active=server.EVIDENCE_ADMISSION.active_count,
            final_workers=len(server.EVIDENCE_ADMISSION._workers),
            final_rss_bytes=psutil.Process().memory_info().rss,
        )
        if torch.cuda.is_available():
            obs.update(
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_name=torch.cuda.get_device_name(),
            )
        write_json(output, obs)


async def request(client, path, text=None):
    start = time.perf_counter()
    response = await (
        client.get(path) if text is None else client.post(path, json={"text": text})
    )
    require(response.status_code == 200, f"http_{response.status_code}")
    return {
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "payload": response.json(),
    }


def validate_main(item):
    from loadtest_detector_api import validate_detector_payload

    p = item["payload"]
    require(validate_detector_payload(p) is None, "main_contract")
    require(
        set(p) == {"score", "threshold", "label", "model_name", "score_type"},
        "main_fields",
    )
    require(
        type(p["score"]) in (int, float)
        and math.isfinite(p["score"])
        and 0 <= p["score"] <= 1,
        "main_score",
    )


def validate_evidence(engine, item, status, reason=None):
    p = item["payload"]
    require(engine.validate_router_response(p) == p, "backend_contract_rejected")
    require(
        p["status"] == status and p["reason"] == reason,
        f"evidence_expected_{status}_{reason}",
    )
    if status == "routed":
        require(item["elapsed_ms"] < 10000, "evidence_deadline")


async def measure(client, engine, mode, baseline, result):
    from loadtest_detector_api import build_text, summarize_latencies

    texts, long_text = main_samples(), build_text(20000, 0)
    for text in texts[:3]:
        validate_main(await request(client, "/detect", text))
    if mode == "off":
        item = await request(client, "/evidence/route", texts[0])
        validate_evidence(engine, item, "failed", "model_unavailable")
        require(item["payload"]["routerArtifactSha256"] is None, "off_sha")
    elif mode == "on":
        result["functional"] = []
        for name, text in [
            *SUPPORTED.items(),
            ("long_en", long_text),
            ("long_zh", (SUPPORTED["zh"] * 1000)[:20000]),
        ]:
            item = await request(client, "/evidence/route", text)
            result["functional"].append({"case": name, **item})
            validate_evidence(engine, item, "routed")
        for name, text, status, reason in (
            ("italian", ITALIAN, "unsupported", "unsupported_language"),
            (
                "no_information",
                "!? ... --- ### *** @@@ !!!",
                "failed",
                "language_undetermined",
            ),
        ):
            item = await request(client, "/evidence/route", text)
            result["functional"].append({"case": name, **item})
            validate_evidence(engine, item, status, reason)
    else:
        # Only fault injection: a shorter deadline, with the real model untouched.
        first = await request(client, "/evidence/route", long_text)
        result["timeout_recovery"] = {"timeout": first}
        validate_evidence(engine, first, "failed", "timeout")
        busy = await request(client, "/evidence/route", SUPPORTED["en"])
        validate_evidence(engine, busy, "failed", "busy")
        main = await request(client, "/detect", texts[0])
        validate_main(main)
        require(
            same_detection(baseline[0]["payload"], main["payload"]),
            "main_changed_during_timeout",
        )
        probes, deadline = 0, time.monotonic() + 30
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            item = await request(client, "/evidence/route", ITALIAN)
            if item["payload"]["reason"] != "busy":
                validate_evidence(engine, item, "unsupported", "unsupported_language")
                result["timeout_recovery"] = {
                    "timeout": first,
                    "busy": busy,
                    "main": main,
                    "recovered": item,
                    "busy_probes": probes,
                }
                return
            validate_evidence(engine, item, "failed", "busy")
            probes += 1
        raise RuntimeError("worker_did_not_recover_within_30s")

    result["main_idle"] = []
    for i, text in enumerate(texts):
        item = await request(client, "/detect", text)
        validate_main(item)
        if baseline:
            require(
                same_detection(baseline[i]["payload"], item["payload"]),
                "main_changed_idle",
            )
        result["main_idle"].append(item)
    result["main_idle_latency"] = summarize_latencies(
        [r["elapsed_ms"] for r in result["main_idle"]]
    )
    if mode == "off":
        return
    result["paired"] = []
    for i, text in enumerate(texts):
        evidence, main = await asyncio.gather(
            request(client, "/evidence/route", long_text),
            request(client, "/detect", text),
        )
        validate_evidence(engine, evidence, "routed")
        validate_main(main)
        require(
            same_detection(baseline[i]["payload"], main["payload"]),
            "main_changed_concurrent",
        )
        result["paired"].append({"main": main, "evidence": evidence})
    for name in ("main", "evidence"):
        result[name + "_concurrent_latency"] = summarize_latencies(
            [p[name]["elapsed_ms"] for p in result["paired"]]
        )
    burst = await asyncio.gather(
        *(request(client, "/evidence/route", long_text) for _ in range(4))
    )
    for item in burst:
        if item["payload"]["status"] == "routed":
            validate_evidence(engine, item, "routed")
        else:
            validate_evidence(engine, item, "failed", "busy")
    require(
        any(item["payload"]["reason"] == "busy" for item in burst),
        "burst_did_not_exercise_busy",
    )
    validate_evidence(
        engine, await request(client, "/evidence/route", SUPPORTED["en"]), "routed"
    )
    result["burst"] = burst


async def run_mode(args, mode, engine, baseline, result):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_hex(32)
    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        REPRE_GUARD_LOCAL_FILES_ONLY="true",
        REPRE_GUARD_SERVICE_TOKEN=token,
        REPRE_GUARD_HOST="127.0.0.1",
        REPRE_GUARD_PORT=str(port),
        REPRE_GUARD_EVIDENCE_ENABLED="false" if mode == "off" else "true",
        REPRE_GUARD_EVIDENCE_MODEL_PATH=str(args.router_model),
        REPRE_GUARD_EVIDENCE_ARTIFACT_SHA256=args.router_sha,
        REPRE_GUARD_EVIDENCE_TIMEOUT_SECONDS="0.02" if mode == "timeout" else "10",
    )
    if args.lid_site:
        env["PYTHONPATH"] = str(args.lid_site) + os.pathsep + env.get("PYTHONPATH", "")
    info_path = args.output.with_name(args.output.stem + f"-{mode}-worker.json")
    log_path = args.output.with_name(args.output.stem + f"-{mode}.log")
    result.update(
        mode=mode,
        port=port,
        peak_rss_bytes=0,
        min_available_bytes=psutil.virtual_memory().available,
    )
    stopped, started = threading.Event(), time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--port",
                str(port),
                "--output",
                str(info_path),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result["pid"] = process.pid
        observed = psutil.Process(process.pid)
        reserve = max(2 * 1024**3, psutil.virtual_memory().total * 0.1)

        def monitor():
            while not stopped.wait(0.05):
                try:
                    rss, available = (
                        observed.memory_info().rss,
                        psutil.virtual_memory().available,
                    )
                    result["peak_rss_bytes"] = max(result["peak_rss_bytes"], rss)
                    result["min_available_bytes"] = min(
                        result["min_available_bytes"], available
                    )
                    if available < reserve:
                        result["resource_abort"] = "memory_reserve"
                        process.kill()  # Only the process created immediately above.
                        return
                except psutil.NoSuchProcess:
                    return

        watch = threading.Thread(target=monitor, daemon=True)
        watch.start()
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                headers={"X-RepreGuard-Token": token},
                timeout=40,
                trust_env=False,
            ) as client:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    require(process.poll() is None, "server_exited_during_startup")
                    try:
                        if (
                            await client.get("/health", timeout=0.5)
                        ).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("startup_timeout")
                result["startup_seconds"] = time.perf_counter() - started
                info = json.loads(info_path.read_text(encoding="utf-8"))
                require(
                    info["main_device"] in {"cuda", "cuda:0"},
                    "expected_gpu_main_detector",
                )
                if mode == "off":
                    require(
                        info["router_status"] is None
                        and not info["py3langid_imported"],
                        "off_loaded_evidence",
                    )
                else:
                    require(info["router_status"] == "ready", "router_not_ready")
                    require(
                        info["router_sha"] == args.router_sha
                        and info["router_device"] == "cpu"
                        and info["router_dtype"] == "torch.float32"
                        and info["router_training"] is False,
                        "router_runtime_contract",
                    )
                print(
                    json.dumps(
                        {
                            "stage": mode,
                            "startup_seconds": result["startup_seconds"],
                            "main_device": info["main_device"],
                            "router_status": info["router_status"],
                        }
                    ),
                    flush=True,
                )
                await measure(client, engine, mode, baseline, result)
                result["passed"] = True
        finally:
            if process.poll() is None:
                process.stdin.close()  # EOF requests normal shutdown and admission drain.
                try:
                    await asyncio.to_thread(process.wait, 40)
                except subprocess.TimeoutExpired:
                    result["forced_stop"] = True
                    process.kill()
                    await asyncio.to_thread(process.wait, 5)
            stopped.set()
            watch.join(timeout=1)
            result.update(
                exit_code=process.poll(),
                elapsed_seconds=time.perf_counter() - started,
                process_stopped=process.poll() is not None,
            )
            if info_path.exists():
                result["worker"] = json.loads(info_path.read_text(encoding="utf-8"))
        require(
            result["exit_code"] == 0
            and not result.get("forced_stop")
            and not result.get("resource_abort"),
            "server_not_cleanly_stopped",
        )
        info = result["worker"]
        require(
            all(
                info[k] == 0
                for k in ("final_admitted", "final_active", "final_workers")
            ),
            "admission_not_drained",
        )
        require(info["calls"] == info["completions"], "prediction_not_completed")
        if mode == "timeout":
            require(
                info["forwards"] == 1, "timeout_did_not_run_exactly_one_real_forward"
            )


async def run(args):
    if args.lid_site:
        sys.path.insert(0, str(args.lid_site))
    sys.path.insert(0, str(args.backend_root / "backend"))
    from app.services.evidence_engine import EvidenceEngine

    require(not args.output.exists(), "report_already_exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = EvidenceEngine(
        mode="shadow", bundle_path=str(args.bundle), bundle_sha256=args.bundle_sha
    )
    require(
        engine.status == "ready" and engine.router_artifact_sha256 == args.router_sha,
        "bundle_not_ready_or_router_mismatch",
    )
    report = {
        "schema_version": 1,
        "target_environment_verified": False,
        "purpose": "Local real-model smoke and bounded coexistence screen; not routing accuracy or deployment certification",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "router_sha256": args.router_sha,
        "bundle_sha256": args.bundle_sha,
        "protocol": {
            "samples_per_condition": SAMPLES,
            "main_chars": 500,
            "concurrent_evidence_chars": 20000,
            "main_score_abs_tolerance": 1e-6,
            "p95_increase_budget": "max(baseline*0.2,50ms)",
            "evidence_timeout_seconds": 10,
            "injected_timeout_seconds": 0.02,
            "memory_reserve": "max(2GiB,total*0.1)",
            "warmups": 3,
        },
        "conditions": [],
        "passed": False,
    }
    write_json(args.output, report)  # Freeze the protocol before starting a model.
    try:
        baseline = None
        for mode in ("off", "on", "timeout"):
            result = {}
            report["conditions"].append(result)
            await run_mode(args, mode, engine, baseline, result)
            if mode == "off":
                baseline = result["main_idle"]
            write_json(args.output, report)
        off, on, _ = report["conditions"]
        p95 = off["main_idle_latency"]["p95"]
        allowed = p95 + max(p95 * 0.2, 50)
        report["performance"] = {
            "main_baseline_p95_ms": p95,
            "main_p95_max_ms": allowed,
            "idle_passed": on["main_idle_latency"]["p95"] <= allowed,
            "concurrent_passed": on["main_concurrent_latency"]["p95"] <= allowed,
        }
        report["passed"] = all(
            report["performance"][k] for k in ("idle_passed", "concurrent_passed")
        )
    except Exception as exc:
        report["failure"] = {
            "type": type(exc).__name__,
            "code": str(exc) if type(exc) is RuntimeError else "acceptance_exception",
        }
    finally:
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "failure": report.get("failure"),
                    "performance": report.get("performance"),
                    "report": str(args.output),
                }
            ),
            flush=True,
        )
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("router-model", "bundle", "backend-root", "lid-site"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--router-sha")
    parser.add_argument("--bundle-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--worker", choices=("off", "on", "timeout"), help=argparse.SUPPRESS
    )
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.port, args.output)
    else:
        require(
            all(
                (
                    args.router_model,
                    args.router_sha,
                    args.bundle,
                    args.bundle_sha,
                    args.backend_root,
                )
            ),
            "missing_explicit_artifact_arguments",
        )
        raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()
