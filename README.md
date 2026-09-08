# RepreGuard Server V2.0

FastAPI detection service for AIDetector V2.0.

V2.0 serves the DetectRL-X XLM-RoBERTa detector and keeps a strict probability contract:

```json
{
  "score": 0.037,
  "threshold": 0.0028,
  "label": "AI",
  "model_name": "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All",
  "score_type": "probability"
}
```

For DetectRL-X, `score` is the AI probability for `LABEL_1`; `threshold` is the calibrated probability threshold.
`score_type` is always `probability`; clients should reject missing or different values instead of guessing.

## V2.0 Model

Default model:

```text
WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All
```

The model config does not publish semantic label names, so deployment pins:

```text
AI label = LABEL_1
threshold = 0.0028
```

## Download Model

Recommended local model directory:

```powershell
D:\Anaconda\envs\lab\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All', revision='76649a0257a812a81cf36b5de9cc5f2430aeaa7f', local_dir=r'D:\huggingface\WUJUNCHAO\DetectRL-X-XLM-RoBERTa-Detector-All', allow_patterns=['config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer_config.json'])"
```

If the local directory exists, startup validates the required files and the pinned revision recorded in Hugging Face `*.metadata` files or a local `.repre_guard_model_manifest.json` / `model_manifest.json`. A directory with only `model.safetensors`, missing tokenizer/config files, or files from a different revision is rejected before the server starts.

If the local directory does not exist, the service falls back to the pinned Hugging Face repo revision with cache directory `D:\huggingface`. With the production default `REPRE_GUARD_LOCAL_FILES_ONLY=true`, that fallback still requires the files to be present locally.

## Configuration

Copy `.env.example` to `.env`, then configure the service. RepreGuard always resolves this file next to
`config.py`, so startup does not depend on the current working directory. Explicit process/container environment
variables take precedence over `.env`.

The paths below match the local Windows setup. On Linux, set `REPRE_GUARD_MODEL_CACHE_DIR` and
`REPRE_GUARD_MODEL_PATH` in `.env` to real absolute server paths, such as the commented examples in
`.env.example`.

```text
REPRE_GUARD_MODEL_NAME=WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All
REPRE_GUARD_MODEL_REVISION=76649a0257a812a81cf36b5de9cc5f2430aeaa7f
REPRE_GUARD_MODEL_CACHE_DIR=D:\huggingface
REPRE_GUARD_MODEL_PATH=D:\huggingface\WUJUNCHAO\DetectRL-X-XLM-RoBERTa-Detector-All
REPRE_GUARD_LOCAL_FILES_ONLY=true
REPRE_GUARD_THRESHOLD=0.0028
REPRE_GUARD_AI_LABEL_ID=1
REPRE_GUARD_TOKENIZER_USE_FAST=false
REPRE_GUARD_MAX_INPUT_TOKENS=512
REPRE_GUARD_MAX_PENDING_REQUESTS=3
REPRE_GUARD_QUEUE_TIMEOUT_SECONDS=15
REPRE_GUARD_HOST=0.0.0.0
REPRE_GUARD_PORT=9000
REPRE_GUARD_SERVICE_TOKEN=<same value as AIDetector-Back/.env>
```

Production should run from the pinned local model files. Set `REPRE_GUARD_LOCAL_FILES_ONLY=false` only for an explicit download/cache warm-up flow.

`REPRE_GUARD_SERVICE_TOKEN` is mandatory and must contain at least 32 printable ASCII characters without whitespace. Generate it once, then put the same value in `AIDetector-Back/.env` and this repository's `.env`. RepreGuard validates it before loading the model; never commit it or print it in logs.

The default `0.0.0.0` bind is intentional for the local Docker backend to reach the Windows-hosted detector through `host.docker.internal`. Keep the port blocked from untrusted networks with the host firewall.

Inputs longer than `REPRE_GUARD_MAX_INPUT_TOKENS` are rejected with `INPUT_TOO_LONG`; the service no longer silently truncates detection input.

Inference admission is bounded to one active GPU request plus `REPRE_GUARD_MAX_PENDING_REQUESTS` queued requests. The defaults allow one active request and three queued requests, matching the backend's four concurrent text segments. A fifth request is rejected immediately with `503 DETECT_QUEUE_FULL`; a request waiting longer than `REPRE_GUARD_QUEUE_TIMEOUT_SECONDS` returns `503 DETECT_QUEUE_TIMEOUT`. Both responses include an integer `Retry-After` header (5 seconds with the defaults). A queued request that disconnects is removed and never reaches the model.

Tune the queue from a warmed-up local single-request measurement: `max pending <= floor(max acceptable wait / p95 inference time)`. The default `3` and `15` seconds assume p95 inference is at most 5 seconds. Keep the queue timeout below the backend's detect-service timeout.

## Start on Windows

```powershell
D:\Anaconda\envs\lab\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and copy the existing REPRE_GUARD_SERVICE_TOKEN from AIDetector-Back/.env.
D:\Anaconda\envs\lab\python.exe .\run_roberta_server.py
```

PowerShell wrapper:

```powershell
.\run_local_roberta_server.ps1
```

## Start on Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
# Edit .env: set the shared token and Linux model paths before starting.
python run_roberta_server.py
```

## Endpoints

```text
GET  /health
POST /detect
POST /evidence/route
```

All three endpoints are internal and require this header. Evidence is disabled by default:

```text
X-RepreGuard-Token: <REPRE_GUARD_SERVICE_TOKEN>
```

Missing, incorrect, or duplicate token headers return `401` before the request body or model is touched. `/detect` accepts at most 131072 request-body bytes; the limit is checked from both `Content-Length` and the actual ASGI body stream before JSON parsing. Invalid or duplicate `Content-Length` returns `400`, and an oversized body returns `413`.

Authenticated health probe:

```powershell
$serviceToken = $env:REPRE_GUARD_SERVICE_TOKEN
if ([string]::IsNullOrWhiteSpace($serviceToken)) {
  $serviceToken = (Get-Content .env | Where-Object { $_ -like "REPRE_GUARD_SERVICE_TOKEN=*" } | Select-Object -First 1) `
    -replace "^REPRE_GUARD_SERVICE_TOKEN=", ""
}
Invoke-RestMethod http://127.0.0.1:9000/health `
  -Headers @{ "X-RepreGuard-Token" = $serviceToken }
```

Request:

```json
{
  "text": "Sample text to classify."
}
```

Response:

```json
{
  "score": 0.0012,
  "threshold": 0.0028,
  "label": "HUMAN",
  "model_name": "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All",
  "score_type": "probability"
}
```

## Direct Load Test

This bypasses the AIDetector backend quota, user auth, database, and history path. It still uses the internal detector service token from `REPRE_GUARD_SERVICE_TOKEN`:

```powershell
D:\Anaconda\envs\lab\python.exe .\loadtest_detector_api.py `
  --url http://127.0.0.1:9000/detect `
  --users 100 `
  --rounds 1 `
  --chars 500
```

The report is written to `loadtest_results/detector-api-YYYYMMDD-HHMMSS.json`.

With bounded admission, a burst larger than the configured active-plus-pending capacity is expected to contain `503` responses instead of building an unbounded wait queue.

## Evidence V1 internal Router (EV4-07 DONE_LOCAL)

`evidence_router.py` loads the existing production-final Evidence Router on CPU.
When explicitly enabled, server startup constructs one instance and the internal
`POST /evidence/route` endpoint checks language support before model inference.
The backend connects its detection flow to this endpoint once per full document
(EV4-02) and persists validated Evidence snapshots for history/replay (EV4-04).
The local real-model backend business chain passed on 2026-09-07. Main detector
scores, thresholds, labels and `/health` behavior are unchanged; default is off.
Local real-model acceptance passed on 2026-09-07, completing EV4-07 locally.
The owner deferred deployment work; the target environment remains unverified.
Persistent Evidence configuration remains off; local acceptance does not enable public Evidence.

```python
from config import settings
from evidence_router import EvidenceRouter

router = EvidenceRouter(
    enabled=settings.evidence_enabled,
    model_path=settings.evidence_model_path,
    artifact_sha256=settings.evidence_artifact_sha256,
)
result = router.predict_route(original_text)
```

| Configuration | Default | Meaning |
|---|---|---|
| `REPRE_GUARD_EVIDENCE_ENABLED` | `false` | Load at server startup; accepts true/false, 1/0, yes/no, on/off |
| `REPRE_GUARD_EVIDENCE_MODEL_PATH` | empty | Immutable local `production-final/model` directory, not the parent containing research reports |
| `REPRE_GUARD_EVIDENCE_ARTIFACT_SHA256` | empty | One deployment pin for the six-file Router artifact |
| `REPRE_GUARD_EVIDENCE_TIMEOUT_SECONDS` | `10` | Finite response deadline, greater than 0 and at most 60 seconds |

These optional settings remain raw strings until the consumer validates them, so
invalid Evidence settings do not fail the main service's configuration import.
The local instance has `off/ready/failed` loading states. Off performs no filesystem
or model work. Construction verifies and loads once; failure is also retained, with
an explicit new instance/process restart required to retry loading. Invalid settings
or loading errors disable Evidence without raising into the main startup handler.
Requests reuse both models and the verified identity without filesystem access or
repeated SHA calculation. Off does not import or load the language identifier.

The directory must contain exactly `config.json`, `router_config.json`,
`router_model.safetensors`, `sentencepiece.bpe.model`, `tokenizer.json`, and
`tokenizer_config.json`. Regular files only; symlinks/reparse points, missing or extra
entries, changed files, incorrect SHA and incompatible configuration are rejected.
The two model/router configuration JSON files are canonical and limited to 64 KiB
each. Router class order, model label maps, and the fast tokenizer's vocabulary,
special tokens and 512-token contract are validated before inference.

The deployment SHA is the SHA-256 of the canonical six-file claim mapping
`{filename: {bytes, sha256}}`: UTF-8 JSON with `ensure_ascii=False`, sorted keys,
`allow_nan=False`, two-space indentation and a trailing newline. It is **not** the
weight file's SHA. The frozen production Router pin is
`9a1a5be8f0d7599e682d30c7f50e3deb5f747b59ef720e288f6342e7c944f896`.
File hashes are computed once to derive that claim; there are no additional
per-file deployment pins or research-lineage checks. Ordinary changes during load
are checked by file identity/stat snapshots. Deployment must provide an immutable
directory; those metadata checks cannot prove immutability against hostile writes.

The consumer preserves the published encoder and nine head state names, loads
safetensors strictly in CPU FP32, and uses eval/inference mode. It does not import
the research Trainer or set training seeds, CUDA state or process-wide thread
limits. Existing `torch/transformers/safetensors/tokenizers/sentencepiece`
dependencies suffice for this model; no separate Router requirements file is used.

`py3langid==0.4.0` performs the language-support check and is declared
in the existing `requirements.txt`. It requires Python >=3.10 and NumPy >=2.0.
The 2026-09-07 isolated candidate screen passed its predeclared engineering budgets:
191/192 supported-language documents retained, 24/24 synthetic out-of-scope examples
rejected, and 16/16 formatting variants retained. The one rejection was a Chinese
product listing classified as `wuu`; it also fails the existing ten-sentence rule,
but remains counted as a language-check error. Local CPU P95 was 1.84 ms for normal
inputs and 5.78 ms for repeated 20,000-character timing inputs, with 75.46 MiB added
RSS after loading (NumPy already imported). These are small-sample local results,
not production compatibility, concurrency or accuracy guarantees. The research
repository records the protocol and results in `outputs/py3langid_screen_20260907.json`.
Only the isolated research cache received an installation; the existing lab/service
environment has not been changed. Tests can use that existing cache through their
process-local Python path; deployment must provide the declared dependency.

The identifier retains its full candidate language set with `norm_probs=True` and
no confidence threshold. Startup caches its public `rank("")` result. Each request
calls `rank` once on the entire normalized document, before XLM-R tokenization:

- An exact match to the empty-input ranking, or top label `und/zxx`, returns
  `failed/language_undetermined`. This catches no-information outputs, including
  the library's alias-folded `sr` result, without labeling them unsupported.
- A top label outside `ar/de/en/es/fr/pt/ru/zh` returns
  `unsupported/unsupported_language`; `wuu/yue/ary/arz` are not mapped to `zh/ar`.
- A supported top label allows the frozen XLM-R route unchanged, even if its
  language differs from the identifier. Neither confidence nor agreement gates it.

Short and mixed texts follow the same whole-document rule, with no segment voting
or Mixed result. This fallible classifier is not a universal unknown-language or
nonlinguistic detector; feature-free checks do not identify all meaningless input.

`predict_route` accepts one original string of 1–20,000 code points. Empty or
whitespace-only input is invalid. The frozen preprocessing applies NFKC and whitespace
normalization, tokenizes once, keeps the first 255 and last 255 payload tokens when
needed, adds two special tokens, and pads to a multiple of eight up to 512 tokens.
This model input policy does not truncate the backend's full-text feature analysis.
Each supported request performs one model forward, chooses language from aggregated domain mass,
then chooses domain within that language. Temperature is 1; low confidence never
causes abstention. A global 48-class argmax is not the operational route.

Internal results contain only `status`, `routerArtifactSha256`, `route`, and
`reason`. Successful `status=predicted` returns `{language, domain,
confidence:{language,domain}}`; confidence values are diagnostic, not correctness
probabilities. The endpoint wraps a successful internal prediction as `routed`.
No original text, generator, local path, AI/Human label, score or threshold is returned.

Off returns no route or SHA. Failed loading returns no route/SHA and a fixed
`invalid_evidence_router_config` or `model_unavailable` reason. Invalid text returns
`failed/invalid_text`; inference exceptions and invalid model outputs return
`failed/model_failure`, preserving the loaded SHA. Failures expose no partial route
or exception text and do not poison subsequent predictions.

The endpoint accepts exactly `{"text":"original full document"}` and requires the
existing `X-RepreGuard-Token`. The existing ingress enforces 128 KiB including
streamed bodies; the strict request requires 1–20,000 Unicode code points and
rejects blank text, coercions and extra fields. Validation errors use a fixed
message without echoing text. Authentication, body and input failures are non-2xx.

All D1 business results, including `busy/timeout`, use HTTP 200 and exactly five
fields: `schemaVersion=1`, `status`, `routerArtifactSha256`, `route`, `reason`.
`routed` has a route and null reason; `unsupported` has null route and
`unsupported_language`; `failed` has null route and one of `language_undetermined`,
`model_unavailable`, `model_failure`, `busy`, `timeout`. Off/loading failure returns
`failed/model_unavailable` with null SHA; ready-instance responses use its cached
SHA. No LID confidence, original text or internal exception is exposed. The matching
consumer contract is in AIDetector-Back's `docs/detection-contract.md`, section 10.

Evidence reuses `InferenceAdmission` with a separate one-active, zero-queued slot.
Its CPU work runs off the event loop. Timeout or request cancellation does not
release capacity until the worker thread actually exits; further requests return
`busy`, preventing abandoned work from accumulating. Main detection has its own
capacity. Shutdown drains both. This bounds concurrent Evidence work but cannot
kill a hung native thread or isolate process CPU/RSS. Local coexistence and startup
costs passed the local acceptance budgets in batch 3 below.

Run the synthetic-file and tiny-model checks in the existing development environment:

```powershell
python -B -m unittest discover -s tests -p test_evidence_router.py -v
python -B -m unittest discover -s tests -v
```

These tests exercise actual ASGI handlers without listeners, fake/tiny XLM-R models,
language decisions, authentication/body limits, private failures, timeout/cancellation
capacity retention and main-detector regression. The real py3langid API check skips
if the package is absent; for local batch-2 acceptance it was run using the existing
isolated cache and passed without skips. No test downloads or loads the real
production model. The opt-in real-model run below separately covers local tokenizer/
model compatibility, startup memory, latency and main-detector coexistence.
CPU inference avoids adding GPU-resident
weights but still shares process memory and CPU resources. The frozen EV3 waiver
and non-certification facts are unchanged; local completion is not serving approval.

### Opt-in local real-model acceptance (batch 3)

`smoke_evidence_runtime.py` uses the existing development environment and local
model files. It starts its own single-worker RepreGuard processes on ephemeral
`127.0.0.1` ports, with temporary tokens and process-local configuration; it does
not change `.env`, start AIDetector-Back, install packages or download models.
Normal EOF shutdown drains admissions; a memory-reserve breach or stuck shutdown
stops only the child created by the script. No production inference code is patched.
The temporary worker observes call/forward counts and actual model devices.

The three runs are Evidence off, on, and on with an injected 20 ms deadline to
exercise a real timed-out forward. Main-model samples are 500 characters and are
checked against its 512-token limit. Eight supported-language examples, Italian,
no-information input and 20,000-character EN/ZH inputs exercise Evidence; these are
functional cases, not a new accuracy evaluation. After three main warmups, each
idle/concurrent main condition has 30 requests; concurrent Evidence uses 20,000
characters. Actual HTTP results pass the backend's real Bundle-bound validator.

The predeclared local budgets are identical main labels/thresholds, score difference
at most 1e-6, main P95 increase at most `max(baseline*20%, 50 ms)`, successful Evidence
responses below the normal 10-second deadline, and available memory above
`max(2 GiB, total RAM*10%)`. The script leaves the existing CPU thread settings
unchanged. A failed budget remains a failed report; no automatic retuning occurs.

Run explicitly from this repository; use a new output filename for each run:

```powershell
D:\Anaconda\envs\lab\python.exe -B smoke_evidence_runtime.py `
  --router-model D:\Code\AIDetector-evidence-research\outputs\detectrl_x_xlmr_router\production-final\model `
  --router-sha 9a1a5be8f0d7599e682d30c7f50e3deb5f747b59ef720e288f6342e7c944f896 `
  --bundle D:\Code\AIDetector-evidence-research\dist\evidence-v1.bundle `
  --bundle-sha 8abe24fc7e7747f4e2e9b90a80bf26b7999726373fa80e8519a97dffbe7014b2 `
  --backend-root D:\Code\AIDetector-Back `
  --lid-site D:\Code\AIDetector-evidence-research\data\cache\py3langid-screen-20260907\site `
  --output loadtest_results\evidence-runtime-local.json
```

The script's current local profile requires the main model on CUDA and the Router
on CPU FP32. Reports contain timings, response metadata and counts, not input text
or authentication tokens. The JSON report, per-worker observations and diagnostic
logs use the existing ignored `loadtest_results/` directory. CPU/RSS observations
include both models in one service process; RSS peaks are sampled every 50 ms.

2026-09-07 result (`evidence-runtime-20260907-batch3-01.json`, SHA-256
`72211f3f8ec3624d63acb762e1d53f4d0a3bc61b7edf6ea4af579091fc0b5f86`):

| Measurement | Evidence off | Evidence on |
|---|---:|---:|
| Startup, including imports/model load | 15.94 s | 18.27 s |
| Main P95, Evidence idle | 14.83 ms | 19.51 ms |
| Main P95, concurrent Evidence | — | 31.96 ms |
| Evidence P95, concurrent 20k-character input | — | 215.32 ms |
| Sampled process RSS peak | 1909.94 MiB | 3266.87 MiB |
| Process RSS after requests/drain | 1240.40 MiB | 2783.46 MiB |

These are process-start timings; OS file-cache state was not controlled.
Main score differences were exactly zero, with identical labels and thresholds.
The main concurrent P95 increased by 17.13 ms and passed the approved 50 ms absolute
allowance; this is a measurable slowdown, not a claim of zero performance cost.
All 12 functional cases passed. Burst admission returned one routed result and
three busy results. The 20 ms timeout run performed exactly one real XLM-R forward,
kept the next request busy, preserved main detection and recovered after completion.
All workers finished with zero admitted/active/pending work, normal exit and no
remaining listener; the lowest available memory across runs was 15.42 GiB.

This was Windows, Python 3.10.19, RTX 5070 Ti, existing lab plus the isolated py3langid
0.4.0 cache. `pip check` found no broken installed requirements, but the actual
environment differs from the declared deployment requirements:

| Package | requirements.txt | Tested lab |
|---|---|---|
| torch | 2.7.1 | 2.9.1+cu128 |
| safetensors | 0.5.3 | 0.7.0 |
| fastapi | 0.115.14 | 0.128.0 |
| pydantic | 2.11.7 | 2.12.5 |
| starlette | 0.46.2 | 0.50.0 |
| uvicorn | 0.35.0 | 0.40.0 |
| py3langid | 0.4.0 | isolated cache only; absent from lab installation |

No package versions were changed. This small local run is not a production latency
distribution or validation of the declared requirements on another deployment.
The owner accepted local completion and deferred deployment questions until needed;
they do not block further local development. Backend mode projection and network
orchestration remain later governed items.

## Legacy Files

`repreGuard_detector.py`, `init_tiny_model.py`, `repe/`, and `saved_rep_reader.pt` belong to the V1 representation-reading detector path. The V2.0 RoBERTa service path does not require `saved_rep_reader.pt`.
