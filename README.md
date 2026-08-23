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
```

Both endpoints are internal and require this header:

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

## Legacy Files

`repreGuard_detector.py`, `init_tiny_model.py`, `repe/`, and `saved_rep_reader.pt` belong to the V1 representation-reading detector path. The V2.0 RoBERTa service path does not require `saved_rep_reader.pt`.
