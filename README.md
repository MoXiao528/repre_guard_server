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

Environment variables:

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
```

Production should run from the pinned local model files. Set `REPRE_GUARD_LOCAL_FILES_ONLY=false` only for an explicit download/cache warm-up flow.

Inputs longer than `REPRE_GUARD_MAX_INPUT_TOKENS` are rejected with `INPUT_TOO_LONG`; the service no longer silently truncates detection input.

## Start

```powershell
pip install -r requirements.txt
D:\Anaconda\envs\lab\python.exe .\run_roberta_server.py
```

PowerShell wrapper:

```powershell
.\run_local_roberta_server.ps1
```

## Endpoints

```text
GET  /health
POST /detect
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

## Legacy Files

`repreGuard_detector.py`, `init_tiny_model.py`, `repe/`, and `saved_rep_reader.pt` belong to the V1 representation-reading detector path. The V2.0 RoBERTa service path does not require `saved_rep_reader.pt`.
