from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


DEFAULT_MODEL_NAME = "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All"
DEFAULT_MODEL_REVISION = "76649a0257a812a81cf36b5de9cc5f2430aeaa7f"
DEFAULT_MODEL_CACHE_DIR = Path(r"D:\huggingface")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_CACHE_DIR / "WUJUNCHAO" / "DetectRL-X-XLM-RoBERTa-Detector-All"
DEFAULT_MODEL_THRESHOLD = 0.0028
DEFAULT_AI_LABEL_ID = 1
MIN_SERVICE_TOKEN_LENGTH = 32
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
MODEL_MANIFEST_FILES = (".repre_guard_model_manifest.json", "model_manifest.json")


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_valid_service_token(token: str) -> bool:
    return len(token) >= MIN_SERVICE_TOKEN_LENGTH and all(0x21 <= ord(char) <= 0x7E for char in token)


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_metadata_revision(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


@dataclass(frozen=True)
class RepreGuardConfig:
    model_name: str = os.getenv("REPRE_GUARD_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_revision: str = os.getenv("REPRE_GUARD_MODEL_REVISION", DEFAULT_MODEL_REVISION)
    model_cache_dir: Path = Path(os.getenv("REPRE_GUARD_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR)))
    model_path: Path = Path(os.getenv("REPRE_GUARD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    local_files_only: bool = _bool_env("REPRE_GUARD_LOCAL_FILES_ONLY", True)
    threshold: float = _float_env("REPRE_GUARD_THRESHOLD", DEFAULT_MODEL_THRESHOLD)
    ai_label_id: int = _int_env("REPRE_GUARD_AI_LABEL_ID", DEFAULT_AI_LABEL_ID)
    tokenizer_use_fast: bool = _bool_env("REPRE_GUARD_TOKENIZER_USE_FAST", False)
    max_input_tokens: int = _int_env("REPRE_GUARD_MAX_INPUT_TOKENS", 512)
    device: str = os.getenv("REPRE_GUARD_DEVICE", "auto").strip().lower() or "auto"
    host: str = os.getenv("REPRE_GUARD_HOST", "0.0.0.0")
    port: int = _int_env("REPRE_GUARD_PORT", 9000)
    log_level: str = os.getenv("REPRE_GUARD_LOG_LEVEL", "info")
    service_token: str = field(
        default_factory=lambda: os.getenv("REPRE_GUARD_SERVICE_TOKEN", ""),
        repr=False,
    )

    def require_service_token(self) -> str:
        token = self.service_token
        if not is_valid_service_token(token):
            raise RuntimeError(
                "REPRE_GUARD_SERVICE_TOKEN must contain at least 32 printable ASCII characters without whitespace."
            )
        return token

    def validate_model_path(self) -> None:
        missing_files = [name for name in REQUIRED_MODEL_FILES if not (self.model_path / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Local RepreGuard model directory is incomplete: {self.model_path}. "
                f"Missing files: {', '.join(missing_files)}"
            )

        manifest_revision = ""
        for manifest_name in MODEL_MANIFEST_FILES:
            manifest = _read_json_file(self.model_path / manifest_name)
            manifest_revision = str(manifest.get("revision") or manifest.get("model_revision") or "").strip()
            if manifest_revision:
                break

        if manifest_revision:
            if manifest_revision != self.model_revision:
                raise RuntimeError(
                    f"Local RepreGuard model manifest revision mismatch: expected {self.model_revision}, "
                    f"got {manifest_revision} in {self.model_path}"
                )
            return

        metadata_dir = self.model_path / ".cache" / "huggingface" / "download"
        missing_metadata: list[str] = []
        mismatched_metadata: list[str] = []
        for file_name in REQUIRED_MODEL_FILES:
            revision = _read_metadata_revision(metadata_dir / f"{file_name}.metadata")
            if not revision:
                missing_metadata.append(file_name)
            elif revision != self.model_revision:
                mismatched_metadata.append(f"{file_name}={revision}")

        if missing_metadata or mismatched_metadata:
            detail_parts = []
            if missing_metadata:
                detail_parts.append(f"missing metadata for: {', '.join(missing_metadata)}")
            if mismatched_metadata:
                detail_parts.append(f"revision mismatch: {', '.join(mismatched_metadata)}")
            raise RuntimeError(
                f"Local RepreGuard model directory does not match pinned revision {self.model_revision}: "
                + "; ".join(detail_parts)
            )

    def resolve_model_source(self) -> str:
        if self.model_path.exists():
            self.validate_model_path()
            return str(self.model_path)
        return self.model_name


settings = RepreGuardConfig()
