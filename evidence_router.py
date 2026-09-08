"""Evidence-only language support check and SHA-bound local Router inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from itertools import product
from pathlib import Path
from typing import Any


LANGUAGES = ("ar", "de", "en", "es", "fr", "pt", "ru", "zh")
DOMAINS = ("academic", "news", "novel", "seo", "webtext", "wiki")
CLASSES = tuple(
    f"{language}:{domain}" for language, domain in product(LANGUAGES, DOMAINS)
)
MODEL_FILES = (
    "config.json",
    "router_config.json",
    "router_model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
MAX_INPUT_CHARS = 20_000
MAX_LENGTH = 512
ROUTER_CONFIG = {
    "schema_version": 1,
    "architecture": "one_xlmr_encoder_one_language_head_eight_domain_heads",
    "language_heads": 1,
    "domain_heads": 8,
    "domains_per_head": 6,
    "max_length": MAX_LENGTH,
    "truncation": "canonical_nfkc_whitespace_then_head_255_tail_255",
    "classes": list(CLASSES),
}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("Invalid Evidence Router contract")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        + "\n"
    ).encode("utf-8")


def _signature(info: os.stat_result) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _snapshot(directory: Path) -> dict[str, tuple]:
    entries = {"": directory, **{name: directory / name for name in MODEL_FILES}}
    result = {}
    for name, path in entries.items():
        info = path.lstat()
        _require(
            not getattr(info, "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
        _require(stat.S_ISREG(info.st_mode) if name else stat.S_ISDIR(info.st_mode))
        result[name] = _signature(info)
    _require({path.name for path in directory.iterdir()} == set(MODEL_FILES))
    return result


def _verify_artifact(directory: Path, expected_sha256: str) -> tuple[dict, dict]:
    snapshot = _snapshot(directory)
    claims, configs = {}, {}
    for name in MODEL_FILES:
        digest = hashlib.sha256()
        size = 0
        is_config = name in ("config.json", "router_config.json")
        _require(snapshot[name][3] > 0)
        if is_config:
            _require(snapshot[name][3] <= 64 * 1024)
        with (directory / name).open("rb") as source:
            _require(_signature(os.fstat(source.fileno())) == snapshot[name])
            chunks = []
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                if is_config:
                    chunks.append(chunk)
            _require(_signature(os.fstat(source.fileno())) == snapshot[name])
        _require(size == snapshot[name][3])
        claims[name] = {"bytes": size, "sha256": digest.hexdigest()}
        if is_config:
            configs[name] = b"".join(chunks)
    _require(hashlib.sha256(_canonical_bytes(claims)).hexdigest() == expected_sha256)
    _require(_snapshot(directory) == snapshot)
    # Exact canonical bytes also reject duplicate keys, bool/int drift and unknown fields.
    _require(configs["router_config.json"] == _canonical_bytes(ROUTER_CONFIG))
    config = json.loads(configs["config.json"].decode("utf-8"))
    _require(
        type(config) is dict and _canonical_bytes(config) == configs["config.json"]
    )
    bindings = {
        "architectures": ["XLMRobertaHierarchicalRouter"],
        "model_type": "xlm-roberta",
        "id2label": {str(index): label for index, label in enumerate(CLASSES)},
        "label2id": {label: index for index, label in enumerate(CLASSES)},
    }
    _require(
        _canonical_bytes({key: config.get(key) for key in bindings})
        == _canonical_bytes(bindings)
    )
    return config, snapshot


def _make_model(encoder: Any) -> Any:
    """Keep the published EV3 state names and forward; omit all training controls."""
    import torch

    class HierarchicalRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(encoder.config.hidden_dropout_prob)
            self.language_classifier = torch.nn.Linear(encoder.config.hidden_size, 8)
            self.domain_classifiers = torch.nn.ModuleList(
                torch.nn.Linear(encoder.config.hidden_size, 6) for _ in LANGUAGES
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> dict:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = self.dropout(outputs.last_hidden_state[:, 0, :])
            language_logits = self.language_classifier(hidden)
            domain_logits = torch.stack(
                [head(hidden) for head in self.domain_classifiers], dim=1
            )
            joint = torch.log_softmax(language_logits, dim=-1).unsqueeze(
                -1
            ) + torch.log_softmax(domain_logits, dim=-1)
            return {"logits": joint.reshape(-1, 48)}

    return HierarchicalRouter()


def _load_model(directory: Path, config: dict) -> tuple[Any, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    _require(
        type(tokenizer).__name__ == "XLMRobertaTokenizerFast"
        and tokenizer.is_fast is True
        and len(tokenizer) == 250_002
        and tokenizer.model_max_length == MAX_LENGTH
        and tokenizer.num_special_tokens_to_add(pair=False) == 2
        and (
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
            tokenizer.mask_token_id,
        )
        == (0, 1, 2, 3, 250_001)
    )
    # Do not seed training RNGs, touch CUDA, or change process-wide thread settings.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = _make_model(XLMRobertaModel(XLMRobertaConfig.from_dict(config))).float()
    state = load_file(directory / "router_model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def _tokenize_head_tail(tokenizer: Any, canonical_text: str) -> dict:
    tokenized = tokenizer(
        [canonical_text],
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = list(tokenized["input_ids"][0])
    if len(ids) > 510:
        ids = ids[:255] + ids[-255:]
    item = tokenizer.prepare_for_model(
        ids,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    _require(len(item["input_ids"]) == len(ids) + 2 <= MAX_LENGTH)
    return tokenizer.pad([item], pad_to_multiple_of=8, return_tensors="pt")


def _route_from_logits(logits: Any) -> dict:
    import torch

    _require(
        isinstance(logits, torch.Tensor)
        and logits.is_floating_point()
        and tuple(logits.shape) == (1, 48)
    )
    _require(bool(torch.isfinite(logits).all()))
    # EV3 converts to float64 before temperature=1 softmax and language aggregation.
    joint = torch.softmax(
        logits.detach().to(device="cpu", dtype=torch.float64), dim=1
    ).reshape(8, 6)
    language_mass = joint.sum(dim=1)
    language = int(language_mass.argmax())
    domain = int(joint[language].argmax())
    confidence = {
        "language": float(language_mass[language]),
        "domain": float(joint[language, domain] / language_mass[language]),
    }
    _require(
        all(math.isfinite(value) and 0 <= value <= 1 for value in confidence.values())
    )
    return {
        "language": LANGUAGES[language],
        "domain": DOMAINS[domain],
        "confidence": confidence,
    }


def _load_language_identifier() -> Any:
    from py3langid.langid import LanguageIdentifier, MODEL_FILE

    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    _require(set(LANGUAGES) < set(identifier.labels))
    return identifier


class EvidenceRouter:
    """Explicitly owned CPU instance; support classification is fallible."""

    def __init__(
        self,
        *,
        enabled: bool | str = False,
        model_path: str | Path = "",
        artifact_sha256: str = "",
    ) -> None:
        self.status = "off"
        self.reason: str | None = None
        self.artifact_sha256: str | None = None
        self._tokenizer = None
        self._model = None
        self._language_identifier = None
        self._uninformative_ranking = None
        if type(enabled) not in (bool, str):
            enabled = None
        if type(enabled) is str:
            enabled = enabled.strip().lower()
        if enabled is False or enabled in ("", "0", "false", "no", "off"):
            return
        self.status = "failed"
        if (
            not (enabled is True or enabled in ("1", "true", "yes", "on"))
            or not isinstance(model_path, (str, Path))
            or not model_path
            or type(artifact_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
        ):
            self.reason = "invalid_evidence_router_config"
            return
        try:
            directory = Path(model_path).absolute()
            config, snapshot = _verify_artifact(directory, artifact_sha256)
            identifier = _load_language_identifier()
            uninformative = identifier.rank("")
            tokenizer, model = _load_model(directory, config)
            # ponytail: immutable deployment required; stat catches ordinary edits only.
            # Writable deployment would need an immutable snapshot for verification/load.
            _require(_snapshot(directory) == snapshot)
        except Exception:
            self.reason = "model_unavailable"
            return
        self._tokenizer, self._model = tokenizer, model
        self._language_identifier = identifier
        self._uninformative_ranking = uninformative
        self.artifact_sha256 = artifact_sha256
        self.status = "ready"

    def predict_route(self, text: Any) -> dict:
        """Check the whole text once, then preserve the frozen XLM-R route unchanged."""
        result = {
            "status": self.status,
            "routerArtifactSha256": self.artifact_sha256,
            "route": None,
            "reason": self.reason,
        }
        if self.status != "ready":
            return result
        result["status"] = "failed"
        if type(text) is not str or not 0 < len(text) <= MAX_INPUT_CHARS:
            result["reason"] = "invalid_text"
            return result
        canonical = " ".join(unicodedata.normalize("NFKC", text).split())
        if not canonical:
            result["reason"] = "invalid_text"
            return result
        try:
            ranking = self._language_identifier.rank(canonical)
            language, confidence = ranking[0]
            _require(
                type(language) is str
                and math.isfinite(confidence)
                and 0 <= confidence <= 1
            )
            # Exact no-information output, not a confidence threshold. Alias folding
            # can otherwise make an empty-feature input look like Serbian (sr).
            if ranking == self._uninformative_ranking or language in {"und", "zxx"}:
                result["reason"] = "language_undetermined"
                return result
            if language not in LANGUAGES:
                result.update(status="unsupported", reason="unsupported_language")
                return result

            import torch

            batch = _tokenize_head_tail(self._tokenizer, canonical)
            with torch.inference_mode():
                route = _route_from_logits(self._model(**batch)["logits"])
        except Exception:
            result["reason"] = "model_failure"
            return result
        result.update(status="predicted", route=route, reason=None)
        return result
