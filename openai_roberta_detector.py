from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging


@dataclass(frozen=True)
class RobertaDetection:
    ai_probability: float
    raw_score: float
    threshold: float
    label: str
    score_type: str


class OpenAIRobertaDetector:
    """Hugging Face sequence-classification detector wrapper.

    DetectRL-X does not publish id2label metadata, so the AI label id is an
    explicit deployment setting. Its threshold is calibrated on probability.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        revision: str | None = None,
        local_files_only: bool = True,
        threshold: float = 0.0028,
        ai_label_id: int | None = 1,
        tokenizer_use_fast: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_name = Path(model_name_or_path).name if Path(model_name_or_path).exists() else model_name_or_path
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if revision and not Path(model_name_or_path).exists():
            load_kwargs["revision"] = revision
        load_kwargs["local_files_only"] = local_files_only
        transformers_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=tokenizer_use_fast, **load_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.ai_label_id = self._resolve_ai_label_id(explicit_label_id=ai_label_id)
        self.threshold = float(threshold)

    def _resolve_label_id(self, label: str) -> int | None:
        label2id = getattr(self.model.config, "label2id", None) or {}
        for key, value in label2id.items():
            if str(key).lower() == label.lower():
                return int(value)
        return None

    def _resolve_ai_label_id(self, *, explicit_label_id: int | None) -> int:
        num_labels = int(getattr(self.model.config, "num_labels", 2) or 2)
        if explicit_label_id is not None:
            label_id = int(explicit_label_id)
            if 0 <= label_id < num_labels:
                return label_id
            raise ValueError(f"ai_label_id={label_id} is outside model label range 0..{num_labels - 1}")

        for label in ("AI", "Fake", "Generated", "Machine", "LABEL_1"):
            label_id = self._resolve_label_id(label)
            if label_id is not None:
                return label_id
        return 1 if num_labels > 1 else 0

    def count_tokens(self, text: str, *, max_tokens: int | None = None) -> int:
        tokenize_kwargs = {"truncation": False}
        if max_tokens is not None:
            tokenize_kwargs = {"truncation": True, "max_length": max_tokens}
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            **tokenize_kwargs,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    def predict_text(self, text: str) -> RobertaDetection:
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits.float()
            probabilities = torch.softmax(logits, dim=-1)[0]
            ai_probability = float(probabilities[self.ai_label_id].detach().cpu().item())

        raw_score = ai_probability
        label = "AI" if raw_score >= self.threshold else "HUMAN"
        return RobertaDetection(
            ai_probability=ai_probability,
            raw_score=raw_score,
            threshold=self.threshold,
            label=label,
            score_type="probability",
        )

    def score_text(self, text: str) -> float:
        return self.predict_text(text).raw_score
