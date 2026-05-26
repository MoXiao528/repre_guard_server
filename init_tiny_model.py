import json
import logging
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "microsoft/phi-2"
TRAIN_DATA_ENV_VAR = "REPRE_GUARD_TRAIN_DATA_PATH"
TRAIN_DATA_CANDIDATES = (
    Path("../data_local/repre_train_data.json"),
    Path("../data_local/processed_datasets/repre_train_data.json"),
    Path("train_MIXED_ALL.json"),
)
READER_OUTPUT_PATH = Path("saved_rep_reader.pt")


def _build_dummy_samples() -> list[dict]:
    dummy_data = []
    for idx in range(10):
        dummy_data.append(
            {
                "direct_prompt": f"AI synthetic sample {idx}: {random.random()}",
                "human_text": f"Human synthetic sample {idx}: {random.random()}",
            }
        )
    return dummy_data


def _resolve_train_data_path() -> Path | None:
    override = str(os.getenv(TRAIN_DATA_ENV_VAR, "")).strip()
    candidates: list[Path] = []

    if override:
        candidates.append(Path(override).expanduser())

    candidates.extend(TRAIN_DATA_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _normalize_train_item(item: dict) -> tuple[dict | None, str | None]:
    if not isinstance(item, dict):
        return None, None

    split = str(item.get("split", "")).strip().lower()
    if split and split != "train":
        return None, None

    direct_prompt = item.get("direct_prompt")
    human_text = item.get("human_text")
    if isinstance(direct_prompt, str) and direct_prompt.strip() and isinstance(human_text, str) and human_text.strip():
        return {
            "direct_prompt": direct_prompt.strip(),
            "human_text": human_text.strip(),
        }, "legacy"

    rewritten_text = item.get("rewritten_text")
    original_text = item.get("original_text")
    if isinstance(rewritten_text, str) and rewritten_text.strip() and isinstance(original_text, str) and original_text.strip():
        return {
            "direct_prompt": rewritten_text.strip(),
            "human_text": original_text.strip(),
        }, "rewrite"

    return None, None


def _load_train_data() -> list[dict]:
    train_data_path = _resolve_train_data_path()
    if train_data_path is None:
        LOGGER.warning("Training data not found. Using synthetic dummy samples.")
        return _build_dummy_samples()

    LOGGER.info("Loading training data from %s", train_data_path)
    with train_data_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise RuntimeError(f"Training data must be a list, got: {type(raw_data)!r}")

    normalized_data: list[dict] = []
    dropped_count = 0
    source_formats: set[str] = set()

    for item in raw_data:
        normalized_item, source_format = _normalize_train_item(item)
        if normalized_item is None:
            dropped_count += 1
            continue
        normalized_data.append(normalized_item)
        if source_format:
            source_formats.add(source_format)

    if not normalized_data:
        raise RuntimeError(
            f"No compatible train records found in {train_data_path}. Expected direct_prompt/human_text or rewritten_text/original_text."
        )

    LOGGER.info(
        "Prepared %d train pairs from %s (source_formats=%s, dropped=%d)",
        len(normalized_data),
        train_data_path,
        ",".join(sorted(source_formats)) or "unknown",
        dropped_count,
    )
    return normalized_data


def truncate_data(data: list[dict], model_name: str, max_length: int = 400) -> list[dict]:
    LOGGER.info(f"Initializing tokenizer for truncation (max_length={max_length})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        LOGGER.warning(f"Failed to load tokenizer for truncation: {e}. Skipping truncation.")
        return data

    truncated_count = 0
    for item in data:
        for key, value in item.items():
            if isinstance(value, str):
                tokens = tokenizer.encode(value, add_special_tokens=False)
                if len(tokens) > max_length:
                    item[key] = tokenizer.decode(tokens[:max_length])
                    truncated_count += 1

    if truncated_count > 0:
        LOGGER.info(f"Truncated {truncated_count} fields to {max_length} tokens to prevent overflow.")

    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_data = _load_train_data()

    # phi-2 先控制在 1024 tokens，优先保证能够稳定生成新的 rep_reader。
    # 如果你的显存余量足够，再提升到 2048。
    train_data = truncate_data(train_data, MODEL_NAME, max_length=1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")

    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=len(train_data),
        rep_token=-1,
        # phi-2 先用 batch_size=1，稳定优先。
        batch_size=1,
        random_seed=2025,
        device=device,
    )

    LOGGER.info("Fitting rep_reader directions with %d samples.", len(train_data))

    try:
        model.fit_rep_reader(train_data)
    except RuntimeError as e:
        if "device-side assert" in str(e):
            LOGGER.error("!!! CUDA ERROR DETECTED !!!")
            LOGGER.error("Please RESTART your Python kernel/terminal explicitly.")
            LOGGER.error("The GPU state is corrupted from a previous error and cannot recover without a restart.")
        raise e

    READER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.rep_reader, READER_OUTPUT_PATH)
    LOGGER.info("Saved rep_reader to %s", READER_OUTPUT_PATH)


if __name__ == "__main__":
    main()
