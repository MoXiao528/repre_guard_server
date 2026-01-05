import json
import logging
import random
from pathlib import Path

import torch

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

# TODO: Update MODEL_NAME to the production model path when training the final detector.
MODEL_NAME = "sshleifer/tiny-gpt2"

# TODO: Point TRAIN_DATA_PATH to the full, real training dataset when ready for production.
TRAIN_DATA_PATH = Path("direct_prompt_train.json")

READER_OUTPUT_PATH = Path("saved_rep_reader.pt")


def _load_train_data() -> list[dict]:
    if TRAIN_DATA_PATH.exists():
        LOGGER.info("Loading training data from %s", TRAIN_DATA_PATH)
        with TRAIN_DATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    LOGGER.warning("Training data not found. Using synthetic dummy samples.")
    dummy_data = []
    for idx in range(10):
        dummy_data.append(
            {
                "direct_prompt": f"AI synthetic sample {idx}: {random.random()}",
                "human_text": f"Human synthetic sample {idx}: {random.random()}",
            }
        )
    return dummy_data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_data = _load_train_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=len(train_data),
        rep_token=-1,
        batch_size=16,
        random_seed=2025,
        device=device,
    )

    LOGGER.info("Fitting rep_reader directions with %d samples.", len(train_data))
    model.fit_rep_reader(train_data)

    READER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.rep_reader, READER_OUTPUT_PATH)
    LOGGER.info("Saved rep_reader to %s", READER_OUTPUT_PATH)


if __name__ == "__main__":
    main()
