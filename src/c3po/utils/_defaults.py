from typing import Any

DEFAULT_TRAIN_ARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_epochs": 1000,
    "initial_batch_size": 64,
    "buffer_size": 8,
    "min_batch_size": 32,
    "max_n_neg": 128,
    "initial_n_neg": 8,
    "multi_gpu": False,
}