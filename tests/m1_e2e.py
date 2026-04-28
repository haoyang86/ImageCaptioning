"""
End-to-end smoke test for M1.

Builds a tiny mock Flickr8k under datasets/flickr8k_mock/, runs
training/m1_train.main() against it, then loads the saved checkpoint back
and runs greedy_decode on a few images. Shared plumbing lives in
tests/e2e_helpers.py.

Using random embeddings here to avoid pulling GloVe (not in the repo).
This is purely a plumbing check, not a model-quality check.

Run from the repo root:
    python tests/m1_e2e.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.e2e_helpers import (
    build_mock_dataset,
    patch_config_for_e2e,
    reload_and_decode,
)


def build_m1(vocab, ckpt):
    from configs import m1_config as cfg
    from models.model_m1 import ModelM1

    return ModelM1(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        encoder_dim=cfg.encoder_dim,
        hidden_dim=cfg.hidden_dim,
        num_lstm_layers=cfg.num_lstm_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=False,
        embedding_type="random",
        embedding_dim=cfg.embedding_dim,
    )


def main():
    print(f"repo root: {ROOT}")
    build_mock_dataset()

    from configs import m1_config as cfg
    patch_config_for_e2e(cfg, experiment_name="m1_e2e_mock", model_dir="M1")
    print(f"using dataset: {cfg.dataset_name}, embedding: {cfg.embedding_type}")

    # Late import so the cfg patches are visible to the trainer
    from training.m1_train import main as train_main
    train_main()

    reload_and_decode(cfg, build_m1)
    print("\nend-to-end ok.")


if __name__ == "__main__":
    main()
