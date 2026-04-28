"""
End-to-end smoke test for M4.

Same idea as the M1 e2e test; uses the shared mock-dataset and
reload-and-decode helpers in tests/e2e_helpers.py.

Run from the repo root:
    python tests/m4_e2e.py
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


def build_m4(vocab, ckpt):
    from configs import m4_config as cfg
    from models.model_m4 import ModelM4

    return ModelM4(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        encoder_dim=cfg.encoder_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
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

    from configs import m4_config as cfg
    patch_config_for_e2e(cfg, experiment_name="m4_e2e_mock", model_dir="M4")
    print(f"using dataset: {cfg.dataset_name}, embedding: {cfg.embedding_type}")

    # Late import so the cfg patches are visible to the trainer
    from training.m4_train import main as train_main
    train_main()

    reload_and_decode(cfg, build_m4)
    print("\nend-to-end ok.")


if __name__ == "__main__":
    main()
