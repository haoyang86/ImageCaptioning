"""
Evaluation entrypoint for M1 (Show and Tell baseline).

Loads a saved m1 checkpoint, decodes captions over the val set, saves the
predictions, and computes BLEU / METEOR / CIDEr / ROUGE.

Mirrors evaluation/m3_eval.py so the team gets the same eval format across
models. Architecture is rebuilt from the saved checkpoint config so we don't
have to keep configs/m1_config.py in lockstep with the trained model.

Usage:
    uv run python -m evaluation.m1_eval [decode_strategy] [beam_size] [length_penalty]
        decode_strategy : "greedy" (default) or "beam"
        beam_size       : default 3   (only used when strategy=beam)
        length_penalty  : default 0.7 (only used when strategy=beam)

Examples:
    uv run python -m evaluation.m1_eval               # greedy
    uv run python -m evaluation.m1_eval beam          # beam, size 3, lp 0.7
    uv run python -m evaluation.m1_eval beam 5 0.6    # beam, size 5, lp 0.6

Set cfg.debug=True to limit to 5 batches for a quick sanity check.
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs import m1_config as cfg
from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate
from models.model_m1 import ModelM1
from models.embedding_utils import (
    build_embedding_matrix,
    should_use_pretrained_embedding,
)
from evaluation.eval_utils import run_evaluation


def main():

    # CLI: decode_strategy, beam_size, length_penalty (all optional)
    decode_strategy = sys.argv[1].lower() if len(sys.argv) > 1 else "greedy"
    beam_size       = int(sys.argv[2])    if len(sys.argv) > 2 else 3
    length_penalty  = float(sys.argv[3])  if len(sys.argv) > 3 else 0.7
    assert decode_strategy in {"greedy", "beam"}, (
        f"decode_strategy must be 'greedy' or 'beam', got {decode_strategy!r}"
    )

    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------------------
    # Paths
    # -----------------------------------------

    dataset_root = project_root / "datasets" / cfg.dataset_name
    image_dir = dataset_root / cfg.image_folder
    eval_csv = dataset_root / "val.csv"

    checkpoint_path = (
        project_root /
        cfg.checkpoint_dir /
        cfg.best_checkpoint_name
    )

    prediction_dir = project_root / cfg.prediction_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_dir / cfg.val_prediction_name
    metrics_path    = prediction_dir / cfg.metrics_result_name

    # Adding support for mps as I have a macbook pro, but I think my teammates have CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Eval csv:", eval_csv)
    print("Checkpoint:", checkpoint_path)
    print("Prediction path:", prediction_path)
    print("Metrics path:", metrics_path)

    # -----------------------------------------
    # Transform + dataset
    # -----------------------------------------

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    eval_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=eval_csv,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
    )

    # -----------------------------------------
    # Load checkpoint metadata + restore vocab
    # -----------------------------------------

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    eval_dataset.vocab.stoi = checkpoint["vocab_stoi"]
    eval_dataset.vocab.itos = checkpoint["vocab_itos"]

    vocab = eval_dataset.vocab
    vocab_size = len(vocab)

    # Pull architecture from the saved checkpoint config so we always
    # match the trained model, even if cfg has drifted since training.
    ckpt_config = checkpoint.get("config", {})

    embedding_type   = ckpt_config.get("embedding_type",   cfg.embedding_type)
    embedding_dim    = ckpt_config.get("embedding_dim",    cfg.embedding_dim)
    embedding_path   = ckpt_config.get("embedding_path",   cfg.embedding_path)
    encoder_dim      = ckpt_config.get("encoder_dim",      cfg.encoder_dim)
    hidden_dim       = ckpt_config.get("hidden_dim",       cfg.hidden_dim)
    num_lstm_layers  = ckpt_config.get("num_lstm_layers",  cfg.num_lstm_layers)
    dropout          = ckpt_config.get("dropout",          cfg.dropout)
    encoder_dropout  = ckpt_config.get("encoder_dropout",  cfg.encoder_dropout)
    max_len          = ckpt_config.get("max_len",          cfg.max_len)

    print("Eval samples:", len(eval_dataset))
    print("Vocab size:", vocab_size)
    print("Checkpoint embedding type:", embedding_type)
    print("Checkpoint hidden_dim:", hidden_dim, "num_lstm_layers:", num_lstm_layers)

    # -----------------------------------------
    # Optional pretrained embedding matrix
    # (only needed if the checkpoint was trained with GloVe)
    # -----------------------------------------

    pretrained_embedding_matrix = None
    if should_use_pretrained_embedding(embedding_type):
        glove_path = project_root / embedding_path
        pretrained_embedding_matrix, found_count, oov_count = build_embedding_matrix(
            vocab=vocab,
            glove_path=glove_path,
            embedding_dim=embedding_dim,
        )
        print("Pretrained embedding matrix:", pretrained_embedding_matrix.shape)
        print("GloVe found:", found_count, "OOV:", oov_count)

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.debug_batch_size if cfg.debug else cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=FlickrCollate(pad_idx=vocab.pad_idx),
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = ModelM1(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        max_len=max_len,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=False,            # we are loading a state_dict; no need for torchvision weights
        encoder_dropout=encoder_dropout,
        embedding_type=embedding_type,
        embedding_dim=embedding_dim,
        pretrained_embedding_matrix=pretrained_embedding_matrix,
    )

    # In debug mode just run 5 batches so we can sanity-check the pipeline fast.
    max_batches = 5 if cfg.debug else None

    # -----------------------------------------
    # Run evaluation
    # -----------------------------------------

    run_evaluation(
        model=model,
        checkpoint_path=checkpoint_path,
        loader=eval_loader,
        dataset=eval_dataset,
        vocab=vocab,
        device=device,
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        max_len=max_len,
        max_batches=max_batches,
        decode_strategy=decode_strategy,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )


if __name__ == "__main__":
    main()
