"""
Quick smoke test for M4 (Global encoder + Transformer decoder).

Same idea as the M1 smoke test; random images and random caption ids.
We check:
    1. Model builds for all 3 embedding strategies.
    2. forward returns logits of shape (B, T, vocab_size).
    3. greedy_decode returns ids of shape (B, L) with L <= max_len.
    4. A short overfit-one-batch run produces a strictly lower loss.

Run from the repo root:
    python tests/m4_smoke.py
"""

import sys
from pathlib import Path

# Allow running this file directly from the repo root
# (I flattened the repo when I cloned it and I don't know how this will affect y'all)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from models.model_m4 import ModelM4


# --------------------------------------------------------------------------
# Tiny fake setup; we don't need real data here.
# --------------------------------------------------------------------------

VOCAB_SIZE = 100
PAD_IDX = 0
START_IDX = 1
END_IDX = 2

B = 4         # batch size
T = 12        # caption length (incl <start> and <end>)
EMBED_DIM = 300


def make_fake_batch(device):
    images = torch.randn(B, 3, 224, 224, device=device)

    # random caption ids in vocab range, with <start>/<end> at the boundaries
    captions = torch.randint(
        low=4,                # skip special tokens
        high=VOCAB_SIZE,
        size=(B, T),
        device=device,
    )
    captions[:, 0] = START_IDX
    captions[:, -1] = END_IDX

    return images, captions


def make_model(embedding_type, device):
    """
    Build an M4 with a random GloVe-shaped matrix when needed so the
    pretrained branches can also be exercised.
    """
    pretrained = None
    if embedding_type in {"pretrained_frozen", "pretrained_finetune"}:
        pretrained = torch.randn(VOCAB_SIZE, EMBED_DIM)

    model = ModelM4(
        vocab_size=VOCAB_SIZE,
        pad_idx=PAD_IDX,
        start_idx=START_IDX,
        end_idx=END_IDX,
        # tiny network so this runs in seconds on CPU
        encoder_dim=128,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        max_len=20,
        freeze_backbone=True,
        pretrained=False,        # don't pull torchvision weights for the smoke test
        embedding_type=embedding_type,
        embedding_dim=EMBED_DIM,
        pretrained_embedding_matrix=pretrained,
    ).to(device)

    return model


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_forward_shape(device):
    print("\n[test] forward shape across embedding strategies")

    for embedding_type in ["random", "pretrained_frozen", "pretrained_finetune"]:

        model = make_model(embedding_type, device)
        images, captions = make_fake_batch(device)

        # the trainer normally feeds captions[:, :-1]
        decoder_input = captions[:, :-1]
        logits = model(images, decoder_input)

        expected = (B, T - 1, VOCAB_SIZE)
        assert logits.shape == expected, (
            f"[{embedding_type}] expected {expected}, got {tuple(logits.shape)}"
        )
        print(f"  {embedding_type:22s} ok  logits {tuple(logits.shape)}")


def test_greedy_decode_shape(device):
    print("\n[test] greedy_decode shape")

    model = make_model("random", device)
    images, _ = make_fake_batch(device)

    generated = model.greedy_decode(images, max_len=20)

    assert generated.dim() == 2,            f"expected 2D, got {generated.shape}"
    assert generated.size(0) == B,          f"batch dim mismatch: {generated.shape}"
    assert generated.size(1) <= 20,         f"L exceeded max_len: {generated.shape}"
    assert (generated[:, 0] == START_IDX).all(), "first token should be <start>"

    print(f"  ok  generated {tuple(generated.shape)}")


def test_overfit_one_batch(device):
    print("\n[test] overfit one batch (loss should drop)")

    torch.manual_seed(6739)

    model = make_model("random", device)
    images, captions = make_fake_batch(device)

    decoder_input = captions[:, :-1]
    targets = captions[:, 1:]

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(30):
        logits = model(images, decoder_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"  initial loss: {losses[0]:.4f}")
    print(f"  final   loss: {losses[-1]:.4f}")

    assert losses[-1] < losses[0] - 0.5, (
        f"loss did not drop enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print("  ok  loss decreased")


# --------------------------------------------------------------------------

def pick_device():
    # Adding support for mps as I have a macbook pro, but I think my teammates have CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = pick_device()
    print(f"device: {device}")

    test_forward_shape(device)
    test_greedy_decode_shape(device)
    test_overfit_one_batch(device)

    print("\nall good.")


if __name__ == "__main__":
    main()
