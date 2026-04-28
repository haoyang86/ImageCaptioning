"""
Shared helpers for the per-model end-to-end smoke tests.

Builds a tiny mock Flickr8k on disk, patches a model's config in-place to
use it (random embeddings, debug mode, separate output prefix), and
exposes a reload-and-decode helper that takes a model factory so each
model's specific constructor signature stays in its own test file.
"""

import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Mock dataset settings
# --------------------------------------------------------------------------

MOCK_DATASET_NAME = "flickr8k_mock"
N_IMAGES = 30
CAPTIONS_PER_IMAGE = 5

# Small word bank so the vocab builder has enough words above freq_threshold
WORDS = [
    "a", "the", "is", "on", "in", "with", "of",
    "dog", "cat", "boy", "girl", "man", "woman",
    "running", "playing", "sitting", "standing", "walking",
    "ball", "park", "beach", "field", "grass",
    "red", "blue", "white", "black", "small", "large",
]


def build_mock_dataset():
    """
    Create a tiny mock Flickr8k under datasets/flickr8k_mock/ that mirrors
    the real layout: a Flicker8k_Dataset/ folder with jpg images plus
    train.csv and val.csv with (image, caption) columns.
    """
    random.seed(42)
    np.random.seed(42)

    dataset_root = ROOT / "datasets" / MOCK_DATASET_NAME
    image_dir = dataset_root / "Flicker8k_Dataset"

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(N_IMAGES):
        name = f"img_{i:03d}.jpg"
        # tiny random RGB image
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(image_dir / name)

        for _ in range(CAPTIONS_PER_IMAGE):
            n_tokens = random.randint(5, 9)
            caption = " ".join(random.choices(WORDS, k=n_tokens))
            records.append({"image": name, "caption": caption})

    random.shuffle(records)
    split = int(0.8 * len(records))
    train = records[:split]
    val = records[split:]

    for rows, csv_name in [(train, "train.csv"), (val, "val.csv")]:
        with open(dataset_root / csv_name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "caption"])
            w.writeheader()
            w.writerows(rows)

    print(f"mock data: {N_IMAGES} images, {len(records)} captions, "
          f"train={len(train)}, val={len(val)}")
    return dataset_root


def patch_config_for_e2e(cfg, experiment_name: str, model_dir: str):
    """
    In-place patch a model's config module to use the mock dataset, random
    embeddings (no GloVe needed), debug-mode training, and a separate output
    prefix so we don't clobber any real runs.

    Args:
        cfg               : the imported config module (e.g. configs.m1_config)
        experiment_name   : e.g. "m1_e2e_mock"
        model_dir         : top-level output namespace, e.g. "M1" or "M4"
    """
    cfg.dataset_name = MOCK_DATASET_NAME
    cfg.embedding_type = "random"
    cfg.freq_threshold = 2
    cfg.debug = True
    cfg.debug_batch_size = 4
    cfg.debug_num_epochs = 1
    cfg.debug_pretrained_encoder = False

    cfg.experiment_name = experiment_name
    cfg.best_checkpoint_name = f"{experiment_name}_best.pt"
    cfg.last_checkpoint_name = f"{experiment_name}_last.pt"
    cfg.log_dir = f"outputs/logs/{model_dir}/{experiment_name}"
    cfg.prediction_dir = f"outputs/predictions/{model_dir}/{experiment_name}"


def reload_and_decode(cfg, build_model, n_samples: int = 3):
    """
    Load the best checkpoint back, rebuild the model via build_model(),
    and run greedy_decode on a few mock images.

    Args:
        cfg          : the model's config module (post-patching)
        build_model  : factory(vocab, ckpt) -> nn.Module
                       The factory owns the model-specific constructor args.
                       The returned model will be load_state_dict'd here.
        n_samples    : how many images to decode from val
    """
    from data.dataset import FlickrDataset
    from data.vocab import Vocabulary

    print("\n[verify] loading best checkpoint and decoding")

    dataset_root = ROOT / "datasets" / cfg.dataset_name
    image_dir = dataset_root / cfg.image_folder
    val_csv = dataset_root / "val.csv"

    ckpt_path = ROOT / cfg.checkpoint_dir / cfg.best_checkpoint_name
    print(f"  checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  saved epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f}")

    vocab = Vocabulary(freq_threshold=cfg.freq_threshold)
    vocab.stoi = ckpt["vocab_stoi"]
    vocab.itos = ckpt["vocab_itos"]

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])
    val_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=val_csv,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
    )
    val_dataset.vocab = vocab

    model = build_model(vocab, ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    images = torch.stack([val_dataset[i][0] for i in range(n_samples)], dim=0)
    generated = model.greedy_decode(images, max_len=20)

    print(f"  generated shape: {tuple(generated.shape)}")
    for i in range(n_samples):
        ids = generated[i].tolist()
        decoded = vocab.decode_ids(ids, remove_special_tokens=True)
        print(f"    sample {i}: {decoded[:80]!r}")
