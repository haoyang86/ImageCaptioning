from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs import m3_config as cfg

from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate
from data.vocab import Vocabulary

from models.model_m3 import ModelM3
from models.embedding_utils import (
    build_embedding_matrix,
    should_use_pretrained_embedding,
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    vocab_size,
    epoch,
):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, captions) in enumerate(loader, start=1):

        images = images.to(device)
        captions = captions.to(device)

        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        logits = model(
            images,
            decoder_input,
        )

        loss = criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.grad_clip,
        )

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss {loss.item():.4f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    vocab_size,
):
    model.eval()
    total_loss = 0.0

    for images, captions in loader:

        images = images.to(device)
        captions = captions.to(device)

        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        logits = model(
            images,
            decoder_input,
        )

        loss = criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    vocab,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "train_loss": train_loss,
        "val_loss": val_loss,

        "vocab_stoi": vocab.stoi,
        "vocab_itos": vocab.itos,
        "pad_idx": vocab.pad_idx,
        "start_idx": vocab.start_idx,
        "end_idx": vocab.end_idx,
        "unk_idx": vocab.unk_idx,

        "config": {
            "experiment_name": cfg.experiment_name,

            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_decoder_layers": cfg.num_decoder_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,

            "encoder_dim": cfg.encoder_dim,
            "freq_threshold": cfg.freq_threshold,

            "embedding_type": cfg.embedding_type,
            "embedding_dim": cfg.embedding_dim,
            "embedding_path": cfg.embedding_path,

            "freeze_backbone": cfg.freeze_backbone,
            "pretrained_encoder": cfg.pretrained_encoder,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def main():

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    dataset_root = (
        project_root /
        "datasets" /
        cfg.dataset_name
    )

    image_dir = dataset_root / cfg.image_folder
    train_csv = dataset_root / "train.csv"
    val_csv = dataset_root / "val.csv"
    shared_vocab_path = dataset_root / "shared_vocab.pkl"

    output_dir = (
        project_root /
        cfg.checkpoint_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_ckpt_path = (
        output_dir /
        cfg.best_checkpoint_name
    )

    last_ckpt_path = (
        output_dir /
        cfg.last_checkpoint_name
    )

    # --------------------------------------------------
    # Debug mode settings
    # --------------------------------------------------

    if cfg.debug:
        batch_size = cfg.debug_batch_size
        num_epochs = cfg.debug_num_epochs
        pretrained_encoder = cfg.debug_pretrained_encoder
    else:
        batch_size = cfg.batch_size
        num_epochs = cfg.num_epochs
        pretrained_encoder = cfg.pretrained_encoder

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Debug mode:", cfg.debug)
    print("Experiment:", cfg.experiment_name)
    print("Embedding type:", cfg.embedding_type)
    print("Image dir:", image_dir)
    print("Train csv:", train_csv)
    print("Val csv:", val_csv)
    print("Shared vocab:", shared_vocab_path)
    print("Batch size:", batch_size)
    print("Epochs:", num_epochs)
    print("Pretrained encoder:", pretrained_encoder)
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)

    # --------------------------------------------------
    # Transform
    # --------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (
                cfg.image_size,
                cfg.image_size,
            )
        ),
        transforms.ToTensor(),
    ])

    # --------------------------------------------------
    # Shared Vocabulary
    # --------------------------------------------------

    if not shared_vocab_path.exists():
        raise FileNotFoundError(
            f"Shared vocabulary not found: {shared_vocab_path}\n"
            f"Please run data/prepare_flickr8k.py first."
        )

    vocab = Vocabulary.load(shared_vocab_path)
    vocab_size = len(vocab)

    print("Vocab size:", vocab_size)
    print("Pad idx:", vocab.pad_idx)
    print("Start idx:", vocab.start_idx)
    print("End idx:", vocab.end_idx)
    print("Unk idx:", vocab.unk_idx)

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=train_csv,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
        vocab=vocab,
    )

    val_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=val_csv,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
        vocab=vocab,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # --------------------------------------------------
    # Optional pretrained embedding matrix
    # --------------------------------------------------

    pretrained_embedding_matrix = None

    if should_use_pretrained_embedding(
        cfg.embedding_type
    ):
        glove_path = (
            project_root /
            cfg.embedding_path
        )

        pretrained_embedding_matrix, found_count, oov_count = (
            build_embedding_matrix(
                vocab=vocab,
                glove_path=glove_path,
                embedding_dim=cfg.embedding_dim,
            )
        )

        print(
            "Pretrained embedding matrix:",
            pretrained_embedding_matrix.shape
        )
        print("GloVe found:", found_count)
        print("GloVe OOV:", oov_count)

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=FlickrCollate(
            pad_idx=vocab.pad_idx,
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=FlickrCollate(
            pad_idx=vocab.pad_idx,
        ),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ModelM3(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,

        encoder_dim=cfg.encoder_dim,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=pretrained_encoder,

        embedding_type=cfg.embedding_type,
        embedding_dim=cfg.embedding_dim,
        pretrained_embedding_matrix=pretrained_embedding_matrix,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, num_epochs + 1):

        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            vocab_size=vocab_size,
            epoch=epoch,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            vocab_size=vocab_size,
        )

        elapsed = time.time() - start_time

        print(
            f"\nEpoch {epoch}/{num_epochs} finished | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            vocab=vocab,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                vocab=vocab,
            )

            print(
                f"Saved best checkpoint: {best_ckpt_path}"
            )

        print("-" * 80)

    print("Training finished.")
    print("Best val loss:", best_val_loss)
    print("Best checkpoint:", best_ckpt_path)


if __name__ == "__main__":
    main()