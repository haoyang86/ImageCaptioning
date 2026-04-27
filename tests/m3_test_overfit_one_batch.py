from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs import m3_config as cfg

from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate

from models.model_m3 import ModelM3
from models.embedding_utils import (
    build_embedding_matrix,
    should_use_pretrained_embedding,
)


def main():

    project_root = Path(__file__).resolve().parents[1]

    image_dir = (
        project_root
        / "datasets"
        / "flickr8k"
        / "Flicker8k_Dataset"
    )

    caption_file = (
        project_root
        / "datasets"
        / "flickr8k"
        / "train.csv"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Embedding type:", cfg.embedding_type)
    print("Image dir:", image_dir)
    print("Caption file:", caption_file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=FlickrCollate(
            pad_idx=dataset.vocab.pad_idx
        ),
    )

    images, captions = next(iter(loader))

    images = images.to(device)
    captions = captions.to(device)

    vocab = dataset.vocab
    vocab_size = len(vocab)

    pad_idx = vocab.pad_idx
    start_idx = vocab.start_idx
    end_idx = vocab.end_idx

    print("\nBatch:")
    print("images:", images.shape)
    print("captions:", captions.shape)
    print("vocab_size:", vocab_size)

    # --------------------------------------------------
    # Optional pretrained embedding matrix
    # --------------------------------------------------

    pretrained_embedding_matrix = None

    if should_use_pretrained_embedding(
        cfg.embedding_type
    ):
        glove_path = (
            project_root
            / cfg.embedding_path
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
            pretrained_embedding_matrix.shape,
        )
        print("GloVe found:", found_count)
        print("GloVe OOV:", oov_count)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ModelM3(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,

        pretrained=False,
        freeze_backbone=True,

        embedding_type=cfg.embedding_type,
        embedding_dim=cfg.embedding_dim,
        pretrained_embedding_matrix=pretrained_embedding_matrix,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    model.train()

    num_steps = 200

    print("\nStart one-batch overfit test")
    print("Embedding type:", cfg.embedding_type)
    print("images:", images.shape)
    print("captions:", captions.shape)
    print("vocab_size:", vocab_size)

    for step in range(1, num_steps + 1):

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
            max_norm=1.0,
        )

        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(
                f"Step {step:03d} | "
                f"loss = {loss.item():.4f}"
            )

    print("\nDecode after overfitting:")

    model.eval()

    with torch.no_grad():
        generated = model.greedy_decode(
            images,
            max_len=20,
        )

    for i in range(generated.shape[0]):

        print("-" * 60)

        print(
            "Target:",
            vocab.decode_ids(
                captions[i].detach().cpu().tolist()
            )
        )

        print(
            "Pred:",
            vocab.decode_ids(
                generated[i].detach().cpu().tolist()
            )
        )

    print(
        "\nOne-batch overfit test finished "
        f"with embedding_type={cfg.embedding_type}."
    )


if __name__ == "__main__":
    main()