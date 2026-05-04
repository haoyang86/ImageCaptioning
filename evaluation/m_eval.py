from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs import m_config as cfg
from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate

from models.model import Model_IC
from models.embedding_utils import (
    build_embedding_matrix,
    should_use_pretrained_embedding,
)

from evaluation.eval_utils import run_evaluation


def main():

    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------------------
    # Paths
    # -----------------------------------------

    dataset_root = (
        project_root /
        "datasets" /
        cfg.dataset_name
    )

    image_dir = dataset_root / cfg.image_folder
    eval_csv = dataset_root / "val.csv"

    checkpoint_path = (
        project_root /
        cfg.checkpoint_dir /
        cfg.best_checkpoint_name
    )

    prediction_dir = project_root / cfg.prediction_dir

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_path = (
        prediction_dir /
        cfg.val_prediction_name
    )

    metrics_path = (
        prediction_dir /
        cfg.metrics_result_name
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Eval csv:", eval_csv)
    print("Checkpoint:", checkpoint_path)
    print("Prediction path:", prediction_path)
    print("Metrics path:", metrics_path)
    print("Encoder type:", cfg.encoder_type)
    print("Decoder type:", cfg.decoder_type)
    print("Decode strategy:", cfg.decode_strategy)

    if cfg.decode_strategy == "beam":
        print("Beam size:", cfg.beam_size)
        print("Length penalty:", cfg.length_penalty)

    # -----------------------------------------
    # Transform
    # -----------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (
                cfg.image_size,
                cfg.image_size
            )
        ),
        transforms.ToTensor(),
    ])

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    eval_dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=eval_csv,
        transform=transform,
        freq_threshold=cfg.freq_threshold,
    )

    # -----------------------------------------
    # Load checkpoint metadata
    # -----------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Restore training vocab
    eval_dataset.vocab.stoi = checkpoint["vocab_stoi"]
    eval_dataset.vocab.itos = checkpoint["vocab_itos"]

    vocab = eval_dataset.vocab
    vocab_size = len(vocab)

    ckpt_config = checkpoint.get(
        "config",
        {}
    )

    embedding_type = ckpt_config.get(
        "embedding_type",
        cfg.embedding_type,
    )

    embedding_dim = ckpt_config.get(
        "embedding_dim",
        cfg.embedding_dim,
    )

    embedding_path = ckpt_config.get(
        "embedding_path",
        cfg.embedding_path,
    )

    encoder_dim = ckpt_config.get(
        "encoder_dim",
        cfg.encoder_dim,
    )

    encoder_type = ckpt_config.get(
        "encoder_type",
        cfg.encoder_type,
    )

    decoder_type = ckpt_config.get(
        "decoder_type",
        cfg.decoder_type,
    )

    print("Eval samples:", len(eval_dataset))
    print("Vocab size:", vocab_size)
    print("Checkpoint embedding type:", embedding_type)
    print("Checkpoint embedding dim:", embedding_dim)
    print("Checkpoint encoder type:", encoder_type)
    print("Checkpoint decoder type:", decoder_type)

    # -----------------------------------------
    # Optional pretrained embedding matrix
    # -----------------------------------------

    pretrained_embedding_matrix = None

    if should_use_pretrained_embedding(
        embedding_type
    ):
        glove_path = (
            project_root /
            embedding_path
        )

        pretrained_embedding_matrix, found_count, oov_count = (
            build_embedding_matrix(
                vocab=vocab,
                glove_path=glove_path,
                embedding_dim=embedding_dim,
            )
        )

        print(
            "Pretrained embedding matrix:",
            pretrained_embedding_matrix.shape,
        )
        print("GloVe found:", found_count)
        print("GloVe OOV:", oov_count)

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=(
            cfg.debug_batch_size
            if cfg.debug
            else cfg.batch_size
        ),
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=FlickrCollate(
            pad_idx=vocab.pad_idx
        ),
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = Model_IC(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_idx,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,

        encoder_dim=encoder_dim,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=False,

        encoder_type=encoder_type,
        decoder_type=decoder_type,

        embedding_type=embedding_type,
        embedding_dim=embedding_dim,
        pretrained_embedding_matrix=pretrained_embedding_matrix,
    )

    max_batches = (
        5 if cfg.debug
        else None
    )

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
        max_len=cfg.max_len,
        max_batches=max_batches,

        decode_strategy=cfg.decode_strategy,
        beam_size=cfg.beam_size,
        length_penalty=cfg.length_penalty,
    )


if __name__ == "__main__":
    main()