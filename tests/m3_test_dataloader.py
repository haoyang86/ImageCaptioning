from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import FlickrDataset
from data.collate_fn import FlickrCollate


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    image_dir = PROJECT_ROOT / "datasets" / "flickr8k" / "Flicker8k_Dataset"
    caption_file = PROJECT_ROOT / "datasets" / "flickr8k" / "train.csv"

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
        freq_threshold=5,
    )

    print("=" * 60)
    print("Dataset size:", len(dataset))
    print("Vocab size:", len(dataset.vocab))

    image, caption = dataset[0]

    print("\nSingle sample:")
    print("Image shape:", image.shape)
    print("Caption ids:", caption)
    print("Decoded caption:", dataset.vocab.decode_ids(caption.tolist()))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=FlickrCollate(
            pad_idx=dataset.vocab.pad_idx
        )
    )

    images, captions = next(iter(loader))

    print("\nBatch test:")
    print("Images shape:", images.shape)
    print("Captions shape:", captions.shape)

    print("\nBatch captions:")
    for i in range(4):
        print(i, ":", dataset.vocab.decode_ids(captions[i].tolist()))

    print("\nPAD index:", dataset.vocab.pad_idx)
    print("START index:", dataset.vocab.start_idx)
    print("END index:", dataset.vocab.end_idx)

    print("=" * 60)
    print("Data pipeline test passed.")


if __name__ == "__main__":
    main()