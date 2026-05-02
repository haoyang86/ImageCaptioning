import sys
import zipfile
import pandas as pd
from pathlib import Path

# Allow running as `uv run data/prepare_flickr8k.py` (script mode) by putting
# the project root on sys.path. Running via `-m data.prepare_flickr8k` also still works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.vocab import Vocabulary


def unzip_if_needed(zip_path, extract_to):
    """
    Unzip only if not already extracted.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Cannot find {zip_path}"
        )

    print(f"Extracting {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    print("Done.")


def find_file(root, filename):
    """
    Recursively find a file.
    """
    for p in root.rglob(filename):
        return p
    return None


def build_captions_csv(token_file, output_csv):

    records = []

    with open(token_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # image.jpg#0 \t caption
            img_id, caption = line.split("\t")

            image_name = img_id.split("#")[0]

            records.append(
                {
                    "image": image_name,
                    "caption": caption
                }
            )

    df = pd.DataFrame(records)

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Saved captions.csv "
        f"({len(df)} rows)"
    )

    return df


def build_split_csv(
    captions_df,
    split_txt,
    output_csv
):

    with open(
        split_txt,
        "r",
        encoding="utf-8"
    ) as f:

        image_names = set(
            line.strip()
            for line in f
            if line.strip()
        )

    split_df = captions_df[
        captions_df["image"].isin(
            image_names
        )
    ].copy()

    split_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"{output_csv.name} saved: "
        f"{split_df['image'].nunique()} images, "
        f"{len(split_df)} captions"
    )

    return split_df


def build_shared_vocab(
    train_csv,
    output_vocab,
    freq_threshold=5
):
    """
    Build shared vocabulary from training captions only.
    """

    train_df = pd.read_csv(train_csv)

    vocab = Vocabulary(
        freq_threshold=freq_threshold
    )

    vocab.build_vocabulary(
        train_df["caption"].tolist()
    )

    vocab.save(output_vocab)

    print(
        f"Saved shared vocabulary: {output_vocab.name} "
        f"(vocab size = {len(vocab)})"
    )

    return vocab


def main():

    project_root = Path(
        __file__
    ).resolve().parents[1]

    dataset_root = (
        project_root /
        "datasets" /
        "flickr8k"
    )

    raw_dir = dataset_root / "raw"

    image_zip = raw_dir / "Flickr8k_Dataset.zip"
    text_zip = raw_dir / "Flickr8k_text.zip"

    # --------------------------------
    # unzip images
    # --------------------------------
    image_folder = (
        dataset_root /
        "Flicker8k_Dataset"
    )

    if not image_folder.exists():
        unzip_if_needed(
            image_zip,
            dataset_root
        )
    else:
        print(
            "Image folder exists."
        )

    # --------------------------------
    # unzip text files
    # --------------------------------
    token_file = find_file(
        dataset_root,
        "Flickr8k.token.txt"
    )

    if token_file is None:
        unzip_if_needed(
            text_zip,
            dataset_root
        )

    # --------------------------------
    # find annotation files
    # --------------------------------
    token_file = find_file(
        dataset_root,
        "Flickr8k.token.txt"
    )

    train_file = find_file(
        dataset_root,
        "Flickr_8k.trainImages.txt"
    )

    val_file = find_file(
        dataset_root,
        "Flickr_8k.devImages.txt"
    )

    test_file = find_file(
        dataset_root,
        "Flickr_8k.testImages.txt"
    )

    if token_file is None:
        raise FileNotFoundError(
            "Flickr8k.token.txt not found"
        )

    if train_file is None:
        raise FileNotFoundError(
            "Flickr_8k.trainImages.txt not found"
        )

    if val_file is None:
        raise FileNotFoundError(
            "Flickr_8k.devImages.txt not found"
        )

    if test_file is None:
        raise FileNotFoundError(
            "Flickr_8k.testImages.txt not found"
        )

    # --------------------------------
    # captions.csv
    # --------------------------------
    captions_csv = (
        dataset_root /
        "captions.csv"
    )

    captions_df = build_captions_csv(
        token_file,
        captions_csv
    )

    # --------------------------------
    # train / val / test split
    # --------------------------------
    train_csv = dataset_root / "train.csv"
    val_csv = dataset_root / "val.csv"
    test_csv = dataset_root / "test.csv"

    build_split_csv(
        captions_df,
        train_file,
        train_csv
    )

    build_split_csv(
        captions_df,
        val_file,
        val_csv
    )

    build_split_csv(
        captions_df,
        test_file,
        test_csv
    )

    # --------------------------------
    # shared vocabulary
    # build from train captions only
    # --------------------------------
    shared_vocab_path = (
        dataset_root /
        "shared_vocab.pkl"
    )

    build_shared_vocab(
        train_csv=train_csv,
        output_vocab=shared_vocab_path,
        freq_threshold=5,
    )

    print("\nDone.")
    print(
        "Generated:\n"
        "  captions.csv\n"
        "  train.csv\n"
        "  val.csv\n"
        "  test.csv\n"
        "  shared_vocab.pkl"
    )


if __name__ == "__main__":
    main()