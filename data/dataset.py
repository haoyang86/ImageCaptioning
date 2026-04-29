import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.vocab import Vocabulary


class FlickrDataset(Dataset):
    """
    Flickr8K / Flickr30K dataset

    returns:
        image
        caption_ids
    """

    def __init__(
        self,
        image_dir,
        caption_file,
        transform=None,
        freq_threshold=5,
        vocab=None,
    ):
        """
        Args
        ----
        image_dir:
            image folder

        caption_file:
            csv annotations with columns:
            image, caption

        transform:
            image transforms

        freq_threshold:
            word frequency threshold used only if vocab is not provided

        vocab:
            existing shared vocabulary.
            If provided, this dataset will use it directly.
        """

        self.image_dir = image_dir
        self.transform = transform

        # expected csv:
        # image,caption
        self.df = pd.read_csv(caption_file)

        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Use shared vocabulary if provided.
        # Otherwise build vocabulary from this dataset's captions.
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(
                freq_threshold=freq_threshold
            )

            self.vocab.build_vocabulary(
                self.captions.tolist()
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        caption = self.captions[idx]
        img_name = self.images[idx]

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        image = Image.open(
            img_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption_ids = torch.tensor(
            self.vocab.numericalize(
                caption
            ),
            dtype=torch.long
        )

        return image, caption_ids