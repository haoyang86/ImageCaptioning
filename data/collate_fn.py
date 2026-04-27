import torch
from torch.nn.utils.rnn import pad_sequence


class FlickrCollate:
    """
    Collate function for Flickr image captioning dataset.

    Input from dataset:
        image:       (3, 224, 224)
        caption_ids: (T,)

    Output batch:
        images:      (B, 3, 224, 224)
        captions:    (B, T_max)
    """

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = []
        captions = []

        for image, caption in batch:
            images.append(image)
            captions.append(caption)

        images = torch.stack(images, dim=0)

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return images, captions