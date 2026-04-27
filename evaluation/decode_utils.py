from typing import List, Sequence, Any, Optional

import torch


def decode_caption(
    token_ids: Sequence[int],
    vocab: Any,
    remove_special_tokens: bool = True,
) -> str:
    """
    Convert one sequence of token ids into a caption string.

    Args:
        token_ids:
            A list / tuple / tensor of token ids.

        vocab:
            Vocabulary object with decode_ids() method.

        remove_special_tokens:
            If True, remove <pad>, <start>, <end>.

    Returns:
        Decoded caption string.
    """

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()

    return vocab.decode_ids(
        token_ids,
        remove_special_tokens=remove_special_tokens,
    )


def decode_batch(
    batch_token_ids: torch.Tensor,
    vocab: Any,
    remove_special_tokens: bool = True,
) -> List[str]:
    """
    Convert a batch of token id sequences into caption strings.

    Args:
        batch_token_ids:
            Tensor with shape (B, T).

        vocab:
            Vocabulary object.

    Returns:
        List of decoded captions.
    """

    captions = []

    for token_ids in batch_token_ids:
        caption = decode_caption(
            token_ids=token_ids,
            vocab=vocab,
            remove_special_tokens=remove_special_tokens,
        )
        captions.append(caption)

    return captions


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    images: torch.Tensor,
    vocab: Any,
    device: torch.device,
    max_len: Optional[int] = None,
) -> List[str]:
    """
    Generate captions using model.greedy_decode().

    This assumes each model implements:

        model.greedy_decode(images, max_len=max_len)

    Args:
        model:
            Captioning model.

        images:
            Image batch, shape (B, 3, H, W).

        vocab:
            Vocabulary object.

        device:
            torch device.

        max_len:
            Maximum decoding length.

    Returns:
        List of predicted caption strings.
    """

    model.eval()

    images = images.to(device)

    generated_ids = model.greedy_decode(
        images,
        max_len=max_len,
    )

    predictions = decode_batch(
        batch_token_ids=generated_ids,
        vocab=vocab,
        remove_special_tokens=True,
    )

    return predictions


def clean_caption(caption: str) -> str:
    """
    Basic caption cleanup.

    Useful before saving predictions or computing metrics.
    """

    caption = caption.strip()
    caption = " ".join(caption.split())

    return caption


def clean_caption_list(captions: List[str]) -> List[str]:
    """
    Clean a list of captions.
    """

    return [
        clean_caption(caption)
        for caption in captions
    ]