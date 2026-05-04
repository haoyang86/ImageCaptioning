from typing import List, Sequence, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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

@torch.no_grad()
def beam_generate(
    model: torch.nn.Module,
    images: torch.Tensor,
    vocab: Any,
    device: torch.device,
    max_len: int = 50,
    beam_size: int = 3,
    length_penalty: float = 0.7,
) -> List[str]:
    """
    Generate captions using beam search.

    Important:
        This implementation assumes the decoder supports:

            decoder(
                tgt_ids=tgt_ids,
                memory=memory,
                memory_key_padding_mask=None,
            )

        If your LSTM decoder does not support this interface,
        you need to add a step-level decoding method to the decoder.
    """

    model.eval()
    images = images.to(device)

    start_idx = vocab.start_idx
    end_idx = vocab.end_idx
    pad_idx = vocab.pad_idx

    encoder_output, _ = model.encoder(images)

    batch_generated = []

    for i in range(images.size(0)):
        memory_i = encoder_output[i:i + 1]

        generated_i = beam_search_single(
            decoder=model.decoder,
            memory=memory_i,
            start_idx=start_idx,
            end_idx=end_idx,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            device=device,
        )

        batch_generated.append(generated_i.cpu())

    generated_ids = pad_sequence(
        batch_generated,
        batch_first=True,
        padding_value=pad_idx,
    )

    predictions = decode_batch(
        batch_token_ids=generated_ids,
        vocab=vocab,
        remove_special_tokens=True,
    )

    return predictions


@torch.no_grad()
def beam_search_single(
    decoder: torch.nn.Module,
    memory: torch.Tensor,
    start_idx: int,
    end_idx: int,
    max_len: int,
    beam_size: int,
    length_penalty: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Beam search for one image.

    Returns:
        Tensor of token ids with shape (L,)
    """

    beams: List[Tuple[torch.Tensor, float, bool]] = [
        (
            torch.tensor(
                [start_idx],
                dtype=torch.long,
                device=device,
            ),
            0.0,
            False,
        )
    ]

    for _ in range(max_len - 1):

        candidates: List[Tuple[torch.Tensor, float, bool]] = []

        for seq, score, finished in beams:

            if finished:
                candidates.append((seq, score, True))
                continue

            tgt_ids = seq.unsqueeze(0)

            logits = decoder(
                tgt_ids=tgt_ids,
                memory=memory,
                memory_key_padding_mask=None,
            )

            next_token_logits = logits[:, -1, :]

            log_probs = F.log_softmax(
                next_token_logits,
                dim=-1,
            ).squeeze(0)

            top_log_probs, top_indices = torch.topk(
                log_probs,
                beam_size,
            )

            for log_prob, token_idx in zip(
                top_log_probs,
                top_indices,
            ):

                token_idx = token_idx.view(1)

                new_seq = torch.cat(
                    [seq, token_idx],
                    dim=0,
                )

                new_score = score + float(log_prob.item())

                new_finished = int(token_idx.item()) == end_idx

                candidates.append(
                    (
                        new_seq,
                        new_score,
                        new_finished,
                    )
                )

        candidates = sorted(
            candidates,
            key=lambda x: normalized_beam_score(
                score=x[1],
                length=len(x[0]),
                length_penalty=length_penalty,
            ),
            reverse=True,
        )

        beams = candidates[:beam_size]

        if all(finished for _, _, finished in beams):
            break

    best_seq, _, _ = max(
        beams,
        key=lambda x: normalized_beam_score(
            score=x[1],
            length=len(x[0]),
            length_penalty=length_penalty,
        ),
    )

    return best_seq


def normalized_beam_score(
    score: float,
    length: int,
    length_penalty: float,
) -> float:
    """
    Normalize beam score to reduce the bias toward short captions.
    """

    if length_penalty <= 0:
        return score

    return score / (length ** length_penalty)


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