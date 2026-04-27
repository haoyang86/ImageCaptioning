import torch

from models.decoder_transformer import TransformerCaptionDecoder


def test_decoder_forward():
    batch_size = 2
    seq_len = 10
    spatial_tokens = 49
    vocab_size = 5000
    pad_idx = 0

    # fake inputs
    tgt_ids = torch.randint(
        1,
        vocab_size,
        (batch_size, seq_len)
    )

    memory = torch.randn(
        batch_size,
        spatial_tokens,
        512
    )

    decoder = TransformerCaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        encoder_dim=512
    )

    decoder.eval()

    with torch.no_grad():
        logits = decoder(
            tgt_ids=tgt_ids,
            memory=memory
        )

    print("logits shape:", logits.shape)

    assert logits.shape == (
        batch_size,
        seq_len,
        vocab_size
    )

    print("Transformer decoder test passed!")


if __name__ == "__main__":
    test_decoder_forward()