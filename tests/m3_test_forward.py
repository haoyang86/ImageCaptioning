import torch
import torch.nn as nn

from models.model_m3 import ModelM3


def test_m3_forward():
    batch_size = 2
    seq_len = 12
    vocab_size = 5000

    pad_idx = 0
    start_idx = 1
    end_idx = 2

    images = torch.randn(batch_size, 3, 224, 224)

    captions = torch.randint(
        low=3,
        high=vocab_size,
        size=(batch_size, seq_len)
    )

    captions[:, 0] = start_idx

    model = ModelM3(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        pretrained=False,   # avoid downloading pretrained ResNet weights
    )

    model.eval()

    inputs = captions[:, :-1]
    targets = captions[:, 1:]

    with torch.no_grad():
        logits = model(images, inputs)

    print("logits shape:", logits.shape)

    assert logits.shape == (
        batch_size,
        seq_len - 1,
        vocab_size
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    loss = criterion(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    print("loss:", loss.item())

    with torch.no_grad():
        generated = model.greedy_decode(images, max_len=15)

    print("generated shape:", generated.shape)

    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= 15

    print("ModelM3 forward test passed!")


if __name__ == "__main__":
    test_m3_forward()