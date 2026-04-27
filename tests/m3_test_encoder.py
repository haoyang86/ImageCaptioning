import torch

from models.encoder_spatial import EncoderSpatial


def test_encoder_spatial_forward():
    batch_size = 2
    encoder_dim = 512

    images = torch.randn(batch_size, 3, 224, 224)

    encoder = EncoderSpatial(
        encoder_dim=encoder_dim,
        freeze_backbone=True,
        pretrained=False,   # test 时建议 False，避免下载 pretrained weights
        dropout=0.2,
    )

    encoder.eval()

    with torch.no_grad():
        encoder_output, global_feat = encoder(images)

    print("encoder_output shape:", encoder_output.shape)
    print("global_feat shape:", global_feat.shape)

    assert encoder_output.shape == (batch_size, 49, encoder_dim)
    assert global_feat.shape == (batch_size, encoder_dim)

    print("EncoderSpatial test passed!")


if __name__ == "__main__":
    test_encoder_spatial_forward()