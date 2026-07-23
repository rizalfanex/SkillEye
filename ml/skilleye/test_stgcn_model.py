import torch

from stgcn_model import STGCN


def test_extract_features_shape():
    model = STGCN(num_classes=2, base_channels=32)
    x = torch.randn(4, 4, 64, 17)
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)


def test_forward_equals_fc_of_extract_features():
    model = STGCN(num_classes=2, base_channels=32)
    model.eval()
    x = torch.randn(4, 4, 64, 17)
    with torch.no_grad():
        feats = model.extract_features(x)
        logits_via_features = model.fc(feats)
        logits_direct = model.forward(x)
    assert torch.allclose(logits_via_features, logits_direct)


def test_forward_output_shape_unchanged():
    model = STGCN(num_classes=6, base_channels=32)
    x = torch.randn(2, 4, 64, 17)
    logits = model(x)
    assert logits.shape == (2, 6)
