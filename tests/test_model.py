import torch

from models.model import build_model, save_checkpoint, load_checkpoint


def test_forward_shapes():
    model = build_model(num_classes=43, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 43)
    assert features.shape == (2, 512)


def test_forward_without_features_returns_logits_only():
    model = build_model(num_classes=43, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 43)


def test_frozen_layers_have_no_grad():
    model = build_model(num_classes=43, pretrained=False)
    for name in ("conv1", "bn1", "layer1", "layer2"):
        module = getattr(model.backbone, name)
        assert all(not p.requires_grad for p in module.parameters())
    for name in ("layer3", "layer4"):
        module = getattr(model.backbone, name)
        assert all(p.requires_grad for p in module.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_train_mode_keeps_frozen_stages_in_eval():
    model = build_model(num_classes=43, pretrained=False)
    model.train()
    for name in ("conv1", "bn1", "layer1", "layer2"):
        assert getattr(model.backbone, name).training is False
    assert model.backbone.layer3.training is True


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    model = build_model(num_classes=43, pretrained=False)
    model.eval()
    ckpt_path = tmp_path / "model.ckpt"
    save_checkpoint(ckpt_path, model, extra={"val_acc": 0.5, "epoch": 1})

    loaded_model, payload = load_checkpoint(ckpt_path, device="cpu")
    assert payload["val_acc"] == 0.5
    assert payload["num_classes"] == 43

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        original_out = model(x)
        loaded_out = loaded_model(x)
    torch.testing.assert_close(original_out, loaded_out)
