"""Fine-tune a pretrained ResNet18 on GTSRB and save a checkpoint.

Run from repo root:
    python -m models.train
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, make_loader
from models.inference import pick_device
from models.model import build_model, save_checkpoint


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])

    device = pick_device()
    print(f"device: {device}")

    data_root = resolve_path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    num_classes = cfg["data"]["num_classes"]

    splits = get_gtsrb_splits(data_root, image_size, seed=cfg["seed"],
                               fractions=cfg["data"]["train_val_calib_split"])
    mcfg = cfg["model"]
    train_loader = make_loader(splits["train"], mcfg["batch_size"], shuffle=True, num_workers=mcfg["num_workers"])
    val_loader = make_loader(splits["val"], mcfg["batch_size"], shuffle=False, num_workers=mcfg["num_workers"])
    print(f"train={len(splits['train'])} val={len(splits['val'])} "
          f"calib={len(splits['calib'])} test={len(splits['test'])}")

    model = build_model(num_classes=num_classes, pretrained=mcfg["pretrained"],
                         freeze=tuple(mcfg["freeze"])).to(device)

    optimizer = torch.optim.Adam([
        {"params": model.backbone.layer3.parameters(), "lr": mcfg["lr_backbone"]},
        {"params": model.backbone.layer4.parameters(), "lr": mcfg["lr_backbone"]},
        {"params": model.classifier.parameters(), "lr": mcfg["lr_head"]},
    ], weight_decay=mcfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = resolve_path(mcfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_acc = evaluate(model, val_loader, device)
        train_loss = running_loss / seen
        dt = time.time() - t0
        print(f"epoch {epoch}/{mcfg['epochs']}  train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_path, model, extra={
                "val_acc": val_acc, "epoch": epoch, "image_size": image_size,
                "arch": mcfg["arch"],
            })
            print(f"  saved checkpoint ({checkpoint_path}) val_acc={val_acc:.4f}")

    test_loader = make_loader(splits["test"], mcfg["batch_size"], shuffle=False, num_workers=mcfg["num_workers"])
    model, _ = build_model(num_classes=num_classes, pretrained=False).to(device), None
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_acc = evaluate(model, test_loader, device)
    print(f"final best-checkpoint test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
