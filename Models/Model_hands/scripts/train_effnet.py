from __future__ import annotations
from pathlib import Path
import json
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, top_k_accuracy_score
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from train.dataset import DatasetConfig, SlovoNPZDataset
from train.model_effnet import SlovoEfficientNetB0

def count_topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    if k == 1:
        pred = torch.argmax(logits, dim=1)
        return int((pred == y).sum().item())
    topk_idx = torch.topk(logits, k=k, dim=1).indices
    return int((topk_idx == y.unsqueeze(1)).any(dim=1).sum().item())

def build_letter_downweight(id2label, num_classes: int, letter_weight: float = 0.7) -> torch.Tensor:
    letters = set(list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"))
    w = torch.ones(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        name = str(id2label[i]).strip()
        if len(name) == 1 and name in letters:
            w[i] = letter_weight
    return w
@torch.no_grad()

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum = 0.0
    n = 0
    top1 = 0
    top5 = 0
    all_true = []
    all_pred = []
    all_proba = []
    for x, m, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        top1 += int((pred == y).sum().item())
        top5 += count_topk_correct(logits, y, k=5)
        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())
        all_proba.append(probs.detach().cpu())
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    proba = torch.cat(all_proba).numpy()
    top1_acc = (y_true == y_pred).mean()
    top5_acc = top_k_accuracy_score(y_true, proba, k=5, labels=list(range(num_classes)))
    return {
        "loss": loss_sum / max(1, n),
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "y_true": y_true,
        "y_pred": y_pred,
    }

def main():
    data_dir = PROJECT_ROOT / "keypoints_out"
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = 10
    batch_size_train = 16
    batch_size_eval = 32
    num_workers = 2
    lr = 5e-4
    weight_decay = 2e-2
    label_smoothing = 0.1
    effnet_dropout = 0.3
    best_ckpt_path = save_dir / "effnetb0_best_val_top1_hands.pt"
    plots_loss_path = save_dir / "curves_loss_hands.png"
    plots_acc_path = save_dir / "curves_acc_hands.png"
    report_path = save_dir / "classification_report_test_hands.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    num_classes = len(labels["label2id"])
    if isinstance(labels["id2label"], list):
        target_names = [str(x) for x in labels["id2label"]]
    else:
        id2label = {int(k): v for k, v in labels["id2label"].items()}
        target_names = [id2label[i] for i in range(num_classes)]
    print("num_classes:", num_classes)
    train_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="train"))
    val_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="val"))
    test_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="test"))
    pin = (device.type == "cuda")
    pw = (num_workers > 0)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pw
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pw
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pw
    )
    model = SlovoEfficientNetB0(num_classes=num_classes, pretrained=True, drop=effnet_dropout).to(device)
    class_weights = build_letter_downweight(id2label=target_names, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    hist = {
        "train_loss": [], "train_top1": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top5": [],
        "lr": [],
    }
    best_val_top1 = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n = 0
        top1 = 0
        top5 = 0
        for x, m, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
            top1 += count_topk_correct(logits.detach(), y, k=1)
            top5 += count_topk_correct(logits.detach(), y, k=5)
        train_loss = loss_sum / max(1, n)
        train_top1 = top1 / max(1, n)
        train_top5 = top5 / max(1, n)
        val_metrics = evaluate(model, val_dl, criterion, device, num_classes)
        scheduler.step()
        lr_now = optim.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | {dt:.1f}s | lr {lr_now:.6f}\n"
            f"  train: loss {train_loss:.4f} | top1 {train_top1:.4f} | top5 {train_top5:.4f}\n"
            f"  val:   loss {val_metrics['loss']:.4f} | top1 {val_metrics['top1']:.4f} | top5 {val_metrics['top5']:.4f}\n"
        )
        hist["train_loss"].append(train_loss)
        hist["train_top1"].append(train_top1)
        hist["train_top5"].append(train_top5)
        hist["val_loss"].append(val_metrics["loss"])
        hist["val_top1"].append(val_metrics["top1"])
        hist["val_top5"].append(val_metrics["top5"])
        hist["lr"].append(lr_now)
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "hist": hist}, best_ckpt_path)
            print(f"Saved BEST (epoch={best_epoch}, val_top1={best_val_top1:.4f}) -> {best_ckpt_path}\n")
    ep = list(range(1, epochs + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(ep, hist["train_loss"], label="train loss")
    plt.plot(ep, hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_loss_path, dpi=150)
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(ep, hist["train_top1"], label="train top-1")
    plt.plot(ep, hist["val_top1"], label="val top-1")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Top-1 accuracy curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_acc_path, dpi=150)
    plt.show()
    print(f"Saved plots:\n  {plots_loss_path}\n  {plots_acc_path}")
    print(f"Best epoch: {best_epoch} | best val_top1: {best_val_top1:.4f}")
    print("\n=== Final TEST evaluation (once, after training) ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    test_metrics = evaluate(model, test_dl, criterion, device, num_classes)
    print(f"TEST: loss {test_metrics['loss']:.4f} | top1 {test_metrics['top1']:.4f} | top5 {test_metrics['top5']:.4f}")
    rep = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print("\n=== Classification report (ALL classes) ===")
    print(rep)
    report_path.write_text(
        f"Best epoch: {best_epoch} | best val_top1: {best_val_top1:.4f}\n"
        f"TEST: loss {test_metrics['loss']:.4f} | top1 {test_metrics['top1']:.4f} | top5 {test_metrics['top5']:.4f}\n\n"
        f"{rep}\n",
        encoding="utf-8",
    )
    print(f"Saved classification report to: {report_path}")

if __name__ == "__main__":
    main()
