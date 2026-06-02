from __future__ import annotations
from pathlib import Path
import json
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, top_k_accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from train.dataset import DatasetConfig, SlovoNPZDataset
from train.model_effnetv2_s import SlovoEfficientNetV2S
INPUT_SIZE = 384

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    try:
        import torch_directml
        return torch_directml.device(), "directml"
    except Exception:
        return torch.device("cpu"), "cpu"

def move_to_device(x: torch.Tensor, device, backend: str) -> torch.Tensor:
    return x.to(device, non_blocking=(backend == "cuda"))

def resize_for_model(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)

def count_topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    if k == 1:
        pred = torch.argmax(logits, dim=1)
        return int((pred == y).sum().item())
    topk_idx = torch.topk(logits, k=k, dim=1).indices
    return int((topk_idx == y.unsqueeze(1)).any(dim=1).sum().item())

def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def build_letter_downweight(id2label, num_classes: int, letter_weight: float = 0.7) -> torch.Tensor:
    letters = set(
        "\u0410\u0411\u0412\u0413\u0414\u0415\u0401\u0416\u0417\u0418\u0419"
        "\u041a\u041b\u041c\u041d\u041e\u041f\u0420\u0421\u0422\u0423\u0424"
        "\u0425\u0426\u0427\u0428\u0429\u042a\u042b\u042c\u042d\u042e\u042f"
    )
    weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        name = str(id2label[idx]).strip()
        if len(name) == 1 and name in letters:
            weights[idx] = letter_weight
    return weights
@torch.no_grad()

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum = 0.0
    n = 0
    y_true_all = []
    y_pred_all = []
    proba_all = []
    for x, _m, y in loader:
        x = resize_for_model(move_to_device(x, device, backend_name))
        y = move_to_device(y, device, backend_name)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        y_true_all.append(y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())
        proba_all.append(probs.detach().cpu())
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    proba = torch.cat(proba_all).numpy()
    return {
        "loss": loss_sum / max(1, n),
        "top1": float((y_true == y_pred).mean()),
        "top5": float(top_k_accuracy_score(y_true, proba, k=5, labels=list(range(num_classes)))),
        "y_true": y_true,
        "y_pred": y_pred,
    }

def main() -> None:
    data_dir = PROJECT_ROOT / "keypoints_out"
    save_dir = PROJECT_ROOT / "checkpoints_effnetv2_s_384"
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = 15
    batch_size_train = 12
    batch_size_eval = 24
    num_workers = 2
    lr = 3e-4
    weight_decay = 2e-2
    label_smoothing = 0.1
    dropout = 0.3
    best_ckpt_path = save_dir / "effnetv2_s_384_best_val_top1.pt"
    plots_loss_path = save_dir / "curves_loss_effnetv2_s_384.png"
    plots_acc_path = save_dir / "curves_acc_effnetv2_s_384.png"
    report_path = save_dir / "classification_report_test_effnetv2_s_384.txt"
    global backend_name
    device, backend_name = choose_device()
    print("device:", device)
    print("backend:", backend_name)
    if backend_name == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    elif backend_name == "directml":
        print("gpu: AMD/DirectML")
        num_workers = 0
        print("num_workers forced to 0 for DirectML")
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    num_classes = len(labels["label2id"])
    if isinstance(labels["id2label"], list):
        target_names = [str(x) for x in labels["id2label"]]
    else:
        id2label = {int(k): v for k, v in labels["id2label"].items()}
        target_names = [id2label[i] for i in range(num_classes)]
    train_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="train"))
    val_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="val"))
    test_ds = SlovoNPZDataset(DatasetConfig(data_dir=data_dir, split="test"))
    pin = backend_name == "cuda"
    persistent = num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    val_dl = DataLoader(val_ds, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    test_dl = DataLoader(test_ds, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    weights_path = PROJECT_ROOT / "pretrained" / "efficientnet_v2_s-dd5fe13b.pth"
    model = SlovoEfficientNetV2S(
        num_classes=num_classes,
        pretrained=True,
        drop=dropout,
        weights_path=weights_path,
    ).to(device)
    class_weights = build_letter_downweight(target_names, num_classes=num_classes, letter_weight=0.7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hist = {"train_loss": [], "train_top1": [], "train_top5": [], "val_loss": [], "val_top1": [], "val_top5": [], "lr": []}
    best_val_top1 = -1.0
    best_epoch = -1
    train_started_at = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n = 0
        top1 = 0
        top5 = 0
        train_bar = tqdm(
            train_dl,
            desc=f"Epoch {epoch:02d}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for x, _m, y in train_bar:
            x = resize_for_model(move_to_device(x, device, backend_name))
            y = move_to_device(y, device, backend_name)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
            top1 += count_topk_correct(logits.detach(), y, k=1)
            top5 += count_topk_correct(logits.detach(), y, k=5)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = loss_sum / max(1, n)
        train_top1 = top1 / max(1, n)
        train_top5 = top5 / max(1, n)
        val_metrics = evaluate(model, val_dl, criterion, device, num_classes)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | {time.time() - t0:.1f}s | lr {lr_now:.6f}\n"
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
    train_elapsed_sec = time.time() - train_started_at
    train_elapsed_text = format_duration(train_elapsed_sec)
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
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(ep, hist["train_top1"], label="train top-1")
    plt.plot(ep, hist["val_top1"], label="val top-1")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Top-1 accuracy curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_acc_path, dpi=150)
    plt.close()
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_dl, criterion, device, num_classes)
    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_path.write_text(
        f"Best epoch: {best_epoch} | best val_top1: {best_val_top1:.4f}\n"
        f"Training time: {train_elapsed_text} ({train_elapsed_sec:.2f} sec)\n"
        f"TEST: loss {test_metrics['loss']:.4f} | top1 {test_metrics['top1']:.4f} | top5 {test_metrics['top5']:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )
    print(f"Training time: {train_elapsed_text} ({train_elapsed_sec:.2f} sec)")
    print(f"TEST: loss {test_metrics['loss']:.4f} | top1 {test_metrics['top1']:.4f} | top5 {test_metrics['top5']:.4f}")
    print(f"Saved report to: {report_path}")

if __name__ == "__main__":
    main()
