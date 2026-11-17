"""Colab baseline runner for the authors' ISIC-DICM-17K train/valid manifests.

This uses a ResNet18 backbone and a short training budget. It is a documented
deviation from Ahammed et al. (2025), not an exact score reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


BASE = "https://raw.githubusercontent.com/mmu-dermatology-research/isic-dicm-17k/main/"
MODES = ("image", "metadata", "fusion")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_manifests(image_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(BASE + "train-set-metadata.csv", low_memory=False)
    valid = pd.read_csv(BASE + "valid-set-metadata.csv", low_memory=False)
    needed = {"isic_id", "class", "age_approx", "sex", "anatom_site_general",
              "patient_id", "lesion_id"}
    for name, frame in (("train", train), ("valid", valid)):
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
        if frame["isic_id"].duplicated().any() or frame["isic_id"].isna().any():
            raise ValueError(f"{name} has duplicate or missing image IDs")
        if set(frame["class"].dropna().unique()) != {0, 1} or frame["class"].isna().any():
            raise ValueError(f"{name} requires complete binary class labels")
        missing_images = [x for x in frame["isic_id"] if not (image_dir / f"{x}.jpg").is_file()]
        if missing_images:
            raise FileNotFoundError(f"{name}: {len(missing_images)} images missing; first: {missing_images[:3]}")
    overlap = {}
    for col in ("isic_id", "patient_id", "lesion_id"):
        a = set(train[col].dropna().astype(str))
        b = set(valid[col].dropna().astype(str))
        overlap[col] = len(a & b)
    if overlap["isic_id"]:
        raise ValueError("Image IDs overlap across train and validation")
    audit = {
        "train_rows": len(train), "valid_rows": len(valid),
        "train_classes": train["class"].value_counts().sort_index().to_dict(),
        "valid_classes": valid["class"].value_counts().sort_index().to_dict(),
        "missing_patient_id": {"train": int(train.patient_id.isna().sum()),
                               "valid": int(valid.patient_id.isna().sum())},
        "distinct_cross_split_id_overlap": overlap,
        "warning": "Published validation is not patient/lesion independent when group overlap is nonzero.",
    }
    return train, valid, audit


class MetadataEncoder:
    """Fit age imputation and categorical vocabulary on training rows only."""

    def fit(self, train: pd.DataFrame) -> "MetadataEncoder":
        self.age_median = float(pd.to_numeric(train.age_approx, errors="coerce").median())
        self.age_mean = float(pd.to_numeric(train.age_approx, errors="coerce").fillna(self.age_median).mean())
        self.age_std = float(pd.to_numeric(train.age_approx, errors="coerce").fillna(self.age_median).std())
        if not np.isfinite(self.age_std) or self.age_std == 0:
            self.age_std = 1.0
        self.categories = {}
        for col in ("sex", "anatom_site_general"):
            values = train[col].fillna("unknown").astype(str).str.lower().str.strip()
            self.categories[col] = sorted(set(values) | {"unknown"})
        return self

    @property
    def dimension(self) -> int:
        return 1 + sum(len(v) for v in self.categories.values())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        age = pd.to_numeric(frame.age_approx, errors="coerce").fillna(self.age_median)
        columns = [((age.to_numpy(dtype=np.float32) - self.age_mean) / self.age_std)[:, None]]
        for col, categories in self.categories.items():
            values = frame[col].fillna("unknown").astype(str).str.lower().str.strip().to_numpy()
            columns.append(np.column_stack([(values == category).astype(np.float32)
                                            for category in categories]))
        return np.concatenate(columns, axis=1).astype(np.float32)

    def config(self) -> dict:
        return {"age_median": self.age_median, "age_mean": self.age_mean,
                "age_std": self.age_std, "categories": self.categories,
                "dimension": self.dimension}


def image_transform(training: bool):
    # Paper: shortest side 224, center crop 224. Train augmentation below is
    # a practical ResNet18 deviation and is explicitly logged in run.json.
    if training:
        ops = [transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip()]
    else:
        ops = [transforms.Resize(224), transforms.CenterCrop(224)]
    ops += [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, metadata: np.ndarray, image_dir: Path,
                 mode: str, training: bool):
        self.ids = frame.isic_id.astype(str).tolist()
        self.labels = frame["class"].astype(int).to_numpy()
        self.metadata = torch.from_numpy(metadata)
        self.image_dir = image_dir
        self.mode = mode
        self.transform = image_transform(training)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image = torch.empty(0)
        if self.mode != "metadata":
            with Image.open(self.image_dir / f"{self.ids[index]}.jpg") as source:
                image = self.transform(source.convert("RGB"))
        return image, self.metadata[index], int(self.labels[index]), self.ids[index]


class Baseline(nn.Module):
    def __init__(self, mode: str, metadata_dim: int, pretrained: bool = True):
        super().__init__()
        self.mode = mode
        if mode != "metadata":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            self.backbone.fc = nn.Identity()
        if mode != "image":
            self.meta_branch = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(), nn.Dropout(0.2))
        width = 512 if mode == "image" else 64 if mode == "metadata" else 576
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, 2))

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if self.mode == "image":
            features = self.backbone(image)
        elif self.mode == "metadata":
            features = self.meta_branch(meta)
        else:
            features = torch.cat([self.backbone(image), self.meta_branch(meta)], dim=1)
        return self.head(features)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probs, ids = [], [], []
    for image, meta, target, batch_ids in loader:
        image, meta = image.to(device), meta.to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(image, meta)
        probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().tolist())
        labels.extend(target.tolist())
        ids.extend(batch_ids)
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= 0.5).astype(int)
    metrics = {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": classification_report(y, pred, labels=[0, 1], output_dict=True,
                                            zero_division=0),
    }
    predictions = pd.DataFrame({"isic_id": ids, "y_true": y, "p_class_1": p,
                                "y_pred": pred})
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    image_dir = Path(args.data_dir) / "images"
    train, valid, audit = load_manifests(image_dir)
    print("Split audit:", json.dumps(audit, indent=2))
    encoder = MetadataEncoder().fit(train)
    train_data = LesionDataset(train, encoder.transform(train), image_dir, args.mode, True)
    valid_data = LesionDataset(valid, encoder.transform(valid), image_dir, args.mode, False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                              generator=generator)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline(args.mode, encoder.dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    output = Path(args.output_dir) / args.mode
    output.mkdir(parents=True, exist_ok=True)
    commit = None
    if "__file__" in globals():
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False).stdout.strip() or None
    code_sha = globals().get("SOURCE_SHA256")
    if code_sha is None and "__file__" in globals():
        code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {
        "protocol": "authors_train_valid_manifest_group_overlap_present",
        "paper_score_reproduction": False, "mode": args.mode, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "backbone": None if args.mode == "metadata" else "ImageNet pretrained ResNet18",
        "optimizer": "AdamW(weight_decay=1e-4)", "loss": "CrossEntropyLoss",
        "metadata_dropout_p": float(getattr(args, "metadata_dropout_p", 0.0)),
        "reported_epoch_rule": "fixed final epoch; no best-validation-epoch selection",
        "image_preprocessing": "train: resize256/random_resized_crop224/hflip; valid: shortest-side resize224/center_crop224; ImageNet normalization",
        "metadata_features": ["age_approx", "sex", "anatom_site_general"],
        "metadata_preprocessing": encoder.config(), "split_audit": audit,
        "data_dir": str(args.data_dir), "git_commit": commit,
        "source_sha256": code_sha,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (output / "run.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, seen = 0.0, 0
        epoch_start = time.perf_counter()
        for image, meta, target, _ in train_loader:
            image, meta, target = image.to(device), meta.to(device), target.to(device)
            drop_p = float(getattr(args, "metadata_dropout_p", 0.0))
            if drop_p:
                if args.mode != "fusion" or not (0 < drop_p < 1):
                    raise ValueError("metadata dropout requires fusion mode and 0 < p < 1")
                missing = torch.rand((len(meta), 1), device=device) < drop_p
                meta = meta.masked_fill(missing, 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(image, meta)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += float(loss.item()) * len(target)
            seen += len(target)
        metrics, predictions = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": losses / seen,
               "seconds": time.perf_counter() - epoch_start,
               **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(args.mode, row, flush=True)
        # The fixed final epoch is the reported result. Selecting the best
        # validation epoch and reporting on that same set would add bias.
        if epoch == args.epochs:
            metrics["reported_epoch"] = epoch
            metrics["total_train_seconds"] = time.perf_counter() - started
            metrics["peak_gpu_memory_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                                              if device.type == "cuda" else None)
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            predictions.to_csv(output / "validation_predictions.csv", index=False)
            if args.save_checkpoint:
                torch.save(model.state_dict(), output / "final_model.pt")
    print("Saved", output, "total seconds", round(time.perf_counter() - started, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--data-dir", default="/content/isic-dicm-17k")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/YZM5110/results/baselines")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--metadata-dropout-p", type=float, default=0.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()

