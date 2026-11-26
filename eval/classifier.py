"""CNN classifier Fashion-MNIST dari nol sebagai evaluator Conditional DDPM (pemilik: Aksa).

Input memakai normalisasi [-1, 1] yang sama persis dengan DDPM lewat
`data.dataset.get_dataloaders(cfg)`. Bobot terbaik disimpan ke `eval/classifier.pt`
(di-.gitignore sehingga skrip ini dirancang cepat dijalankan ulang).

Pemakaian (dari root repo):
    python -m eval.classifier
    python -m eval.classifier --device cpu --clf_epochs 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_config
from data.dataset import CLASS_NAMES, get_dataloaders

# Ukuran pool spasial 2x2 menjaga informasi posisi atas/bawah pakaian tanpa hardcode 7x7.
_POOL_GRID: int = 2
_DEFAULT_CHANNELS: tuple[int, int, int] = (32, 64, 128)
_DEFAULT_CKPT: str = "eval/classifier.pt"


class ConvBNReLU(nn.Module):
    """Satu blok Conv3x3 (tanpa bias) -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)  # [B, in_ch, H, W] -> [B, out_ch, H, W]


class FashionCNN(nn.Module):
    """CNN ringkas 4-layer konvolusi untuk klasifikasi Fashion-MNIST berskala [-1, 1]."""

    def __init__(
        self,
        cfg: Config,
        channels: Sequence[int] = _DEFAULT_CHANNELS,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError(f"channels harus berisi 3 bilangan bulat positif, didapat {channels}")

        c1, c2, c3 = (int(c) for c in channels)
        self.in_channels = cfg.in_channels
        self.img_size = cfg.img_size
        self.num_classes = cfg.num_classes
        self.channels = (c1, c2, c3)

        self.stage1 = nn.Sequential(
            ConvBNReLU(cfg.in_channels, c1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(c1, c2),
            ConvBNReLU(c2, c2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(c2, c3),
            nn.AdaptiveAvgPool2d((_POOL_GRID, _POOL_GRID)),
        )

        flat_dim = c3 * _POOL_GRID * _POOL_GRID
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(cfg.dropout),
            nn.Linear(flat_dim, c3),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(c3, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Petakan gambar float32 [B, C, H, W] dalam [-1, 1] ke logits [B, num_classes]."""
        h = self.stage1(x)  # [B, C, 28, 28] -> [B, c1, 14, 14]
        h = self.stage2(h)  # [B, c1, 14, 14] -> [B, c2, 7, 7]
        h = self.stage3(h)  # [B, c2, 7, 7] -> [B, c3, 2, 2]
        return self.head(h)  # [B, c3, 2, 2] -> [B, num_classes]


def resolve_device(cfg: Config) -> torch.device:
    """Pilih device dari cfg; beri peringatan jelas jika CUDA diminta tetapi tidak tersedia."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("PERINGATAN: cfg.device='cuda' tetapi CUDA tidak tersedia, berjalan di CPU")
        return torch.device("cpu")
    return torch.device(cfg.device)


def set_seed(seed: int) -> None:
    """Atur seed PyTorch agar pelatihan classifier dapat direproduksi."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Jalankan satu epoch pelatihan dan kembalikan (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d}", leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=True)  # [B, 1, 28, 28]
        y = y.to(device, non_blocking=True)  # [B]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # [B, num_classes]
        loss = criterion(logits, y)  # skalar
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)  # [B]
        correct += (preds == y).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[float]]:
    """Evaluasi classifier pada DataLoader; kembalikan (mean_loss, overall_acc, per_class_acc)."""
    model.eval()
    total_loss = 0.0
    total = 0
    class_correct = torch.zeros(num_classes, dtype=torch.int64)  # [num_classes]
    class_total = torch.zeros(num_classes, dtype=torch.int64)  # [num_classes]

    for x, y in loader:
        x = x.to(device, non_blocking=True)  # [B, 1, 28, 28]
        y = y.to(device, non_blocking=True)  # [B]

        logits = model(x)  # [B, num_classes]
        loss = criterion(logits, y)  # skalar
        preds = logits.argmax(dim=1)  # [B]

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        y_cpu = y.cpu()  # [B]
        p_cpu = preds.cpu()  # [B]
        class_total += torch.bincount(y_cpu, minlength=num_classes)
        class_correct += torch.bincount(y_cpu[p_cpu == y_cpu], minlength=num_classes)

    overall_acc = class_correct.sum().item() / total
    per_class_acc = [
        (class_correct[c].item() / class_total[c].item()) if class_total[c].item() > 0 else 0.0
        for c in range(num_classes)
    ]
    return total_loss / total, overall_acc, per_class_acc


def load_classifier(
    ckpt_path: str | Path,
    cfg: Config,
    device: torch.device,
) -> tuple[FashionCNN, float]:
    """Muat bobot classifier terlatih ke mode eval; kembalikan (model, classifier_test_acc)."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint classifier tidak ditemukan di {path}. "
            "Jalankan `python -m eval.classifier` terlebih dahulu."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    channels = tuple(payload.get("channels", _DEFAULT_CHANNELS))
    model = FashionCNN(cfg, channels=channels).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, float(payload["test_acc"])


def train_and_save(
    cfg: Config,
    clf_epochs: int,
    clf_lr: float,
    weight_decay: float,
    channels: Sequence[int],
    out_path: Path,
) -> dict[str, object]:
    """Latih FashionCNN pada Fashion-MNIST, laporkan metrik test, dan simpan bobot terbaik."""
    set_seed(cfg.seed)
    device = resolve_device(cfg)
    train_loader, test_loader = get_dataloaders(cfg)

    model = FashionCNN(cfg, channels=channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | FashionCNN channels={tuple(channels)} | params={n_params:,}")

    # Uji cepat shape output sebelum loop training
    dummy = torch.zeros(2, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)  # [2, 1, 28, 28]
    with torch.no_grad():
        dummy_out = model(dummy)  # [2, num_classes]
    assert dummy_out.shape == (2, cfg.num_classes), f"shape logits salah: {tuple(dummy_out.shape)}"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=clf_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=clf_epochs)

    best_acc = -1.0
    best_per_class: list[float] = []
    best_state: dict[str, Tensor] = {}
    t_start = time.perf_counter()

    for epoch in range(1, clf_epochs + 1):
        t0 = time.perf_counter()
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip, epoch, clf_epochs
        )
        te_loss, te_acc, per_class = evaluate_classifier(
            model, test_loader, criterion, device, cfg.num_classes
        )
        scheduler.step()
        dt = time.perf_counter() - t0

        marker = ""
        if te_acc > best_acc:
            best_acc = te_acc
            best_per_class = per_class
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        print(
            f"Epoch {epoch:02d}/{clf_epochs:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc * 100:.2f}% | "
            f"test_loss={te_loss:.4f} test_acc={te_acc * 100:.2f}% | "
            f"lr={cur_lr:.2e} | {dt:.1f}s{marker}"
        )

    total_time = time.perf_counter() - t_start
    payload = {
        "model_state_dict": best_state,
        "test_acc": best_acc,
        "per_class_acc": best_per_class,
        "channels": list(channels),
        "in_channels": cfg.in_channels,
        "img_size": cfg.img_size,
        "num_classes": cfg.num_classes,
        "dropout": cfg.dropout,
        "clf_epochs": clf_epochs,
        "clf_lr": clf_lr,
        "weight_decay": weight_decay,
        "seed": cfg.seed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"\nSelesai dalam {total_time:.1f}s | Best test accuracy: {best_acc * 100:.2f}%")
    print("Akurasi per kelas (test set):")
    for idx, acc in enumerate(best_per_class):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        print(f"  {idx}: {name:<12} {acc * 100:6.2f}%")
    print(f"Bobot terbaik disimpan ke {out_path}")
    return payload


def _parse_args() -> tuple[argparse.Namespace, Config]:
    """Parse argumen khusus classifier dulu lalu teruskan sisanya ke get_config."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--clf_epochs", type=int, default=10, help="jumlah epoch pelatihan classifier")
    parser.add_argument("--clf_lr", type=float, default=1e-3, help="learning rate AdamW classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay AdamW classifier")
    parser.add_argument(
        "--clf_channels",
        type=int,
        nargs=3,
        default=list(_DEFAULT_CHANNELS),
        help="jumlah channel untuk 3 stage konvolusi",
    )
    parser.add_argument("--out", type=str, default=_DEFAULT_CKPT, help="path simpan checkpoint classifier")
    extra, rest = parser.parse_known_args()
    return extra, get_config(rest)


if __name__ == "__main__":  # wajib di Windows karena num_workers > 0 pada DataLoader
    extra_args, config = _parse_args()
    train_and_save(
        cfg=config,
        clf_epochs=extra_args.clf_epochs,
        clf_lr=extra_args.clf_lr,
        weight_decay=extra_args.weight_decay,
        channels=tuple(extra_args.clf_channels),
        out_path=Path(extra_args.out),
    )
