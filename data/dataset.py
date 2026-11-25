"""Data pipeline Fashion-MNIST untuk Conditional DDPM (pemilik: Aksa)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tanpa jendela GUI, aman dijalankan dari terminal
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config, get_config

# ToTensor menghasilkan [0, 1]. (x - 0.5) / 0.5 memetakannya persis ke [-1, 1].
# Dua konstanta ini mendefinisikan pemetaan tersebut, bukan hyperparameter.
_NORM_MEAN: float = 0.5
_NORM_STD: float = 0.5

# Nama kelas bawaan Fashion-MNIST (urutan label 0..9), hanya untuk label plot.
CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)

_CELL_INCH: float = 0.8  # ukuran satu sel gambar pada plot pemeriksaan


def _build_transform() -> transforms.Compose:
    """Transform deterministik untuk train dan test (tanpa augmentasi)."""
    return transforms.Compose([
        transforms.ToTensor(),  # PIL uint8 [28, 28] -> float32 [1, 28, 28] dalam [0, 1]
        transforms.Normalize((_NORM_MEAN,), (_NORM_STD,)),  # [1, 28, 28] dalam [-1, 1]
    ])


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Kembalikan (train_loader, test_loader).

    Batch: x float32 [B, 1, 28, 28] dalam [-1, 1]; y int64 [B] bernilai 0..9.
    Train: shuffle (diatur seed dari cfg), drop_last=True. Test: urut tetap, semua sampel.
    """
    tf = _build_transform()
    train_set = datasets.FashionMNIST(root=cfg.data_root, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(root=cfg.data_root, train=False, download=True, transform=tf)

    gen = torch.Generator().manual_seed(cfg.seed)  # urutan shuffle bisa diulang
    # persistent_workers: worker tidak dibuat ulang tiap epoch (spawn di Windows lambat)
    persistent = cfg.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        generator=gen,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, test_loader


def denormalize(x: Tensor) -> Tensor:
    """[-1, 1] -> [0, 1] untuk disimpan sebagai gambar. Nilai di luar rentang di-clamp."""
    return (x * _NORM_STD + _NORM_MEAN).clamp(0.0, 1.0)  # shape sama dengan x


# ---------------------------------------------------------------------------
# Skrip pemeriksaan: python -m data.dataset [--n_per_class 10]
# ---------------------------------------------------------------------------

def _first_n_per_class(dataset: datasets.FashionMNIST, n: int, num_classes: int) -> Tensor:
    """Ambil n gambar pertama tiap kelas (sudah ternormalisasi)."""
    targets = dataset.targets  # int64 [N]
    rows = []
    for c in range(num_classes):
        idx = (targets == c).nonzero(as_tuple=True)[0][:n]  # [n]
        rows.append(torch.stack([dataset[int(i)][0] for i in idx]))  # [n, 1, 28, 28]
    return torch.stack(rows)  # [num_classes, n, 1, 28, 28]


def _save_check_grid(cfg: Config, train_set: datasets.FashionMNIST, n: int, out: Path) -> None:
    """Simpan grid [kelas x n contoh] dengan nama kelas di sumbu-y."""
    imgs = denormalize(_first_n_per_class(train_set, n, cfg.num_classes))  # [C, n, 1, 28, 28]
    c, _, _, h, w = imgs.shape
    grid = imgs.squeeze(2).permute(0, 2, 1, 3).reshape(c * h, n * w)  # [C*28, n*28]

    fig, ax = plt.subplots(figsize=(n * _CELL_INCH, c * _CELL_INCH))
    ax.imshow(grid.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_yticks([(r + 0.5) * h for r in range(c)])
    ax.set_yticklabels([f"{r}: {CLASS_NAMES[r]}" for r in range(c)])
    ax.set_xticks([])
    ax.set_title(f"Fashion-MNIST train, {n} contoh pertama per kelas")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _check(cfg: Config, n_per_class: int) -> None:
    train_loader, test_loader = get_dataloaders(cfg)
    x, y = next(iter(train_loader))

    print(f"x: shape={tuple(x.shape)} dtype={x.dtype} min={x.min():.4f} max={x.max():.4f} "
          f"mean={x.mean():.4f} std={x.std():.4f}")
    print(f"y: shape={tuple(y.shape)} dtype={y.dtype} min={y.min().item()} max={y.max().item()}")
    print(f"batch ke-1, jumlah per label : {torch.bincount(y, minlength=cfg.num_classes).tolist()}")
    print(f"train penuh, jumlah per label: "
          f"{torch.bincount(train_loader.dataset.targets, minlength=cfg.num_classes).tolist()}")
    print(f"test penuh, jumlah per label : "
          f"{torch.bincount(test_loader.dataset.targets, minlength=cfg.num_classes).tolist()}")
    print(f"jumlah batch: train={len(train_loader)} (drop_last), test={len(test_loader)}")

    assert x.shape == (cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size), "shape x salah"
    assert x.dtype == torch.float32, "dtype x harus float32"
    assert x.min() >= -1.0 and x.max() <= 1.0, "x di luar rentang [-1, 1]"
    assert y.dtype == torch.int64, "dtype y harus int64"
    assert 0 <= y.min().item() and y.max().item() < cfg.num_classes, "label di luar 0..9"
    d = denormalize(x)
    assert d.min() >= 0.0 and d.max() <= 1.0, "denormalize di luar [0, 1]"

    out = Path(cfg.runs_dir) / "data_check.png"
    _save_check_grid(cfg, train_loader.dataset, n_per_class, out)
    print(f"grid disimpan ke {out}")
    print("SEMUA PEMERIKSAAN LULUS")


if __name__ == "__main__":  # wajib di Windows karena num_workers > 0
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--n_per_class", type=int, default=10)
    extra, rest = parser.parse_known_args()
    _check(get_config(rest), extra.n_per_class)