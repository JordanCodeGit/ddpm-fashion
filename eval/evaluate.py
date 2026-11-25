"""Evaluasi sampel Conditional DDPM dengan FashionCNN classifier (pemilik: Aksa).

Menghasilkan `eval.json` (`overall_acc`, `per_class_acc`, `classifier_test_acc`)
dan `confusion.png` di dalam `cfg.run_dir` (`runs/<run_name>/`).

Selama `diffusion/ddpm.py` milik Axan belum tersedia, gunakan flag `--use_stub`
untuk menguji pipeline memakai gambar test set asli sesuai label yang diminta.

Pemakaian (dari root repo):
    # Uji kewajaran dengan stub (gambar test set asli)
    python -m eval.evaluate --use_stub --n_per_class 100 --run_name stub_test --device cpu

    # Evaluasi checkpoint final DDPM
    python -m eval.evaluate --n_per_class 100 --from_json runs/<run_name>/config.json
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")  # tanpa jendela GUI, aman dijalankan dari terminal
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from config import Config, get_config
from data.dataset import CLASS_NAMES, get_dataloaders
from eval.classifier import load_classifier, resolve_device, set_seed
from models.unet import UNet

_DEFAULT_CLASSIFIER_CKPT: str = "eval/classifier.pt"
_DEFAULT_DDPM_CKPT_NAME: str = "ckpt_ema.pt"
_CKPT_STATE_KEYS: tuple[str, ...] = (
    "ema_state_dict",
    "model_state_dict",
    "state_dict",
    "ema",
    "model",
)
_CM_FIGSIZE: tuple[float, float] = (9.5, 8.0)


class DiffusionSampler(Protocol):
    """Kontrak minimal metode sample() yang dipatuhi GaussianDiffusion dan stub."""

    def sample(self, y: Tensor, guidance_scale: float = 0.0) -> Tensor:
        ...


# ---------------------------------------------------------------------------
# STUB (SEMENTARA): dipakai hanya saat --use_stub aktif sebelum ddpm.py ada
# ---------------------------------------------------------------------------

class StubTestSetDiffusion(nn.Module):
    """STUB pengganti GaussianDiffusion sebelum `diffusion/ddpm.py` tersedia.

    Mengembalikan gambar test set Fashion-MNIST asli (skala [-1, 1]) sesuai
    label `y` yang diminta, mematuhi kontrak `GaussianDiffusion.sample`.
    Akurasi evaluasi dengan stub ini harus mendekati `classifier_test_acc`.
    """

    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        _, test_loader = get_dataloaders(cfg)

        # Kumpulkan semua gambar test set asli per kelas dalam rentang [-1, 1]
        buckets: list[list[Tensor]] = [[] for _ in range(cfg.num_classes)]
        for x, y in test_loader:
            for cls_id in range(cfg.num_classes):
                mask = y == cls_id  # [B]
                if mask.any():
                    buckets[cls_id].append(x[mask])  # [K, 1, 28, 28]

        gen = torch.Generator().manual_seed(cfg.seed)
        self.bank: list[Tensor] = []
        self.cursors: list[int] = [0] * cfg.num_classes
        for cls_id in range(cfg.num_classes):
            cls_imgs = torch.cat(buckets[cls_id], dim=0)  # [N_c, 1, 28, 28]
            perm = torch.randperm(cls_imgs.size(0), generator=gen)  # [N_c]
            self.bank.append(cls_imgs[perm])

    @torch.no_grad()
    def sample(self, y: Tensor, guidance_scale: float = 0.0) -> Tensor:
        """Kembalikan batch gambar test set asli [N, C, H, W] dalam [-1, 1] sesuai label y."""
        _validate_guidance_scale(guidance_scale, self.cfg)
        if y.dim() != 1 or y.dtype != torch.int64:
            raise ValueError(f"y harus int64 1D [N], didapat shape={tuple(y.shape)} dtype={y.dtype}")
        if ((y < 0) | (y >= self.cfg.num_classes)).any().item():
            raise ValueError(f"label y harus di 0..{self.cfg.num_classes - 1}")

        chosen: list[Tensor] = []
        for cls_id in y.detach().cpu().tolist():
            pool = self.bank[cls_id]  # [N_c, 1, 28, 28]
            idx = self.cursors[cls_id] % pool.size(0)
            self.cursors[cls_id] += 1
            chosen.append(pool[idx])

        return torch.stack(chosen, dim=0).to(self.device)  # [N, 1, 28, 28]


# ---------------------------------------------------------------------------
# Pemuatan model DDPM asli (UNet asli dari models/unet.py + GaussianDiffusion)
# ---------------------------------------------------------------------------

def _validate_guidance_scale(guidance_scale: float, cfg: Config) -> None:
    """Validasi aturan kontrak guidance_scale: w >= 0, dan w > 0 butuh cond_drop_prob > 0."""
    if guidance_scale < 0.0:
        raise ValueError(f"guidance_scale wajib >= 0, didapat {guidance_scale}")
    if guidance_scale > 0.0 and cfg.cond_drop_prob == 0.0:
        raise ValueError(
            "guidance_scale > 0 tidak valid untuk run dengan cond_drop_prob == 0 "
            "(token tanpa kelas tidak pernah dilatih)."
        )


def _extract_state_dict(raw_ckpt: Any) -> dict[str, Tensor]:
    """Ambil state_dict U-Net baik checkpoint disimpan langsung maupun di dalam dict."""
    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"Format checkpoint tidak dikenali: {type(raw_ckpt).__name__}")
    for key in _CKPT_STATE_KEYS:
        if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
            return raw_ckpt[key]
    if raw_ckpt and all(isinstance(v, Tensor) for v in raw_ckpt.values()):
        return raw_ckpt
    raise ValueError(
        f"state_dict model tidak ditemukan di checkpoint; kunci yang ada: {sorted(raw_ckpt.keys())}"
    )


def build_ddpm_sampler(
    cfg: Config,
    device: torch.device,
    use_stub: bool,
    ckpt_override: str | None,
) -> DiffusionSampler:
    """Bangun sampler evaluasi: StubTestSetDiffusion (--use_stub) atau GaussianDiffusion asli."""
    # Selalu pastikan UNet asli dari models/unet.py terbangun sesuai cfg (bukan stub)
    unet = UNet(cfg).to(device)
    unet.eval()

    if use_stub:
        print("MODE STUB AKTIF: memakai StubTestSetDiffusion (gambar test set asli, tanpa ddpm.py)")
        return StubTestSetDiffusion(cfg, device)

    ckpt_path = Path(ckpt_override) if ckpt_override else (cfg.run_dir / _DEFAULT_DDPM_CKPT_NAME)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint DDPM tidak ditemukan di {ckpt_path}. "
            "Pastikan training sudah menghasilkan ckpt_ema.pt, atau gunakan --use_stub untuk uji pipeline."
        )

    try:
        ddpm_mod = importlib.import_module("diffusion.ddpm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Modul `diffusion/ddpm.py` (milik Axan) belum tersedia di repo. "
            "Gunakan flag `--use_stub` untuk menguji pipeline evaluasi terlebih dahulu."
        ) from exc

    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = _extract_state_dict(raw_ckpt)
    unet.load_state_dict(state_dict)
    unet.eval()

    diffusion = ddpm_mod.GaussianDiffusion(unet, cfg).to(device)
    diffusion.eval()
    print(f"Memuat checkpoint DDPM dari {ckpt_path}")
    return diffusion


# ---------------------------------------------------------------------------
# Sampling, klasifikasi, metrik, dan plot confusion matrix
# ---------------------------------------------------------------------------

def _check_generated_batch(x_gen: Tensor, expected_n: int, cfg: Config) -> None:
    """Validasi kontrak output sample(): float32 [N, C, H, W] finite dalam [-1, 1]."""
    expected_shape = (expected_n, cfg.in_channels, cfg.img_size, cfg.img_size)
    if tuple(x_gen.shape) != expected_shape:
        raise ValueError(f"Output sample() harus {expected_shape}, didapat {tuple(x_gen.shape)}")
    if x_gen.dtype != torch.float32:
        raise ValueError(f"Output sample() harus float32, didapat {x_gen.dtype}")
    if not torch.isfinite(x_gen).all().item():
        raise ValueError("Output sample() mengandung NaN/Inf")
    x_min = x_gen.min().item()
    x_max = x_gen.max().item()
    if x_min < -1.0 - 1e-4 or x_max > 1.0 + 1e-4:
        raise ValueError(f"Output sample() di luar rentang [-1, 1]: [{x_min:.4f}, {x_max:.4f}]")


@torch.no_grad()
def generate_and_classify(
    sampler: DiffusionSampler,
    classifier: nn.Module,
    cfg: Config,
    n_per_class: int,
    guidance_scale: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate n_per_class sampel tiap kelas dalam chunk cfg.batch_size lalu klasifikasikan."""
    if n_per_class <= 0:
        raise ValueError(f"n_per_class harus > 0, didapat {n_per_class}")

    # Susun label target: [0..0, 1..1, ..., 9..9] sepanjang num_classes * n_per_class
    y_all = torch.arange(cfg.num_classes, dtype=torch.int64).repeat_interleave(n_per_class)  # [N]
    total = y_all.size(0)
    preds_chunks: list[Tensor] = []

    classifier.eval()
    starts = range(0, total, cfg.batch_size)
    for start in tqdm(starts, desc="Sampling & evaluasi"):
        y_chunk = y_all[start : start + cfg.batch_size].to(device)  # [B]
        x_gen = sampler.sample(y_chunk, guidance_scale=guidance_scale)  # [B, 1, 28, 28]
        _check_generated_batch(x_gen, y_chunk.size(0), cfg)

        logits = classifier(x_gen.to(device))  # [B, num_classes]
        preds_chunks.append(logits.argmax(dim=1).cpu())  # [B]

    y_pred = torch.cat(preds_chunks, dim=0)  # [N]
    return y_all, y_pred


def compute_metrics(
    y_true: Tensor,
    y_pred: Tensor,
    num_classes: int,
) -> tuple[float, dict[str, float], Tensor]:
    """Hitung overall_acc, per_class_acc (dict nama kelas -> akurasi), dan confusion matrix."""
    idx = y_true * num_classes + y_pred  # [N]
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)  # [C, C]

    total = cm.sum().item()
    overall_acc = cm.diag().sum().item() / total if total > 0 else 0.0

    per_class_acc: dict[str, float] = {}
    for cls_id in range(num_classes):
        row_sum = cm[cls_id].sum().item()
        acc = (cm[cls_id, cls_id].item() / row_sum) if row_sum > 0 else 0.0
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        per_class_acc[name] = round(acc, 6)

    return overall_acc, per_class_acc, cm


def save_confusion_matrix(
    cm: Tensor,
    out_path: Path,
    run_name: str,
    overall_acc: float,
    n_per_class: int,
    guidance_scale: float,
    use_stub: bool,
) -> None:
    """Plot confusion matrix (warna ternormalisasi per baris, teks jumlah + persen) ke PNG."""
    cm_np = cm.numpy().astype(np.int64)  # [C, C]
    row_sums = np.maximum(cm_np.sum(axis=1, keepdims=True), 1)  # [C, 1]
    cm_norm = cm_np / row_sums  # [C, C] dalam [0, 1]

    num_classes = cm_np.shape[0]
    labels = [
        f"{i}: {CLASS_NAMES[i]}" if i < len(CLASS_NAMES) else str(i)
        for i in range(num_classes)
    ]

    fig, ax = plt.subplots(figsize=_CM_FIGSIZE)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporsi per kelas target")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Kelas Prediksi Classifier")
    ax.set_ylabel("Kelas Kondisi Target")

    tag = " [STUB: Test Set Asli]" if use_stub else f" (w={guidance_scale:g})"
    ax.set_title(
        f"Confusion Matrix - {run_name}{tag}\n"
        f"Overall Acc: {overall_acc * 100:.2f}% | n_per_class={n_per_class}"
    )

    for r in range(num_classes):
        for c in range(num_classes):
            val = cm_np[r, c]
            pct = cm_norm[r, c] * 100.0
            color = "white" if cm_norm[r, c] >= 0.5 else "black"
            ax.text(c, r, f"{val}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    cfg: Config,
    n_per_class: int,
    guidance_scale: float,
    classifier_ckpt: Path,
    use_stub: bool,
    ckpt_override: str | None,
) -> dict[str, Any]:
    """Jalankan pipeline evaluasi lengkap dan simpan eval.json + confusion.png di cfg.run_dir."""
    _validate_guidance_scale(guidance_scale, cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg)

    classifier, classifier_test_acc = load_classifier(classifier_ckpt, cfg, device)
    sampler = build_ddpm_sampler(cfg, device, use_stub=use_stub, ckpt_override=ckpt_override)

    y_true, y_pred = generate_and_classify(
        sampler=sampler,
        classifier=classifier,
        cfg=cfg,
        n_per_class=n_per_class,
        guidance_scale=guidance_scale,
        device=device,
    )
    overall_acc, per_class_acc, cm = compute_metrics(y_true, y_pred, cfg.num_classes)

    out_dir = cfg.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = out_dir / "eval.json"
    cm_png_path = out_dir / "confusion.png"

    result: dict[str, Any] = {
        "overall_acc": round(overall_acc, 6),
        "per_class_acc": per_class_acc,
        "classifier_test_acc": round(classifier_test_acc, 6),
        "n_per_class": n_per_class,
        "guidance_scale": guidance_scale,
        "run_name": cfg.run_name,
        "used_stub": use_stub,
        "confusion_matrix": cm.tolist(),
    }
    eval_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_confusion_matrix(
        cm=cm,
        out_path=cm_png_path,
        run_name=cfg.run_name,
        overall_acc=overall_acc,
        n_per_class=n_per_class,
        guidance_scale=guidance_scale,
        use_stub=use_stub,
    )

    diff_pp = abs(overall_acc - classifier_test_acc) * 100.0
    print(f"\nRun: {cfg.run_name} | device={device} | n_per_class={n_per_class} | stub={use_stub}")
    print(f"Overall accuracy      : {overall_acc * 100:.2f}%")
    print(f"Classifier test acc   : {classifier_test_acc * 100:.2f}% (selisih {diff_pp:.2f} pp)")
    print("Akurasi per kelas:")
    for idx, (name, acc) in enumerate(per_class_acc.items()):
        print(f"  {idx}: {name:<12} {acc * 100:6.2f}%")
    print(f"Hasil disimpan ke {eval_json_path} dan {cm_png_path}")
    return result


def _parse_args() -> tuple[argparse.Namespace, Config]:
    """Parse argumen khusus evaluate.py lalu teruskan sisanya ke get_config (sesuai kontrak)."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--n_per_class", type=int, default=100, help="jumlah sampel per kelas")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="skala CFG (w >= 0)")
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default=_DEFAULT_CLASSIFIER_CKPT,
        help="path checkpoint classifier FashionCNN",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="override path checkpoint DDPM (default: runs/<run_name>/ckpt_ema.pt)",
    )
    parser.add_argument(
        "--use_stub",
        action="store_true",
        help="uji pipeline memakai StubTestSetDiffusion (gambar test set asli)",
    )
    extra, rest = parser.parse_known_args()
    return extra, get_config(rest)


if __name__ == "__main__":  # wajib di Windows karena DataLoader dengan num_workers > 0
    extra_args, config = _parse_args()
    run_evaluation(
        cfg=config,
        n_per_class=extra_args.n_per_class,
        guidance_scale=extra_args.guidance_scale,
        classifier_ckpt=Path(extra_args.classifier_ckpt),
        use_stub=extra_args.use_stub,
        ckpt_override=extra_args.ckpt,
    )
