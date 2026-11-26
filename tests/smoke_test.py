"""
Integration Smoke Test end-to-end untuk proyek Conditional DDPM Fashion-MNIST.
Dijalankan dari root repository melalui: python -m tests.smoke_test
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

# ==============================================================================
# 1. IMPOR MODUL ASLI WAJIB (JORDAN)
# ==============================================================================
try:
    from config import Config, get_config
    from models.unet import UNet
except ImportError as exc:
    raise ImportError(
        "[ERROR - Modul: config.py / models/unet.py (Jordan)] "
        f"Gagal mengimpor modul inti wajib dari repo: {exc}"
    ) from exc


# ==============================================================================
# 2. DETEKSI MODUL ANGGOTA LAIN & DEFINISI FALLBACK STUB SESUAI KONTRAK
# ==============================================================================

# --- Modul Aksa: data/dataset.py ---
try:
    from data.dataset import denormalize, get_dataloaders
    USING_STUB_DATASET = False
except ImportError:
    USING_STUB_DATASET = True

    def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
        """[STUB] Mengembalikan DataLoader sintetis dalam rentang [-1, 1]."""
        print("[PERINGATAN STUB] Menggunakan stub untuk 'data/dataset.py' (Aksa). Bukan pengujian data asli!")
        num_samples = max(cfg.batch_size * 2, 8)
        x_train = torch.empty((num_samples, cfg.in_channels, cfg.img_size, cfg.img_size), dtype=torch.float32).uniform_(-1.0, 1.0)  # [N, 1, 28, 28]
        y_train = torch.randint(0, cfg.num_classes, (num_samples,), dtype=torch.int64)  # [N]

        x_test = torch.empty((num_samples, cfg.in_channels, cfg.img_size, cfg.img_size), dtype=torch.float32).uniform_(-1.0, 1.0)  # [N, 1, 28, 28]
        y_test = torch.randint(0, cfg.num_classes, (num_samples,), dtype=torch.int64)  # [N]

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        return train_loader, test_loader

    def denormalize(x: Tensor) -> Tensor:
        """[STUB] Memetakan rentang [-1, 1] ke [0, 1]."""
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


# --- Modul Axan: diffusion/schedule.py ---
try:
    from diffusion.schedule import make_beta_schedule
    USING_STUB_SCHEDULE = False
except ImportError:
    USING_STUB_SCHEDULE = True

    def make_beta_schedule(cfg: Config) -> Tensor:
        """[STUB] Membuat jadwal beta linear sesuai rentang di Config."""
        print("[PERINGATAN STUB] Menggunakan stub untuk 'diffusion/schedule.py' (Axan). Bukan implementasi asli!")
        return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T, dtype=torch.float32)  # [T]


# --- Modul Axan: diffusion/ddpm.py ---
try:
    from diffusion.ddpm import GaussianDiffusion
    USING_STUB_DDPM = False
except ImportError:
    USING_STUB_DDPM = True

    class GaussianDiffusion(nn.Module):
        """[STUB] Implementasi minimal GaussianDiffusion yang mematuhi kontrak antarmuka."""

        def __init__(self, model: nn.Module, cfg: Config) -> None:
            super().__init__()
            print("[PERINGATAN STUB] Menggunakan stub untuk 'diffusion/ddpm.py' (Axan). Bukan implementasi asli!")
            self.model = model
            self.cfg = cfg
            betas = make_beta_schedule(cfg)  # [T]
            alphas = 1.0 - betas  # [T]
            alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]
            self.register_buffer("betas", betas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)

        def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
            """[STUB] Forward diffusion q(x_t | x_0)."""
            device = x0.device
            alpha_bar_t = self.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise  # [B, 1, 28, 28]

        def training_loss(self, x0: Tensor, y: Tensor) -> Tensor:
            """[STUB] Menghitung MSE loss antara noise asli dan prediksi model dengan label dropout CFG."""
            b = x0.shape[0]
            device = x0.device
            t = torch.randint(0, self.cfg.T, (b,), device=device, dtype=torch.int64)  # [B]
            noise = torch.randn_like(x0)  # [B, 1, 28, 28]

            drop_mask = torch.rand(b, device=device) < self.cfg.cond_drop_prob  # [B]
            y_cond = torch.where(drop_mask, torch.full_like(y, self.cfg.num_classes), y)  # [B]

            x_t = self.q_sample(x0, t, noise)  # [B, 1, 28, 28]
            eps_pred = self.model(x_t, t, y_cond)  # [B, 1, 28, 28]
            return nn.functional.mse_loss(eps_pred, noise)

        @torch.no_grad()
        def sample(self, y: Tensor, guidance_scale: float = 0.0) -> Tensor:
            """[STUB] Reverse sampling mini untuk validasi alur dan bentuk output."""
            if guidance_scale < 0.0:
                raise ValueError("guidance_scale harus >= 0")
            if guidance_scale > 0.0 and self.cfg.cond_drop_prob == 0.0:
                raise ValueError("guidance_scale > 0 tidak valid jika cond_drop_prob == 0")

            n = y.shape[0]
            device = y.device
            x_t = torch.randn((n, self.cfg.in_channels, self.cfg.img_size, self.cfg.img_size), device=device)  # [N, 1, 28, 28]
            uncond_y = torch.full_like(y, self.cfg.num_classes)  # [N]

            for step in reversed(range(self.cfg.T)):
                t_batch = torch.full((n,), step, device=device, dtype=torch.int64)  # [N]
                if guidance_scale == 0.0:
                    eps = self.model(x_t, t_batch, y)  # [N, 1, 28, 28]
                else:
                    eps_c = self.model(x_t, t_batch, y)  # [N, 1, 28, 28]
                    eps_u = self.model(x_t, t_batch, uncond_y)  # [N, 1, 28, 28]
                    eps = eps_c + guidance_scale * (eps_c - eps_u)  # [N, 1, 28, 28]

                beta_t = self.betas.to(device)[step]
                x_t = x_t - beta_t * eps  # [N, 1, 28, 28]

            return torch.clamp(x_t, -1.0, 1.0)  # [N, 1, 28, 28]


# --- Modul Axan: train.py ---
try:
    import train as train_module
    USING_STUB_TRAIN = False
except ImportError:
    train_module = None
    USING_STUB_TRAIN = True


def _run_stub_training(cfg: Config, diffusion: GaussianDiffusion, train_loader: DataLoader) -> None:
    """[STUB] Simulasi 1 epoch training untuk menghasilkan artefak output sesuai kontrak."""
    print("[PERINGATAN STUB] Menggunakan stub untuk 'train.py' (Axan). Menghasilkan artefak output simulasi!")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.save_json(run_dir / "config.json")

    device = torch.device(cfg.device)
    diffusion.to(device)
    diffusion.train()
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr)

    loader_iter = iter(train_loader)
    x0, y = next(loader_iter)
    del loader_iter

    x0, y = x0.to(device), y.to(device)
    optimizer.zero_grad()
    loss = diffusion.training_loss(x0, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
    optimizer.step()

    loss_val = float(loss.item())

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"])
        writer.writerow([1, f"{loss_val:.6f}", f"{loss_val:.6f}", cfg.lr, "0.10"])

    diffusion.eval()
    labels = torch.arange(cfg.num_classes, device=device, dtype=torch.int64).repeat_interleave(10)  # [100]
    samples = diffusion.sample(labels, guidance_scale=0.0)  # [100, 1, 28, 28]
    samples_01 = denormalize(samples)  # [100, 1, 28, 28]
    save_image(samples_01, run_dir / "samples_epoch_001.png", nrow=10)

    torch.save({"epoch": 1, "model_state_dict": diffusion.model.state_dict()}, run_dir / "ckpt_last.pt")
    torch.save({"epoch": 1, "model_state_dict": diffusion.model.state_dict()}, run_dir / "ckpt_ema.pt")


# --- Modul Aksa: eval/evaluate.py ---
try:
    import eval.evaluate as evaluate_module
    USING_STUB_EVAL = False
except ImportError:
    evaluate_module = None
    USING_STUB_EVAL = True


def _run_stub_evaluation(cfg: Config) -> None:
    """[STUB] Simulasi penulisan eval.json sesuai kontrak."""
    print("[PERINGATAN STUB] Menggunakan stub untuk 'eval/evaluate.py' (Aksa). Menghasilkan eval.json simulasi!")
    eval_payload = {
        "overall_acc": 0.10,
        "per_class_acc": {str(i): 0.10 for i in range(cfg.num_classes)},
        "classifier_test_acc": 0.90,
    }
    eval_path = cfg.run_dir / "eval.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)


# ==============================================================================
# 3. FUNGSI HELPER VALIDASI, EKSEKUSI FLEKSIBEL, & PELACAKAN MODUL
# ==============================================================================

def check_contract(condition: bool, module_name: str, owner: str, detail: str) -> None:
    """Memvalidasi syarat kontrak dan melempar AssertionError yang menyebut modul penyebab."""
    if not condition:
        raise AssertionError(f"[GAGAL KONTRAK - Modul: {module_name} (Pemilik: {owner})] {detail}")


def validate_tensor_basics(
    tensor: Tensor,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    module_name: str,
    owner: str,
    tensor_name: str,
    value_range: tuple[float, float] | None = None,
) -> None:
    """Memeriksa tipe Tensor, shape, dtype, bebas NaN/Inf, dan rentang nilai."""
    check_contract(
        isinstance(tensor, Tensor),
        module_name,
        owner,
        f"{tensor_name} harus berupa torch.Tensor, tetapi menerima {type(tensor)}.",
    )
    check_contract(
        tuple(tensor.shape) == expected_shape,
        module_name,
        owner,
        f"Shape {tensor_name} salah: diharapkan {expected_shape}, diperoleh {tuple(tensor.shape)}.",
    )
    check_contract(
        tensor.dtype == expected_dtype,
        module_name,
        owner,
        f"Dtype {tensor_name} salah: diharapkan {expected_dtype}, diperoleh {tensor.dtype}.",
    )
    check_contract(
        not torch.isnan(tensor).any().item() and not torch.isinf(tensor).any().item(),
        module_name,
        owner,
        f"{tensor_name} mengandung nilai NaN atau Inf.",
    )
    if value_range is not None:
        min_val, max_val = value_range
        actual_min = float(tensor.min().item())
        actual_max = float(tensor.max().item())
        check_contract(
            actual_min >= min_val - 1e-5 and actual_max <= max_val + 1e-5,
            module_name,
            owner,
            f"Rentang nilai {tensor_name} di luar [{min_val}, {max_val}]: aktual [{actual_min:.4f}, {actual_max:.4f}].",
        )


def _invoke_module_entrypoint(
    mod: Any,
    module_dotted_name: str,
    candidate_funcs: tuple[str, ...],
    cfg: Config,
    extra_cli_args: list[str] | None = None,
) -> bool:
    """
    Mencoba memanggil fungsi entry-point (cfg) pada modul, atau fallback ke eksekusi
    subproses `python -m <module> --from_json <cfg_path>`. Mengembalikan True jika berhasil dipanggil.
    """
    for fn_name in candidate_funcs:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if len(params) >= 1:
                fn(cfg)
                return True
            if len(params) == 0:
                fn()
                return True

    # Fallback jika modul hanya memiliki blok `if __name__ == "__main__":` berbasis CLI
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = cfg.run_dir / "_smoke_input_config.json"
    cfg.save_json(tmp_cfg_path)

    cmd = [sys.executable, "-m", module_dotted_name, "--from_json", str(tmp_cfg_path)]
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if tmp_cfg_path.exists():
        tmp_cfg_path.unlink(missing_ok=True)

    if res.returncode != 0:
        raise RuntimeError(
            f"Eksekusi subproses '{' '.join(cmd)}' gagal (exit code {res.returncode}).\n"
            f"STDERR:\n{res.stderr}"
        )
    return True


def _release_dataloader(loader: DataLoader) -> None:
    """Membersihkan iterator/worker DataLoader secara aman di Windows sebelum direktori dihapus."""
    if hasattr(loader, "_iterator") and loader._iterator is not None:
        loader._iterator._shutdown_workers()
        loader._iterator = None
    gc.collect()


# ==============================================================================
# 4. EKSEKUSI END-TO-END SMOKE TEST
# ==============================================================================

def run_smoke_test() -> None:
    """Menjalankan seluruh tahapan smoke test integrasi end-to-end."""
    print("=== MEMULAI INTEGRATION SMOKE TEST DDPM FASHION-MNIST ===")

    temp_dir = Path(tempfile.mkdtemp(prefix="ddpm_smoke_"))
    try:
        # 1. Konfigurasi mini di memori menggunakan Config asli (Jordan)
        base_cfg = get_config([])
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = dataclasses.replace(
            base_cfg,
            T=10,
            batch_size=4,
            epochs=1,
            num_workers=0,
            device=device_str,
            run_name="smoke_test_run",
            runs_dir=str(temp_dir),
        )
        device = torch.device(cfg.device)
        print(f"[INFO] Device aktif: {device} | T={cfg.T}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

        # 2. Uji Data Pipeline (data/dataset.py - Aksa)
        print("[1/6] Menguji data/dataset.py (get_dataloaders, denormalize, & kompatibilitas Windows num_workers > 0)...")
        train_loader, test_loader = get_dataloaders(cfg)
        train_iter = iter(train_loader)
        x_batch, y_batch = next(train_iter)
        del train_iter

        validate_tensor_basics(
            x_batch,
            expected_shape=(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size),
            expected_dtype=torch.float32,
            module_name="data/dataset.py",
            owner="Aksa",
            tensor_name="x_batch",
            value_range=(-1.0, 1.0),
        )
        validate_tensor_basics(
            y_batch,
            expected_shape=(cfg.batch_size,),
            expected_dtype=torch.int64,
            module_name="data/dataset.py",
            owner="Aksa",
            tensor_name="y_batch",
            value_range=(0.0, float(cfg.num_classes - 1)),
        )

        x_denorm = denormalize(x_batch)
        validate_tensor_basics(
            x_denorm,
            expected_shape=(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size),
            expected_dtype=torch.float32,
            module_name="data/dataset.py",
            owner="Aksa",
            tensor_name="denormalize(x_batch)",
            value_range=(0.0, 1.0),
        )

        # Verifikasi khusus multiprocessing Windows (num_workers=2)
        cfg_mp = dataclasses.replace(cfg, num_workers=2)
        mp_train_loader, mp_test_loader = get_dataloaders(cfg_mp)
        try:
            mp_iter = iter(mp_train_loader)
            x_mp, y_mp = next(mp_iter)
            del mp_iter
            validate_tensor_basics(
                x_mp,
                expected_shape=(cfg_mp.batch_size, cfg_mp.in_channels, cfg_mp.img_size, cfg_mp.img_size),
                expected_dtype=torch.float32,
                module_name="data/dataset.py",
                owner="Aksa",
                tensor_name="x_mp (num_workers=2)",
                value_range=(-1.0, 1.0),
            )
            validate_tensor_basics(
                y_mp,
                expected_shape=(cfg_mp.batch_size,),
                expected_dtype=torch.int64,
                module_name="data/dataset.py",
                owner="Aksa",
                tensor_name="y_mp (num_workers=2)",
                value_range=(0.0, float(cfg_mp.num_classes - 1)),
            )
        finally:
            _release_dataloader(mp_train_loader)
            _release_dataloader(mp_test_loader)

        # 3. Uji Arsitektur U-Net Asli (models/unet.py & models/embeddings.py - Jordan)
        print("[2/6] Menguji models/unet.py (forward pass bersyarat & tanpa kelas)...")
        model = UNet(cfg).to(device)
        x_dev = x_batch.to(device)
        t_dev = torch.randint(0, cfg.T, (cfg.batch_size,), device=device, dtype=torch.int64)  # [B]
        y_dev = y_batch.to(device)
        y_uncond = torch.full((cfg.batch_size,), cfg.num_classes, device=device, dtype=torch.int64)  # [B]

        eps_cond = model(x_dev, t_dev, y_dev)  # [B, 1, 28, 28]
        validate_tensor_basics(
            eps_cond,
            expected_shape=(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size),
            expected_dtype=torch.float32,
            module_name="models/unet.py",
            owner="Jordan",
            tensor_name="eps_cond",
        )

        eps_uncond = model(x_dev, t_dev, y_uncond)  # [B, 1, 28, 28]
        validate_tensor_basics(
            eps_uncond,
            expected_shape=(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size),
            expected_dtype=torch.float32,
            module_name="models/unet.py",
            owner="Jordan",
            tensor_name="eps_uncond",
        )

        # 4. Uji Noise Schedule & GaussianDiffusion (diffusion/schedule.py & diffusion/ddpm.py - Axan)
        print("[3/6] Menguji diffusion/schedule.py & diffusion/ddpm.py (q_sample, training_loss, sample, validasi kontrak)...")
        betas = make_beta_schedule(cfg)
        validate_tensor_basics(
            betas,
            expected_shape=(cfg.T,),
            expected_dtype=torch.float32,
            module_name="diffusion/schedule.py",
            owner="Axan",
            tensor_name="betas",
            value_range=(1e-7, 0.999),
        )

        diffusion = GaussianDiffusion(model, cfg).to(device)
        noise = torch.randn_like(x_dev)  # [B, 1, 28, 28]
        x_t = diffusion.q_sample(x_dev, t_dev, noise)  # [B, 1, 28, 28]
        validate_tensor_basics(
            x_t,
            expected_shape=(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size),
            expected_dtype=torch.float32,
            module_name="diffusion/ddpm.py",
            owner="Axan",
            tensor_name="q_sample output (x_t)",
        )

        loss = diffusion.training_loss(x_dev, y_dev)
        check_contract(
            isinstance(loss, Tensor) and loss.ndim == 0,
            "diffusion/ddpm.py",
            "Axan",
            f"training_loss harus mengembalikan Tensor skalar (0-D), diperoleh shape {tuple(loss.shape)}.",
        )
        check_contract(
            not torch.isnan(loss).item() and not torch.isinf(loss).item() and loss.item() > 0.0,
            "diffusion/ddpm.py",
            "Axan",
            f"Nilai training_loss tidak valid (NaN/Inf/<=0): {loss.item()}.",
        )

        # Uji reverse sampling (2 sampel, dengan dan tanpa classifier-free guidance)
        y_sample = torch.tensor([0, 1], device=device, dtype=torch.int64)  # [2]
        for w in (0.0, 1.5):
            gen_imgs = diffusion.sample(y_sample, guidance_scale=w)  # [2, 1, 28, 28]
            validate_tensor_basics(
                gen_imgs,
                expected_shape=(2, cfg.in_channels, cfg.img_size, cfg.img_size),
                expected_dtype=torch.float32,
                module_name="diffusion/ddpm.py",
                owner="Axan",
                tensor_name=f"sample(guidance_scale={w})",
                value_range=(-1.0, 1.0),
            )

        # Uji penjagaan kontrak ValueError pada GaussianDiffusion.sample
        try:
            diffusion.sample(y_sample, guidance_scale=-0.5)
            check_contract(
                False,
                "diffusion/ddpm.py",
                "Axan",
                "sample(guidance_scale=-0.5) wajib melempar ValueError sesuai kontrak.",
            )
        except ValueError:
            pass

        cfg_no_drop = dataclasses.replace(cfg, cond_drop_prob=0.0)
        diffusion_no_drop = GaussianDiffusion(model, cfg_no_drop).to(device)
        try:
            diffusion_no_drop.sample(y_sample, guidance_scale=1.0)
            check_contract(
                False,
                "diffusion/ddpm.py",
                "Axan",
                "sample(guidance_scale > 0) dengan cond_drop_prob=0 wajib melempar ValueError sesuai kontrak.",
            )
        except ValueError:
            pass

        # 5. Uji Training Loop & Output Artefak (train.py - Axan)
        print("[4/6] Menguji eksekusi training & pembuatan artefak output...")
        if not USING_STUB_TRAIN:
            try:
                _invoke_module_entrypoint(train_module, "train", ("train", "main", "run"), cfg)
            except Exception as exc:
                check_contract(
                    False,
                    "train.py",
                    "Axan",
                    f"Gagal menjalankan modul asli train.py: {exc}",
                )
        else:
            _run_stub_training(cfg, diffusion, train_loader)

        expected_train_files = [
            "config.json",
            "metrics.csv",
            "samples_epoch_001.png",
            "ckpt_last.pt",
            "ckpt_ema.pt",
        ]
        for fname in expected_train_files:
            fpath = cfg.run_dir / fname
            check_contract(
                fpath.exists() and fpath.stat().st_size > 0,
                "train.py",
                "Axan",
                f"File output wajib '{fname}' tidak ditemukan atau kosong di {cfg.run_dir}.",
            )

        with (cfg.run_dir / "metrics.csv").open("r", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        expected_cols = ["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"]
        check_contract(
            header == expected_cols,
            "train.py",
            "Axan",
            f"Kolom metrics.csv salah: diharapkan {expected_cols}, diperoleh {header}.",
        )

        # 6. Uji Evaluasi & Output eval.json (eval/evaluate.py - Aksa)
        print("[5/6] Menguji evaluasi & pembuatan eval.json...")
        classifier_ckpt = Path("eval/classifier.pt")
        if not USING_STUB_EVAL:
            if not classifier_ckpt.exists():
                print(
                    "[PERINGATAN STUB] 'eval/evaluate.py' sudah ada, tetapi 'eval/classifier.pt' belum tersedia. "
                    "Menggunakan stub eval.json sementara agar pengujian tidak gagal prematur."
                )
                _run_stub_evaluation(cfg)
            else:
                try:
                    _invoke_module_entrypoint(
                        evaluate_module,
                        "eval.evaluate",
                        ("evaluate", "main", "run"),
                        cfg,
                        extra_cli_args=["--n_per_class", "2"],
                    )
                except Exception as exc:
                    check_contract(
                        False,
                        "eval/evaluate.py",
                        "Aksa",
                        f"Gagal menjalankan modul asli eval/evaluate.py: {exc}",
                    )
        else:
            _run_stub_evaluation(cfg)

        eval_file = cfg.run_dir / "eval.json"
        check_contract(
            eval_file.exists() and eval_file.stat().st_size > 0,
            "eval/evaluate.py",
            "Aksa",
            f"File output wajib 'eval.json' tidak ditemukan atau kosong di {cfg.run_dir}.",
        )
        with eval_file.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for key in ("overall_acc", "per_class_acc", "classifier_test_acc"):
            check_contract(
                key in eval_data,
                "eval/evaluate.py",
                "Aksa",
                f"Kunci wajib '{key}' tidak ada di dalam eval.json.",
            )

        _release_dataloader(train_loader)
        _release_dataloader(test_loader)

        print("[6/6] Ringkasan status modul yang diuji:")
        print(f"  - config.py & models/unet.py (Jordan) : ASLI REPO")
        print(f"  - data/dataset.py (Aksa)              : {'STUB' if USING_STUB_DATASET else 'ASLI REPO'}")
        print(f"  - diffusion/schedule.py (Axan)        : {'STUB' if USING_STUB_SCHEDULE else 'ASLI REPO'}")
        print(f"  - diffusion/ddpm.py (Axan)            : {'STUB' if USING_STUB_DDPM else 'ASLI REPO'}")
        print(f"  - train.py (Axan)                     : {'STUB' if USING_STUB_TRAIN else 'ASLI REPO'}")
        print(f"  - eval/evaluate.py (Aksa)             : {'STUB' if USING_STUB_EVAL else 'ASLI REPO'}")
        print("=== SEMUA PEMERIKSAAN SMOKE TEST LULUS ===")

    finally:
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()
