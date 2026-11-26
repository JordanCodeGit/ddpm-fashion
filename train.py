"""Training pipeline Conditional DDPM untuk Fashion-MNIST.

Training normal: python -m train [argumen Config]
Self-test ringan: python -m train --self_test --device cuda --num_workers 0
"""

import argparse
import copy
import csv
import math
import random
from collections.abc import Callable, Iterator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from struct import unpack
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, Subset, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config, get_config
from diffusion.ddpm import GaussianDiffusion
from models.unet import UNet


# Kebijakan monitoring yang disepakati (Config tidak diubah).
SAMPLE_EVERY = 5
IMAGES_PER_CLASS = 10  # Format grid 10 x 10 pada Fashion-MNIST.
METRIC_FIELDS = ("epoch", "train_loss", "val_loss", "lr", "epoch_time_s")


def set_seed(seed: int) -> None:
    """Mengatur seed training dan backend cuDNN."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(cfg: Config) -> torch.device:
    """Memvalidasi device tanpa fallback CUDA ke CPU."""
    try:
        device = torch.device(cfg.device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Device tidak valid: {cfg.device!r}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config meminta CUDA, tetapi CUDA tidak tersedia. "
                "Periksa instalasi PyTorch/driver atau pilih --device cpu secara eksplisit."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index tidak tersedia: {index}")
        return torch.device("cuda", index)
    if device.type == "cpu":
        return torch.device("cpu")
    raise ValueError("Training hanya mendukung CPU dan CUDA.")


def synchronize(device: torch.device) -> None:
    """Menunggu operasi CUDA selesai sebelum mengukur waktu."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def isolated_rng(device: torch.device, seed: int | None = None) -> Iterator[None]:
    """Memulihkan RNG Python, NumPy, CPU dan GPU aktif setelah blok selesai."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                # Seed hanya CPU dan GPU aktif, bukan semua GPU.
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class EMA:
    """Menyimpan salinan U-Net dengan exponential moving average."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        """Menginisialisasi EMA dengan bobot model awal."""
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Memperbarui EMA setelah satu optimizer step."""
        source_params = dict(model.named_parameters())
        for name, averaged in self.ema_model.named_parameters():
            averaged.mul_(self.decay).add_(
                source_params[name].detach(), alpha=1.0 - self.decay
            )  # shape parameter tidak berubah

        # Buffer bukan parameter terlatih; salin nilai aktualnya.
        source_buffers = dict(model.named_buffers())
        for name, averaged in self.ema_model.named_buffers():
            averaged.copy_(source_buffers[name])


@dataclass(frozen=True)
class ValidationBank:
    """Random inputs tetap pada CPU, satu entri per sampel test set."""

    timesteps: Tensor  # [N_test], int64
    noise: Tensor  # [N_test, C, H, W], float32
    drop_mask: Tensor  # [N_test], bool


def prepare_validation_bank(cfg: Config, test_loader: DataLoader) -> ValidationBank:
    """Membuat validation bank independen dari ukuran batch dan RNG training."""
    if not isinstance(test_loader.sampler, SequentialSampler) or test_loader.drop_last:
        raise ValueError(
            "Validation memerlukan test_loader berurutan (shuffle=False) "
            "dan drop_last=False agar noise tetap terikat pada sampel yang sama."
        )
    count = len(test_loader.dataset)
    if count == 0:
        raise ValueError("Test dataset kosong.")

    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    timesteps = torch.randint(0, cfg.T, (count,), generator=generator, dtype=torch.int64, device="cpu")  # [N_test]
    noise = torch.randn(
        (count, cfg.in_channels, cfg.img_size, cfg.img_size),
        generator=generator, dtype=torch.float32, device="cpu",
    )  # [N_test, C, H, W]
    drop_mask = torch.rand((count,), generator=generator, dtype=torch.float32, device="cpu") < cfg.cond_drop_prob  # [N_test]
    return ValidationBank(timesteps, noise, drop_mask)


def move_batch(batch: tuple[Tensor, Tensor] | list[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """Memindahkan batch sesuai kontrak dataset ke device model."""
    x0, y = batch
    if x0.dtype != torch.float32 or y.dtype != torch.int64:
        raise ValueError("Dataset harus menghasilkan gambar float32 dan label int64.")
    x0 = x0.to(device, non_blocking=device.type == "cuda")  # [B, C, H, W]
    y = y.to(device, non_blocking=device.type == "cuda")  # [B]
    return x0, y


def clip_gradients(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    """Membatasi global L2 gradient norm; tolak gradient non-finite."""
    return nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)  # []


def train_one_epoch(
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: AdamW,
    ema: EMA,
    cfg: Config,
    device: torch.device,
    epoch: int,
) -> float:
    """Menjalankan training satu epoch dan mengembalikan MSE rata-rata."""
    diffusion.train()
    total_loss, total_count = 0.0, 0

    with tqdm(loader, desc=f"Train {epoch}/{cfg.epochs}", unit="batch") as progress:
        for batch in progress:
            x0, y = move_batch(batch, device)  # [B, C, H, W], [B]
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(x0, y)  # []
            value = loss.detach().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Train loss non-finite pada epoch {epoch}: {value}")

            loss.backward()
            clip_gradients(diffusion.model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(diffusion.model)

            count = x0.shape[0]
            total_loss += value * count
            total_count += count
            progress.set_postfix(loss=f"{total_loss / total_count:.6f}")

    if total_count == 0:
        raise ValueError("Train DataLoader tidak menghasilkan batch.")
    # Lepaskan gradient sebelum validasi dan sampling untuk menghemat VRAM.
    optimizer.zero_grad(set_to_none=True)
    return total_loss / total_count


@torch.no_grad()
def validate_one_epoch(
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    bank: ValidationBank,
    device: torch.device,
) -> float:
    """Menghitung MSE dengan timestep, noise dan mask tetap per sampel."""
    was_training = diffusion.training
    # DataLoader dapat memiliki generator eksplisit selain global RNG.
    loader_generator = loader.generator
    loader_state = loader_generator.get_state() if loader_generator is not None else None
    total_loss, offset = 0.0, 0
    diffusion.eval()
    try:
        with isolated_rng(device, diffusion.cfg.seed):
            with tqdm(loader, desc="Validation", unit="batch", leave=False) as progress:
                for batch in progress:
                    x0, y = move_batch(batch, device)  # [B, C, H, W], [B]
                    count = x0.shape[0]
                    end = offset + count
                    if end > bank.timesteps.numel():
                        raise ValueError("Jumlah sampel validation melebihi ukuran bank.")

                    t = bank.timesteps[offset:end].to(device)  # [B]
                    noise = bank.noise[offset:end].to(device)  # [B, C, H, W]
                    mask = bank.drop_mask[offset:end].to(device)  # [B], bool
                    y_cond = torch.where(mask, torch.full_like(y, diffusion.cfg.num_classes), y)  # [B]
                    x_t = diffusion.q_sample(x0, t, noise)  # [B, C, H, W]
                    predicted_noise = diffusion.model(x_t, t, y_cond)  # [B, C, H, W]
                    if predicted_noise.shape != noise.shape:
                        raise ValueError("Shape prediksi noise validation tidak sesuai.")
                    loss = F.mse_loss(predicted_noise, noise, reduction="mean")  # []
                    value = loss.item()
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Validation loss non-finite: {value}")
                    total_loss += value * count
                    offset = end
        if offset != bank.timesteps.numel() or offset == 0:
            raise ValueError("Test DataLoader tidak mencakup seluruh validation bank.")
        return total_loss / offset
    finally:
        if loader_generator is not None and loader_state is not None:
            loader_generator.set_state(loader_state)
        diffusion.train(was_training)


def prepare_run_dir(cfg: Config) -> None:
    """Membuat direktori run baru tanpa menimpa eksperimen lama."""
    directory = cfg.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Run directory sudah berisi file: {directory}. Pilih --run_name baru.")
    cfg.save_json(directory / "config.json")
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()


def append_metrics(cfg: Config, epoch: int, train_loss: float, val_loss: float, lr: float, elapsed: float) -> None:
    """Menambahkan satu baris dengan lima kolom sesuai kontrak."""
    row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr, epoch_time_s=elapsed)
    with (cfg.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)


def _save_state_dict(model: nn.Module, path: Path) -> None:
    """Menyimpan raw U-Net state_dict melalui file sementara."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(model.state_dict(), temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_checkpoints(cfg: Config, model: nn.Module, ema: EMA) -> None:
    """Menyimpan bobot U-Net biasa dan EMA untuk inference."""
    _save_state_dict(model, cfg.run_dir / "ckpt_last.pt")
    _save_state_dict(ema.ema_model, cfg.run_dir / "ckpt_ema.pt")


@torch.no_grad()
def save_sample_grid(
    cfg: Config,
    ema_diffusion: GaussianDiffusion,
    denormalize: Callable[[Tensor], Tensor],
    device: torch.device,
    epoch: int,
) -> Path:
    """Menghasilkan 10 gambar per kelas dengan urutan baris tetap."""
    labels = torch.arange(cfg.num_classes, device=device, dtype=torch.int64).repeat_interleave(IMAGES_PER_CLASS)  # [K*10]
    chunks: list[Tensor] = []
    with isolated_rng(device, cfg.seed):
        # Batch sampling tidak melebihi cfg.batch_size.
        for label_batch in labels.split(cfg.batch_size):  # [N_chunk]
            images = ema_diffusion.sample(label_batch)  # [N_chunk, C, H, W]; w=0 (default)
            chunks.append(images.cpu())
        images = torch.cat(chunks, dim=0)  # [K*10, C, H, W]
        images = denormalize(images).clamp(0.0, 1.0)  # [K*10, C, H, W]
        expected_shape = (cfg.num_classes * IMAGES_PER_CLASS, cfg.in_channels, cfg.img_size, cfg.img_size)
        if tuple(images.shape) != expected_shape or not torch.isfinite(images).all():
            raise ValueError("Grid sampel memiliki shape atau nilai yang tidak valid.")
        path = cfg.run_dir / f"samples_epoch_{epoch:03d}.png"
        save_image(images, path, nrow=IMAGES_PER_CLASS)
    return path


def build_training(cfg: Config, device: torch.device) -> tuple[GaussianDiffusion, EMA, AdamW]:
    """Membuat U-Net asli, GaussianDiffusion, EMA, dan AdamW."""
    model = UNet(cfg).to(device)
    diffusion = GaussianDiffusion(model, cfg).to(device)
    ema = EMA(model, cfg.ema_decay)
    # Default AdamW lain dipertahankan tetap untuk semua run (opsi A).
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    return diffusion, ema, optimizer


def run_training(
    cfg: Config,
    train_loader: DataLoader,
    test_loader: DataLoader,
    denormalize: Callable[[Tensor], Tensor],
    diffusion: GaussianDiffusion,
    ema: EMA,
    optimizer: AdamW,
    device: torch.device,
    audit_rng: bool = False,
) -> None:
    """Mengatur epoch, validation, logging, checkpoint, dan grid EMA."""
    bank = prepare_validation_bank(cfg, test_loader)
    prepare_run_dir(cfg)
    ema_diffusion = GaussianDiffusion(ema.ema_model, cfg).to(device)
    print(f"Run: {cfg.run_dir} | device={device} | T={cfg.T} | batch={cfg.batch_size}")

    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = perf_counter()
        train_loss = train_one_epoch(diffusion, train_loader, optimizer, ema, cfg, device, epoch)
        val_loss = validate_one_epoch(diffusion, test_loader, bank, device)
        synchronize(device)
        elapsed = perf_counter() - started
        lr = float(optimizer.param_groups[0]["lr"])
        append_metrics(cfg, epoch, train_loss, val_loss, lr, elapsed)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.6g} | epoch_time_s={elapsed:.3f}")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"  Peak allocated VRAM (training + validation): {peak:.2f} MiB")

        save_checkpoints(cfg, diffusion.model, ema)

        if epoch % SAMPLE_EVERY == 0 or epoch == cfg.epochs:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            sample_start = perf_counter()
            rng_before = _rng_snapshot(device) if audit_rng else None
            path = save_sample_grid(cfg, ema_diffusion, denormalize, device, epoch)
            if rng_before is not None:
                _assert_rng_same(rng_before, _rng_snapshot(device))
                print("[PASS] Sample grid preserves training RNG")
            synchronize(device)
            print(f"  Saved {path.name} | sampling + save time={perf_counter() - sample_start:.3f} s")
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                print(f"  Peak allocated VRAM (sampling): {peak:.2f} MiB")

    print(f"TRAINING COMPLETED | {cfg.run_dir}")


# ============================================================
# SELF-TEST: U-Net dan GaussianDiffusion asli, T dan data kecil.
# Stub dataset hanya jika modul dataset milik Aksa belum tersedia.
# ============================================================


def _stub_denormalize(x: Tensor) -> Tensor:
    """STUB SELF-TEST ONLY: mengubah [-1,1] menjadi [0,1]."""
    return (x + 1.0) / 2.0  # [B, C, H, W]


def _self_test_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, Callable[[Tensor], Tensor]]:
    """Memakai subset dataset asli atau stub sintetis khusus self-test."""
    train_count, test_count = 8, 4  # ukuran fixture pengujian, bukan hyperparameter training
    try:
        from data.dataset import get_dataloaders, denormalize
    except ModuleNotFoundError as exc:
        if exc.name not in ("data", "data.dataset"):
            raise
        print("[STUB DATASET - SELF-TEST ONLY] data/dataset.py belum tersedia.")
        generator = torch.Generator().manual_seed(cfg.seed)
        def synthetic_dataset(count: int) -> TensorDataset:
            """Membuat gambar dan label sintetis sesuai kontrak."""
            x = torch.rand((count, cfg.in_channels, cfg.img_size, cfg.img_size), generator=generator, dtype=torch.float32) * 2.0 - 1.0  # [N, C, H, W]
            y = torch.arange(count, dtype=torch.int64) % cfg.num_classes  # [N]
            return TensorDataset(x, y)
        train_data = synthetic_dataset(train_count)
        test_data = synthetic_dataset(test_count)
        denormalize = _stub_denormalize
    else:
        print("[REAL DATASET] Menggunakan subset dataset asli; download dapat terjadi jika cache belum ada.")
        original_train, original_test = get_dataloaders(cfg)
        if not isinstance(original_test.sampler, SequentialSampler) or original_test.drop_last:
            raise ValueError("Test loader asli harus shuffle=False dan drop_last=False.")
        if len(original_train.dataset) < train_count or len(original_test.dataset) < test_count:
            raise ValueError("Dataset asli terlalu kecil untuk fixture self-test.")
        train_data = Subset(original_train.dataset, range(train_count))
        test_data = Subset(original_test.dataset, range(test_count))

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed))
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed))
    return train_loader, test_loader, denormalize


def _rng_snapshot(device: torch.device) -> tuple[Any, ...]:
    """Mengambil snapshot RNG proses utama untuk pengujian isolasi."""
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_state


def _assert_rng_same(before: tuple[Any, ...], after: tuple[Any, ...]) -> None:
    """Memastikan RNG Python, NumPy, CPU, dan GPU aktif tidak berubah."""
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    if before[3] is not None:
        assert torch.equal(before[3], after[3])


def run_self_test(base_cfg: Config) -> None:
    """Menjalankan satu optimizer step dan pengujian seluruh output run."""
    cfg = replace(base_cfg, T=3, epochs=1, batch_size=8, run_name="smoke_test" if base_cfg.run_name == "baseline" else base_cfg.run_name)
    device = resolve_device(cfg)
    set_seed(cfg.seed)
    print(f"SELF-TEST | device={device} | T={cfg.T} | batch={cfg.batch_size} | workers={cfg.num_workers}")
    train_loader, test_loader, denormalize = _self_test_loaders(cfg)
    diffusion, ema, optimizer = build_training(cfg, device)
    initial = {name: param.detach().clone() for name, param in diffusion.model.named_parameters()}
    calls = 0

    def check_optimizer_step(opt: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Memeriksa gradient aktual sesudah clipping, sebelum AdamW step."""
        nonlocal calls
        grads = [p.grad.detach() for p in diffusion.model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads)
        assert any(torch.count_nonzero(g).item() > 0 for g in grads)
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()  # []
        assert norm <= cfg.grad_clip * (1.0 + 1e-5) + 1e-7
        calls += 1

    # Uji clipping aktif secara terkontrol, tidak bergantung gradient acak U-Net.
    probe = nn.Parameter(torch.zeros(2, device=device))  # [2]
    probe.grad = torch.tensor([3.0, 4.0], device=device) * cfg.grad_clip  # [2]
    previous_norm = clip_gradients([probe], cfg.grad_clip)  # []
    assert previous_norm.item() > cfg.grad_clip
    assert probe.grad.norm().item() <= cfg.grad_clip * (1.0 + 1e-5)
    print("[PASS] Controlled gradient clipping")

    hook = optimizer.register_step_pre_hook(check_optimizer_step)
    try:
        run_training(cfg, train_loader, test_loader, denormalize, diffusion, ema, optimizer, device, audit_rng=True)
    finally:
        hook.remove()
    assert calls == 1, f"Expected one optimizer step, got {calls}"
    updated = dict(diffusion.model.named_parameters())
    assert any(not torch.equal(initial[name], p) for name, p in updated.items())
    print("[PASS] Training, backward, clipped finite gradients and parameter update")

    for name, averaged in ema.ema_model.named_parameters():
        expected = initial[name] * cfg.ema_decay + updated[name].detach() * (1.0 - cfg.ema_decay)
        torch.testing.assert_close(averaged, expected)
        assert not averaged.requires_grad
    print("[PASS] EMA formula after one optimizer step")

    before = _rng_snapshot(device)
    bank = prepare_validation_bank(cfg, test_loader)
    second_bank = prepare_validation_bank(cfg, test_loader)
    assert torch.equal(bank.timesteps, second_bank.timesteps)
    assert torch.equal(bank.noise, second_bank.noise)
    assert torch.equal(bank.drop_mask, second_bank.drop_mask)
    model_before = {name: p.detach().clone() for name, p in diffusion.model.state_dict().items()}
    loader_before = test_loader.generator.get_state()
    mode_before = diffusion.training
    val_a = validate_one_epoch(diffusion, test_loader, bank, device)
    val_b = validate_one_epoch(diffusion, test_loader, bank, device)
    assert math.isfinite(val_a) and math.isclose(val_a, val_b, rel_tol=1e-6, abs_tol=1e-7)
    assert diffusion.training == mode_before
    assert torch.equal(loader_before, test_loader.generator.get_state())
    _assert_rng_same(before, _rng_snapshot(device))
    for name, value in diffusion.model.state_dict().items():
        torch.testing.assert_close(value, model_before[name], rtol=0, atol=0)
    print(f"[PASS] Repeatable validation, unchanged model and RNG | val_a={val_a:.8f} | val_b={val_b:.8f}")

    # Isolasi RNG sampling telah diperiksa saat run_training(audit_rng=True).
    path = cfg.run_dir / "samples_epoch_001.png"

    expected_files = ("config.json", "metrics.csv", "ckpt_last.pt", "ckpt_ema.pt", "samples_epoch_001.png")
    for filename in expected_files:
        file = cfg.run_dir / filename
        assert file.is_file() and file.stat().st_size > 0, filename
    assert Config.load_json(cfg.run_dir / "config.json") == cfg
    with (cfg.run_dir / "metrics.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == METRIC_FIELDS
        rows = list(reader)
    assert len(rows) == 1 and int(rows[0]["epoch"]) == 1
    for key in METRIC_FIELDS[1:]:
        assert math.isfinite(float(rows[0][key])), key
    assert float(rows[0]["train_loss"]) >= 0 and float(rows[0]["val_loss"]) >= 0
    assert math.isclose(float(rows[0]["val_loss"]), val_a, rel_tol=1e-6, abs_tol=1e-7)
    with path.open("rb") as image_file:
        header = image_file.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
    width, height = unpack(">II", header[16:24])
    # torchvision.save_image menggunakan padding default 2 piksel.
    assert width == IMAGES_PER_CLASS * (cfg.img_size + 2) + 2
    assert height == cfg.num_classes * (cfg.img_size + 2) + 2
    print("[PASS] All five output files, grid dimensions, Config roundtrip and finite CSV metrics")

    restored = UNet(cfg).cpu()
    for filename, source in (("ckpt_last.pt", diffusion.model), ("ckpt_ema.pt", ema.ema_model)):
        weights = torch.load(cfg.run_dir / filename, map_location="cpu", weights_only=True)
        restored.load_state_dict(weights, strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name].detach().cpu(), rtol=0, atol=0)
        print(f"[PASS] Strict checkpoint reload: {filename}")

    try:
        prepare_run_dir(cfg)
    except FileExistsError:
        print("[PASS] Existing run protected from overwrite")
    else:
        raise AssertionError("Existing run directory must not be overwritten.")
    print(f"ALL TRAIN SELF-TESTS PASSED | DEVICE={device}")


def main() -> None:
    """Parse argumen khusus self-test, lalu teruskan sisanya ke Config asli."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--self_test", "--self-test", action="store_true")
    extra, rest = parser.parse_known_args()
    cfg = get_config(rest)
    if extra.self_test:
        run_self_test(cfg)
        return

    device = resolve_device(cfg)
    set_seed(cfg.seed)
    # Training normal tidak pernah menggunakan stub dataset.
    from data.dataset import get_dataloaders, denormalize

    train_loader, test_loader = get_dataloaders(cfg)
    diffusion, ema, optimizer = build_training(cfg, device)
    run_training(cfg, train_loader, test_loader, denormalize, diffusion, ema, optimizer, device)


if __name__ == "__main__":
    main()
