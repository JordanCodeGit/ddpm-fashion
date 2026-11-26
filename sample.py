"""Generate class-conditional Fashion-MNIST PNG grids from a DDPM checkpoint.

Normal sampling uses the original training Config and raw U-Net state_dict.
Self-test uses the original U-Net with a small-T checkpoint, never a stub U-Net.
"""

import argparse
import math
import random
import re
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config, get_config
from diffusion.ddpm import GaussianDiffusion
from models.unet import UNet

# Agreed grid format and operational inference batch limit.
NUM_CLASSES: int = 10
IMAGES_PER_CLASS: int = 10
MAX_SAMPLE_BATCH: int = 32
GRID_PADDING: int = 2
INFERENCE_OVERRIDES: frozenset[str] = frozenset({"device", "seed", "batch_size"})


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, Config, Path | None]:
    """Parse sampling arguments first, then forward Config arguments unchanged."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--ckpt", type=Path, help="Raw U-Net state_dict checkpoint")
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--self_test", action="store_true")
    extra, rest = parser.parse_known_args(argv)

    # Inspect --from_json without removing it from Config arguments.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--from_json", type=Path)
    source, _ = probe.parse_known_args(rest)
    config_path = source.from_json

    if not extra.self_test and (config_path is None or extra.ckpt is None):
        parser.error("Sampling normal wajib menggunakan --from_json dan --ckpt.")
    if extra.self_test and ((config_path is None) != (extra.ckpt is None)):
        parser.error("Self-test memakai kedua --from_json dan --ckpt, atau tanpa keduanya.")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")

    cfg = get_config(rest)
    if config_path is not None:
        original = Config.load_json(config_path)
        validate_sampling_config(original, cfg)
        if extra.ckpt.resolve().parent != config_path.resolve().parent:
            print("[WARNING] Checkpoint dan Config berbeda folder; pastikan berasal dari run yang sama.")
    validate_guidance(cfg, extra.guidance_scale)
    return extra, cfg, config_path


def validate_sampling_config(original: Config, cfg: Config) -> None:
    """Reject overrides except device, sampling seed and batch size."""
    changed = [
        field.name for field in fields(original)
        if field.name not in INFERENCE_OVERRIDES
        and getattr(original, field.name) != getattr(cfg, field.name)
    ]
    if changed:
        raise ValueError("Config training tidak boleh diubah saat sampling: " + ", ".join(changed))
    if cfg.num_classes != NUM_CLASSES:
        raise ValueError("Grid Fashion-MNIST membutuhkan tepat 10 kelas.")


def validate_guidance(cfg: Config, guidance_scale: float) -> None:
    """Enforce the project CFG convention and valid guidance values."""
    if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
        raise ValueError("guidance_scale harus finite dan >= 0.")
    if guidance_scale > 0.0 and cfg.cond_drop_prob == 0.0:
        raise ValueError("CFG membutuhkan cond_drop_prob > 0 pada Config training.")


def resolve_device(cfg: Config) -> torch.device:
    """Resolve CPU/CUDA explicitly, without silently falling back to CPU."""
    device = torch.device(cfg.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"Device tidak didukung: {cfg.device}. Gunakan cpu atau cuda.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diminta tetapi tidak tersedia. Gunakan --device cpu jika memang diinginkan.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"GPU CUDA index {index} tidak tersedia.")
        return torch.device("cuda", index)
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed inference RNGs; exact reproducibility across devices is not promised."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(model: UNet, path: Path) -> None:
    """Load raw U-Net weights on CPU with strict key and shape checking."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("Checkpoint bukan raw state_dict yang valid.")
        model.load_state_dict(weights, strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint tidak dapat dimuat: {path}. Pastikan raw U-Net state_dict "
            "dan Config berasal dari training yang kompatibel. "
            f"Detail: {exc}"
        ) from exc


def build_diffusion(cfg: Config, checkpoint: Path, device: torch.device) -> GaussianDiffusion:
    """Construct the real U-Net, load weights, and attach diffusion buffers."""
    model = UNet(cfg).cpu()
    load_checkpoint(model, checkpoint)
    model.requires_grad_(False)
    model.eval()
    diffusion = GaussianDiffusion(model, cfg).to(device)
    diffusion.eval()
    return diffusion


def make_class_labels(cfg: Config) -> Tensor:
    """Create contiguous blocks of ten labels in Fashion-MNIST class order."""
    if cfg.num_classes != NUM_CLASSES:
        raise ValueError("Grid Fashion-MNIST membutuhkan tepat 10 kelas.")
    labels = torch.arange(cfg.num_classes, dtype=torch.int64)  # [K]
    return labels.repeat_interleave(IMAGES_PER_CLASS)  # [100]


@torch.no_grad()
def generate_samples(diffusion: GaussianDiffusion, cfg: Config, guidance_scale: float) -> Tensor:
    """Run full DDPM in bounded batches and return a CPU image tensor."""
    validate_guidance(cfg, guidance_scale)
    labels = make_class_labels(cfg)  # [100], CPU
    batch_size = min(cfg.batch_size, MAX_SAMPLE_BATCH)
    if batch_size <= 0:
        raise ValueError("Sampling batch size harus positif.")
    diffusion.eval()
    # Seed once, not once per batch (which could repeat initial noise).
    set_seed(cfg.seed)
    device = diffusion.betas.device
    chunks: list[Tensor] = []
    try:
        for offset in tqdm(range(0, labels.numel(), batch_size), desc="Sampling", unit="batch"):
            batch_labels = labels[offset:offset + batch_size].to(device)  # [N]
            images = diffusion.sample(batch_labels, guidance_scale=guidance_scale)  # [N,C,H,W]
            expected = (batch_labels.numel(), cfg.in_channels, cfg.img_size, cfg.img_size)
            if tuple(images.shape) != expected or images.dtype != torch.float32:
                raise ValueError(f"Output sampling tidak sesuai kontrak: {images.shape}, {images.dtype}")
            if not torch.isfinite(images).all():
                raise FloatingPointError("Output sampling mengandung NaN atau Inf.")
            if not ((images >= -1.0) & (images <= 1.0)).all():
                raise ValueError("Output DDPM harus berada pada rentang [-1, 1].")
            chunks.append(images.cpu())  # [N,C,H,W], CPU
            del images
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory saat sampling. Coba --batch_size yang lebih kecil.") from exc
    return torch.cat(chunks, dim=0)  # [100,C,H,W], CPU


def resolve_output_path(run_dir: Path, checkpoint: Path, guidance_scale: float, seed: int) -> Path:
    """Create a descriptive, unused filename distinct from training grids."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = {"ckpt_ema": "ema", "ckpt_last": "last"}.get(checkpoint.stem, checkpoint.stem)
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    guidance_text = format(0.0 if guidance_scale == 0.0 else guidance_scale, ".17g")
    # repr gives a short round-trip float spelling for nonintegral values.
    if guidance_scale != int(guidance_scale):
        guidance_text = repr(guidance_scale)
    base = f"samples_final_{stem}_g{guidance_text}_seed{seed}"
    path = run_dir / f"{base}.png"
    counter = 1
    while path.exists():
        path = run_dir / f"{base}_{counter:03d}.png"
        counter += 1
    return path


@torch.no_grad()
def save_sample_grid(images: Tensor, cfg: Config, checkpoint: Path, guidance_scale: float,
                     denormalize: Callable[[Tensor], Tensor]) -> Path:
    """Denormalize and write a 10x10 PNG without overwriting an existing file."""
    expected = (NUM_CLASSES * IMAGES_PER_CLASS, cfg.in_channels, cfg.img_size, cfg.img_size)
    if tuple(images.shape) != expected:
        raise ValueError(f"Shape grid tidak valid: {images.shape}; expected {expected}.")
    pixels = denormalize(images.cpu())  # [100,C,H,W]
    if pixels.shape != images.shape or not torch.isfinite(pixels).all():
        raise ValueError("Output denormalize tidak valid.")
    if not ((pixels >= 0.0) & (pixels <= 1.0)).all():
        raise ValueError("Output denormalize harus berada pada rentang [0, 1].")
    # Exclusive creation prevents overwrites even if two processes race.
    while True:
        path = resolve_output_path(cfg.run_dir, checkpoint, guidance_scale, cfg.seed)
        try:
            stream = path.open("xb")
        except FileExistsError:
            continue
        try:
            with stream:
                save_image(pixels, stream, nrow=IMAGES_PER_CLASS, padding=GRID_PADDING, format="PNG")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path


def _stub_denormalize(x: Tensor) -> Tensor:
    """STUB ONLY: normalization inverse when dataset module is absent in self-test."""
    return (x + 1.0) / 2.0  # [N,C,H,W]


def get_denormalize(*, self_test: bool) -> Callable[[Tensor], Tensor]:
    """Use Aksa's original helper; allow a labeled stub only in self-test."""
    try:
        from data.dataset import denormalize
    except ModuleNotFoundError as exc:
        if self_test and exc.name in {"data", "data.dataset"}:
            print("[STUB DENORMALIZE - SELF-TEST ONLY] data/dataset.py belum tersedia.")
            return _stub_denormalize
        raise
    return denormalize


def _expect_error(action: Callable[[], object], error: type[Exception], message: str) -> None:
    """Check that invalid input raises the expected exception."""
    try:
        action()
    except error:
        print(f"[PASS] {message}")
    else:
        raise AssertionError(f"Expected {error.__name__}: {message}")


def _check_png(path: Path, cfg: Config) -> None:
    """Check PNG signature, IHDR dimensions, and nonempty image data."""
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        assert stream.read(8) == b"\x00\x00\x00\x0dIHDR"
        width, height = struct.unpack(">II", stream.read(8))
    expected = IMAGES_PER_CLASS * (cfg.img_size + GRID_PADDING) + GRID_PADDING
    assert (width, height) == (expected, expected)
    assert path.stat().st_size > 33


def _audit_sampling(diffusion: GaussianDiffusion, cfg: Config, guidance: float) -> Tensor:
    """Inspect actual U-Net forward calls and label order without editing U-Net."""
    labels = make_class_labels(cfg)  # [100]
    expected_chunks = list(labels.split(min(cfg.batch_size, MAX_SAMPLE_BATCH)))  # each [N]
    calls = 0
    calls_per_step = 1 if guidance == 0.0 else 2
    calls_per_chunk = cfg.T * calls_per_step

    def inspect_forward(module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
        """Verify labels, timesteps, mode, device and gradient state."""
        nonlocal calls
        x, t, y = inputs  # [N,C,H,W], [N], [N]
        chunk_index, local_call = divmod(calls, calls_per_chunk)
        step_index, branch = divmod(local_call, calls_per_step)
        expected_y = expected_chunks[chunk_index]  # [N]
        if branch == 1:
            expected_y = torch.full_like(expected_y, cfg.num_classes)  # [N]
        torch.testing.assert_close(y.cpu(), expected_y, rtol=0, atol=0)
        assert (t == cfg.T - 1 - step_index).all()
        assert t.dtype == y.dtype == torch.int64
        assert x.device == t.device == y.device == diffusion.betas.device
        assert not module.training and not torch.is_grad_enabled()
        calls += 1

    hook = diffusion.model.register_forward_pre_hook(inspect_forward)
    try:
        images = generate_samples(diffusion, cfg, guidance)  # [100,C,H,W]
    finally:
        hook.remove()
    assert calls == len(expected_chunks) * calls_per_chunk
    assert not images.requires_grad and images.device.type == "cpu"
    print(f"[PASS] Reverse DDPM and label order | guidance={guidance} | forward_calls={calls}")
    return images


def run_self_test(base_cfg: Config, checkpoint: Path | None) -> None:
    """Test real U-Net checkpoints with small T and isolated temporary outputs."""
    device = resolve_device(base_cfg)
    denormalize = get_denormalize(self_test=True)
    with tempfile.TemporaryDirectory(prefix="kaja_sample_test_") as temp:
        root = Path(temp)
        if checkpoint is None:
            cfg = replace(base_cfg, T=3, batch_size=32, cond_drop_prob=0.1,
                          runs_dir=str(root), run_name="self_test")
            cfg.run_dir.mkdir(parents=True)
            set_seed(cfg.seed)
            model = UNet(cfg).cpu()
            checkpoint = cfg.run_dir / "ckpt_ema.pt"
            # Both filenames contain untrained weights; no actual EMA training occurred.
            torch.save(model.state_dict(), checkpoint)
            torch.save(model.state_dict(), cfg.run_dir / "ckpt_last.pt")
            cfg.save_json(cfg.run_dir / "config.json")
            cfg = get_config(["--from_json", str(cfg.run_dir / "config.json")])
            del model
            print("[SELF-TEST ONLY] Checkpoint sementara U-Net asli, BELUM DILATIH; bukan hasil eksperimen.")
        else:
            if base_cfg.T > 10:
                raise ValueError("Self-test checkpoint eksternal membutuhkan T <= 10. Gunakan checkpoint smoke test, bukan checkpoint final.")
            cfg = replace(base_cfg, runs_dir=str(root), run_name="self_test")
            cfg.run_dir.mkdir(parents=True)
            cfg.save_json(cfg.run_dir / "config.json")
            cfg = get_config(["--from_json", str(cfg.run_dir / "config.json")])
            print("[SELF-TEST ONLY] Memakai checkpoint yang diberikan; output sementara.")
        assert Config.load_json(cfg.run_dir / "config.json") == cfg
        print("[PASS] Config JSON roundtrip")

        labels = make_class_labels(cfg)  # [100]
        assert labels.dtype == torch.int64
        assert labels.tolist() == [i for i in range(NUM_CLASSES) for _ in range(IMAGES_PER_CLASS)]
        print("[PASS] 100 labels, 10 per class, correct order")

        diffusion = build_diffusion(cfg, checkpoint, device)
        assert all(p.device == device and not p.requires_grad for p in diffusion.model.parameters())
        assert all(b.device == device for b in diffusion.buffers())
        before = {name: value.detach().cpu().clone() for name, value in diffusion.model.state_dict().items()}
        sibling = checkpoint.parent / ("ckpt_last.pt" if checkpoint.name == "ckpt_ema.pt" else "ckpt_ema.pt")
        if sibling.is_file() and sibling != checkpoint:
            sibling_model = UNet(cfg).cpu()
            load_checkpoint(sibling_model, sibling)
            del sibling_model
            print("[PASS] Strict raw state_dict loading for both checkpoint files")
        else:
            print("[PASS] Strict raw state_dict loading for supplied checkpoint")
            print("[SKIP] Second checkpoint file unavailable")

        _expect_error(lambda: load_checkpoint(UNet(cfg).cpu(), root / "missing.pt"), FileNotFoundError, "Missing checkpoint rejected")
        bad_path = root / "bad.pt"
        torch.save({"not_a_model_parameter": torch.zeros(1)}, bad_path)
        _expect_error(lambda: load_checkpoint(UNet(cfg).cpu(), bad_path), RuntimeError, "Incompatible checkpoint rejected")
        _expect_error(lambda: validate_guidance(cfg, -1.0), ValueError, "Negative guidance rejected")
        _expect_error(lambda: validate_guidance(replace(cfg, cond_drop_prob=0.0), 2.0), ValueError, "CFG without conditional dropout rejected")

        ordinary = _audit_sampling(diffusion, cfg, 0.0)  # [100,C,H,W]
        repeated = generate_samples(diffusion, cfg, 0.0)  # [100,C,H,W]
        torch.testing.assert_close(ordinary, repeated, rtol=1e-5, atol=1e-6)
        print("[PASS] Same seed and sampling batch reproduce output within tolerance")

        guided: Tensor | None = None
        if cfg.cond_drop_prob > 0:
            guided = _audit_sampling(diffusion, cfg, 2.0)  # [100,C,H,W]
        else:
            print("[SKIP] Positive CFG: supplied Config has cond_drop_prob=0")

        # Existing training grid must not be touched, even by repeated sampling.
        training_grid = cfg.run_dir / "samples_epoch_001.png"
        sentinel = b"existing training grid (self-test sentinel)"
        training_grid.write_bytes(sentinel)
        p1 = save_sample_grid(ordinary, cfg, checkpoint, 0.0, denormalize)
        original_png = p1.read_bytes()
        p2 = save_sample_grid(ordinary, cfg, checkpoint, 0.0, denormalize)
        assert p1 != p2 and p1.read_bytes() == original_png
        assert training_grid.read_bytes() == sentinel
        for path in (p1, p2):
            _check_png(path, cfg)
        if guided is not None:
            _check_png(save_sample_grid(guided, cfg, checkpoint, 2.0, denormalize), cfg)
        print("[PASS] Denormalized 10x10 PNG and no overwrite")

        for name, value in diffusion.model.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), before[name], rtol=0, atol=0)
        print("[PASS] Sampling did not change U-Net weights or buffers")
        print(f"[PASS] Output shape={list(ordinary.shape)} | range=[{ordinary.min().item():.6f}, {ordinary.max().item():.6f}]")
        print(f"ALL SAMPLE SELF-TESTS PASSED | DEVICE={device}")
    print("Temporary self-test files removed; no training run was modified.")


def main() -> None:
    """Run normal checkpoint sampling or the isolated lightweight self-test."""
    args, cfg, _ = parse_args()
    if args.self_test:
        run_self_test(cfg, args.ckpt)
        return

    device = resolve_device(cfg)
    if cfg.num_classes != NUM_CLASSES:
        raise ValueError("Grid Fashion-MNIST membutuhkan tepat 10 kelas.")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.ckpt}")
    denormalize = get_denormalize(self_test=False)
    diffusion = build_diffusion(cfg, args.ckpt, device)
    print(f"Checkpoint: {args.ckpt} | Device: {device} | T={cfg.T} | guidance={args.guidance_scale}")
    print(f"Sampling batch: {min(cfg.batch_size, MAX_SAMPLE_BATCH)} | seed={cfg.seed}")
    if args.ckpt.name == "ckpt_last.pt":
        print("[INFO] Menggunakan bobot biasa; ckpt_ema.pt disarankan untuk sampel final.")
    images = generate_samples(diffusion, cfg, args.guidance_scale)  # [100,C,H,W]
    output = save_sample_grid(images, cfg, args.ckpt, args.guidance_scale, denormalize)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
