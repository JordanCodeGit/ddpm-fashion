
"""Noise schedule dan koefisien matematis untuk Gaussian Diffusion.

Mendukung linear schedule dan cosine schedule.
Referensi:
- Ho et al. (2020), Denoising Diffusion Probabilistic Models.
- Nichol & Dhariwal (2021), Improved DDPM.
"""

from dataclasses import replace

import torch
from torch import Tensor

from config import Config, get_config


# Konstanta algoritma cosine berdasarkan Nichol & Dhariwal (2021).
# Penggunaan konstanta ini telah disetujui melalui opsi A.
COSINE_OFFSET: float = 0.008
COSINE_MAX_BETA: float = 0.999


def _linear_beta_schedule(cfg: Config) -> Tensor:
    """Menghasilkan beta yang meningkat secara linier."""

    betas = torch.linspace(
        start=cfg.beta_start,
        end=cfg.beta_end,
        steps=cfg.T,
        dtype=torch.float64,
        device="cpu",
    )  # [T]

    return betas


def _cosine_beta_schedule(cfg: Config) -> Tensor:
    """Menghasilkan beta berdasarkan cosine cumulative alpha."""

    # T+1 titik diperlukan untuk menghitung T nilai beta.
    progress = torch.linspace(
        start=0.0,
        end=1.0,
        steps=cfg.T + 1,
        dtype=torch.float64,
        device="cpu",
    )  # [T+1]

    angles = (
        (progress + COSINE_OFFSET)
        / (1.0 + COSINE_OFFSET)
        * (torch.pi / 2.0)
    )  # [T+1]

    alpha_bars = torch.cos(angles).square()  # [T+1]

    # Normalisasi agar alpha_bar sebelum diffusion bernilai 1.
    alpha_bars = alpha_bars / alpha_bars[0]  # [T+1]

    # Rasio dua timestep berurutan menghasilkan T nilai beta.
    betas = 1.0 - (
        alpha_bars[1:] / alpha_bars[:-1]
    )  # [T]

    # Mencegah beta mencapai 1 dan menyebabkan singularitas.
    betas = torch.clamp(
        betas,
        max=COSINE_MAX_BETA,
    )  # [T]

    return betas


def make_beta_schedule(cfg: Config) -> Tensor:
    """Menghasilkan tensor beta float32 berukuran [T] pada CPU."""

    if cfg.schedule == "linear":
        betas = _linear_beta_schedule(cfg)  # [T]

    elif cfg.schedule == "cosine":
        betas = _cosine_beta_schedule(cfg)  # [T]

    else:
        raise ValueError(
            f"Schedule '{cfg.schedule}' tidak valid. "
            "Gunakan 'linear' atau 'cosine'."
        )

    betas = betas.to(dtype=torch.float32)  # [T]

    # Validasi output sesuai kontrak.
    if betas.shape != (cfg.T,):
        raise ValueError(
            f"Shape beta tidak valid: {betas.shape}. "
            f"Seharusnya ({cfg.T},)."
        )

    if not torch.isfinite(betas).all():
        raise ValueError("Beta mengandung NaN atau Inf.")

    if not ((betas > 0.0) & (betas < 1.0)).all():
        raise ValueError(
            "Seluruh nilai beta harus berada pada rentang (0, 1)."
        )

    return betas


def precompute_coefficients(cfg: Config) -> dict[str, Tensor]:
    """Menghitung koefisien forward dan reverse diffusion."""

    # Gunakan beta sesuai kontrak, lalu naikkan presisi perhitungan.
    betas = make_beta_schedule(cfg).to(torch.float64)  # [T]

    # Alpha dan cumulative alpha.
    alphas = 1.0 - betas  # [T]
    alpha_bars = torch.cumprod(alphas, dim=0)  # [T]

    # Alpha bar sebelum timestep pertama adalah 1.
    alpha_bars_prev = torch.cat(
        (
            torch.ones_like(alpha_bars[:1]),  # [1]
            alpha_bars[:-1],  # [T-1]
        ),
        dim=0,
    )  # [T]

    one_minus_alpha_bars = 1.0 - alpha_bars  # [T]

    # Koefisien forward diffusion.
    sqrt_alpha_bars = torch.sqrt(alpha_bars)  # [T]

    sqrt_one_minus_alpha_bars = torch.sqrt(
        one_minus_alpha_bars
    )  # [T]

    # Koefisien reverse diffusion berbasis prediksi noise.
    sqrt_recip_alphas = torch.rsqrt(alphas)  # [T]

    noise_pred_coef = (
        betas / sqrt_one_minus_alpha_bars
    )  # [T]

    # Varians posterior q(x_{t-1} | x_t, x_0).
    posterior_variance = (
        betas
        * (1.0 - alpha_bars_prev)
        / one_minus_alpha_bars
    )  # [T]

    # Koefisien posterior mean untuk x_0.
    posterior_mean_coef1 = (
        betas
        * torch.sqrt(alpha_bars_prev)
        / one_minus_alpha_bars
    )  # [T]

    # Koefisien posterior mean untuk x_t.
    posterior_mean_coef2 = (
        torch.sqrt(alphas)
        * (1.0 - alpha_bars_prev)
        / one_minus_alpha_bars
    )  # [T]

    coefficients: dict[str, Tensor] = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "noise_pred_coef": noise_pred_coef,
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }

    # Float32 agar kompatibel dengan tensor model.
    result = {
        name: value.to(dtype=torch.float32)
        for name, value in coefficients.items()
    }  # 11 tensor, masing-masing [T]

    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Koefisien '{name}' mengandung NaN atau Inf."
            )

    return result


def _test_schedule(cfg: Config) -> None:
    """Menguji validitas satu konfigurasi noise schedule."""

    betas = make_beta_schedule(cfg)  # [T]
    coefficients = precompute_coefficients(cfg)

    # Shape, dtype, device, dan rentang beta.
    assert betas.shape == (cfg.T,)
    assert betas.dtype == torch.float32
    assert betas.device.type == "cpu"
    assert torch.isfinite(betas).all()
    assert ((betas > 0.0) & (betas < 1.0)).all()

    # Memastikan beta meningkat secara monoton.
    assert (betas[1:] >= betas[:-1]).all()

    # Memastikan seluruh koefisien memiliki shape dan dtype benar.
    assert len(coefficients) == 11

    for name, value in coefficients.items():
        assert value.shape == (cfg.T,), name
        assert value.dtype == torch.float32, name
        assert value.device.type == "cpu", name
        assert torch.isfinite(value).all(), name

    torch.testing.assert_close(
        coefficients["betas"],
        betas,
    )

    # Validasi alpha dan cumulative alpha.
    alphas = coefficients["alphas"]  # [T]
    alpha_bars = coefficients["alpha_bars"]  # [T]
    alpha_bars_prev = coefficients["alpha_bars_prev"]  # [T]

    torch.testing.assert_close(alphas, 1.0 - betas)

    expected_alpha_bars = torch.cumprod(
        1.0 - betas.to(torch.float64),
        dim=0,
    ).float()  # [T]

    torch.testing.assert_close(
        alpha_bars,
        expected_alpha_bars,
    )

    assert ((alpha_bars > 0.0) & (alpha_bars <= 1.0)).all()
    assert (alpha_bars[1:] <= alpha_bars[:-1]).all()
    assert alpha_bars_prev[0].item() == 1.0

    torch.testing.assert_close(
        alpha_bars_prev[1:],  # [T-1]
        alpha_bars[:-1],  # [T-1]
    )

    # Validasi posterior variance.
    posterior_variance = coefficients["posterior_variance"]

    assert (posterior_variance >= 0.0).all()
    assert (posterior_variance <= betas).all()
    assert posterior_variance[0].item() == 0.0

    # Pada timestep pertama, posterior mean kembali ke x_0.
    torch.testing.assert_close(
        coefficients["posterior_mean_coef1"][0],
        torch.ones_like(posterior_variance[0]),
    )

    assert coefficients["posterior_mean_coef2"][0].item() == 0.0

    # Kedua koefisien forward harus memenuhi identitas varians.
    torch.testing.assert_close(
        coefficients["sqrt_alpha_bars"].square()
        + coefficients["sqrt_one_minus_alpha_bars"].square(),
        torch.ones_like(betas),
    )

    # Pemeriksaan khusus setiap jenis schedule.
    if cfg.schedule == "linear":
        torch.testing.assert_close(
            betas[0],
            torch.tensor(cfg.beta_start, dtype=torch.float32),
        )

        if cfg.T > 1:
            torch.testing.assert_close(
                betas[-1],
                torch.tensor(cfg.beta_end, dtype=torch.float32),
            )

    elif cfg.schedule == "cosine":
        max_beta = torch.tensor(
            COSINE_MAX_BETA,
            dtype=torch.float32,
        )

        assert (betas <= max_beta).all()

        torch.testing.assert_close(
            betas[-1],
            max_beta,
        )

    print(f"[PASS] {cfg.schedule.upper()} | T={cfg.T}")
    print(f"  Shape               : {list(betas.shape)}")
    print(f"  Dtype               : {betas.dtype}")
    print(f"  Beta awal           : {betas[0].item():.8f}")
    print(f"  Beta akhir          : {betas[-1].item():.8f}")
    print(f"  Alpha bar akhir     : {alpha_bars[-1].item():.8e}")
    print(f"  Posterior var awal  : {posterior_variance[0].item():.8f}")
    print(f"  Jumlah koefisien    : {len(coefficients)}")
    print()


def _run_self_tests() -> None:
    """Menjalankan pengujian linear, cosine, dan invalid schedule."""

    base_cfg = get_config([])

    print("=" * 55)
    print("DDPM NOISE SCHEDULE SELF-TEST")
    print("=" * 55)

    # Uji konfigurasi normal dan jumlah timestep kecil.
    for schedule_name in ("linear", "cosine"):
        for num_steps in (1, 10, 100, base_cfg.T):
            cfg = replace(
                base_cfg,
                schedule=schedule_name,
                T=num_steps,
            )

            _test_schedule(cfg)

    # Uji penolakan jenis schedule yang tidak valid.
    invalid_cfg = replace(
        base_cfg,
        schedule="invalid",
    )

    try:
        make_beta_schedule(invalid_cfg)

    except ValueError as error:
        print(f"[PASS] Invalid schedule: {error}")

    else:
        raise AssertionError(
            "Schedule tidak valid seharusnya menghasilkan ValueError."
        )

    print("=" * 55)
    print("ALL SCHEDULE TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    _run_self_tests()
