
"""Conditional Gaussian Diffusion untuk Fashion-MNIST.

Mendukung:
- Forward diffusion.
- MSE noise prediction loss.
- Conditional dropout.
- Full reverse DDPM sampling.
- Classifier-free guidance.

Koefisien diffusion menggunakan diffusion/schedule.py.
Noise prediction menggunakan U-Net asli milik proyek.
"""

import math
from collections.abc import Callable
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config, get_config
from diffusion.schedule import precompute_coefficients


class GaussianDiffusion(nn.Module):
    """Conditional DDPM dengan classifier-free guidance."""

    def __init__(self, model: nn.Module, cfg: Config) -> None:
        """Menyimpan model, Config, dan koefisien diffusion."""

        super().__init__()

        self.model = model
        self.cfg = cfg

        # Mengikuti device aktual U-Net saat inisialisasi.
        device = next(model.parameters()).device

        # Gunakan seluruh 11 koefisien dari schedule.py.
        coefficients = precompute_coefficients(cfg)

        for name, value in coefficients.items():
            self.register_buffer(
                name,
                value.to(device=device),
            )  # masing-masing [T]

    def _extract(self, coefficient: Tensor, t: Tensor) -> Tensor:
        """Mengambil koefisien timestep untuk setiap sampel."""

        values = coefficient.gather(0, t)  # [B]

        return values.reshape(-1, 1, 1, 1)  # [B, 1, 1, 1]

    def _validate_image(self, x: Tensor) -> None:
        """Memastikan tensor gambar sesuai dengan Config."""

        expected_shape = (
            self.cfg.in_channels,
            self.cfg.img_size,
            self.cfg.img_size,
        )

        if x.ndim != 4 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Shape gambar tidak valid: {tuple(x.shape)}. "
                f"Seharusnya [B, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}]."
            )

        if x.shape[0] == 0:
            raise ValueError("Batch gambar tidak boleh kosong.")

        if x.dtype != torch.float32:
            raise ValueError("Tensor gambar harus torch.float32.")

        if x.device != self.betas.device:
            raise ValueError(
                f"Device gambar {x.device} berbeda dengan "
                f"device diffusion {self.betas.device}."
            )

    def _validate_labels(
        self,
        y: Tensor,
        batch_size: int,
        allow_unconditional: bool = False,
    ) -> None:
        """Memastikan shape, dtype, dan rentang label valid."""

        if y.shape != (batch_size,):
            raise ValueError(
                f"Shape label tidak valid: {tuple(y.shape)}. "
                f"Seharusnya ({batch_size},)."
            )

        if y.dtype != torch.int64:
            raise ValueError("Label harus torch.int64.")

        max_label = (
            self.cfg.num_classes
            if allow_unconditional
            else self.cfg.num_classes - 1
        )

        if ((y < 0) | (y > max_label)).any():
            raise ValueError(
                f"Rentang label tidak valid. "
                f"Gunakan label 0 sampai {max_label}."
            )

    def _apply_conditional_dropout(self, y: Tensor) -> Tensor:
        """Mengganti label secara acak dengan unconditional token."""

        if self.cfg.cond_drop_prob == 0.0:
            return y

        drop_mask = (
            torch.rand(y.shape, device=y.device)
            < self.cfg.cond_drop_prob
        )  # [B], bool

        unconditional_labels = torch.full_like(
            y,
            self.cfg.num_classes,
        )  # [B], int64

        # Tidak mengubah label asli secara in-place.
        y_cond = torch.where(
            drop_mask,
            unconditional_labels,
            y,
        )  # [B]

        return y_cond

    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Menghasilkan noisy image pada timestep tertentu."""

        self._validate_image(x0)

        batch_size = x0.shape[0]

        if noise.shape != x0.shape:
            raise ValueError(
                "Shape noise harus sama dengan shape x0."
            )

        if noise.dtype != x0.dtype:
            raise ValueError(
                "Dtype noise harus sama dengan dtype x0."
            )

        if noise.device != x0.device:
            raise ValueError(
                "Device noise harus sama dengan device x0."
            )

        if t.shape != (batch_size,):
            raise ValueError(
                f"Shape timestep harus ({batch_size},)."
            )

        if t.dtype != torch.int64:
            raise ValueError("Timestep harus torch.int64.")

        if t.device != x0.device:
            raise ValueError(
                "Device timestep harus sama dengan device x0."
            )

        if ((t < 0) | (t >= self.cfg.T)).any():
            raise ValueError(
                f"Rentang timestep tidak valid. "
                f"Gunakan 0 sampai {self.cfg.T - 1}."
            )

        signal_coef = self._extract(
            self.sqrt_alpha_bars,
            t,
        )  # [B, 1, 1, 1]

        noise_coef = self._extract(
            self.sqrt_one_minus_alpha_bars,
            t,
        )  # [B, 1, 1, 1]

        # Closed-form forward diffusion.
        x_t = (
            signal_coef * x0
            + noise_coef * noise
        )  # [B, C, H, W]

        return x_t

    def training_loss(self, x0: Tensor, y: Tensor) -> Tensor:
        """Menghitung MSE antara noise asli dan prediksi U-Net."""

        self._validate_image(x0)

        batch_size = x0.shape[0]

        self._validate_labels(y, batch_size)

        if y.device != x0.device:
            raise ValueError(
                "Device label harus sama dengan device gambar."
            )

        # Setiap sampel memperoleh timestep acak independen.
        t = torch.randint(
            low=0,
            high=self.cfg.T,
            size=(batch_size,),
            device=x0.device,
            dtype=torch.int64,
        )  # [B]

        # Gaussian noise dengan mean 0 dan variance 1.
        noise = torch.randn_like(x0)  # [B, C, H, W]

        # Forward diffusion.
        x_t = self.q_sample(
            x0,
            t,
            noise,
        )  # [B, C, H, W]

        # Conditional dropout per sampel.
        y_cond = self._apply_conditional_dropout(y)  # [B]

        # Prediksi noise menggunakan U-Net asli.
        eps_pred = self.model(
            x_t,
            t,
            y_cond,
        )  # [B, C, H, W]

        if eps_pred.shape != noise.shape:
            raise ValueError(
                "Shape prediksi noise U-Net tidak sesuai "
                "dengan shape noise asli."
            )

        # Mean reduction menghasilkan tensor skalar.
        loss = F.mse_loss(
            eps_pred,
            noise,
            reduction="mean",
        )  # []

        return loss

    def _predict_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        y: Tensor,
        guidance_scale: float,
    ) -> Tensor:
        """Menghasilkan prediksi noise dengan atau tanpa CFG."""

        eps_c = self.model(
            x_t,
            t,
            y,
        )  # [N, C, H, W]

        # w = 0 berarti conditional sampling biasa.
        if guidance_scale == 0.0:
            return eps_c

        unconditional_labels = torch.full_like(
            y,
            self.cfg.num_classes,
        )  # [N]

        eps_u = self.model(
            x_t,
            t,
            unconditional_labels,
        )  # [N, C, H, W]

        # Konvensi classifier-free guidance kelompok KAJA.
        eps = (
            eps_c
            + guidance_scale * (eps_c - eps_u)
        )  # [N, C, H, W]

        return eps

    @torch.no_grad()
    def sample(
        self,
        y: Tensor,
        guidance_scale: float = 0.0,
    ) -> Tensor:
        """Menghasilkan gambar melalui T langkah reverse DDPM."""

        if (
            not math.isfinite(guidance_scale)
            or guidance_scale < 0.0
        ):
            raise ValueError(
                "guidance_scale harus finite dan >= 0."
            )

        if (
            guidance_scale > 0.0
            and self.cfg.cond_drop_prob == 0.0
        ):
            raise ValueError(
                "Classifier-free guidance memerlukan "
                "cfg.cond_drop_prob > 0."
            )

        if y.ndim != 1 or y.numel() == 0:
            raise ValueError(
                "Label sampling harus memiliki shape [N] "
                "dan tidak boleh kosong."
            )

        num_samples = y.shape[0]

        self._validate_labels(
            y,
            num_samples,
            allow_unconditional=True,
        )

        if (
            self.cfg.cond_drop_prob == 0.0
            and (y == self.cfg.num_classes).any()
        ):
            raise ValueError(
                "Unconditional token tidak boleh digunakan "
                "ketika cond_drop_prob == 0."
            )

        # Gunakan device aktual buffer, bukan string cfg.device.
        device = self.betas.device

        y = y.to(device=device)  # [N]

        image_shape = (
            num_samples,
            self.cfg.in_channels,
            self.cfg.img_size,
            self.cfg.img_size,
        )

        # Simpan mode model sebelum sampling.
        was_training = self.model.training

        self.model.eval()

        try:
            # Mulai reverse diffusion dari Gaussian noise.
            x = torch.randn(
                image_shape,
                device=device,
                dtype=torch.float32,
            )  # [N, C, H, W]

            # Reverse loop penuh: T-1, T-2, ..., 0.
            for step in range(self.cfg.T - 1, -1, -1):

                t = torch.full(
                    (num_samples,),
                    step,
                    device=device,
                    dtype=torch.int64,
                )  # [N]

                eps_pred = self._predict_noise(
                    x,
                    t,
                    y,
                    guidance_scale,
                )  # [N, C, H, W]

                if eps_pred.shape != x.shape:
                    raise ValueError(
                        "Shape prediksi noise tidak sesuai "
                        "dengan shape noisy image."
                    )

                # Koefisien reverse mean.
                sqrt_recip_alpha = self._extract(
                    self.sqrt_recip_alphas,
                    t,
                )  # [N, 1, 1, 1]

                noise_coef = self._extract(
                    self.noise_pred_coef,
                    t,
                )  # [N, 1, 1, 1]

                # Reverse mean berbasis prediksi noise.
                mean = sqrt_recip_alpha * (
                    x - noise_coef * eps_pred
                )  # [N, C, H, W]

                if step > 0:
                    variance = self._extract(
                        self.posterior_variance,
                        t,
                    )  # [N, 1, 1, 1]

                    z = torch.randn(
                        x.shape,
                        device=x.device,
                        dtype=x.dtype,
                    )  # [N, C, H, W]

                    x = (
                        mean
                        + torch.sqrt(variance) * z
                    )  # [N, C, H, W]

                else:
                    # Timestep terakhir tidak menambah noise.
                    x = mean  # [N, C, H, W]

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    "Hasil sampling mengandung NaN atau Inf."
                )

            # Clipping hanya pada hasil akhir.
            return x.clamp(-1.0, 1.0)  # [N, C, H, W]

        finally:
            # Pulihkan mode model sebelum sampling.
            self.model.train(was_training)


# ============================================================
# SELF-TEST
# Menggunakan U-Net asli dan synthetic tensors.
# Tidak dijalankan ketika modul ini di-import.
# ============================================================


def _expect_value_error(
    action: Callable[[], object],
    description: str,
) -> None:
    """Memastikan input invalid menghasilkan ValueError."""

    try:
        action()

    except ValueError:
        print(f"[PASS] {description}")

    else:
        raise AssertionError(
            f"{description}: seharusnya menghasilkan ValueError."
        )


def _test_buffers(diffusion: GaussianDiffusion) -> None:
    """Menguji shape, dtype, dan device diffusion buffers."""

    buffers = dict(
        diffusion.named_buffers(recurse=False)
    )

    assert len(buffers) == 11

    device = diffusion.betas.device

    parameter_names = dict(diffusion.named_parameters())

    for name, value in buffers.items():
        assert value.shape == (diffusion.cfg.T,), name
        assert value.dtype == torch.float32, name
        assert value.device == device, name
        assert not value.requires_grad, name
        assert torch.isfinite(value).all(), name
        assert name not in parameter_names, name

    for parameter in diffusion.model.parameters():
        assert parameter.device == device

    print("[PASS] 11 diffusion buffers and model device")


def _test_forward(diffusion: GaussianDiffusion) -> None:
    """Menguji closed-form forward diffusion."""

    cfg = diffusion.cfg
    device = diffusion.betas.device

    x0 = (
        torch.rand(
            cfg.batch_size,
            cfg.in_channels,
            cfg.img_size,
            cfg.img_size,
            device=device,
        ) * 2.0 - 1.0
    )  # [B, C, H, W]

    noise = torch.randn_like(x0)  # [B, C, H, W]

    # Gunakan timestep berbeda pada satu batch.
    t = torch.tensor(
        [0, cfg.T - 1],
        dtype=torch.int64,
        device=device,
    )  # [2]

    x_t = diffusion.q_sample(
        x0,
        t,
        noise,
    )  # [B, C, H, W]

    alpha_bar = diffusion.alpha_bars[t].reshape(
        -1, 1, 1, 1
    )  # [B, 1, 1, 1]

    expected = (
        torch.sqrt(alpha_bar) * x0
        + torch.sqrt(1.0 - alpha_bar) * noise
    )  # [B, C, H, W]

    assert x_t.shape == x0.shape
    assert x_t.dtype == torch.float32
    assert x_t.device == device
    assert torch.isfinite(x_t).all()

    # Toleransi mengakomodasi pembulatan float32 alpha_bar.
    torch.testing.assert_close(
        x_t,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    print("[PASS] Forward diffusion formula and shape")

    invalid_t = torch.full_like(t, cfg.T)  # [B]

    _expect_value_error(
        lambda: diffusion.q_sample(x0, invalid_t, noise),
        "Invalid timestep rejected",
    )


def _test_training_loss(
    diffusion: GaussianDiffusion,
) -> None:
    """Menguji loss, conditional labels, dan backward pass."""

    cfg = diffusion.cfg
    device = diffusion.betas.device

    diffusion.train()
    diffusion.zero_grad(set_to_none=True)

    x0 = (
        torch.rand(
            cfg.batch_size,
            cfg.in_channels,
            cfg.img_size,
            cfg.img_size,
            device=device,
        ) * 2.0 - 1.0
    )  # [B, C, H, W]

    y = (
        torch.arange(
            cfg.batch_size,
            device=device,
            dtype=torch.int64,
        ) % cfg.num_classes
    )  # [B]

    original_y = y.clone()  # [B]

    # Rekonstruksi random inputs dengan seed yang sama.
    torch.manual_seed(cfg.seed)

    expected_t = torch.randint(
        0,
        cfg.T,
        (cfg.batch_size,),
        device=device,
        dtype=torch.int64,
    )  # [B]

    expected_noise = torch.randn_like(x0)  # [B, C, H, W]

    expected_mask = (
        torch.rand(y.shape, device=device)
        < cfg.cond_drop_prob
    )  # [B], bool

    expected_y = torch.where(
        expected_mask,
        torch.full_like(y, cfg.num_classes),
        y,
    )  # [B]

    records: list[
        tuple[Tensor, Tensor, Tensor, Tensor]
    ] = []

    def capture_forward(
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Merekam input dan output U-Net asli."""

        image, timestep, labels = inputs

        records.append((
            image.detach().clone(),
            timestep.detach().clone(),
            labels.detach().clone(),
            output.detach().clone(),
        ))

    hook = diffusion.model.register_forward_hook(
        capture_forward
    )

    # Reset seed agar random inputs dapat dibandingkan.
    torch.manual_seed(cfg.seed)

    try:
        loss = diffusion.training_loss(x0, y)

    finally:
        hook.remove()

    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert len(records) == 1

    actual_x_t, actual_t, actual_y, eps_pred = records[0]

    torch.testing.assert_close(actual_t, expected_t)
    torch.testing.assert_close(actual_y, expected_y)
    torch.testing.assert_close(y, original_y)

    expected_x_t = diffusion.q_sample(
        x0,
        expected_t,
        expected_noise,
    )  # [B, C, H, W]

    torch.testing.assert_close(
        actual_x_t,
        expected_x_t,
    )

    expected_loss = F.mse_loss(
        eps_pred,
        expected_noise,
    )  # []

    torch.testing.assert_close(
        loss.detach(),
        expected_loss,
    )

    print(
        f"[PASS] Training loss and MSE formula "
        f"| loss={loss.item():.6f}"
    )

    # Uji computational graph dan gradient flow.
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in diffusion.model.parameters()
        if parameter.grad is not None
    ]

    assert len(gradients) > 0

    assert all(
        torch.isfinite(grad).all().item()
        for grad in gradients
    )

    assert any(
        torch.count_nonzero(grad).item() > 0
        for grad in gradients
    )

    print("[PASS] Backward pass and finite gradients")

    diffusion.zero_grad(set_to_none=True)


def _test_conditional_dropout(
    diffusion: GaussianDiffusion,
) -> GaussianDiffusion:
    """Menguji conditional dropout aktif dan nonaktif."""

    cfg = diffusion.cfg
    device = diffusion.betas.device

    # Batch label lebih besar khusus untuk menguji dropout.
    num_labels = 512

    y = (
        torch.arange(
            num_labels,
            device=device,
            dtype=torch.int64,
        ) % cfg.num_classes
    )  # [512]

    original_y = y.clone()  # [512]

    torch.manual_seed(cfg.seed)

    expected_mask = (
        torch.rand(y.shape, device=device)
        < cfg.cond_drop_prob
    )  # [512], bool

    expected_y = torch.where(
        expected_mask,
        torch.full_like(y, cfg.num_classes),
        y,
    )  # [512]

    torch.manual_seed(cfg.seed)

    y_cond = diffusion._apply_conditional_dropout(y)  # [512]

    torch.testing.assert_close(y_cond, expected_y)
    torch.testing.assert_close(y, original_y)

    assert y_cond.dtype == torch.int64
    assert y_cond.shape == y.shape

    valid_labels = (
        (y_cond == y)
        | (y_cond == cfg.num_classes)
    )  # [512], bool

    assert valid_labels.all()

    dropped = int(
        (y_cond == cfg.num_classes).sum().item()
    )

    assert 0 < dropped < num_labels

    print(
        f"[PASS] Conditional dropout "
        f"| dropped={dropped}/{num_labels}"
    )

    # Gunakan model asli yang sama dengan Config dropout nol.
    cfg_no_drop = replace(
        cfg,
        cond_drop_prob=0.0,
    )

    diffusion_no_drop = GaussianDiffusion(
        diffusion.model,
        cfg_no_drop,
    )

    unchanged_y = (
        diffusion_no_drop._apply_conditional_dropout(y)
    )  # [512]

    torch.testing.assert_close(unchanged_y, original_y)

    print("[PASS] Zero conditional dropout preserves labels")

    return diffusion_no_drop


def _test_sampling(
    diffusion: GaussianDiffusion,
    guidance_scale: float,
) -> None:
    """Menguji reverse steps, CFG, output, dan model mode."""

    cfg = diffusion.cfg
    device = diffusion.betas.device

    y = (
        torch.arange(
            cfg.batch_size,
            device=device,
            dtype=torch.int64,
        ) % cfg.num_classes
    )  # [N]

    shape = (
        y.shape[0],
        cfg.in_channels,
        cfg.img_size,
        cfg.img_size,
    )

    # Rekam forward asli untuk memverifikasi setiap timestep.
    records: list[
        tuple[Tensor, Tensor, Tensor, Tensor]
    ] = []

    def capture_forward(
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Merekam noisy image, timestep, label, dan prediksi."""

        assert not module.training
        assert not torch.is_grad_enabled()

        image, timestep, labels = inputs

        records.append((
            image.detach().clone(),
            timestep.detach().clone(),
            labels.detach().clone(),
            output.detach().clone(),
        ))

    # Mulai dalam training mode untuk menguji pemulihan mode.
    diffusion.model.train()

    hook = diffusion.model.register_forward_hook(
        capture_forward
    )

    torch.manual_seed(cfg.seed)

    try:
        samples = diffusion.sample(
            y,
            guidance_scale=guidance_scale,
        )  # [N, C, H, W]

    finally:
        hook.remove()

    assert diffusion.model.training
    assert samples.shape == shape
    assert samples.dtype == torch.float32
    assert samples.device == device
    assert not samples.requires_grad
    assert torch.isfinite(samples).all()
    assert ((samples >= -1.0) & (samples <= 1.0)).all()

    calls_per_step = (
        1 if guidance_scale == 0.0 else 2
    )

    expected_calls = cfg.T * calls_per_step

    assert len(records) == expected_calls

    # Generator terpisah untuk merekonstruksi Gaussian noise.
    generator = torch.Generator(
        device=device
    ).manual_seed(cfg.seed)

    expected_x = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )  # [N, C, H, W]

    # Verifikasi seluruh reverse step berdasarkan hasil
    # prediksi U-Net asli yang direkam melalui forward hook.
    for index, step in enumerate(
        range(cfg.T - 1, -1, -1)
    ):
        record_index = index * calls_per_step

        x_c, t_c, y_c, eps_c = records[record_index]

        expected_t = torch.full_like(
            y,
            step,
        )  # [N]

        torch.testing.assert_close(x_c, expected_x)
        torch.testing.assert_close(t_c, expected_t)
        torch.testing.assert_close(y_c, y)

        if guidance_scale > 0.0:
            x_u, t_u, y_u, eps_u = records[
                record_index + 1
            ]

            torch.testing.assert_close(x_u, x_c)
            torch.testing.assert_close(t_u, t_c)

            torch.testing.assert_close(
                y_u,
                torch.full_like(y, cfg.num_classes),
            )

            # Rumus CFG dalam bentuk ekuivalen.
            expected_eps = (
                (1.0 + guidance_scale) * eps_c
                - guidance_scale * eps_u
            )  # [N, C, H, W]

        else:
            expected_eps = eps_c  # [N, C, H, W]

        # Rekonstruksi reverse mean.
        expected_mean = (
            diffusion.sqrt_recip_alphas[step]
            * (
                x_c
                - diffusion.noise_pred_coef[step]
                * expected_eps
            )
        )  # [N, C, H, W]

        if step > 0:
            expected_z = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )  # [N, C, H, W]

            expected_x = (
                expected_mean
                + torch.sqrt(
                    diffusion.posterior_variance[step]
                ) * expected_z
            )  # [N, C, H, W]

        else:
            expected_x = expected_mean  # [N, C, H, W]

    # Output akhir harus sesuai reverse steps dan clipping.
    torch.testing.assert_close(
        samples,
        expected_x.clamp(-1.0, 1.0),
        rtol=1e-4,
        atol=1e-5,
    )

    # Pastikan tidak ada random noise tambahan pada step 0.
    if device.type == "cuda":
        actual_rng_state = torch.cuda.get_rng_state(device)

    else:
        actual_rng_state = torch.get_rng_state()

    assert torch.equal(
        actual_rng_state,
        generator.get_state(),
    ), "Jumlah pengambilan Gaussian noise tidak sesuai."

    print(
        f"[PASS] Reverse sampling | "
        f"schedule={cfg.schedule} | T={cfg.T} | "
        f"guidance={guidance_scale} | "
        f"forward_calls={len(records)}"
    )

    print(
        f"  Shape       : {list(samples.shape)}\n"
        f"  Min         : {samples.min().item():.6f}\n"
        f"  Max         : {samples.max().item():.6f}\n"
        f"  Device      : {samples.device}\n"
        f"  Requires grad: {samples.requires_grad}"
    )


def _run_device_tests(device: torch.device) -> None:
    """Menjalankan self-test dengan U-Net asli pada satu device."""

    from models.unet import UNet

    base_cfg = get_config([])

    # Timestep dan batch kecil khusus untuk self-test.
    # Arsitektur U-Net tetap mengikuti Config asli.
    cfg = replace(
        base_cfg,
        T=3,
        batch_size=2,
        schedule="linear",
        device=str(device),
    )

    torch.manual_seed(cfg.seed)

    print()
    print("=" * 60)
    print(f"GAUSSIAN DIFFUSION SELF-TEST | DEVICE={device}")
    print("=" * 60)

    # U-Net asli, bukan stub.
    model = UNet(cfg)

    diffusion = GaussianDiffusion(
        model,
        cfg,
    ).to(device)

    _test_buffers(diffusion)
    _test_forward(diffusion)
    _test_training_loss(diffusion)

    diffusion_no_drop = _test_conditional_dropout(
        diffusion
    )

    # Reverse sampling tanpa dan dengan CFG.
    _test_sampling(
        diffusion,
        guidance_scale=0.0,
    )

    _test_sampling(
        diffusion,
        guidance_scale=2.0,
    )

    # Validasi guidance scale.
    y = torch.zeros(
        cfg.batch_size,
        dtype=torch.int64,
        device=device,
    )  # [N]

    _expect_value_error(
        lambda: diffusion.sample(y, guidance_scale=-1.0),
        "Negative guidance scale rejected",
    )

    _expect_value_error(
        lambda: diffusion.sample(
            y,
            guidance_scale=float("nan"),
        ),
        "Non-finite guidance scale rejected",
    )

    _expect_value_error(
        lambda: diffusion_no_drop.sample(
            y,
            guidance_scale=2.0,
        ),
        "CFG with zero conditional dropout rejected",
    )

    # Conditional sampling tetap valid tanpa conditional dropout.
    _test_sampling(
        diffusion_no_drop,
        guidance_scale=0.0,
    )

    # Uji compatibility dengan cosine schedule.
    cfg_cosine = replace(
        cfg,
        schedule="cosine",
    )

    diffusion_cosine = GaussianDiffusion(
        model,
        cfg_cosine,
    )

    _test_sampling(
        diffusion_cosine,
        guidance_scale=0.0,
    )

    # Edge case T=1 dengan U-Net asli yang sesuai Config.
    cfg_single_step = replace(
        cfg,
        T=1,
    )

    model_single_step = UNet(
        cfg_single_step
    ).to(device)

    diffusion_single_step = GaussianDiffusion(
        model_single_step,
        cfg_single_step,
    )

    _test_sampling(
        diffusion_single_step,
        guidance_scale=0.0,
    )

    # Pastikan buffer dan model dapat kembali ke CPU.
    diffusion.to("cpu")

    assert diffusion.betas.device.type == "cpu"

    _test_buffers(diffusion)

    print("[PASS] Model and buffers compatible with CPU transfer")
    print(f"ALL {str(device).upper()} TESTS PASSED")


def _run_self_tests() -> None:
    """Menjalankan pengujian CPU dan CUDA apabila tersedia."""

    _run_device_tests(torch.device("cpu"))

    if torch.cuda.is_available():
        _run_device_tests(torch.device("cuda"))

    else:
        print("[SKIP] CUDA tests: CUDA is not available.")

    print()
    print("=" * 60)
    print("ALL DDPM SELF-TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
