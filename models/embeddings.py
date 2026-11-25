"""Embedding timestep sinusoidal untuk U-Net Conditional DDPM.

Mengubah indeks timestep diskrit t (int64, 0..T-1) menjadi vektor kontinu
berisi pasangan sin/cos dengan frekuensi berbeda. Modul ini sengaja tidak
punya parameter terlatih; MLP dan embedding kelas ada di models/unet.py.

Tes cepat (dari root repo):
    python -m models.embeddings
    python -m models.embeddings --time_emb_dim 128
"""

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    """Petakan t int64 [B] ke embedding float32 [B, dim].

    emb[:, :dim/2] = sin(t * freq), emb[:, dim/2:] = cos(t * freq),
    dengan freq_i = max_period^(-i / (dim/2)) untuk i = 0..dim/2-1.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim harus genap dan > 0, didapat {dim}")
        if max_period <= 1.0:
            raise ValueError(f"max_period harus > 1, didapat {max_period}")
        self.dim = dim
        half = dim // 2
        # frekuensi turun geometris dari 1 (periode 2*pi) ke ~1/max_period
        exponent = torch.arange(half, dtype=torch.float32) / half  # [half]
        freqs = torch.exp(-math.log(max_period) * exponent)  # [half]
        # buffer: ikut .to(device), tidak dilatih, tidak disimpan di state_dict
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] (int64 sesuai kontrak) -> [B, dim] float32."""
        if t.dim() != 1:
            raise ValueError(f"t harus 1D [B], didapat shape {tuple(t.shape)}")
        args = t.float()[:, None] * self.freqs[None, :]  # [B, 1] * [1, half] -> [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


def _self_test() -> None:
    """Cek shape, dtype, rentang nilai, dan keunikan embedding tiap timestep."""
    from config import get_config

    cfg = get_config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    emb = SinusoidalTimeEmbedding(cfg.time_emb_dim).to(device)
    batch = 4

    t = torch.randint(0, cfg.T, (batch,), dtype=torch.int64, device=device)  # [4]
    out = emb(t)  # [4, dim]
    print(f"device={device}  t={t.tolist()}  out.shape={tuple(out.shape)}  dtype={out.dtype}")
    assert out.shape == (batch, cfg.time_emb_dim), out.shape
    assert out.dtype == torch.float32, out.dtype
    assert out.abs().max().item() <= 1.0 + 1e-6, "sin/cos harus di [-1, 1]"

    # t = 0 -> separuh pertama sin(0) = 0, separuh kedua cos(0) = 1
    half = cfg.time_emb_dim // 2
    zero = emb(torch.zeros(1, dtype=torch.int64, device=device))[0]  # [dim]
    assert torch.allclose(zero[:half], torch.zeros(half, device=device))
    assert torch.allclose(zero[half:], torch.ones(half, device=device))

    # setiap pasangan timestep bertetangga harus punya embedding berbeda
    all_t = torch.arange(cfg.T, dtype=torch.int64, device=device)  # [T]
    table = emb(all_t)  # [T, dim]
    step_dist = (table[1:] - table[:-1]).norm(dim=-1)  # [T-1]
    print(f"jarak L2 antar timestep bertetangga: min={step_dist.min():.4f}  max={step_dist.max():.4f}")
    assert step_dist.min().item() > 1e-3, "ada timestep yang embedding-nya nyaris sama"

    n_params = sum(p.numel() for p in emb.parameters())
    assert n_params == 0, f"embedding sinusoidal seharusnya tanpa parameter, didapat {n_params}"
    print("SinusoidalTimeEmbedding: semua tes lulus")


if __name__ == "__main__":
    _self_test()