"""U-Net bersyarat (waktu + kelas) yang memprediksi noise untuk Conditional DDPM.

Kontrak:
    UNet(cfg).forward(x [B, C, H, W] float32, t int64 [B] di 0..T-1, y int64 [B] di 0..num_classes)
        -> eps_pred [B, C, H, W]
    y == num_classes (default 10) adalah token "tanpa kelas" untuk classifier-free guidance.

Struktur (default cfg): 28x28 (64 ch) -> 14x14 (128) -> 7x7 (128, + attention) -> bottleneck
-> decoder simetris dengan skip connection dari setiap blok encoder.

Tes (dari root repo):
    python -m models.unet
    python -m models.unet --base_ch 32 --ch_mult 1 2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config
from models.embeddings import SinusoidalTimeEmbedding


class ConditionEmbedding(nn.Module):
    """cond = MLP(SinusoidalTimeEmbedding(t)) + Embedding(num_classes + 1)(y)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        dim = cfg.time_emb_dim
        self.time_sin = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # indeks num_classes (=10) adalah token "tanpa kelas"
        self.class_emb = nn.Embedding(cfg.num_classes + 1, dim)

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        t_emb = self.time_mlp(self.time_sin(t))  # [B] -> [B, dim]
        y_emb = self.class_emb(y)  # [B] -> [B, dim]
        return t_emb + y_emb  # [B, dim]


class ResBlock(nn.Module):
    """GN-SiLU-Conv, + proyeksi kondisi, GN-SiLU-Dropout-Conv, + shortcut."""

    def __init__(self, in_ch: int, out_ch: int, cfg: Config) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(cfg.num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cfg.time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(cfg.num_groups, out_ch)
        self.dropout = nn.Dropout(cfg.dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # conv 1x1 hanya jika jumlah channel berubah (selalu terjadi di decoder karena concat)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))  # [B, in, H, W] -> [B, out, H, W]
        h = h + self.cond_proj(F.silu(cond))[:, :, None, None]  # [B, out] -> [B, out, 1, 1], broadcast
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))  # [B, out, H, W]
        return h + self.shortcut(x)  # [B, out, H, W]


class AttentionBlock(nn.Module):
    """Multi-head self-attention antar posisi spasial, dengan residual."""

    def __init__(self, ch: int, cfg: Config) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(cfg.num_groups, ch)
        self.attn = nn.MultiheadAttention(ch, cfg.num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)  # [B, H*W, C]
        out = out.transpose(1, 2).reshape(b, c, h, w)  # [B, H*W, C] -> [B, C, H, W]
        return x + out  # [B, C, H, W]


class ResAttnBlock(nn.Module):
    """ResBlock, diikuti AttentionBlock bila tingkat ini memakai attention."""

    def __init__(self, in_ch: int, out_ch: int, use_attn: bool, cfg: Config) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cfg)
        self.attn = AttentionBlock(out_ch, cfg) if use_attn else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.attn(self.res(x, cond))  # [B, in, H, W] -> [B, out, H, W]


class Downsample(nn.Module):
    """Conv3x3 stride 2: resolusi /2, channel tetap."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # [B, C, H, W] -> [B, C, H/2, W/2]


class Upsample(nn.Module):
    """Nearest x2 lalu Conv3x3: resolusi x2, channel tetap."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # [B, C, H, W] -> [B, C, 2H, 2W]
        return self.conv(x)  # [B, C, 2H, 2W]


class DownLevel(nn.Module):
    """Satu tingkat encoder: num_res_blocks ResAttnBlock (+ Downsample kecuali tingkat terakhir).

    Mengembalikan (h, skips): skips berisi output setiap blok dan output Downsample, berurutan.
    """

    def __init__(self, in_ch: int, out_ch: int, use_attn: bool, downsample: bool, cfg: Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResAttnBlock(in_ch if i == 0 else out_ch, out_ch, use_attn, cfg)
            for i in range(cfg.num_res_blocks)
        )
        self.down = Downsample(out_ch) if downsample else None

    def forward(self, h: Tensor, cond: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips: list[Tensor] = []
        for block in self.blocks:
            h = block(h, cond)  # [B, out, H, W]
            skips.append(h)
        if self.down is not None:
            h = self.down(h)  # [B, out, H, W] -> [B, out, H/2, W/2]
            skips.append(h)
        return h, skips


class MidBlock(nn.Module):
    """Bottleneck: ResBlock -> Attention -> ResBlock, resolusi dan channel tetap."""

    def __init__(self, ch: int, cfg: Config) -> None:
        super().__init__()
        self.res1 = ResBlock(ch, ch, cfg)
        self.attn = AttentionBlock(ch, cfg)
        self.res2 = ResBlock(ch, ch, cfg)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return self.res2(self.attn(self.res1(h, cond)), cond)  # [B, C, H, W]


class UpLevel(nn.Module):
    """Satu tingkat decoder: num_res_blocks + 1 ResAttnBlock, tiap blok concat satu skip.

    skip_chs adalah channel skip dalam urutan pop (LIFO). Upsample di akhir kecuali tingkat 0.
    """

    def __init__(self, in_ch: int, skip_chs: list[int], out_ch: int, use_attn: bool,
                 upsample: bool, cfg: Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResAttnBlock((in_ch if i == 0 else out_ch) + s, out_ch, use_attn, cfg)
            for i, s in enumerate(skip_chs)
        )
        self.up = Upsample(out_ch) if upsample else None

    def forward(self, h: Tensor, skips: list[Tensor], cond: Tensor) -> Tensor:
        """skips adalah stack bersama dari encoder; fungsi ini mem-pop dari ujungnya."""
        for block in self.blocks:
            h = torch.cat([h, skips.pop()], dim=1)  # [B, ch_h + ch_skip, H, W]
            h = block(h, cond)  # [B, out, H, W]
        if self.up is not None:
            h = self.up(h)  # [B, out, H, W] -> [B, out, 2H, 2W]
        return h


class UNet(nn.Module):
    """U-Net bersyarat pemrediksi noise. Semua ukuran dibaca dari cfg."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.in_channels = cfg.in_channels
        self.img_size = cfg.img_size
        self.num_classes = cfg.num_classes
        self.T = cfg.T
        n_levels = len(cfg.ch_mult)
        level_ch = [cfg.base_ch * m for m in cfg.ch_mult]
        use_attn = [r in cfg.attn_resolutions for r in cfg.resolutions]

        self.cond = ConditionEmbedding(cfg)
        self.conv_in = nn.Conv2d(cfg.in_channels, cfg.base_ch, kernel_size=3, padding=1)

        # encoder: catat channel setiap tensor yang di-push ke stack skip
        skip_chs = [cfg.base_ch]  # output conv_in
        ch = cfg.base_ch
        self.down_levels = nn.ModuleList()
        for i in range(n_levels):
            is_last = i == n_levels - 1
            self.down_levels.append(DownLevel(ch, level_ch[i], use_attn[i], not is_last, cfg))
            skip_chs += [level_ch[i]] * cfg.num_res_blocks
            if not is_last:
                skip_chs.append(level_ch[i])  # output Downsample
            ch = level_ch[i]

        self.mid = MidBlock(ch, cfg)

        # decoder: pop channel skip dengan urutan yang sama seperti forward
        self.up_levels = nn.ModuleList()
        for i in reversed(range(n_levels)):
            level_skips = [skip_chs.pop() for _ in range(cfg.num_res_blocks + 1)]
            self.up_levels.append(UpLevel(ch, level_skips, level_ch[i], use_attn[i], i > 0, cfg))
            ch = level_ch[i]
        assert not skip_chs, f"sisa skip tidak terpakai: {skip_chs}"  # pengaman logika push/pop

        self.head = nn.Sequential(
            nn.GroupNorm(cfg.num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, cfg.in_channels, kernel_size=3, padding=1),  # tidak di-zero-init
        )

    def _check_inputs(self, x: Tensor, t: Tensor, y: Tensor) -> None:
        """Validasi kontrak input; pesan jelas lebih baik daripada 'device-side assert' CUDA."""
        expected = (self.in_channels, self.img_size, self.img_size)
        if x.dim() != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x harus [B, {expected[0]}, {expected[1]}, {expected[2]}], didapat {tuple(x.shape)}")
        if t.shape != (x.shape[0],) or y.shape != (x.shape[0],):
            raise ValueError(f"t dan y harus [B={x.shape[0]}], didapat t{tuple(t.shape)} y{tuple(y.shape)}")
        if t.dtype != torch.int64 or y.dtype != torch.int64:
            raise ValueError(f"t dan y harus int64, didapat t={t.dtype} y={y.dtype}")
        # satu sinkronisasi GPU->CPU untuk kedua cek rentang
        invalid = ((y < 0) | (y > self.num_classes)).any() | ((t < 0) | (t >= self.T)).any()
        if invalid.item():
            raise ValueError(f"rentang tidak valid: y harus 0..{self.num_classes}, t harus 0..{self.T - 1}; "
                             f"didapat y=[{y.min().item()}, {y.max().item()}] t=[{t.min().item()}, {t.max().item()}]")

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        self._check_inputs(x, t, y)
        cond = self.cond(t, y)  # [B, time_emb_dim]
        h = self.conv_in(x)  # [B, C_in, 28, 28] -> [B, base_ch, 28, 28]
        skips = [h]
        for level in self.down_levels:
            h, level_skips = level(h, cond)  # 28 -> 14 -> 7 (tingkat terakhir tanpa downsample)
            skips.extend(level_skips)
        h = self.mid(h, cond)  # [B, ch_last, 7, 7]
        for level in self.up_levels:
            h = level(h, skips, cond)  # 7 -> 14 -> 28
        return self.head(h)  # [B, base_ch, 28, 28] -> [B, C_in, 28, 28]


# ----------------------------------------------------------------------------- tes

def _register_shape_hooks(model: UNet, cfg: Config) -> list[torch.utils.hooks.RemovableHandle]:
    """Pasang forward hook yang mencetak shape output conv_in, tiap tingkat, bottleneck, dan head."""
    def make_hook(label: str):
        def hook(_module: nn.Module, _inputs: tuple, output) -> None:
            if isinstance(output, tuple):  # DownLevel -> (h, skips)
                h, skips = output
                print(f"  {label:<26} h={tuple(h.shape)}  skip={[s.shape[1] for s in skips]}")
            else:
                print(f"  {label:<26} {tuple(output.shape)}")
        return hook

    res = cfg.resolutions
    targets = [("conv_in", model.conv_in)]
    targets += [(f"encoder L{i} ({res[i]}x{res[i]})", m) for i, m in enumerate(model.down_levels)]
    targets += [("bottleneck", model.mid)]
    n = len(res)
    targets += [(f"decoder L{n - 1 - j} ({res[n - 1 - j]}x{res[n - 1 - j]})", m)
                for j, m in enumerate(model.up_levels)]
    targets += [("head (eps_pred)", model.head)]
    return [m.register_forward_hook(make_hook(label)) for label, m in targets]


def _self_test() -> None:
    from config import get_config

    cfg = get_config()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("PERINGATAN: cfg.device='cuda' tetapi CUDA tidak tersedia, tes berjalan di CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = UNet(cfg).to(device)
    batch = 4

    x = torch.randn(batch, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)  # [4, 1, 28, 28]
    t = torch.randint(0, cfg.T, (batch,), dtype=torch.int64, device=device)  # [4]
    y = torch.randint(0, cfg.num_classes, (batch,), dtype=torch.int64, device=device)  # [4]

    # 1-2. forward pass + shape tiap tingkat lewat hook
    print(f"[1-2] device={device}  t={t.tolist()}  y={y.tolist()}")
    handles = _register_shape_hooks(model, cfg)
    out = model(x, t, y)  # [4, 1, 28, 28]
    for hd in handles:
        hd.remove()

    # 3. jumlah parameter
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[3] parameter total: {n_params:,}")
    for name in ("cond", "conv_in", "down_levels", "mid", "up_levels", "head"):
        part = sum(p.numel() for p in getattr(model, name).parameters())
        print(f"      {name:<12} {part:>10,}")
    assert n_params < 15_000_000, f"parameter {n_params:,} melebihi 15 juta"

    # 4. kontrak output
    assert out.shape == x.shape, out.shape
    assert out.dtype == torch.float32, out.dtype
    assert torch.isfinite(out).all(), "output mengandung NaN/Inf"
    print(f"[4] output {tuple(out.shape)} {out.dtype}, semua finite, std={out.std().item():.4f}")

    # 5. token tanpa kelas
    y_null = torch.full((batch,), cfg.num_classes, dtype=torch.int64, device=device)  # [4] semua = 10
    out_null = model(x, t, y_null)  # [4, 1, 28, 28]
    n_mix = cfg.num_classes + 1
    x_mix = torch.randn(n_mix, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)  # [11, 1, 28, 28]
    t_mix = torch.randint(0, cfg.T, (n_mix,), dtype=torch.int64, device=device)  # [11]
    y_mix = torch.arange(n_mix, dtype=torch.int64, device=device)  # [11] = 0..10
    out_mix = model(x_mix, t_mix, y_mix)  # [11, 1, 28, 28]
    assert torch.isfinite(out_null).all() and torch.isfinite(out_mix).all()
    print(f"[5] y=10 semua: {tuple(out_null.shape)} OK; campuran y=0..{cfg.num_classes}: {tuple(out_mix.shape)} OK")

    # 6. kondisi kelas sampai ke output (eval: dropout mati, jadi beda hanya dari y)
    model.eval()
    with torch.no_grad():
        y3 = torch.full((batch,), 3, dtype=torch.int64, device=device)  # [4]
        e_null = model(x, t, y_null)  # [4, 1, 28, 28]
        e_null_again = model(x, t, y_null)
        e_3 = model(x, t, y3)
    assert torch.equal(e_null, e_null_again), "eval mode tidak deterministik"
    diff = (e_null - e_3).abs().max().item()
    print(f"[6] eval: |eps(y=10) - eps(y=3)| max = {diff:.6f} (harus > 0)")
    assert diff > 1e-6, "output tidak berubah saat y diganti: kondisi kelas tidak sampai ke output"
    model.train()

    # tambahan: input di luar kontrak harus ditolak dengan pesan jelas
    for bad_y, bad_t in ((cfg.num_classes + 1, 0), (0, cfg.T)):
        try:
            model(x, torch.full_like(t, bad_t), torch.full_like(y, bad_y))
        except ValueError as e:
            print(f"[+] input tidak valid ditolak: {e}")
        else:
            raise AssertionError("input di luar kontrak tidak ditolak")

    # tambahan: gradien mengalir ke semua parameter
    model.zero_grad(set_to_none=True)
    model(x, t, y).pow(2).mean().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"parameter tanpa gradien: {no_grad}"
    print("[+] backward OK, semua parameter menerima gradien")

    # tambahan (CUDA saja): puncak VRAM satu langkah forward+backward di cfg.batch_size
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        xb = torch.randn(cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)
        tb = torch.randint(0, cfg.T, (cfg.batch_size,), dtype=torch.int64, device=device)
        yb = torch.randint(0, cfg.num_classes + 1, (cfg.batch_size,), dtype=torch.int64, device=device)
        model.zero_grad(set_to_none=True)
        model(xb, tb, yb).pow(2).mean().backward()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"[+] puncak VRAM forward+backward batch={cfg.batch_size}: {peak_gb:.2f} GB")

    print("UNet: semua tes lulus")


if __name__ == "__main__":
    _self_test()