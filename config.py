"""Satu-satunya sumber hyperparameter proyek Conditional DDPM Fashion-MNIST.

Urutan prioritas nilai (yang belakang menimpa yang depan):
    1. default di dataclass Config
    2. file JSON via --from_json (mis. runs/<run_name>/config.json)
    3. argumen command-line, mis. --lr 1e-4 --ch_mult 1 2 2

Pemakaian:
    from config import get_config
    cfg = get_config()                 # membaca sys.argv
    cfg = get_config(["--T", "500"])   # untuk tes/notebook
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Config:
    """Hyperparameter global. Field dan default mengikuti kontrak antarmuka."""

    # data
    img_size: int = 28
    in_channels: int = 1
    num_classes: int = 10

    # diffusion
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"

    # model
    base_ch: int = 64
    ch_mult: tuple = (1, 2, 2)
    time_emb_dim: int = 256
    dropout: float = 0.1
    num_res_blocks: int = 2          # ResBlock per tingkat resolusi
    num_groups: int = 8              # grup GroupNorm; harus membagi semua jumlah channel
    num_heads: int = 4               # head self-attention
    attn_resolutions: tuple = (7,)   # resolusi encoder/decoder yang diberi attention (bottleneck selalu)

    # training
    batch_size: int = 128
    lr: float = 2e-4
    epochs: int = 20
    num_workers: int = 2
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    cond_drop_prob: float = 0.0

    # misc
    seed: int = 42
    device: str = "cuda"
    run_name: str = "baseline"
    data_root: str = "datasets"      # folder unduhan torchvision (jangan "data": bentrok dengan package data/)
    runs_dir: str = "runs"           # induk folder output runs/<run_name>/

    def __post_init__(self) -> None:
        # JSON tidak punya tuple; list dari JSON/CLI dinormalkan ke tuple
        for name in ("ch_mult", "attn_resolutions"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        self._validate()

    @property
    def resolutions(self) -> tuple:
        """Resolusi tiap tingkat U-Net, mis. (28, 14, 7) untuk ch_mult panjang 3."""
        return tuple(self.img_size // (2 ** i) for i in range(len(self.ch_mult)))

    @property
    def run_dir(self) -> Path:
        """Folder output run ini: runs_dir/run_name."""
        return Path(self.runs_dir) / self.run_name

    def _validate(self) -> None:
        """Tolak kombinasi nilai yang pasti merusak run, sebelum training dimulai."""
        _check(self.img_size > 0 and self.in_channels > 0, "img_size dan in_channels harus > 0")
        _check(self.num_classes > 0, "num_classes harus > 0")
        _check(self.T > 0, "T harus > 0")
        _check(0.0 < self.beta_start < self.beta_end < 1.0,
               "harus berlaku 0 < beta_start < beta_end < 1")
        _check(len(self.ch_mult) > 0 and all(m > 0 for m in self.ch_mult),
               "ch_mult harus berisi bilangan bulat positif")
        # setiap tingkat setelah tingkat pertama melakukan satu downsample x2
        n_down = len(self.ch_mult) - 1
        _check(self.img_size % (2 ** n_down) == 0,
               f"img_size={self.img_size} tidak habis dibagi 2^{n_down}; "
               "skip connection U-Net tidak akan cocok ukurannya")
        _check(self.base_ch > 0 and self.time_emb_dim > 0, "base_ch dan time_emb_dim harus > 0")
        _check(self.time_emb_dim % 2 == 0, "time_emb_dim harus genap (pasangan sin/cos)")
        _check(0.0 <= self.dropout < 1.0, "dropout harus di [0, 1)")
        _check(self.batch_size > 0 and self.epochs > 0, "batch_size dan epochs harus > 0")
        _check(self.lr > 0, "lr harus > 0")
        _check(self.num_workers >= 0, "num_workers harus >= 0")
        _check(0.0 < self.ema_decay < 1.0, "ema_decay harus di (0, 1)")
        _check(self.grad_clip > 0, "grad_clip harus > 0")
        _check(0.0 <= self.cond_drop_prob < 1.0, "cond_drop_prob harus di [0, 1)")
        _check(self.run_name.strip() != "", "run_name tidak boleh kosong")
        _check(self.data_root.strip() != "" and self.runs_dir.strip() != "",
               "data_root dan runs_dir tidak boleh kosong")
        self._validate_unet()

    def _validate_unet(self) -> None:
        """Cek kecocokan channel dengan GroupNorm/attention dan resolusi attention."""
        _check(self.num_res_blocks > 0, "num_res_blocks harus > 0")
        _check(self.num_groups > 0 and self.num_heads > 0, "num_groups dan num_heads harus > 0")
        level_ch = [self.base_ch * m for m in self.ch_mult]
        # skip connection di-concat: decoder melihat ch_level + ch_skip (tingkat sama atau lebih tinggi)
        all_ch = set(level_ch) | {a + b for a in level_ch for b in level_ch}
        bad = sorted(c for c in all_ch if c % self.num_groups != 0)
        _check(not bad, f"num_groups={self.num_groups} tidak membagi channel {bad}")
        _check(all(r in self.resolutions for r in self.attn_resolutions),
               f"attn_resolutions {self.attn_resolutions} harus subset dari {self.resolutions}")
        attn_ch = [ch for ch, r in zip(level_ch, self.resolutions) if r in self.attn_resolutions]
        attn_ch.append(level_ch[-1])  # bottleneck
        bad = sorted(c for c in set(attn_ch) if c % self.num_heads != 0)
        _check(not bad, f"num_heads={self.num_heads} tidak membagi channel attention {bad}")

    def to_dict(self) -> dict[str, Any]:
        """Semua field sebagai dict biasa (siap di-dump ke JSON)."""
        d = dataclasses.asdict(self)
        d["ch_mult"] = list(self.ch_mult)
        d["attn_resolutions"] = list(self.attn_resolutions)
        return d

    def save_json(self, path: str | Path) -> None:
        """Tulis config ke JSON. Dipakai train.py untuk runs/<run_name>/config.json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        """Baca config dari JSON; kunci yang tidak dikenal dianggap eror (kemungkinan typo)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        _check(not unknown, f"kunci tidak dikenal di {path}: {sorted(unknown)}")
        return cls(**raw)


def _check(cond: bool, msg: str) -> None:
    """Pengganti assert yang tetap aktif walau Python dijalankan dengan -O."""
    if not cond:
        raise ValueError(f"[Config] {msg}")


def _str2bool(s: str) -> bool:
    """argparse type=bool menganggap 'False' bernilai True; ini parser yang benar."""
    low = s.lower()
    if low in ("1", "true", "yes", "y"):
        return True
    if low in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"bukan boolean: {s!r}")


def _add_field_arg(parser: argparse.ArgumentParser, field: dataclasses.Field) -> None:
    """Buat satu argumen --<nama_field> dari tipe default field tersebut."""
    default = field.default
    flag = f"--{field.name}"
    help_txt = f"default: {default!r}"
    if isinstance(default, bool):  # cek bool sebelum int, karena bool turunan int
        parser.add_argument(flag, type=_str2bool, help=help_txt)
    elif isinstance(default, tuple):
        elem_type = type(default[0]) if default else str
        parser.add_argument(flag, type=elem_type, nargs="+", help=help_txt)
    else:
        parser.add_argument(flag, type=type(default), help=help_txt)


def build_parser() -> argparse.ArgumentParser:
    """Parser yang dibangkitkan otomatis dari field Config (tidak ada daftar manual)."""
    parser = argparse.ArgumentParser(
        description="Conditional DDPM Fashion-MNIST",
        argument_default=argparse.SUPPRESS,  # argumen yang tidak diketik tidak ikut menimpa
    )
    parser.add_argument("--from_json", type=str,
                        help="muat nilai dasar dari file config.json sebelum override CLI")
    for field in dataclasses.fields(Config):
        _add_field_arg(parser, field)
    return parser


def get_config(argv: Optional[Sequence[str]] = None) -> Config:
    """Bangun Config: default -> --from_json -> override CLI.

    argv=None berarti membaca sys.argv; berikan list untuk tes atau notebook.
    """
    args = vars(build_parser().parse_args(argv))
    json_path = args.pop("from_json", None)
    base = Config.load_json(json_path) if json_path else Config()
    return dataclasses.replace(base, **args)  # replace memanggil __post_init__ lagi -> validasi ulang


if __name__ == "__main__":
    # Cek cepat: python config.py --lr 1e-4 --ch_mult 1 2 4
    print(json.dumps(get_config().to_dict(), indent=2))
