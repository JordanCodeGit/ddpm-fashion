"""Skrip pembuat grafik evaluasi dan tabel ringkasan hasil untuk laporan akhir."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Pastikan root repository masuk ke sys.path meskipun dijalankan via `python scripts/make_figures.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Backend non-interaktif agar aman di CLI/Windows tanpa GUI
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from config import Config, get_config

FASHION_MNIST_CLASSES_ID: list[str] = [
    "0: Kaos/Atasan",
    "1: Celana Panjang",
    "2: Sweter",
    "3: Gaun",
    "4: Mantel",
    "5: Sandal",
    "6: Kemeja",
    "7: Sepatu Kets",
    "8: Tas",
    "9: Sepatu Bot",
]

REQUIRED_METRICS_COLS: set[str] = {
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "epoch_time_s",
}

# Palet Okabe-Ito ramah buta warna + variasi linestyle/marker agar terbaca saat PDF dicetak hitam-putih
OKABE_ITO_PALETTE: list[str] = [
    "#0072B2",  # Biru
    "#D55E00",  # Oranye kemerahan
    "#009E73",  # Hijau kebiruan
    "#CC79A7",  # Merah muda keunguan
    "#E69F00",  # Kuning keemasan
    "#56B4E9",  # Biru langit
    "#332288",  # Nila tua
    "#882255",  # Marun
]
LINE_STYLES: list[str] = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]


@dataclass(frozen=True)
class RunArtifact:
    """Representasi data hasil eksperimen dalam satu folder runs/<run_name>/."""

    run_name: str
    run_dir: Path
    metrics: pd.DataFrame
    config_data: dict[str, Any] | None
    eval_data: dict[str, Any] | None


def parse_script_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Config]:
    """Memisahkan argumen khusus skrip visualisasi dan argumen Config dari repo."""
    parser = argparse.ArgumentParser(
        description="Membuat grafik laporan dan tabel ringkasan dari hasil eksperimen.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--final_run",
        type=str,
        default=None,
        help="Nama folder run final untuk kurva loss & akurasi per kelas (opsional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/figures",
        help="Direktori output gambar dan tabel ringkasan.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolusi DPI gambar yang disimpan.",
    )
    extra_args, rest_args = parser.parse_known_args(argv)
    cfg = get_config(rest_args)
    return extra_args, cfg


def load_json_safe(json_path: Path) -> dict[str, Any] | None:
    """Membaca file JSON secara aman; mengembalikan None jika tidak ada atau korup."""
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[PERINGATAN] Gagal membaca {json_path}: {exc}")
        return None


def discover_runs(runs_root: Path) -> list[RunArtifact]:
    """Memindai semua subfolder di dalam runs/ yang memiliki metrics.csv valid."""
    if not runs_root.is_dir():
        return []

    discovered: list[RunArtifact] = []
    for sub_dir in sorted(runs_root.iterdir()):
        if not sub_dir.is_dir():
            continue

        metrics_path = sub_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            print(f"[PERINGATAN] Gagal memuat {metrics_path}: {exc}")
            continue

        if df.empty or not REQUIRED_METRICS_COLS.issubset(set(df.columns)):
            print(
                f"[PERINGATAN] Melewati {metrics_path} karena kosong atau kolom tidak sesuai kontrak."
            )
            continue

        df = df.sort_values("epoch").reset_index(drop=True)
        config_data = load_json_safe(sub_dir / "config.json")
        eval_data = load_json_safe(sub_dir / "eval.json")

        discovered.append(
            RunArtifact(
                run_name=sub_dir.name,
                run_dir=sub_dir,
                metrics=df,
                config_data=config_data,
                eval_data=eval_data,
            )
        )

    return discovered


def select_final_run(runs: list[RunArtifact], requested_name: str | None) -> RunArtifact:
    """Memilih run utama berdasarkan argumen CLI, folder 'final', keberadaan eval.json, atau val_loss terendah."""
    if requested_name:
        for item in runs:
            if item.run_name == requested_name:
                return item
        raise ValueError(
            f"Run final '{requested_name}' tidak ditemukan. Run tersedia: {[r.run_name for r in runs]}"
        )

    for item in runs:
        if item.run_name.lower() == "final":
            return item

    runs_with_eval = [r for r in runs if r.eval_data is not None]
    candidates = runs_with_eval if runs_with_eval else runs
    return min(candidates, key=lambda r: float(r.metrics["val_loss"].min()))


def select_eval_run(runs: list[RunArtifact], final_run: RunArtifact) -> RunArtifact | None:
    """Mencari run yang memiliki eval.json, memprioritaskan final_run bila tersedia."""
    if final_run.eval_data is not None:
        return final_run

    runs_with_eval = [r for r in runs if r.eval_data is not None]
    if not runs_with_eval:
        return None

    return min(runs_with_eval, key=lambda r: float(r.metrics["val_loss"].min()))


def configure_plot_style() -> None:
    """Mengatur gaya visual grafik akademik agar bersih, konsisten, dan tajam di PDF."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_final_loss_curve(final_run: RunArtifact, out_path: Path, dpi: int) -> None:
    """Membuat grafik kurva train_loss dan val_loss terhadap epoch untuk run final."""
    df = final_run.metrics
    epochs = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy()
    val_loss = df["val_loss"].to_numpy()

    best_idx = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_loss[best_idx])

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(
        epochs,
        train_loss,
        marker="o",
        markersize=4,
        linewidth=2.0,
        color="#0072B2",
        label="Training Loss (train_loss)",
    )
    ax.plot(
        epochs,
        val_loss,
        marker="s",
        markersize=4,
        linestyle="--",
        linewidth=2.0,
        color="#D55E00",
        label="Validation Loss (val_loss)",
    )
    ax.scatter(
        [best_epoch],
        [best_val],
        color="#009E73",
        edgecolor="black",
        linewidth=0.8,
        s=85,
        zorder=5,
        label=f"Val Loss Terbaik: {best_val:.4f} (Epoch {best_epoch})",
    )

    ax.set_title(f"Kurva Training dan Validation Loss — Run '{final_run.run_name}'")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE Prediksi Noise)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_tuning_val_loss_comparison(runs: list[RunArtifact], out_path: Path, dpi: int) -> None:
    """Membuat grafik perbandingan val_loss antar-run dengan zoom inset fase konvergensi."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    max_epoch_overall = 0
    for idx, run in enumerate(runs):
        df = run.metrics
        epochs = df["epoch"].to_numpy()
        val_loss = df["val_loss"].to_numpy()
        best_idx = int(np.argmin(val_loss))
        min_val = float(val_loss[best_idx])
        best_ep = int(epochs[best_idx])
        max_epoch_overall = max(max_epoch_overall, int(epochs.max()))

        color = OKABE_ITO_PALETTE[idx % len(OKABE_ITO_PALETTE)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(
            epochs,
            val_loss,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.8,
            linewidth=1.8,
            label=f"{run.run_name} (min: {min_val:.4f} @ ep {best_ep})",
        )
        ax.scatter(
            [best_ep],
            [min_val],
            color=color,
            edgecolor="black",
            linewidth=0.7,
            s=55,
            zorder=5,
        )

    ax.set_title("Perbandingan Kurva Validation Loss Antar-Eksperimen Tuning")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.legend(
        title="Eksperimen Tuning",
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # Tambahkan inset zoom pada 45% epoch terakhir apabila jumlah epoch >= 5
    if max_epoch_overall >= 5:
        zoom_start_epoch = max(1, int(np.floor(max_epoch_overall * 0.55)))
        zoom_vals: list[float] = []
        for run in runs:
            sub = run.metrics[run.metrics["epoch"] >= zoom_start_epoch]["val_loss"]
            if not sub.empty:
                zoom_vals.extend(sub.astype(float).tolist())

        if zoom_vals:
            y_min, y_max = min(zoom_vals), max(zoom_vals)
            pad = max((y_max - y_min) * 0.18, 1e-4)

            ax_ins = inset_axes(
                ax,
                width="42%",
                height="38%",
                loc="center right",
                borderpad=1.6,
            )
            for idx, run in enumerate(runs):
                df = run.metrics
                epochs = df["epoch"].to_numpy()
                val_loss = df["val_loss"].to_numpy()
                color = OKABE_ITO_PALETTE[idx % len(OKABE_ITO_PALETTE)]
                linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
                marker = MARKERS[idx % len(MARKERS)]
                ax_ins.plot(
                    epochs,
                    val_loss,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.2,
                    linewidth=1.5,
                )

            ax_ins.set_xlim(zoom_start_epoch - 0.3, max_epoch_overall + 0.3)
            ax_ins.set_ylim(y_min - pad, y_max + pad)
            ax_ins.set_title(
                f"Detail Konvergensi (Epoch {zoom_start_epoch}–{max_epoch_overall})",
                fontsize=8.5,
                pad=4,
            )
            ax_ins.tick_params(labelsize=7.5)
            ax_ins.grid(True, linestyle=":", alpha=0.4)
            mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="0.55", linestyle="--", lw=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _extract_per_class_acc(raw_per_class: Any, num_classes: int) -> list[float]:
    """Menormalisasi format per_class_acc dari eval.json (list atau dict) ke list float [10]."""
    if isinstance(raw_per_class, list):
        if len(raw_per_class) != num_classes:
            raise ValueError(
                f"Panjang per_class_acc ({len(raw_per_class)}) tidak sama dengan {num_classes}."
            )
        return [float(v) for v in raw_per_class]

    if isinstance(raw_per_class, dict):
        values: list[float] = []
        for cls_idx in range(num_classes):
            key_str = str(cls_idx)
            if key_str in raw_per_class:
                values.append(float(raw_per_class[key_str]))
            elif cls_idx in raw_per_class:
                values.append(float(raw_per_class[cls_idx]))
            else:
                raise KeyError(f"Kelas {cls_idx} tidak ditemukan di per_class_acc.")
        return values

    raise TypeError("Format per_class_acc di eval.json harus berupa list atau dict.")


def _to_percentage(values: list[float]) -> list[float]:
    """Mengonversi skala [0, 1] menjadi persentase [0, 100] secara otomatis bila diperlukan."""
    if not values:
        return values
    return [v * 100.0 for v in values] if max(values) <= 1.0 else values


def plot_per_class_accuracy(
    eval_run: RunArtifact,
    out_path: Path,
    dpi: int,
    num_classes: int,
) -> bool:
    """Membuat horizontal bar chart akurasi per kelas terurut + garis overall_acc & classifier_test_acc."""
    if eval_run.eval_data is None:
        return False

    data = eval_run.eval_data
    if "per_class_acc" not in data:
        print(f"[PERINGATAN] Key 'per_class_acc' tidak ada di {eval_run.run_dir / 'eval.json'}.")
        return False

    raw_scores = _extract_per_class_acc(data["per_class_acc"], num_classes=num_classes)
    is_unit_scale = max(raw_scores) <= 1.0
    scores_pct = _to_percentage(raw_scores)

    overall_raw = data.get("overall_acc")
    clf_raw = data.get("classifier_test_acc")

    overall_pct = (
        float(overall_raw) * 100.0
        if overall_raw is not None and is_unit_scale
        else (float(overall_raw) if overall_raw is not None else None)
    )
    clf_pct = (
        float(clf_raw) * 100.0
        if clf_raw is not None and is_unit_scale
        else (float(clf_raw) if clf_raw is not None else None)
    )

    base_labels = (
        FASHION_MNIST_CLASSES_ID
        if num_classes == len(FASHION_MNIST_CLASSES_ID)
        else [f"Kelas {i}" for i in range(num_classes)]
    )

    # Urutkan dari akurasi tertinggi (paling atas pada horizontal bar chart) ke terendah
    paired = sorted(zip(base_labels, scores_pct), key=lambda item: item[1], reverse=True)
    sorted_labels = [p[0] for p in paired]
    sorted_scores = [p[1] for p in paired]

    threshold_ref = overall_pct if overall_pct is not None else float(np.mean(sorted_scores))
    bar_colors = ["#0072B2" if val >= threshold_ref else "#E69F00" for val in sorted_scores]

    y_pos = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # Ghost bar latar belakang untuk menunjukkan jarak ke akurasi classifier pada data uji asli
    if clf_pct is not None:
        ax.barh(
            y_pos,
            [clf_pct] * num_classes,
            height=0.64,
            color="#E5E9F0",
            edgecolor="none",
            zorder=1,
            label=f"Batas Referensi Classifier Uji Asli ({clf_pct:.2f}%)",
        )

    bars = ax.barh(
        y_pos,
        sorted_scores,
        height=0.64,
        color=bar_colors,
        edgecolor="#1f2937",
        linewidth=0.7,
        zorder=2,
    )

    # Proxy legend agar warna biru (>= rata-rata) dan oranye (< rata-rata) terjelaskan rapi
    ax.barh([], [], color="#0072B2", edgecolor="#1f2937", label="Akurasi Kelas ≥ Rata-rata")
    ax.barh([], [], color="#E69F00", edgecolor="#1f2937", label="Akurasi Kelas < Rata-rata")

    for bar, val in zip(bars, sorted_scores):
        ax.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#111827",
        )

    if overall_pct is not None:
        ax.axvline(
            overall_pct,
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            zorder=4,
            label=f"Rata-rata Overall Accuracy ({overall_pct:.2f}%)",
        )

    if clf_pct is not None:
        ax.axvline(
            clf_pct,
            color="#009E73",
            linestyle="-.",
            linewidth=2.0,
            zorder=4,
            label=f"Classifier Test Accuracy ({clf_pct:.2f}%)",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()  # Akurasi tertinggi berada di baris paling atas
    ax.set_xlim(0, 108)
    ax.set_xlabel("Akurasi Klasifikasi (%)")
    ax.set_ylabel("Kelas Fashion-MNIST (Terurut)")
    ax.set_title(f"Akurasi Klasifikasi per Kelas Sampel Generate — Run '{eval_run.run_name}'")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=8.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def build_summary_table(runs: list[RunArtifact], output_dir: Path) -> pd.DataFrame:
    """Menyusun dan mengekspor tabel ringkasan metrik seluruh run ke format CSV dan Markdown."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = run.metrics
        cfg = run.config_data or {}
        ev = run.eval_data or {}

        best_val_idx = int(df["val_loss"].idxmin())
        best_val_loss = float(df.loc[best_val_idx, "val_loss"])
        best_epoch = int(df.loc[best_val_idx, "epoch"])
        final_train_loss = float(df["train_loss"].iloc[-1])
        final_val_loss = float(df["val_loss"].iloc[-1])
        total_epochs = int(df["epoch"].max())
        mean_epoch_s = float(df["epoch_time_s"].mean())

        overall_acc = ev.get("overall_acc")
        clf_acc = ev.get("classifier_test_acc")
        if isinstance(overall_acc, (int, float)) and overall_acc <= 1.0:
            overall_acc_str = f"{overall_acc * 100:.2f}%"
        elif isinstance(overall_acc, (int, float)):
            overall_acc_str = f"{overall_acc:.2f}%"
        else:
            overall_acc_str = "-"

        if isinstance(clf_acc, (int, float)) and clf_acc <= 1.0:
            clf_acc_str = f"{clf_acc * 100:.2f}%"
        elif isinstance(clf_acc, (int, float)):
            clf_acc_str = f"{clf_acc:.2f}%"
        else:
            clf_acc_str = "-"

        rows.append(
            {
                "Nama Run": run.run_name,
                "Schedule": cfg.get("schedule", "-"),
                "T": cfg.get("T", "-"),
                "Learning Rate": cfg.get("lr", "-"),
                "Cond Drop": cfg.get("cond_drop_prob", "-"),
                "Total Epoch": total_epochs,
                "Train Loss Akhir": round(final_train_loss, 4),
                "Val Loss Akhir": round(final_val_loss, 4),
                "Val Loss Terbaik": round(best_val_loss, 4),
                "Epoch Terbaik": best_epoch,
                "Rata-rata Waktu/Epoch (detik)": round(mean_epoch_s, 2),
                "Overall Acc": overall_acc_str,
                "Classifier Test Acc": clf_acc_str,
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "ringkasan_eksperimen.csv"
    md_path = output_dir / "ringkasan_eksperimen.md"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    headers = list(summary_df.columns)
    md_lines = [
        "# Tabel Ringkasan Hasil Eksperimen Conditional DDPM",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in summary_df.iterrows():
        cells = [str(row[h]) for h in headers]
        md_lines.append("| " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary_df


def main(argv: list[str] | None = None) -> None:
    """Titik masuk utama pembuatan seluruh gambar dan tabel laporan."""
    extra_args, cfg = parse_script_args(argv)
    runs_root = Path(cfg.runs_dir)
    output_dir = Path(extra_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    runs = discover_runs(runs_root)
    if not runs:
        raise FileNotFoundError(
            f"Tidak ditemukan folder eksperimen valid dengan 'metrics.csv' di '{runs_root}'."
        )

    final_run = select_final_run(runs, extra_args.final_run)
    eval_run = select_eval_run(runs, final_run)

    loss_fig_path = output_dir / "kurva_loss_run_final.png"
    tuning_fig_path = output_dir / "perbandingan_val_loss_tuning.png"
    class_acc_fig_path = output_dir / "akurasi_per_kelas.png"

    plot_final_loss_curve(final_run, loss_fig_path, dpi=extra_args.dpi)
    print(f"[OK] Grafik 1 tersimpan: {loss_fig_path}")

    plot_tuning_val_loss_comparison(runs, tuning_fig_path, dpi=extra_args.dpi)
    print(f"[OK] Grafik 2 tersimpan: {tuning_fig_path}")

    if eval_run is not None:
        created = plot_per_class_accuracy(
            eval_run,
            class_acc_fig_path,
            dpi=extra_args.dpi,
            num_classes=cfg.num_classes,
        )
        if created:
            print(f"[OK] Grafik 3 tersimpan (sumber '{eval_run.run_name}'): {class_acc_fig_path}")
    else:
        print(
            "[INFO] Belum ada file 'eval.json' pada folder run mana pun; grafik akurasi per kelas dilewati."
        )

    build_summary_table(runs, output_dir)
    print(f"[OK] Tabel ringkasan tersimpan di: {output_dir / 'ringkasan_eksperimen.csv'} & .md")


if __name__ == "__main__":
    main()
