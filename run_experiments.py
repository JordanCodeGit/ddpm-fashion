"""Sequential Conditional DDPM experiment orchestration (Python 3.10+).

No training or diffusion implementation is duplicated. Run from the repository
root; each child uses the same Python executable and original train/sample CLI.
The budget is a soft planning limit, not a hard process timeout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
LR_CANDIDATES = (("baseline", 2e-4), ("lr_high", 1e-3), ("lr_low", 5e-5))
BATCH_CANDIDATES = (64, 128, 256)
SAMPLE_EVERY = 5  # Matches train.py; verified against the supplied version.
TARGET_TUNING = 5
MIN_TUNING = 3
MIN_FINAL = 5
DEFAULT_MARGIN = 0.15
FALLBACK_OVERHEAD = 30.0  # seconds per run + per epoch; planning assumption
FINAL_SAMPLE_FACTOR = 3.0  # conservative heuristic, not a measured bound
UNKNOWN_BATCH_FACTOR = 2.0  # planning assumption for unmeasured batches
METRIC_FIELDS = ("epoch", "train_loss", "val_loss", "lr", "epoch_time_s")
ConfigDict = dict[str, Any]


class ExperimentError(RuntimeError):
    """A failed or invalid stage must not silently select a winner."""


class BudgetError(ExperimentError):
    """Estimated remaining cost exceeds the planning budget."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse orchestration options; model settings remain in original Config."""
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--budget_hours", type=float, help="Total hours for this invocation")
    p.add_argument("--tuning_epochs", type=int, help="Fixed epochs for all LR runs")
    p.add_argument("--final_epochs", type=int, help="Fixed final epochs")
    p.add_argument("--batch_tuning", action="store_true")
    p.add_argument("--batch_tuning_epochs", type=int)
    p.add_argument("--timing_run", type=Path, help="Existing timing directory (default: <runs_dir>/timing_test)")
    p.add_argument("--timing_log", type=Path, help="Optional terminal log for an existing timing run")
    p.add_argument("--auto_timing", action="store_true")
    p.add_argument("--safety_margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--tuning_only", action="store_true")
    p.add_argument("--dry_run", "--dry-run", action="store_true")
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--runs_dir", type=Path)
    p.add_argument("--reuse_tuning", action="store_true", help="Read complete LR runs; never resume weights")
    p.add_argument("--self_test", "--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return a
    for name in ("tuning_epochs", "final_epochs", "batch_tuning_epochs"):
        val = getattr(a, name)
        if val is not None and val <= 0:
            p.error(f"--{name} must be positive")
    if a.budget_hours is not None and (not math.isfinite(a.budget_hours) or a.budget_hours <= 0 or not math.isfinite(a.budget_hours * 3600)):
        p.error("--budget_hours must be finite and positive")
    if not math.isfinite(a.safety_margin) or not 0 <= a.safety_margin < 1:
        p.error("--safety_margin must be in [0, 1)")
    if not math.isfinite(a.guidance_scale) or a.guidance_scale < 0:
        p.error("--guidance_scale must be finite and >= 0")
    if a.budget_hours is None and ((a.tuning_epochs is None and not a.reuse_tuning) or (a.final_epochs is None and not a.tuning_only)):
        p.error("Automatic epoch selection requires --budget_hours; otherwise specify epochs explicitly")
    if a.batch_tuning_epochs is not None and not a.batch_tuning:
        p.error("--batch_tuning_epochs requires --batch_tuning")
    return a


def load_base_config(args: argparse.Namespace) -> ConfigDict:
    """Use the original parser, not a copy of Config or the smoke-test JSON."""
    from config import get_config
    overrides: list[str] = []
    for key in ("device", "seed", "runs_dir"):
        value = getattr(args, key)
        if value is not None:
            overrides.extend((f"--{key}", str(value)))
    cfg = json.loads(json.dumps(get_config(overrides).to_dict()))  # Normalize tuple/list fields to JSON form.
    # Resolve paths against the actual repository root, also for subprocesses.
    for key in ("runs_dir", "data_root"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if cfg["batch_size"] != 128 or cfg["T"] != 1000 or cfg["num_classes"] != 10 or cfg["img_size"] != 28 or cfg["in_channels"] != 1:
        raise ExperimentError("Expected default Fashion-MNIST full training (T=1000, batch=128, 10 classes, 1x28x28); check original config.py.")
    if args.guidance_scale > 0 and cfg["cond_drop_prob"] == 0 and not args.tuning_only:
        raise ExperimentError("Positive CFG requires cond_drop_prob > 0.")
    return cfg


def preflight(cfg: ConfigDict, dry: bool) -> None:
    """Check project files and CUDA; dry-run never launches a child process."""
    for name in ("config.py", "train.py", "sample.py", "models/unet.py", "models/embeddings.py", "diffusion/schedule.py", "diffusion/ddpm.py", "data/dataset.py"):
        if not (ROOT / name).is_file():
            if dry:
                print(f"[DRY-RUN WARNING] Missing {name}")
            else:
                raise ExperimentError(f"Required project file missing: {name}")
    if not dry:
        import torch
        device = torch.device(cfg["device"])
        if device.type not in ("cpu", "cuda"):
            raise ExperimentError("Only CPU and CUDA are supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ExperimentError("CUDA requested but not available; no CPU fallback")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ExperimentError(f"CUDA device {index} not available")
            print(f"[GPU] {torch.cuda.get_device_name(index)}", flush=True)


def read_json(path: Path) -> ConfigDict:
    """Read a JSON object; refuse malformed or non-object inputs."""
    with path.open(encoding="utf-8-sig") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ExperimentError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, data: Any) -> None:
    """Atomically update a report owned by this invocation."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write report rows atomically, preserving first-row column order."""
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def expected_config(base: ConfigDict, name: str, epochs: int | str, lr: float | str, batch: int | str) -> ConfigDict:
    """Describe exactly the configuration a child must save."""
    return dict(base, run_name=name, epochs=epochs, lr=lr, batch_size=batch)


def compare_configs(actual: ConfigDict, expected: ConfigDict, ignore: set[str] | None = None) -> None:
    """Reject differences, normalizing relative data and output paths."""
    ignore = ignore or set()
    differences: list[str] = []
    for key in sorted((actual.keys() | expected.keys()) - ignore):
        if key not in actual or key not in expected:
            differences.append(key)
            continue
        a, b = actual[key], expected[key]
        if key in ("runs_dir", "data_root"):
            a, b = (ROOT / str(a)).resolve(), (ROOT / str(b)).resolve()
        if a != b:
            differences.append(key)
    if differences:
        raise ExperimentError("Config mismatch: " + ", ".join(differences))


def read_metrics(path: Path, epochs: int, lr: float) -> list[dict[str, Any]]:
    """Require exactly the five contract columns and complete finite epochs."""
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ExperimentError(f"Invalid CSV columns: {path}")
        for i, row in enumerate(reader, start=1):
            if None in row or any(row.get(k) is None for k in METRIC_FIELDS):
                raise ExperimentError(f"Malformed CSV row {i}: {path}")
            try:
                epoch = int(row["epoch"])
                vals = {k: float(row[k]) for k in METRIC_FIELDS[1:]}
            except (ValueError, TypeError) as exc:
                raise ExperimentError(f"Invalid number in row {i}: {path}") from exc
            if epoch != i or epoch > epochs:
                raise ExperimentError(f"Invalid epoch order or count: {path}")
            if not all(math.isfinite(v) for v in vals.values()):
                raise ExperimentError(f"NaN/Inf in epoch {epoch}: {path}")
            if vals["train_loss"] < 0 or vals["val_loss"] < 0 or vals["epoch_time_s"] <= 0:
                raise ExperimentError(f"Negative loss or nonpositive time in epoch {epoch}: {path}")
            if not math.isclose(vals["lr"], lr, rel_tol=1e-9, abs_tol=0):
                raise ExperimentError(f"Unexpected learning rate in epoch {epoch}: {path}")
            result.append(dict(epoch=epoch, **vals))
    if len(result) != epochs:
        raise ExperimentError(f"Incomplete metrics: expected {epochs}, got {len(result)}: {path}")
    return result


def grid_epochs(epochs: int) -> list[int]:
    """Monitoring schedule in the existing train.py."""
    return [i for i in range(1, epochs + 1) if i % SAMPLE_EVERY == 0 or i == epochs]


def check_png(path: Path, image_size: int) -> None:
    """Check PNG header, expected grid dimensions, and nontrivial file size."""
    with path.open("rb") as f:
        header = f.read(24)
    size = 10 * (image_size + 2) + 2
    if len(header) != 24 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or struct.unpack(">II", header[16:24]) != (size, size) or path.stat().st_size <= 33:
        raise ExperimentError(f"Invalid PNG grid: {path}")


def validate_run(directory: Path, expected: ConfigDict, *, timing: bool = False) -> list[dict[str, Any]]:
    """Check saved config, CSV, checkpoints and every required monitoring PNG.

    This checks checkpoint presence/size, not tensor compatibility; sample.py
    performs strict loading. External runs cannot prove dataset provenance.
    """
    actual = read_json(directory / "config.json")
    compare_configs(actual, expected, {"run_name", "runs_dir"} if timing else None)
    if timing and "smoke" in str(actual.get("run_name", "")).lower():
        raise ExperimentError("Smoke test cannot be used for timing")
    rows = read_metrics(directory / "metrics.csv", expected["epochs"], expected["lr"])
    for name in ("ckpt_last.pt", "ckpt_ema.pt"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ExperimentError(f"Missing/empty checkpoint: {path}")
    for e in grid_epochs(expected["epochs"]):
        check_png(directory / f"samples_epoch_{e:03d}.png", expected["img_size"])
    return rows


def summarize_run(name: str, cfg: ConfigDict, rows: list[dict[str, Any]], wall: float | None) -> dict[str, Any]:
    """Compute comparable numerical statistics without judging image quality."""
    v = [r["val_loss"] for r in rows]
    tr = [r["train_loss"] for r in rows]
    n = len(v)
    center = (n + 1) / 2
    denom = sum((i - center) ** 2 for i in range(1, n + 1))
    slope = sum((i - center) * value for i, value in enumerate(v, 1)) / denom if denom else None
    seconds = sum(r["epoch_time_s"] for r in rows)
    result = dict(run_name=name, lr=cfg["lr"], batch_size=cfg["batch_size"], epochs=n,
                  final_train_loss=tr[-1], final_val_loss=v[-1], min_val_loss=min(v),
                  best_val_epoch=v.index(min(v)) + 1, mean_val_loss=sum(v) / n,
                  train_loss_change=tr[-1] - tr[0], val_loss_change=v[-1] - v[0],
                  val_loss_slope=slope, total_epoch_time_s=seconds, mean_epoch_time_s=seconds/n,
                  wall_time_s=wall, has_nonfinite_loss=False, status="valid")
    if any(isinstance(x, float) and not math.isfinite(x) for x in result.values()):
        raise ExperimentError(f"Nonfinite aggregate: {name}")
    return result


def select_best(runs: list[dict[str, Any]], names: Sequence[str], preferred: str) -> dict[str, Any]:
    """Select by final val loss, min val loss, preferred baseline, fixed order."""
    if len(runs) != len(names) or {r["run_name"] for r in runs} != set(names) or len({r["epochs"] for r in runs}) != 1:
        raise ExperimentError("Selection requires every candidate with equal epoch counts")
    for r in runs:
        if r["status"] != "valid" or r["has_nonfinite_loss"] or not all(math.isfinite(r[k]) and r[k] >= 0 for k in ("final_val_loss", "min_val_loss")):
            raise ExperimentError("Cannot select from invalid results")
    return min(runs, key=lambda r: (r["final_val_loss"], r["min_val_loss"], r["run_name"] != preferred, names.index(r["run_name"])))


@dataclass(frozen=True)
class Timing:
    """Planning model in seconds; assumptions are explicitly recorded."""
    epoch: float
    monitor: float
    startup: float
    per_epoch_overhead: float
    final_sampling: float
    source: str
    assumptions: tuple[str, ...] = ()

    def cost(self, epochs: int) -> float:
        """Include monitoring at every fifth and final epoch."""
        return self.startup + epochs * (self.epoch + self.per_epoch_overhead) + len(grid_epochs(epochs)) * self.monitor

    def scaled(self, factor: float) -> Timing:
        """Use an explicitly heuristic factor for unmeasured batch sizes."""
        return Timing(self.epoch*factor, self.monitor*factor, self.startup, self.per_epoch_overhead,
                      self.final_sampling, self.source, self.assumptions + (f"Unmeasured batch cost multiplied by {factor:g}",))


SAMPLE_RE = re.compile(r"sampling \+ save time=([0-9]+(?:\.[0-9]+)?) s")


def log_times(path: Path | None) -> list[float]:
    """Extract monitoring times printed by the supplied train.py."""
    if path is None or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SELF-TEST" in text or "STUB DATASET" in text:
        raise ExperimentError(f"Self-test log is not a full training measurement: {path}")
    return [float(x) for x in SAMPLE_RE.findall(text)]


def make_timing(rows: list[dict[str, Any]], log: Path | None, wall: float | None, source: str, guidance: float) -> Timing:
    """Estimate with measured epoch times and logged monitoring where available."""
    epoch = max(r["epoch_time_s"] for r in rows)
    times = log_times(log)
    notes: list[str] = []
    if len(times) == len(grid_epochs(len(rows))) and times and all(t > 0 and math.isfinite(t) for t in times):
        monitor = max(times)
    else:
        monitor = epoch
        notes.append("Missing complete monitoring times: assume one epoch duration per grid; not a measured bound")
    if wall is not None and times:
        residual = max(0.0, wall - sum(r["epoch_time_s"] for r in rows) - sum(times))
        # Count the whole observed residual as startup plus a per-epoch allowance.
        startup = max(FALLBACK_OVERHEAD, residual)
    else:
        startup = FALLBACK_OVERHEAD
    notes.append("Startup and per-epoch checkpoint overhead estimated conservatively; not a guarantee")
    final = FINAL_SAMPLE_FACTOR * monitor * (2 if guidance > 0 else 1)
    notes.append(f"Final sample.py time estimated as {FINAL_SAMPLE_FACTOR:g}x monitoring grid, doubled for positive CFG; not measured")
    return Timing(epoch, monitor, startup, FALLBACK_OVERHEAD, final, source, tuple(notes))


class Budget:
    """Track actual elapsed time, reserve a safety margin, never kill for time."""
    def __init__(self, hours: float | None, margin: float, start: float | None = None) -> None:
        self.start = time.monotonic() if start is None else start
        self.usable = math.inf if hours is None else hours * 3600 * (1 - margin)

    def remaining(self) -> float:
        return self.usable - (time.monotonic() - self.start)

    def require(self, cost: float, stage: str) -> None:
        if cost > self.remaining():
            raise BudgetError(f"{stage}: estimated {cost:.1f}s exceeds remaining usable time {max(0, self.remaining()):.1f}s. Completed runs preserved.")


def choose_tuning(t: Timing, remaining: float, args: argparse.Namespace) -> int:
    """Fix equal LR epochs before launching the first candidate."""
    reserve = 0.0 if args.tuning_only else t.cost(args.final_epochs or MIN_FINAL) + t.final_sampling
    candidates = [args.tuning_epochs] if args.tuning_epochs is not None else list(range(TARGET_TUNING, MIN_TUNING - 1, -1))
    for e in candidates:
        if 3 * t.cost(e) + reserve <= remaining:
            return e
    raise BudgetError("Budget cannot fit all three LR runs plus required final reserve. Increase time, choose explicit fewer epochs, or use --tuning_only.")


def choose_final(t: Timing, remaining: float, args: argparse.Namespace, default_epochs: int) -> int:
    """Choose the largest feasible final epoch count, or honor explicit count."""
    candidates = [args.final_epochs] if args.final_epochs is not None else list(range(default_epochs, MIN_FINAL - 1, -1))
    for e in candidates:
        if t.cost(e) + t.final_sampling <= remaining:
            return e
    raise BudgetError("Insufficient time for final training plus final sampling; tuning results preserved")


def train_command(snapshot: Path, cfg: ConfigDict) -> list[str]:
    """Use only the actual Config CLI accepted by train.py."""
    return [sys.executable, "-u", "-m", "train", "--from_json", str(snapshot), "--epochs", str(cfg["epochs"]), "--lr", str(cfg["lr"]), "--batch_size", str(cfg["batch_size"]), "--run_name", cfg["run_name"]]


def sample_command(directory: Path, guidance: float, device: str) -> list[str]:
    """Use the actual sample.py CLI and EMA checkpoint from the same run."""
    return [sys.executable, "-u", "-m", "sample", "--from_json", str(directory/"config.json"), "--ckpt", str(directory/"ckpt_ema.pt"), "--guidance_scale", str(guidance), "--device", device]


def display_command(cmd: Sequence[str]) -> str:
    """Readable commands for Windows and POSIX; never executed through a shell."""
    return subprocess.list2cmdline(list(cmd)) if os.name == "nt" else shlex.join(cmd)


def run_subprocess(cmd: list[str], log: Path, cwd: Path = ROOT) -> float:
    """Stream child output to console/log and propagate failure; no shell."""
    print(f"\n[COMMAND] {display_command(cmd)}\n[LOG] {log}", flush=True)
    start = time.monotonic()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with log.open("x", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            assert p.stdout is not None
            for line in p.stdout:
                f.write(line)
                f.flush()
                try:
                    print(line, end="", flush=True)
                except UnicodeEncodeError:
                    print(line.encode("ascii", "replace").decode(), end="", flush=True)
            code = p.wait()
        except BaseException:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise
        finally:
            if p.stdout is not None:
                p.stdout.close()
    elapsed = time.monotonic() - start
    if code != 0:
        raise ExperimentError(f"Subprocess failed (exit code {code}, {elapsed:.1f}s). Inspect {log}. No further required training started.")
    return elapsed


def protect_directory(path: Path) -> None:
    """Never overwrite existing nonempty runs or non-directory paths."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ExperimentError(f"Existing output protected: {path}. Use another --runs_dir or explicit --reuse_tuning for complete LR results.")


class Reports:
    """Manage unique reports, progress, and provenance for this invocation."""
    def __init__(self, runs: Path, base: ConfigDict, args: argparse.Namespace) -> None:
        folder = runs / "_experiments"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S_") , dir=folder))
        (self.path / "logs").mkdir()
        self.snapshot = self.path / "base_config.json"
        write_json(self.snapshot, base)
        self.plan: dict[str, Any] = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "commands": []}
        self.summary: dict[str, Any] = {"status": "running", "stages": {}, "selection_metric": "final_val_loss", "interpretation": "Noise MSE on fixed validation inputs; not image quality or class accuracy; single seed; test set used for tuning per contract."}
        self.history: list[dict[str, Any]] = []
        self.flush()

    def flush(self) -> None:
        write_json(self.path / "experiment_plan.json", self.plan)
        write_json(self.path / "experiment_summary.json", self.summary)

    def stage(self, name: str, **data: Any) -> None:
        self.summary["stages"].setdefault(name, {}).update(data)
        self.flush()

    def command(self, name: str, cmd: list[str]) -> float:
        self.plan["commands"].append({"stage": name, "argv": cmd})
        self.stage(name, status="running")
        log = self.path / "logs" / f"{name}.log"
        try:
            wall = run_subprocess(cmd, log)
        except BaseException as exc:
            self.stage(name, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), log=str(log))
            raise
        self.stage(name, status="child_completed", wall_time_s=wall, log=str(log))
        return wall

    def training(self, cfg: ConfigDict, reuse: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        name = cfg["run_name"]
        directory = Path(cfg["runs_dir"]) / name
        wall = None
        try:
            if not reuse:
                protect_directory(directory)
                wall = self.command(name, train_command(self.snapshot, cfg))
            rows = validate_run(directory, cfg)
            result = summarize_run(name, cfg, rows, wall)
        except BaseException as exc:
            self.stage(name, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "invalid_or_failed", error=str(exc))
            raise
        self.stage(name, status="valid", reused=reuse, metrics=result)
        self.history.extend(dict(run_name=name, batch_size=cfg["batch_size"], **row) for row in rows)
        write_csv(self.path / "loss_history.csv", self.history)
        print(f"[PASS] {name}: final val_loss={result['final_val_loss']:.8g}", flush=True)
        return result, rows

    def comparison(self, kind: str, results: list[dict[str, Any]], best: dict[str, Any]) -> None:
        reason = "Lowest final validation loss at equal epochs; ties: minimum validation loss, preferred baseline, fixed candidate order."
        write_csv(self.path / f"{kind}_comparison.csv", results)
        write_json(self.path / f"{kind}_comparison.json", {"candidates": results, "selected": best, "reason": reason})
        self.summary[f"{kind}_selection"] = {"selected": best, "reason": reason}
        self.flush()


def get_existing_timing(base: ConfigDict, args: argparse.Namespace) -> Timing:
    """Read a matching existing timing run, rejecting the prior smoke-test files."""
    directory = (ROOT / args.timing_run).resolve() if args.timing_run else Path(base["runs_dir"]) / "timing_test"
    cfg = expected_config(base, "timing_test", 1, base["lr"], base["batch_size"])
    rows = validate_run(directory, cfg, timing=True)
    log = (ROOT / args.timing_log).resolve() if args.timing_log else directory / "training.log"
    if args.timing_log and not log.is_file():
        raise ExperimentError(f"Explicit timing log missing: {log}")
    timing = make_timing(rows, log, None, str(directory), args.guidance_scale)
    return Timing(timing.epoch, timing.monitor, timing.startup, timing.per_epoch_overhead, timing.final_sampling, timing.source,
                  timing.assumptions + ("Existing timing run: config/artifacts checked; actual dataset/hardware provenance requires user confirmation",))


def reuse_epochs(base: ConfigDict, requested: int | None) -> int:
    """Infer equal LR epoch count from existing runs and validate all before reuse."""
    root = Path(base["runs_dir"])
    counts = [read_json(root / n / "config.json")["epochs"] for n, _ in LR_CANDIDATES]
    if any(type(e) is not int or e <= 0 for e in counts) or len(set(counts)) != 1 or (requested is not None and counts[0] != requested):
        raise ExperimentError("Existing LR runs do not have the same requested epoch count")
    for name, lr in LR_CANDIDATES:
        validate_run(root / name, expected_config(base, name, counts[0], lr, base["batch_size"]))
    return counts[0]


def dry_run(base: ConfigDict, args: argparse.Namespace, timing: Timing | None) -> None:
    """Read-only plan; placeholders represent choices requiring real results."""
    runs = Path(base["runs_dir"])
    snapshot = runs / "_experiments" / "<new_report>" / "base_config.json"
    budget = Budget(args.budget_hours, args.safety_margin)
    e: int | str = args.tuning_epochs or "<auto_tuning_epochs>"
    if args.reuse_tuning:
        e = reuse_epochs(base, args.tuning_epochs)
        print(f"[DRY-RUN] Reuse validated LR runs ({e} epochs); no LR training commands.")
    elif timing and args.budget_hours is not None:
        e = choose_tuning(timing, budget.remaining(), args)
    if timing is None and args.auto_timing:
        name = available_timing_name(runs)
        print(display_command(train_command(snapshot, expected_config(base, name, 1, base["lr"], base["batch_size"]))))
    elif timing is None and args.budget_hours is not None:
        print("[DRY-RUN WARNING] Actual execution requires valid timing or --auto_timing.")
    if timing:
        print(f"[TIMING] epoch={timing.epoch:.3f}s grid={timing.monitor:.3f}s source={timing.source}")
        for note in timing.assumptions:
            print(f"[ASSUMPTION] {note}")
    if not args.reuse_tuning:
        for name, lr in LR_CANDIDATES:
            protect_directory(runs / name)
            print(display_command(train_command(snapshot, expected_config(base, name, e, lr, base["batch_size"]))))
    print("[ANALYSIS] All three valid LR runs required; select by final val_loss.")
    if args.batch_tuning:
        print("[OPTIONAL] Entire batch stage only if it fits budget; otherwise skip.")
        for b in BATCH_CANDIDATES:
            print(display_command(train_command(snapshot, expected_config(base, f"batch{b}", args.batch_tuning_epochs or e, "<best_lr>", b))))
    if not args.tuning_only:
        protect_directory(runs / "final")
        print(display_command(train_command(snapshot, expected_config(base, "final", args.final_epochs or "<auto_final_epochs>", "<best_lr>", "<best_batch>" if args.batch_tuning else base["batch_size"]))))
        print(display_command(sample_command(runs / "final", args.guidance_scale, base["device"])))
    print("[DRY-RUN COMPLETE] No experiment files created, no subprocesses started. Placeholders are not executable commands.")


def available_timing_name(root: Path) -> str:
    """Choose a new calibration run without overwriting an invalid prior run."""
    name = "timing_test"
    i = 1
    while (root / name).exists():
        name = f"timing_test_{i:03d}"
        i += 1
    return name


def update_timing(old: Timing, rows: list[dict[str, Any]], log: Path | None, wall: float | None, guidance: float, source: str) -> Timing:
    """Use the largest observed same-batch costs to refresh later plans."""
    new = make_timing(rows, log, wall, source, guidance)
    return Timing(max(old.epoch, new.epoch), max(old.monitor, new.monitor), max(old.startup, new.startup), max(old.per_epoch_overhead, new.per_epoch_overhead), max(old.final_sampling, new.final_sampling), source, tuple(dict.fromkeys(old.assumptions + new.assumptions)))


def execute(base: ConfigDict, args: argparse.Namespace, timing: Timing | None, budget: Budget, report: Reports) -> None:
    """Run mandatory stages sequentially, respecting a single soft time budget."""
    root = Path(base["runs_dir"])
    batch = base["batch_size"]
    # Protect mandatory run names before expensive work starts.
    if args.reuse_tuning:
        epochs = reuse_epochs(base, args.tuning_epochs)
    else:
        for name, _ in LR_CANDIDATES:
            protect_directory(root / name)
        epochs = args.tuning_epochs
    if not args.tuning_only:
        protect_directory(root / "final")
    if timing is None and args.auto_timing:
        if budget.remaining() <= 0:
            raise BudgetError("No usable budget remains for calibration")
        name = available_timing_name(root)
        print("[TIMING] Full one-epoch calibration. Duration unknown; soft budget cannot guarantee it fits.", flush=True)
        result, rows = report.training(expected_config(base, name, 1, base["lr"], batch))
        timing = make_timing(rows, report.path/"logs"/f"{name}.log", result["wall_time_s"], str(root/name), args.guidance_scale)
    if timing is None and args.budget_hours is not None:
        raise ExperimentError("A budgeted run needs valid timing. Use --auto_timing or supply --timing_run.")
    if timing:
        report.plan["timing_initial"] = timing.__dict__
        print(f"[TIMING] epoch={timing.epoch:.3f}s; monitoring={timing.monitor:.3f}s; final sampling estimate={timing.final_sampling:.3f}s")
        for note in timing.assumptions:
            print(f"[ASSUMPTION] {note}")
    if not args.reuse_tuning and args.budget_hours is not None:
        assert timing is not None
        epochs = choose_tuning(timing, budget.remaining(), args)
    assert epochs is not None
    report.plan["tuning_epochs"] = epochs
    report.flush()
    lr_results: list[dict[str, Any]] = []
    for i, (name, lr) in enumerate(LR_CANDIDATES, 1):
        print(f"\n[E2 {i}/3] {name} | lr={lr:g} | epochs={epochs}", flush=True)
        if timing and not args.reuse_tuning:
            reserve = 0.0 if args.tuning_only else timing.cost(args.final_epochs or MIN_FINAL) + timing.final_sampling
            budget.require((4-i)*timing.cost(epochs)+reserve, f"Remaining LR runs ({name}) + final reserve")
        result, rows = report.training(expected_config(base, name, epochs, lr, batch), reuse=args.reuse_tuning)
        lr_results.append(result)
        write_csv(report.path / "lr_comparison.csv", lr_results)
        if timing:
            if args.reuse_tuning:
                # Reused CSV has measured epoch time but no trusted child wall/grid time.
                timing = Timing(max(timing.epoch, result["mean_epoch_time_s"], max(r["epoch_time_s"] for r in rows)), timing.monitor, timing.startup, timing.per_epoch_overhead, timing.final_sampling, timing.source, timing.assumptions)
            else:
                timing = update_timing(timing, rows, report.path/"logs"/f"{name}.log", result["wall_time_s"], args.guidance_scale, str(root/name))
    best = select_best(lr_results, [n for n, _ in LR_CANDIDATES], "baseline")
    report.comparison("lr", lr_results, best)
    best_lr = best["lr"]
    print(f"[E3] Selected LR={best_lr:g} by final validation loss (not visual quality).", flush=True)
    # Preserve useful selection immediately, even if later budget/stage fails.
    write_json(report.path/"best_config.json", expected_config(base, "final", args.final_epochs or base["epochs"], best_lr, batch))
    report.summary["best_config_status"] = "provisional; final epochs not allocated"
    report.flush()
    if args.batch_tuning:
        be = args.batch_tuning_epochs or epochs
        estimates = {b: timing if b == batch else timing.scaled(UNKNOWN_BATCH_FACTOR) for b in BATCH_CANDIDATES} if timing else {}
        if timing:
            reserve = 0.0 if args.tuning_only else max(t.cost(args.final_epochs or MIN_FINAL)+t.final_sampling for t in estimates.values())
            needed = sum(t.cost(be) for t in estimates.values()) + reserve
        else:
            needed = reserve = 0.0
        if needed > budget.remaining():
            report.stage("batch_tuning", status="skipped_budget", estimated_seconds=needed)
            print("[E4 SKIP] Not enough estimated time for all batches plus final reserve.")
        else:
            for b in BATCH_CANDIDATES:
                protect_directory(root / f"batch{b}")
            results: list[dict[str, Any]] = []
            measured: dict[int, Timing] = {}
            for j, b in enumerate(BATCH_CANDIDATES):
                if timing:
                    budget.require(sum(estimates[k].cost(be) for k in BATCH_CANDIDATES[j:])+reserve, "Remaining batch tuning + final reserve")
                name = f"batch{b}"
                print(f"[E4 {j+1}/3] batch={b} | epochs={be}", flush=True)
                result, rows = report.training(expected_config(base, name, be, best_lr, b))
                results.append(result)
                write_csv(report.path / "batch_comparison.csv", results)
                if timing:
                    measured[b] = make_timing(rows, report.path/"logs"/f"{name}.log", result["wall_time_s"], str(root/name), args.guidance_scale)
                    if not args.tuning_only:
                        reserve = max(reserve, measured[b].cost(args.final_epochs or MIN_FINAL)+measured[b].final_sampling)
            chosen = select_best(results, [f"batch{b}" for b in BATCH_CANDIDATES], f"batch{batch}")
            report.comparison("batch", results, chosen)
            batch = chosen["batch_size"]
            if timing:
                # Keep baseline final sampling estimate: sample.py caps batch at 32.
                new = measured[batch]
                timing = Timing(new.epoch, new.monitor, new.startup, new.per_epoch_overhead, max(new.final_sampling, timing.final_sampling), new.source, new.assumptions)
            report.stage("batch_tuning", status="valid", selected_batch=batch)
    else:
        report.stage("batch_tuning", status="disabled")
    if args.tuning_only:
        write_json(report.path/"best_config.json", expected_config(base, "final", args.final_epochs or base["epochs"], best_lr, batch))
        report.summary["best_config_status"] = "provisional recommendation; no final training; epochs default or explicit, not budget-allocated"
        report.summary["status"] = "tuning_complete"
        report.flush()
        return
    final_epochs = args.final_epochs
    if args.budget_hours is not None:
        assert timing is not None
        final_epochs = choose_final(timing, budget.remaining(), args, base["epochs"])
        budget.require(timing.cost(final_epochs) + timing.final_sampling, "Final training and sampling")
    assert final_epochs is not None
    final = expected_config(base, "final", final_epochs, best_lr, batch)
    write_json(report.path/"best_config.json", final)
    report.summary["best_config_status"] = "planned final configuration"
    report.plan["final_epochs"] = final_epochs
    if timing:
        report.plan["timing_final"] = timing.__dict__
    report.flush()
    print(f"[E5] Final training: lr={best_lr:g}, batch={batch}, epochs={final_epochs}", flush=True)
    final_result, final_rows = report.training(final)
    if timing:
        timing = update_timing(timing, final_rows, report.path/"logs"/"final.log", final_result["wall_time_s"], args.guidance_scale, str(root/"final"))
    report.summary["best_config_status"] = "validated against final/config.json"
    report.flush()
    if timing:
        budget.require(timing.final_sampling, "Final sampling")
    directory = root / "final"
    before = set(directory.glob("samples_final_*.png"))
    print("[E6] Final sampling with EMA checkpoint", flush=True)
    report.command("sampling", sample_command(directory, args.guidance_scale, base["device"]))
    try:
        created = set(directory.glob("samples_final_*.png")) - before
        if len(created) != 1:
            raise ExperimentError(f"Expected exactly one new final PNG, found {len(created)}")
        output = created.pop()
        check_png(output, base["img_size"])
    except Exception as exc:
        report.stage("sampling", status="invalid", error=str(exc))
        raise
    report.stage("sampling", status="valid", output=str(output))
    report.summary["status"] = "completed"
    report.flush()


def run_self_test() -> None:
    """Exercise real orchestration with temporary synthetic artifacts, no ML.

    The fixture state_dict files are intentionally not real model weights. No
    fixture is written to runs/ and no train.py or sample.py process is launched.
    """
    from unittest.mock import patch
    from contextlib import redirect_stdout
    from io import StringIO
    import zlib

    def expect_error(action: Any, kind: type[BaseException] = ExperimentError) -> None:
        try:
            action()
        except kind:
            return
        raise AssertionError(f"Expected {kind.__name__}")

    def png(path: Path) -> None:
        """Create a valid tiny-data 302x302 grayscale test PNG."""
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        raw = (b"\x00" + bytes(302)) * 302
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 302, 302, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    def fixture(cfg: ConfigDict, score: float = .2) -> Path:
        """Write labeled synthetic contract artifacts for logic tests only."""
        path = Path(cfg["runs_dir"]) / cfg["run_name"]
        path.mkdir(parents=True, exist_ok=True)
        write_json(path/"config.json", cfg)
        write_csv(path/"metrics.csv", [dict(epoch=i, train_loss=score+.1/i, val_loss=score+.05/i, lr=cfg["lr"], epoch_time_s=10.) for i in range(1,cfg["epochs"]+1)])
        for ckpt in ("ckpt_last.pt", "ckpt_ema.pt"):
            (path/ckpt).write_bytes(b"SYNTHETIC FIXTURE - NOT MODEL WEIGHTS")
        for e in grid_epochs(cfg["epochs"]):
            png(path/f"samples_epoch_{e:03d}.png")
        return path

    with tempfile.TemporaryDirectory(prefix="ddpm_orchestrator_SELF_TEST_") as temp:
        root = Path(temp)
        # Fixture metadata, not a substitute for config.py in production.
        base: ConfigDict = dict(runs_dir=str(root/"runs"), data_root=str(root/"data"), run_name="baseline", epochs=20, T=1000, batch_size=128, lr=2e-4, num_workers=2, img_size=28, in_channels=1, num_classes=10, seed=42, device="cuda", cond_drop_prob=.1, schedule="linear", ch_mult=[1,2,2])
        args = parse_args(["--budget_hours", "2", "--auto_timing"])
        assert not args.batch_tuning
        assert grid_epochs(1) == [1] and grid_epochs(5) == [5] and grid_epochs(11) == [5,10,11]
        t = Timing(100,20,30,5,60,"synthetic")
        assert t.cost(6) == 30+6*105+40
        assert choose_tuning(t, 6120, args) == 5
        assert choose_final(t, 6120, args, 20) == 20
        assert choose_tuning(t, 1800, parse_args(["--budget_hours","2","--tuning_only"])) == 5
        expect_error(lambda: choose_tuning(t, 100, args), BudgetError)
        expect_error(lambda: choose_final(t, 100, args, 20), BudgetError)
        smoke = fixture(dict(base, run_name="smoke_test_cuda", T=3, batch_size=8, num_workers=0, epochs=1))
        expect_error(lambda: validate_run(smoke, expected_config(base,"timing_test",1,base["lr"],128),timing=True))
        print("[PASS] Timing rejects smoke test; equal epochs, monitoring counts and budget planning")
        a = expected_config(base, "baseline", 3, 2e-4, 128)
        path = fixture(a)
        rows = validate_run(path, a)
        assert len(rows) == 3
        expect_error(lambda: compare_configs(dict(a, seed=99), a))
        for bad in (float("nan"), float("inf")):
            invalid = [dict(row) for row in rows]
            invalid[-1]["val_loss"] = bad
            write_csv(path/"metrics.csv", invalid)
            expect_error(lambda: read_metrics(path/"metrics.csv", 3, a["lr"]))
        write_csv(path/"metrics.csv", rows[:-1])
        expect_error(lambda: read_metrics(path/"metrics.csv",3,a["lr"]))
        write_csv(path/"metrics.csv",rows)
        expect_error(lambda: protect_directory(path))
        summary = summarize_run("baseline",a,rows,40.)
        second = dict(summary,run_name="lr_high",final_val_loss=.1,min_val_loss=.1)
        third = dict(summary,run_name="lr_low",final_val_loss=.3,min_val_loss=.05)
        names = [n for n,_ in LR_CANDIDATES]
        assert select_best([summary,second,third],names,"baseline")["run_name"] == "lr_high"
        assert select_best([summary,dict(summary,run_name="lr_high"),dict(summary,run_name="lr_low")],names,"baseline")["run_name"] == "baseline"
        expect_error(lambda: select_best([summary,second],names,"baseline"))
        expect_error(lambda: select_best([summary,dict(second,epochs=2),third],names,"baseline"))
        print("[PASS] Strict config/CSV validation, nonfinite/incomplete rejection, selection and tie-break")
        cmd = train_command(root/"base.json",a)
        assert cmd[:4] == [sys.executable,"-u","-m","train"] and "--self_test" not in cmd and "--ckpt" not in cmd
        assert sample_command(path,0.,"cuda")[:4] == [sys.executable,"-u","-m","sample"]
        run_subprocess([sys.executable,"-c","print('synthetic subprocess OK')"],root/"ok.log",root)
        expect_error(lambda: run_subprocess([sys.executable,"-c","import sys; sys.exit(7)"],root/"fail.log",root))
        print("[PASS] Real harmless subprocess success and nonzero exit; no ML subprocess")

        def fake_child(cmd: list[str], log: Path, cwd: Path = ROOT) -> float:
            """Simulate train/sample output only inside this temporary test."""
            module = cmd[cmd.index("-m")+1]
            def arg(key: str) -> str:
                return cmd[cmd.index(key)+1]
            calls.append(module + (":"+arg("--run_name") if module == "train" else ""))
            if module == "train":
                cfg = read_json(Path(arg("--from_json")))
                cfg.update(epochs=int(arg("--epochs")),lr=float(arg("--lr")),batch_size=int(arg("--batch_size")),run_name=arg("--run_name"))
                if fail_name == cfg["run_name"]:
                    raise ExperimentError("SYNTHETIC subprocess failure")
                fixture(cfg, .1 if cfg["lr"] == 1e-3 else .2)
                log.write_text("\n".join("sampling + save time=2.000 s" for _ in grid_epochs(cfg["epochs"])),encoding="utf-8")
            elif module == "sample":
                directory = Path(arg("--from_json")).parent
                png(directory/"samples_final_ema_g0_seed42.png")
                log.write_text("SYNTHETIC SAMPLE",encoding="utf-8")
            else:
                raise AssertionError("Unexpected subprocess")
            return 40.

        for mode in ("full", "calibration", "tuning_only", "failed", "reuse", "batch", "batch_skip"):
            config = dict(base, runs_dir=str(root/mode))
            cli = ["--tuning_epochs","3","--final_epochs","5"]
            if mode in ("tuning_only","failed"):
                cli.append("--tuning_only" if mode == "tuning_only" else "--auto_timing")
            if mode == "reuse":
                cli.append("--reuse_tuning")
                for n,lr in LR_CANDIDATES:
                    fixture(expected_config(config,n,3,lr,128), .1 if lr==1e-3 else .2)
            if mode in ("batch","batch_skip"):
                cli.append("--batch_tuning")
            if mode == "batch_skip":
                cli += ["--budget_hours", "0.9"]
            if mode == "calibration":
                cli += ["--budget_hours", "2", "--auto_timing"]
            a2 = parse_args(cli)
            rep = Reports(Path(config["runs_dir"]),config,a2)
            calls: list[str] = []
            fail_name = "lr_high" if mode == "failed" else ""
            with patch(__name__+".run_subprocess",side_effect=fake_child), redirect_stdout(StringIO()):
                if mode == "failed":
                    expect_error(lambda: execute(config,a2,t,Budget(None,.15),rep))
                    assert calls == ["train:baseline","train:lr_high"]
                    assert "lr_selection" not in rep.summary
                    continue
                execute(config,a2,None if mode == "calibration" else t,Budget(a2.budget_hours,.15),rep)
            assert read_json(rep.path/"lr_comparison.json")["selected"]["lr"] == 1e-3
            if mode == "tuning_only":
                assert calls == ["train:baseline","train:lr_high","train:lr_low"]
                assert rep.summary["status"] == "tuning_complete"
                assert not (Path(config["runs_dir"])/"final").exists()
            elif mode == "reuse":
                assert calls == ["train:final","sample"]
            else:
                extra = ["train:batch64","train:batch128","train:batch256"] if mode == "batch" else []
                assert calls == (["train:timing_test"] if mode == "calibration" else []) + ["train:baseline","train:lr_high","train:lr_low"] + extra + ["train:final","sample"]
                assert rep.summary["status"] == "completed"
                if mode == "batch_skip":
                    assert rep.summary["stages"]["batch_tuning"]["status"] == "skipped_budget"
                if mode == "batch":
                    assert rep.summary["stages"]["batch_tuning"]["status"] == "valid"
            assert (rep.path/"loss_history.csv").is_file()
        print("[PASS] Synthetic end-to-end: auto timing, full, tuning-only, failure gate, reuse, optional batch and budget skip")
        dry_base = dict(base,runs_dir=str(root/"dry"))
        with patch(__name__+".run_subprocess",side_effect=AssertionError("Dry-run launched a process")), redirect_stdout(StringIO()):
            dry_run(dry_base, parse_args(["--budget_hours","2","--auto_timing","--dry_run"]),None)
        assert not Path(dry_base["runs_dir"]).exists()
        print("[PASS] Dry-run creates no files and launches no subprocess")
    print("ALL ORCHESTRATOR SELF-TESTS PASSED (synthetic only; no training or CUDA test)")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse, preflight, plan, execute and report failures."""
    started = time.monotonic()
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    report: Reports | None = None
    try:
        if args.dry_run:
            sys.dont_write_bytecode = True
        base = load_base_config(args)
        preflight(base,args.dry_run)
        timing: Timing | None = None
        if args.budget_hours is not None or args.auto_timing or args.timing_run is not None:
            try:
                timing = get_existing_timing(base,args)
            except (ExperimentError,OSError,ValueError,KeyError,TypeError) as exc:
                print(f"[TIMING UNAVAILABLE] {exc}",flush=True)
                if args.budget_hours is not None and not args.auto_timing and not args.dry_run:
                    raise ExperimentError("Supply a valid full timing run or use --auto_timing") from exc
        if args.dry_run:
            dry_run(base,args,timing)
            return 0
        budget = Budget(args.budget_hours,args.safety_margin,started)
        report = Reports(Path(base["runs_dir"]),base,args)
        print(f"[REPORT] {report.path}",flush=True)
        if args.budget_hours is not None:
            print(f"[BUDGET] {args.budget_hours:g}h total; {args.safety_margin:.0%} safety margin; soft planning limit.",flush=True)
        execute(base,args,timing,budget,report)
        report.summary["elapsed_seconds"] = time.monotonic()-started
        report.flush()
        print(f"[COMPLETE] {report.summary['status']} | {report.path}",flush=True)
        return 0
    except KeyboardInterrupt:
        if report:
            report.summary.update(status="interrupted",elapsed_seconds=time.monotonic()-started)
            report.flush()
        print("[INTERRUPTED] Existing outputs preserved. Check child workers before retrying.",file=sys.stderr)
        return 130
    except Exception as exc:
        if report:
            report.summary.update(status="stopped_budget" if isinstance(exc,BudgetError) else "failed",error=str(exc),elapsed_seconds=time.monotonic()-started)
            report.flush()
            print(f"[REPORT] {report.path}",file=sys.stderr)
        print(f"[ERROR] {exc}",file=sys.stderr,flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
