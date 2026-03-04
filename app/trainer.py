"""
Core training logic: dataset setup, subprocess management, log parsing.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .config import settings

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised when a cancel_event is set while a subprocess is running."""


def _cancel_watcher(cancel_event: threading.Event, proc: subprocess.Popen) -> None:
    """Block until cancel_event is set, then terminate the subprocess."""
    cancel_event.wait()
    if proc.poll() is None:
        logger.info("Cancellation received — terminating subprocess...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Regex patterns for nnUNet training log
# ---------------------------------------------------------------------------
_RE_EPOCH = re.compile(r'Epoch (\d+)')
_RE_LR = re.compile(r'Current learning rate:\s*([0-9.e+\-]+)')
_RE_TRAIN_LOSS = re.compile(r'train_loss\s+([0-9.e+\-]+)')
_RE_VAL_LOSS = re.compile(r'val_loss\s+([0-9.e+\-]+)')
_RE_DICE = re.compile(r'Pseudo dice \[([^\]]+)\]')
_RE_EPOCH_TIME = re.compile(r'Epoch time:\s*([0-9.]+)\s*s')
_RE_PARENS = re.compile(r'\(([^)]+)\)')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset_num(dataset_name: str) -> str:
    """Extract numeric ID string from 'Dataset###_Name' (e.g. '15' from 'Dataset015_Brain')."""
    m = re.search(r'Dataset(\d+)_', dataset_name)
    if not m:
        raise ValueError(f"Cannot extract dataset number from: {dataset_name!r}")
    return str(int(m.group(1)))  # strip leading zeros; nnUNet accepts plain int string


def get_scripts_dir() -> Path:
    if settings.SCRIPTS_DIR:
        return Path(settings.SCRIPTS_DIR)
    # Default: scripts/ directory next to the project root (parent of app/)
    return Path(__file__).parent.parent / "scripts"


def get_nnunet_env() -> dict:
    """Build environment dict for nnUNet subprocess calls."""
    data_dir = settings.DATA_DIR
    env = os.environ.copy()
    env["NNUNET_DATA_DIR"] = data_dir
    env["nnUNet_raw"] = str(Path(data_dir) / "raw")
    env["nnUNet_preprocessed"] = str(Path(data_dir) / "preprocessed")
    env["nnUNet_results"] = str(Path(data_dir) / "results")
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["CONDA_ENV"] = settings.CONDA_ENV
    env["CONDA_PROFILE"] = settings.CONDA_PROFILE
    env["DEVICE"] = settings.DEVICE
    env["NUM_GPUS"] = str(settings.NUM_GPUS)
    env["NP"] = str(settings.NUM_PREPROCESSING_WORKERS)
    return env


def get_fold_dir(dataset_name: str, configuration: str, fold: int) -> Path:
    return (
        Path(settings.DATA_DIR)
        / "results"
        / dataset_name
        / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
    )


def find_latest_training_log(dataset_name: str, configuration: str, fold: int) -> Optional[Path]:
    """Return the most recently modified training_log_*.txt in the fold directory, or None."""
    fold_dir = get_fold_dir(dataset_name, configuration, fold)
    logs = sorted(fold_dir.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def get_validation_summary_path(dataset_name: str, configuration: str, fold: int) -> Path:
    return (
        Path(settings.DATA_DIR)
        / "results"
        / dataset_name
        / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
        / "validation"
        / "summary.json"
    )


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------

def setup_dataset(zip_path: str, dataset_name: str) -> None:
    """
    Extract dataset ZIP into the nnUNet directory structure.

    ZIP structure (produced by nnunet_server upload):
        Dataset###_Name/imagesTr/...
        Dataset###_Name/labelsTr/...
        Dataset###_Name/dataset.json
        Dataset###_Name/dataset_fingerprint.json   ← goes to preprocessed/
        Dataset###_Name/nnUNetPlans.json            ← goes to preprocessed/
    """
    data_dir = Path(settings.DATA_DIR)
    raw_dest = data_dir / "raw"
    preprocessed_dest = data_dir / "preprocessed" / dataset_name

    raw_dest.mkdir(parents=True, exist_ok=True)
    preprocessed_dest.mkdir(parents=True, exist_ok=True)

    plan_files = {"dataset_fingerprint.json", "nnUNetPlans.json"}
    # dataset.json goes to both raw/ and preprocessed/ (nnUNet reads it from both)
    dual_files = {"dataset.json"}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if not basename:
                continue  # skip directory entries

            if basename in plan_files:
                dest = preprocessed_dest / basename
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info(f"  plan file: {basename} → {dest}")
            elif basename in dual_files:
                # Copy to raw/ (preserving path) and also to preprocessed/
                raw_file = raw_dest / member
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(raw_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                pre_file = preprocessed_dest / basename
                shutil.copy2(raw_file, pre_file)
                logger.info(f"  dual file: {basename} → raw/ + preprocessed/")
            else:
                # Preserve full path under raw/: raw/Dataset###_Name/imagesTr/...
                dest = raw_dest / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    logger.info(f"Dataset {dataset_name} extracted to {data_dir}")


def _read_num_training(dataset_name: str) -> int:
    """Read numTraining from dataset.json."""
    ds_json = Path(settings.DATA_DIR) / "raw" / dataset_name / "dataset.json"
    try:
        with open(ds_json) as f:
            return json.load(f).get("numTraining", 0)
    except Exception:
        return 0


def is_dataset_downloaded(dataset_id: str) -> bool:
    """Return True if the dataset ZIP is already on disk."""
    zip_path = Path(settings.DATA_DIR) / "downloads" / f"{dataset_id}.zip"
    return zip_path.exists() and zip_path.stat().st_size > 0


PREPROCESSING_FLAG = "preprocessing_completed.txt"


def is_preprocessing_done(dataset_name: str) -> bool:
    """Return True if the preprocessing completion flag file exists."""
    flag = Path(settings.DATA_DIR) / "preprocessed" / dataset_name / PREPROCESSING_FLAG
    return flag.exists()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def run_preprocess(
    job_id: str,
    dataset_name: str,
    progress_callback: Callable,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    Run nnUNetv2_preprocess via scripts/preprocess.sh.
    Reads stdout line-by-line; counts "Preprocessing case" lines for progress.
    Calls progress_callback(total_images, done_images, mean_time_s).
    """
    dataset_num = get_dataset_num(dataset_name)
    total_images = _read_num_training(dataset_name)
    logger.info(f"Preprocessing {dataset_name} (num={dataset_num}, total={total_images})")

    log_dir = Path(settings.DATA_DIR) / "logs" / job_id / "preprocess"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess.log"

    script = get_scripts_dir() / "preprocess.sh"
    env = get_nnunet_env()

    proc = subprocess.Popen(
        ["bash", str(script), dataset_num, str(log_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if cancel_event is not None:
        threading.Thread(target=_cancel_watcher, args=(cancel_event, proc), daemon=True).start()

    done_images = 0
    start_time = time.time()
    last_report = time.time()

    with open(log_file, "w") as lf:
        for line in proc.stdout:
            lf.write(line)
            lf.flush()

            # nnUNet prints "Preprocessing case_identifier" for each case
            if "Preprocessing case" in line or "preprocessing case" in line.lower():
                done_images += 1
                now = time.time()
                mean_s = (now - start_time) / done_images if done_images > 0 else None
                if now - last_report >= 10:
                    try:
                        progress_callback(total_images, done_images, mean_s)
                    except Exception as e:
                        logger.warning(f"Preprocessing progress callback failed: {e}")
                    last_report = now

    proc.wait()

    # Final progress report
    try:
        elapsed = time.time() - start_time
        mean_s = elapsed / done_images if done_images > 0 else None
        progress_callback(total_images, done_images or total_images, mean_s)
    except Exception:
        pass

    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Job was cancelled during preprocessing")

    if proc.returncode != 0:
        raise RuntimeError(f"Preprocessing failed (exit {proc.returncode}). See {log_file}")

    # Write flag file so future jobs skip preprocessing for this dataset
    flag = Path(settings.DATA_DIR) / "preprocessed" / dataset_name / PREPROCESSING_FLAG
    flag.write_text(f"Preprocessing completed for job {job_id}\n")
    logger.info(f"Preprocessing complete for {dataset_name}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_train_fold(
    job_id: str,
    dataset_name: str,
    configuration: str,
    fold: int,
    progress_callback: Callable,
    log_upload_callback: Callable,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    Run nnUNetv2_train for a single fold via scripts/train.sh.
    Monitors the nnUNet training_log.txt file in a background thread.
    Calls progress_callback(fold, epoch, lr, train_loss, val_loss, pseudo_dice, epoch_time_s).
    Calls log_upload_callback(fold, text) periodically.
    """
    logger.info(f"Training {dataset_name} {configuration} fold {fold}")

    script = get_scripts_dir() / "train.sh"
    env = get_nnunet_env()

    proc = subprocess.Popen(
        ["bash", str(script), dataset_name, configuration, str(fold)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if cancel_event is not None:
        threading.Thread(target=_cancel_watcher, args=(cancel_event, proc), daemon=True).start()

    stop_event = threading.Event()

    def monitor():
        last_reported_epoch = -1
        last_log_upload = time.time()

        while not stop_event.is_set():
            stop_event.wait(5)

            # Find the most recent training_log_*.txt (name includes datetime)
            log_path = find_latest_training_log(dataset_name, configuration, fold)
            if log_path is None:
                continue

            try:
                content = log_path.read_text(errors="replace")
            except Exception:
                continue

            # Parse all complete epoch blocks from the log
            epoch_data = _parse_all_epochs(content)

            # Report new epochs
            for ep_num in sorted(epoch_data.keys()):
                if ep_num > last_reported_epoch:
                    ep = epoch_data[ep_num]
                    try:
                        progress_callback(
                            fold=fold,
                            epoch=ep_num,
                            learning_rate=ep.get("learning_rate"),
                            train_loss=ep.get("train_loss"),
                            val_loss=ep.get("val_loss"),
                            pseudo_dice=ep.get("pseudo_dice"),
                            epoch_time_s=ep.get("epoch_time_s"),
                        )
                        last_reported_epoch = ep_num
                    except Exception as e:
                        logger.warning(f"Training progress callback failed: {e}")

            # Upload log every 60 seconds
            if time.time() - last_log_upload >= 60:
                try:
                    log_upload_callback(fold, content)
                    last_log_upload = time.time()
                except Exception as e:
                    logger.warning(f"Log upload failed: {e}")

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # Drain subprocess stdout to avoid pipe buffer deadlock
    for _ in proc.stdout:
        pass
    proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=10)

    # Final log upload using the most recent log file
    log_path = find_latest_training_log(dataset_name, configuration, fold)
    if log_path is not None:
        try:
            log_upload_callback(fold, log_path.read_text(errors="replace"))
        except Exception as e:
            logger.warning(f"Final log upload failed: {e}")

    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled(f"Job was cancelled during fold {fold} training")

    if proc.returncode != 0:
        raise RuntimeError(f"Training fold {fold} failed (exit {proc.returncode})")

    logger.info(f"Training fold {fold} complete")


def _parse_all_epochs(log_content: str) -> dict:
    """
    Parse all complete epoch blocks from nnUNet training log.
    Returns dict of {epoch_num: {epoch, learning_rate, train_loss, val_loss, pseudo_dice, epoch_time_s}}.
    An epoch block is complete when epoch_time_s is found.
    """
    epochs = {}
    current = {}

    for line in log_content.splitlines():
        m = _RE_EPOCH.search(line)
        if m:
            current = {"epoch": int(m.group(1))}
            continue
        if not current:
            continue
        m = _RE_LR.search(line)
        if m:
            current["learning_rate"] = float(m.group(1))
            continue
        m = _RE_TRAIN_LOSS.search(line)
        if m:
            current["train_loss"] = float(m.group(1))
            continue
        m = _RE_VAL_LOSS.search(line)
        if m:
            current["val_loss"] = float(m.group(1))
            continue
        m = _RE_DICE.search(line)
        if m:
            vals = [v.strip() for v in m.group(1).split(",")]
            try:
                # Handle both plain floats and np.float32(0.9277) format:
                # prefer the value inside parentheses, fall back to the raw string
                def _to_float(s):
                    pm = _RE_PARENS.search(s)
                    return float(pm.group(1).strip() if pm else s)
                current["pseudo_dice"] = json.dumps([_to_float(v) for v in vals])
            except Exception:
                current["pseudo_dice"] = json.dumps(vals)
            continue
        m = _RE_EPOCH_TIME.search(line)
        if m:
            current["epoch_time_s"] = float(m.group(1))
            ep_num = current.get("epoch", -1)
            if ep_num >= 0:
                epochs[ep_num] = dict(current)
            current = {}

    return epochs


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------

def read_validation_result(dataset_name: str, configuration: str, fold: int) -> Optional[str]:
    """Read fold validation summary.json and return as JSON string, or None if missing."""
    path = get_validation_summary_path(dataset_name, configuration, fold)
    if not path.exists():
        logger.warning(f"Validation summary not found: {path}")
        return None
    return path.read_text()


# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------

def export_model(dataset_name: str, configuration: str) -> Path:
    """
    Export trained model to ZIP using nnUNetv2_export_model_to_zip.
    Returns the path to the created ZIP file.
    """
    output_dir = Path(settings.DATA_DIR) / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / f"{dataset_name}_{configuration}.zip"

    if output_zip.exists():
        output_zip.unlink()

    env = get_nnunet_env()

    # Build a shell command that activates conda (if needed) then runs the export
    conda_profile = settings.CONDA_PROFILE
    conda_env = settings.CONDA_ENV
    activate = (
        f'source "{conda_profile}" && conda activate "{conda_env}" && '
        if conda_profile and Path(conda_profile).exists()
        else ""
    )
    cmd = (
        f'{activate}'
        f'nnUNetv2_export_model_to_zip '
        f'-d "{dataset_name}" '
        f'-c "{configuration}" '
        f'-o "{output_zip}" '
        f'--not_strict'
    )

    logger.info(f"Exporting model: {dataset_name} {configuration} → {output_zip}")
    result = subprocess.run(
        ["bash", "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Model export failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not output_zip.exists():
        raise RuntimeError(f"Export succeeded but ZIP not found at {output_zip}")

    logger.info(f"Model exported: {output_zip} ({output_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_zip
