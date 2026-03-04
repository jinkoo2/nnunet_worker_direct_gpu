# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nnunet_trainer_direct_gpu` — long-running nnUNet training worker with direct host GPU access. Registers with `nnunet_dashboard`, polls for admin-assigned training jobs, downloads datasets, runs nnUNet preprocessing + training via subprocess, streams progress back to the dashboard, and uploads the finished model ZIP for admin approval.

## Commands

```bash
# Setup
conda create -n nnunet_trainer python=3.12 -y
conda activate nnunet_trainer
pip install nnunetv2
pip install -r requirements.txt

# Run
cp .env.example .env
# Edit .env with your DASHBOARD_URL, DASHBOARD_API_KEY, WORKER_NAME, GPU info, DATA_DIR
conda activate nnunet_trainer
python main.py
```

## Architecture

**Stack:** Python 3.12, requests, pydantic-settings. No FastAPI — pure worker process.

**Key files:**
- `main.py` — entry point, sets up logging, calls `worker.run()`
- `app/config.py` — Pydantic `BaseSettings` from `.env`
- `app/dashboard_client.py` — `requests` wrapper for all dashboard API calls
- `app/trainer.py` — dataset extraction, subprocess management, log parsing, model export
- `app/worker.py` — main loop: register → heartbeat thread → poll → execute

**Scripts (conda-aware bash wrappers):**
- `scripts/preprocess.sh <DATASET_NUM> <LOG_DIR>` — runs `nnUNetv2_preprocess`
- `scripts/train.sh <DATASET_ID> <CONFIGURATION> <FOLD>` — runs `nnUNetv2_train`

## Worker Flow

```
startup → register with dashboard (retry up to 10x)
        → start heartbeat daemon thread (every HEARTBEAT_INTERVAL_S)

poll loop (every POLL_INTERVAL_S):
  GET /api/jobs?worker_id={id}&status=pending
  if job found → execute_job (blocking)

execute_job:
  1. PUT /api/jobs/{id}/status  → "assigned"
  2. GET /api/datasets/{id}     → get dataset_name
  3. GET /api/datasets/{id}/download → stream ZIP to disk
  4. Extract ZIP → DATA_DIR/raw/{dataset_name}/ + DATA_DIR/preprocessed/{dataset_name}/
  5. PUT status → "preprocessing"
     subprocess: scripts/preprocess.sh {dataset_num} {log_dir}
     parse stdout for "Preprocessing case" lines → POST preprocessing_progress
  6. PUT status → "training"
     for fold in 0..4:
       subprocess: scripts/train.sh {dataset_name} {configuration} {fold}
       monitor thread: tail training_log.txt → parse epochs → POST training_progress every epoch
       upload log text every 60s
       POST validation_result from fold_{n}/validation/summary.json
  7. PUT status → "uploading"
     nnUNetv2_export_model_to_zip → POST /api/jobs/{id}/model
  8. PUT status → "done"
  on exception → PUT status → "failed", error_message=str(e)
```

## Dataset ZIP Structure

Produced by nnunet_server "Upload to Training Dashboard" feature:
```
Dataset###_Name/imagesTr/case001_0000.nii.gz
Dataset###_Name/labelsTr/case001.nii.gz
Dataset###_Name/dataset.json
Dataset###_Name/dataset_fingerprint.json   → extracted to preprocessed/Dataset###_Name/
Dataset###_Name/nnUNetPlans.json           → extracted to preprocessed/Dataset###_Name/
```

## nnUNet Training Log Format

```
2026-03-01 22:05:50: Epoch 605
2026-03-01 22:05:50: Current learning rate: 0.00433
2026-03-01 22:08:05: train_loss -0.6752
2026-03-01 22:08:05: val_loss -0.4224
2026-03-01 22:08:05: Pseudo dice [0.8876, 0.8706]
2026-03-01 22:05:48: Epoch time: 125.29 s   ← marks end of epoch block
```

Log file location: `DATA_DIR/results/{dataset_name}/nnUNetTrainer__nnUNetPlans__{configuration}/fold_{fold}/training_log.txt`

## Data Layout

```
DATA_DIR/
├── downloads/{dataset_id}.zip       ← downloaded from dashboard
├── raw/Dataset###_Name/             ← raw images + dataset.json
├── preprocessed/Dataset###_Name/    ← nnUNetPlans.json + preprocessed data
├── results/Dataset###_Name/         ← trained models + logs (managed by nnUNet)
├── exports/{dataset_name}_{config}.zip  ← model export zip before upload
└── logs/{job_id}/preprocess/        ← preprocess stdout capture
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_URL` | `http://localhost:9333` | Dashboard server URL |
| `DASHBOARD_API_KEY` | `changeme` | API key (X-Api-Key header) |
| `WORKER_NAME` | `worker-01` | Unique display name in dashboard |
| `WORKER_HOSTNAME` | `""` | Optional hostname info |
| `GPU_NAME` | `""` | GPU model string |
| `GPU_MEMORY_GB` | `0.0` | GPU VRAM in GB |
| `CPU_CORES` | `0` | CPU core count |
| `POLL_INTERVAL_S` | `30` | Seconds between job polls |
| `HEARTBEAT_INTERVAL_S` | `60` | Seconds between heartbeats |
| `DATA_DIR` | `/home/jk/data/nnunet_trainer_data` | Working directory |
| `CONDA_ENV` | `nnunet_trainer` | Conda env with nnunetv2 |
| `CONDA_PROFILE` | `/home/jk/miniconda3/etc/profile.d/conda.sh` | Conda init script |
| `DEVICE` | `cuda` | Training device |
| `NUM_GPUS` | `1` | Number of GPUs |
| `NUM_PREPROCESSING_WORKERS` | `8` | `-np` argument for preprocessing |

## Integration with nnunet_dashboard

- Dashboard URL: `https://nnunet-dashboard-1.apps.myphysics.net`
- Admin creates jobs via the dashboard UI (Datasets tab → Create Job)
- Worker polls `GET /api/jobs?worker_id={id}&status=pending`
- Worker upserts by `WORKER_NAME` on register — safe to restart

## Conventions

- All nnUNet commands run through bash scripts in `scripts/` (handles conda activation)
- `subprocess.Popen` used (not `subprocess.run`) for real-time stdout reading
- Monitor thread uses `threading.Event` for clean shutdown
- Closures in fold loop use factory functions (`make_progress_cb`, `make_log_cb`) to avoid late-binding
