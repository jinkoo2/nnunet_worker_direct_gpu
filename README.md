# nnunet_trainer_direct_gpu

A long-running nnUNet training worker with **direct host GPU access**. It registers with [`nnunet_dashboard`](https://github.com/jinkoo2/nnunet_dashboard), polls for admin-assigned training jobs, runs nnUNet preprocessing and training as subprocesses, streams per-epoch progress back to the dashboard, and uploads the finished model ZIP for admin approval.

> **Direct GPU** means this worker runs natively on the host machine (not inside a container), so it has full access to the system GPU via the installed CUDA driver and the `nnunet_trainer` conda environment.

---

## Requirements

- Linux host with NVIDIA GPU and CUDA drivers installed
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Running [`nnunet_dashboard`](https://github.com/jinkoo2/nnunet_dashboard) instance (local or remote)

---

## Setup

```bash
# 1. Create conda environment and install nnUNet
conda create -n nnunet_trainer python=3.12 -y
conda activate nnunet_trainer
pip install nnunetv2

# 2. Install worker dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set DASHBOARD_URL, DASHBOARD_API_KEY, WORKER_NAME, GPU info, DATA_DIR
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_URL` | `http://localhost:9333` | URL of the nnunet_dashboard server |
| `DASHBOARD_API_KEY` | `changeme` | Shared API key (`X-Api-Key` header) |
| `GOOGLE_CHAT_WEBHOOK_URL` | `""` | Optional Google Chat webhook for notifications |
| `WORKER_NAME` | `worker-01` | Unique display name shown in dashboard |
| `WORKER_HOSTNAME` | `""` | Hostname reported to dashboard |
| `GPU_NAME` | `""` | GPU model string (e.g. `NVIDIA GeForce RTX 3090`) |
| `GPU_MEMORY_GB` | `0.0` | GPU VRAM in GB |
| `CPU_CORES` | `0` | CPU core count |
| `SYSTEM_MEMORY_GB` | `0.0` | System RAM in GB |
| `POLL_INTERVAL_S` | `30` | Seconds between job polls |
| `HEARTBEAT_INTERVAL_S` | `60` | Seconds between heartbeats |
| `DATA_DIR` | `/data/nnunet_trainer_data` | Root directory for datasets, results, exports |
| `CONDA_ENV` | `nnunet_trainer` | Conda environment with nnunetv2 installed |
| `CONDA_PROFILE` | `/opt/miniconda3/etc/profile.d/conda.sh` | Path to conda init script |
| `DEVICE` | `cuda` | Training device (`cuda` or `cpu`) |
| `NUM_GPUS` | `1` | Number of GPUs for training |
| `NUM_PREPROCESSING_WORKERS` | `8` | `-np` workers for `nnUNetv2_preprocess` |

See `.env.example` for a fully annotated template.

---

## Running

```bash
conda activate nnunet_trainer
python main.py
```

The worker will:
1. Register with the dashboard (retries up to 10 times on startup)
2. Send a heartbeat every `HEARTBEAT_INTERVAL_S` seconds (`online` / `busy`)
3. Poll for pending jobs every `POLL_INTERVAL_S` seconds
4. Execute one job at a time (blocking)

---

## Job Execution Flow

```
assigned  →  download dataset ZIP (skipped if already on disk)
          →  extract to DATA_DIR/raw/ and preprocessed/
preprocessing  →  nnUNetv2_preprocess (skipped if preprocessing_completed.txt exists)
training  →  for fold in 0..4:
               nnUNetv2_train  (log tailed every 5s → per-epoch metrics posted to dashboard)
               upload training log every 60s
               post validation summary.json after each fold
uploading →  nnUNetv2_export_model_to_zip → upload ZIP to dashboard
done
```

If **Cancel** is clicked in the dashboard, the worker detects it within 30 seconds and sends `SIGTERM` (then `SIGKILL` after 10 s) to the entire nnUNet process group.

---

## Data Layout

```
DATA_DIR/
├── downloads/{dataset_id}.zip           ← ZIP downloaded from dashboard
├── raw/Dataset###_Name/                 ← extracted images + dataset.json
├── preprocessed/Dataset###_Name/        ← nnUNetPlans.json + preprocessed arrays
│   └── preprocessing_completed.txt      ← flag: skip preprocessing on re-run
├── results/Dataset###_Name/             ← trained models + logs (managed by nnUNet)
├── exports/{dataset_name}_{config}.zip  ← model ZIP before upload
└── logs/{job_id}/preprocess/            ← stdout captured from preprocess.sh
```

---

## Notifications

Set `GOOGLE_CHAT_WEBHOOK_URL` in `.env` to receive Google Chat messages for:
registration, job start/done/failed, download, preprocessing, each fold, model upload.

---

## Project Structure

```
nnunet_trainer_direct_gpu/
├── app/
│   ├── config.py            # Pydantic BaseSettings from .env
│   ├── dashboard_client.py  # HTTP wrapper for all dashboard API calls
│   ├── notifier.py          # Fire-and-forget Google Chat notifications
│   ├── trainer.py           # Dataset setup, subprocess mgmt, log parsing
│   └── worker.py            # Main loop: register → heartbeat → poll → execute
├── scripts/
│   ├── preprocess.sh        # conda-aware nnUNetv2_preprocess wrapper
│   └── train.sh             # conda-aware nnUNetv2_train wrapper
├── main.py                  # Entry point
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Related Projects

- [`nnunet_dashboard`](https://github.com/jinkoo2/nnunet_dashboard) — the training management server this worker connects to
