import os
import logging
import time
import requests
from .config import settings

logger = logging.getLogger(__name__)

_RETRY_EXCEPTIONS = (requests.ConnectionError, requests.Timeout)
_MAX_ATTEMPTS = 5


class DashboardClient:
    def __init__(self):
        self.base = settings.DASHBOARD_URL.rstrip("/")
        self.headers = {"X-Api-Key": settings.DASHBOARD_API_KEY}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """HTTP request with retry on connection errors and 5xx responses."""
        url = f"{self.base}{path}"
        headers = dict(self.headers)
        if kwargs.get("files"):
            headers.pop("Content-Type", None)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = requests.request(method, url, headers=headers, **kwargs)
                r.raise_for_status()
                return r.json()
            except _RETRY_EXCEPTIONS as e:
                if attempt == _MAX_ATTEMPTS:
                    raise
                wait = min(60, 10 * attempt)
                logger.warning(
                    f"Dashboard unreachable ({method} {path}, attempt {attempt}/{_MAX_ATTEMPTS}): {e}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500 and attempt < _MAX_ATTEMPTS:
                    wait = min(60, 10 * attempt)
                    logger.warning(
                        f"Server error {e.response.status_code} ({method} {path}, attempt {attempt}/{_MAX_ATTEMPTS}). "
                        f"Retrying in {wait}s…"
                    )
                    time.sleep(wait)
                else:
                    raise

    def _get(self, path, params=None):
        return self._request("GET", path, params=params, timeout=30)

    def _post(self, path, json=None, data=None, files=None, timeout=30):
        return self._request("POST", path, json=json, data=data, files=files, timeout=timeout)

    def _put(self, path, json=None):
        return self._request("PUT", path, json=json, timeout=30)

    # -------------------------------------------------------------------------
    # Workers
    # -------------------------------------------------------------------------

    def register_worker(self) -> dict:
        return self._post(
            "/api/workers/register",
            json={
                "name": settings.WORKER_NAME,
                "hostname": settings.WORKER_HOSTNAME or None,
                "cpu_cores": settings.CPU_CORES or None,
                "gpu_memory_gb": settings.GPU_MEMORY_GB or None,
                "gpu_name": settings.GPU_NAME or None,
                "system_memory_gb": settings.SYSTEM_MEMORY_GB or None,
            },
        )

    def heartbeat(self, worker_id: str, status: str = "online") -> None:
        self._post(f"/api/workers/{worker_id}/heartbeat", json={"status": status})

    def post_log(self, worker_id: str, worker_name: str, level: str, message: str) -> None:
        self._post("/api/logs/", json={
            "worker_id": worker_id,
            "worker_name": worker_name,
            "level": level,
            "message": message,
        })

    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------

    def get_pending_jobs(self, worker_id: str) -> list:
        return self._get("/api/jobs/", params={"worker_id": worker_id, "status": "pending"})

    def update_job_status(self, job_id: str, status: str, error_message: str = None) -> None:
        body = {"status": status}
        if error_message is not None:
            body["error_message"] = error_message
        self._put(f"/api/jobs/{job_id}/status", json=body)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/api/jobs/{job_id}")

    def get_dataset(self, dataset_id: str) -> dict:
        return self._get(f"/api/datasets/{dataset_id}")

    def download_dataset(self, dataset_id: str, dest_path: str) -> None:
        logger.info(f"Downloading dataset {dataset_id} → {dest_path}")
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = requests.get(
                    f"{self.base}/api/datasets/{dataset_id}/download",
                    headers=self.headers,
                    stream=True,
                    timeout=3600,
                )
                r.raise_for_status()
                downloaded = 0
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                logger.info(f"Download complete: {downloaded / 1024 / 1024:.1f} MB")
                return
            except _RETRY_EXCEPTIONS as e:
                if attempt == _MAX_ATTEMPTS:
                    raise
                wait = min(60, 10 * attempt)
                logger.warning(f"Dataset download failed (attempt {attempt}/{_MAX_ATTEMPTS}): {e}. Retrying in {wait}s…")
                time.sleep(wait)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500 and attempt < _MAX_ATTEMPTS:
                    wait = min(60, 10 * attempt)
                    logger.warning(f"Dataset download server error {e.response.status_code} (attempt {attempt}/{_MAX_ATTEMPTS}). Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    # -------------------------------------------------------------------------
    # Progress reporting
    # -------------------------------------------------------------------------

    def report_preprocessing_progress(
        self, job_id: str, total_images: int, done_images: int, mean_time_s: float = None
    ) -> None:
        self._post(
            f"/api/jobs/{job_id}/preprocessing_progress",
            json={
                "total_images": total_images,
                "done_images": done_images,
                "mean_time_per_image_s": mean_time_s,
            },
        )

    def report_training_progress(
        self, job_id: str, fold: int, epoch: int,
        learning_rate: float = None, train_loss: float = None,
        val_loss: float = None, pseudo_dice: str = None, epoch_time_s: float = None,
    ) -> None:
        self._post(
            f"/api/jobs/{job_id}/training_progress",
            json={
                "fold": fold,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "pseudo_dice": pseudo_dice,
                "epoch_time_s": epoch_time_s,
            },
        )

    def report_validation_result(self, job_id: str, fold: int, summary_json: str) -> None:
        self._post(
            f"/api/jobs/{job_id}/validation_result",
            json={"fold": fold, "summary_json": summary_json},
        )

    def upload_log(self, job_id: str, fold: int, text: str) -> None:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = requests.post(
                    f"{self.base}/api/jobs/{job_id}/log/{fold}",
                    headers=self.headers,
                    data=text.encode("utf-8"),
                    timeout=60,
                )
                r.raise_for_status()
                return
            except _RETRY_EXCEPTIONS as e:
                if attempt == _MAX_ATTEMPTS:
                    raise
                wait = min(60, 10 * attempt)
                logger.warning(f"Log upload failed (attempt {attempt}/{_MAX_ATTEMPTS}): {e}. Retrying in {wait}s…")
                time.sleep(wait)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500 and attempt < _MAX_ATTEMPTS:
                    wait = min(60, 10 * attempt)
                    logger.warning(f"Log upload server error {e.response.status_code} (attempt {attempt}/{_MAX_ATTEMPTS}). Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    def upload_model(self, job_id: str, zip_path: str) -> dict:
        with open(zip_path, "rb") as f:
            return self._post(
                f"/api/jobs/{job_id}/model",
                files={"zip_file": (os.path.basename(zip_path), f, "application/zip")},
                timeout=3600,
            )
