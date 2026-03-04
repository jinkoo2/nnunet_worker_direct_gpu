from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Dashboard connection
    DASHBOARD_URL: str = "http://localhost:9333"
    DASHBOARD_API_KEY: str = "changeme"

    # Notifications (optional)
    GOOGLE_CHAT_WEBHOOK_URL: str = ""

    # Worker identity (reported at registration)
    WORKER_NAME: str = "worker-ai01"
    WORKER_HOSTNAME: str = ""
    GPU_NAME: str = ""
    GPU_MEMORY_GB: float = 0.0
    CPU_CORES: int = 0
    SYSTEM_MEMORY_GB: float = 0.0

    # Timing
    POLL_INTERVAL_S: int = 30       # seconds between job polls
    HEARTBEAT_INTERVAL_S: int = 60  # seconds between heartbeats

    # Paths
    DATA_DIR: str = "/data/nnunet_trainer_data"
    SCRIPTS_DIR: str = ""           # auto-detected relative to main.py if empty
    CONDA_ENV: str = "nnunet_trainer"
    CONDA_PROFILE: str = "/opt/miniconda3/etc/profile.d/conda.sh"

    # nnUNet
    DEVICE: str = "cuda"
    NUM_GPUS: int = 1
    NUM_PREPROCESSING_WORKERS: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
