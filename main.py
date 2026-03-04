#!/usr/bin/env python3
"""
nnunet_trainer_direct_gpu — long-running worker with direct host GPU access for nnunet_dashboard.

Usage:
    conda activate nnunet_trainer
    python main.py
"""
import logging
import sys

# Configure logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from app.worker import run  # noqa: E402

if __name__ == "__main__":
    run()
