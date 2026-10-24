"""
rich/run_bot5.py — Standalone entry point for Bot5 «الغني 💰».

Usage:
  python -m crypto.rich.run_bot5
  or
  python run_bot5.py
"""
from __future__ import annotations

import logging
import sys
import os

# Add parent directory to path to allow direct invocation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from crypto.rich.config import RichConfig, TradingMode
from crypto.rich.engine import RichEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("rich.main")


def main():
    logger.info("==================================================")
    logger.info("       🚀 STRATOCRYPTO — BOT 5 «الغني 💰»          ")
    logger.info("  Autonomous Execution, Risk & Protection Engine  ")
    logger.info("==================================================")

    config = RichConfig.load()
    engine = RichEngine(config)
    engine.start()


if __name__ == "__main__":
    main()
