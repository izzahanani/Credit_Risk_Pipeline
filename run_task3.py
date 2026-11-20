"""Task 3 entry point: train, tune and benchmark XGBoost vs CatBoost.

Usage (from the project root, after run_task2.py):
    python run_task3.py
"""
import logging

from src import config
from src.model_training import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("task3")


def main() -> None:
    _, champion, report = run_benchmark()
    print("\n" + report + "\n")
    logger.info("Champion model saved to %s", config.CHAMPION_MODEL_PATH)
    logger.info("Metadata saved to %s", config.MODEL_METADATA_PATH)
    logger.info("Report saved to %s", config.BENCHMARK_REPORT_PATH)
    logger.info("Top features: %s", list(champion.feature_importance.items())[:5])


if __name__ == "__main__":
    main()
