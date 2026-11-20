"""Task 2 entry point: build the feature matrix with PySpark.

Usage (from the project root, after run_task1.py):
    python run_task2.py
"""
import logging

from pyspark.sql import functions as F

from src import config
from src import transaction_generator
from src.spark_pipeline import create_spark_session, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("task2")


def main() -> None:
    if not config.LOAN_TRANSACTIONS_PARQUET.exists():
        logger.info("loan_transactions.parquet not found - generating mock file")
        transaction_generator.main()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        matrix = run_pipeline(spark)

        total = matrix.count()
        defaults = matrix.agg(F.sum("is_default")).first()[0]
        nulls = matrix.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in matrix.columns]
        ).first().asDict()

        logger.info("Feature matrix: %d rows x %d columns", total, len(matrix.columns))
        logger.info("Default rate: %.2f%% (%d of %d)", 100 * defaults / total, defaults, total)
        logger.info("Remaining nulls: %d", sum(nulls.values()))
        logger.info("Saved to %s", config.FEATURE_MATRIX_PARQUET)
        matrix.show(5, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
