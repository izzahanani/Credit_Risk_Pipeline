"""Task 2: PySpark feature-engineering pipeline.

Each step is a small function that takes and returns a Spark DataFrame, so
steps can be tested and reordered independently.

    load -> clean transactions -> rolling features -> customer-level
    aggregation + target -> join with profiles -> write single Parquet
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Initialise & load
# ---------------------------------------------------------------------------
def create_spark_session(app_name: str = "CreditRiskFeaturePipeline") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")                              # all local cores
        .config("spark.sql.shuffle.partitions", "8")     # default 200 is overkill locally
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_profiles(spark: SparkSession, path: Path) -> DataFrame:
    # Explicit schema: faster than inferSchema and catches bad files early.
    schema = (
        "customer_id STRING, age INT, annual_income DOUBLE, "
        "debt_to_income_ratio DOUBLE, macro_interest_rate DOUBLE"
    )
    return spark.read.csv(str(path), header=True, schema=schema)


def load_transactions(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.parquet(str(path))


# ---------------------------------------------------------------------------
# 2. Cleaning / missing values
# ---------------------------------------------------------------------------
def clean_transactions(tx: DataFrame) -> DataFrame:
    """Deduplicate and impute missing values.

    Imputation choices (each is a business assumption, stated explicitly):
      * duplicate transaction_id       -> keep one copy
      * missing customer_id / date     -> drop (row cannot be joined or ordered)
      * revolving_balance              -> customer's own average balance,
                                          falling back to 0 if they have none
      * days_past_due                  -> 0 (no delinquency was recorded)
      * past_default_count, loan_type  -> customer-level attributes, so take the
                                          value from the customer's other rows;
                                          fall back to 0 / "unknown"
    """
    by_customer = Window.partitionBy("customer_id")

    return (
        tx.dropDuplicates(["transaction_id"])
        .dropna(subset=["customer_id", "transaction_date"])
        .withColumn(
            "revolving_balance",
            F.coalesce(
                F.col("revolving_balance"),
                F.avg("revolving_balance").over(by_customer),
                F.lit(0.0),
            ),
        )
        .withColumn("days_past_due", F.coalesce(F.col("days_past_due"), F.lit(0.0)))
        .withColumn(
            "past_default_count",
            F.coalesce(
                F.col("past_default_count"),
                F.max("past_default_count").over(by_customer),
                F.lit(0.0),
            ).cast("int"),
        )
        .withColumn(
            "loan_type",
            F.coalesce(
                F.col("loan_type"),
                F.first("loan_type", ignorenulls=True).over(by_customer),
                F.lit("unknown"),
            ),
        )
    )


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def add_rolling_balance(tx: DataFrame, months: int = config.ROLLING_WINDOW_MONTHS) -> DataFrame:
    """Rolling mean of revolving balance over the current + previous 5 statements.

    Data is one row per customer per month, so a row-based frame of
    (months - 1) preceding rows equals a 6-month window.
    """
    rolling = (
        Window.partitionBy("customer_id")
        .orderBy("transaction_date")
        .rowsBetween(-(months - 1), Window.currentRow)
    )
    return tx.withColumn(
        f"avg_revolving_balance_last_{months}m",
        F.avg("revolving_balance").over(rolling),
    )


def aggregate_to_customer(tx: DataFrame) -> DataFrame:
    """Collapse monthly rows to one row per customer and build the target."""
    roll_col = f"avg_revolving_balance_last_{config.ROLLING_WINDOW_MONTHS}m"
    latest_first = Window.partitionBy("customer_id").orderBy(F.desc("transaction_date"))

    # The most recent statement carries the "last 6 months" rolling value.
    latest = (
        tx.withColumn("_rn", F.row_number().over(latest_first))
        .filter(F.col("_rn") == 1)
        .select("customer_id", "loan_type", "credit_limit", roll_col)
    )

    history = tx.groupBy("customer_id").agg(
        F.count("*").alias("months_on_book"),
        F.max("past_default_count").alias("past_default_count"),
        F.max("days_past_due").alias("max_days_past_due"),
    )

    return (
        latest.join(history, "customer_id")
        .withColumn(roll_col, F.round(roll_col, 2))
        .withColumn(
            "credit_utilisation_last_6m",
            F.round(
                F.when(F.col("credit_limit") > 0, F.col(roll_col) / F.col("credit_limit"))
                .otherwise(F.lit(0.0)),
                4,
            ),
        )
        # Target: ever more than 90 days past due.
        .withColumn(
            "is_default",
            (F.col("max_days_past_due") > config.DEFAULT_DPD_THRESHOLD).cast("int"),
        )
        # max_days_past_due *defines* the target, so keeping it as a feature
        # would be target leakage. Drop it.
        .drop("max_days_past_due")
    )


def build_feature_matrix(profiles: DataFrame, customer_features: DataFrame) -> DataFrame:
    """Join profiles to transaction features on customer_id.

    Left join first so customers with no transaction history are visible and
    can be counted; they are then removed because they have no label to
    train on (they would be scored by a separate thin-file policy).
    """
    joined = profiles.join(customer_features, on="customer_id", how="left")

    n_no_history = joined.filter(F.col("is_default").isNull()).count()
    logger.info("Customers without transaction history (excluded): %d", n_no_history)

    return joined.filter(F.col("is_default").isNotNull()).select(
        "customer_id",
        "age",
        "annual_income",
        "debt_to_income_ratio",
        "macro_interest_rate",
        "loan_type",
        "credit_limit",
        "months_on_book",
        "past_default_count",
        f"avg_revolving_balance_last_{config.ROLLING_WINDOW_MONTHS}m",
        "credit_utilisation_last_6m",
        "is_default",
    )


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------
def write_single_parquet(df: DataFrame, target: Path) -> None:
    """Write `df` as ONE snappy-compressed Parquet file at `target`.

    Spark always writes a directory of part files. coalesce(1) produces a
    single part file, which is then moved to the requested file name.
    coalesce(1) is safe here because the matrix is one row per customer
    (small); it would not be appropriate for the raw transaction table.
    """
    tmp_dir = target.with_suffix(".tmp")
    try:
        (
            df.coalesce(1)
            .write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(tmp_dir))
        )
    except Exception as exc:  # noqa: BLE001 - Py4J wraps the Java error
        if "HADOOP_HOME" not in str(exc) and "winutils" not in str(exc).lower():
            raise
        # Windows without winutils.exe/hadoop.dll: Spark's Hadoop writer cannot
        # touch the local disk. The matrix is small (one row per customer), so
        # collect it and write the same snappy Parquet with PyArrow instead.
        logger.warning(
            "Spark could not write to local disk (winutils.exe / HADOOP_HOME "
            "missing on Windows). Falling back to a PyArrow write. See README."
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        df.toPandas().to_parquet(target, index=False, compression="snappy")
        return

    part_file = next(tmp_dir.glob("part-*.parquet"))
    if target.exists():
        target.unlink()
    shutil.move(str(part_file), str(target))
    shutil.rmtree(tmp_dir)


def run_pipeline(spark: SparkSession) -> DataFrame:
    profiles = load_profiles(spark, config.CUSTOMER_PROFILES_CSV)
    transactions = load_transactions(spark, config.LOAN_TRANSACTIONS_PARQUET)
    logger.info("Loaded %d profiles, %d transactions",
                profiles.count(), transactions.count())

    cleaned = clean_transactions(transactions)
    with_rolling = add_rolling_balance(cleaned)
    customer_features = aggregate_to_customer(with_rolling)
    matrix = build_feature_matrix(profiles, customer_features).cache()

    write_single_parquet(matrix, config.FEATURE_MATRIX_PARQUET)
    return matrix
