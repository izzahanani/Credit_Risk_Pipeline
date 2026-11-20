"""Central configuration shared by every stage of the pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Task 1: scraping ---------------------------------------------------
MACRO_SOURCE_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_sovereign_states_by_central_bank_interest_rates"
)
MACRO_COUNTRY = "Malaysia"          # Bank Negara Malaysia Overnight Policy Rate
REQUEST_TIMEOUT_SECONDS = 15
# Used only if the live scrape fails (offline machine, page layout change).
# BNM OPR as of the 5 Nov 2026 MPC meeting.
FALLBACK_INTEREST_RATE = 2.75

# --- Task 1: synthetic customers ----------------------------------------
N_CUSTOMERS = 5_000
RANDOM_SEED = 42
CUSTOMER_PROFILES_CSV = DATA_DIR / "customer_profiles.csv"

# --- Task 2: transactions + Spark feature pipeline ----------------------
LOAN_TRANSACTIONS_PARQUET = DATA_DIR / "loan_transactions.parquet"
FEATURE_MATRIX_PARQUET = DATA_DIR / "feature_matrix.parquet"
HISTORY_MONTHS = 24                 # months of history per customer (max)
HISTORY_END_MONTH = "2026-10-31"    # last statement month in the mock data
ROLLING_WINDOW_MONTHS = 6
DEFAULT_DPD_THRESHOLD = 90          # >90 days past due => default

# --- Task 3: modelling --------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
BENCHMARK_REPORT_PATH = REPORTS_DIR / "model_benchmark.txt"

TARGET = "is_default"
ID_COLUMN = "customer_id"
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "debt_to_income_ratio",
    "macro_interest_rate",
    "credit_limit",
    "months_on_book",
    "past_default_count",
    "avg_revolving_balance_last_6m",
    "credit_utilisation_last_6m",
]
CATEGORICAL_FEATURES = ["loan_type"]

TEST_SIZE = 0.20
CV_FOLDS = 5
N_SEARCH_ITERATIONS = 20            # random hyper-parameter draws per model
