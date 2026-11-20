# Bank Credit Risk & Real-Time Decisioning Pipeline

End-to-end prototype: scrape a macro indicator, build customer and transaction
data, engineer features in PySpark, benchmark XGBoost vs CatBoost, and serve a
Streamlit dashboard for credit officers.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

PySpark also needs a Java JDK (17 or 21). Check with `java -version`.

## Run order

| Step | Command | Output |
|---|---|---|
| Task 1 | `python run_task1.py` | `data/customer_profiles.csv` |
| Task 2 | `python run_task2.py` | `data/loan_transactions.parquet`, `data/feature_matrix.parquet` |
| Task 3 | `python run_task3.py` | `models/champion_model.pkl`, `models/model_metadata.json`, `reports/model_benchmark.txt` |
| Task 4 | `streamlit run app.py` | dashboard at http://localhost:8501 |
| Tests | `python -m pytest -q` | |

## Project structure

```
app.py                    Streamlit layout (Task 4)
utils.py                  Dashboard logic: loading, scoring, decision rule
models/                   Saved champion model + metadata
run_task1.py ... 3.py     Entry points for each task
src/
  config.py               Paths and settings shared by all tasks
  scraper.py              BeautifulSoup scraper (Task 1)
  data_generator.py       Pandas/NumPy customer generator (Task 1)
  transaction_generator.py  Mock loan_transactions.parquet (Task 2 input)
  spark_pipeline.py       PySpark feature engineering (Task 2)
  model_training.py       scikit-learn / XGBoost / CatBoost benchmark (Task 3)
tests/                    Parser and decision-rule tests
data/, reports/, docs/    Generated data, benchmark report, screenshots
```

## Tools used per task

| Requirement | Where |
|---|---|
| BeautifulSoup | `src/scraper.py` |
| Pandas, NumPy | `src/data_generator.py`, `src/transaction_generator.py` |
| PySpark | `src/spark_pipeline.py` |
| scikit-learn, XGBoost, CatBoost | `src/model_training.py` |
| Streamlit | `app.py`, `utils.py` |

## Assumptions

- **Macro indicator**: Malaysia's central bank policy rate (Bank Negara OPR),
  scraped from Wikipedia's list of central bank interest rates. If the page is
  unreachable, a logged fallback of 2.75% is used.
- **Transactions file**: no `loan_transactions.parquet` was supplied, so it is
  simulated with one statement row per customer per month, sharing
  `customer_id` with the profiles. Nulls, duplicates and customers with no
  history are injected deliberately for the pipeline to handle.
- **Missing values**: balance -> customer's average; days past due -> 0;
  loan type and past-default count -> taken from the customer's other rows.
- **Target**: `is_default = 1` if the customer was ever more than 90 days past
  due. The days-past-due column is dropped from the features to avoid leakage.
- **Customers with no history** (51) have no label and are excluded.
- **No class re-weighting**: keeps predicted probabilities calibrated so the
  dashboard's 25% / 50% rule is meaningful.
- **Decision rule**: below 25% Approved, 25% to 50% inclusive Manual Review,
  above 50% Rejected.
- All data is synthetic, so the reported scores show that the pipeline works,
  not how it would perform on a real loan book.
