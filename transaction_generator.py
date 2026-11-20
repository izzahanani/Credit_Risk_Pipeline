"""Generate the mock `loan_transactions.parquet` used by Task 2.

The brief calls this file "provided"; since none was supplied, it is
simulated here so that it shares customer_ids with customer_profiles.csv.
One row = one monthly statement for one customer.

Delinquency is driven by a hidden risk score built from the customer's
debt-to-income ratio, income, age and prior defaults, so the models in
Task 3 have a genuine (but noisy) signal to learn.

Deliberate data-quality problems are injected (nulls, duplicate rows,
customers with no history) so the Spark pipeline has something to clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

LOAN_TYPES = ["credit_card", "personal_loan", "auto_loan", "mortgage"]
LOAN_TYPE_PROBS = [0.45, 0.25, 0.18, 0.12]
# Risk adjustment per product (unsecured products default more often).
LOAN_TYPE_RISK = {"credit_card": 0.25, "personal_loan": 0.35,
                  "auto_loan": -0.15, "mortgage": -0.45}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_loan_transactions(
    profiles: pd.DataFrame,
    n_months: int = config.HISTORY_MONTHS,
    seed: int = config.RANDOM_SEED + 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(profiles)

    # ---- customer-level attributes -------------------------------------
    loan_type = rng.choice(LOAN_TYPES, size=n, p=LOAN_TYPE_PROBS)
    past_default_count = rng.choice([0, 1, 2, 3], size=n, p=[0.82, 0.12, 0.045, 0.015])
    months_on_book = rng.integers(6, n_months + 1, size=n)
    has_history = rng.random(n) > 0.01          # ~1% thin-file customers

    income = profiles["annual_income"].to_numpy()
    dti = profiles["debt_to_income_ratio"].to_numpy()
    age = profiles["age"].to_numpy()

    credit_limit = (income * rng.uniform(0.10, 0.45, size=n)).round(-2)
    utilisation = np.clip(rng.beta(2, 3, size=n) + 0.5 * (dti - 0.33), 0.02, 1.0)

    # Hidden risk score -> monthly probability of missing a payment.
    risk = (
        -4.2
        + 4.0 * (dti - 0.33)
        - 0.60 * np.log(income / 58_000)
        - 0.025 * (age - 40)
        + 0.70 * past_default_count
        + 1.20 * (utilisation - 0.45)
        + pd.Series(loan_type).map(LOAN_TYPE_RISK).to_numpy()
        + rng.normal(0, 0.5, size=n)
    )
    p_miss = _sigmoid(risk)

    # ---- simulate month by month ---------------------------------------
    months = pd.date_range(end=config.HISTORY_END_MONTH, periods=n_months, freq="ME")
    dpd = np.zeros(n)
    balance = credit_limit * utilisation
    frames = []
    for m_idx, month in enumerate(months):
        active = has_history & (m_idx >= n_months - months_on_book)

        # Once behind, a customer is far more likely to miss again.
        p_now = np.where(dpd > 0, np.clip(p_miss * 4.0 + 0.35, 0, 0.95), p_miss)
        missed = rng.random(n) < p_now
        dpd = np.where(missed, dpd + 30, 0.0)

        drift = rng.normal(1.0, 0.06, size=n) + np.where(missed, 0.02, 0.0)
        balance = np.clip(balance * drift, 0, credit_limit * 1.15)
        payment = np.where(missed, 0.0, balance * rng.uniform(0.03, 0.15, size=n))

        frames.append(pd.DataFrame({
            "customer_id": profiles["customer_id"].to_numpy()[active],
            "transaction_date": month,
            "loan_type": loan_type[active],
            "credit_limit": credit_limit[active],
            "revolving_balance": balance[active].round(2),
            "payment_amount": payment[active].round(2),
            "days_past_due": dpd[active].astype("float64"),
            "past_default_count": past_default_count[active].astype("float64"),
        }))

    tx = pd.concat(frames, ignore_index=True)
    tx.insert(0, "transaction_id",
              "TXN" + pd.Series(np.arange(1, len(tx) + 1)).astype(str).str.zfill(8))

    # ---- inject realistic data-quality issues --------------------------
    def null_out(col: str, frac: float) -> None:
        tx.loc[rng.random(len(tx)) < frac, col] = np.nan

    null_out("revolving_balance", 0.03)
    null_out("days_past_due", 0.02)
    null_out("past_default_count", 0.02)
    tx["loan_type"] = tx["loan_type"].astype(object)
    tx.loc[rng.random(len(tx)) < 0.01, "loan_type"] = None

    dupes = tx.sample(frac=0.005, random_state=seed)      # duplicate rows
    tx = pd.concat([tx, dupes], ignore_index=True)
    return tx.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    profiles = pd.read_csv(config.CUSTOMER_PROFILES_CSV)
    tx = generate_loan_transactions(profiles)
    tx.to_parquet(config.LOAN_TRANSACTIONS_PARQUET, index=False)
    print(f"Wrote {len(tx):,} rows to {config.LOAN_TRANSACTIONS_PARQUET}")


if __name__ == "__main__":
    main()
