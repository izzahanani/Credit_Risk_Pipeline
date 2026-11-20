"""Generate synthetic retail customer profiles with Pandas and NumPy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def generate_customer_profiles(
    macro_interest_rate: float,
    n_customers: int = config.N_CUSTOMERS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Build a dataframe of synthetic customers.

    Distributions are chosen to resemble a retail loan book rather than
    uniform noise, so downstream models have realistic structure to learn:
      * age            - truncated normal, 21 to 70
      * annual_income  - log-normal (right-skewed), rising then flattening
                         with age, in MYR
      * debt_to_income - beta distribution, negatively related to income
    """
    rng = np.random.default_rng(seed)

    customer_id = np.char.add("CUST", np.char.zfill(
        np.arange(1, n_customers + 1).astype(str), 6))

    age = np.clip(rng.normal(loc=40, scale=11, size=n_customers), 21, 70)
    age = age.round().astype(int)

    # Earnings peak around age 50; multiplier ranges roughly 0.7 to 1.2.
    age_effect = 1.2 - 0.5 * ((age - 50) / 29) ** 2
    annual_income = rng.lognormal(mean=np.log(55_000), sigma=0.55,
                                  size=n_customers) * age_effect
    annual_income = np.clip(annual_income, 18_000, 600_000).round(2)

    # Lower earners tend to carry proportionally more debt.
    income_rank = pd.Series(annual_income).rank(pct=True).to_numpy()
    base_dti = rng.beta(a=2.2, b=4.5, size=n_customers)
    debt_to_income_ratio = np.clip(
        base_dti + 0.12 * (0.5 - income_rank), 0.01, 0.95
    ).round(4)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "annual_income": annual_income,
            "debt_to_income_ratio": debt_to_income_ratio,
            "macro_interest_rate": macro_interest_rate,
        }
    )


def validate_profiles(df: pd.DataFrame, n_expected: int = config.N_CUSTOMERS) -> None:
    """Fail fast if the generated data breaks basic expectations."""
    assert len(df) == n_expected, f"Expected {n_expected} rows, got {len(df)}"
    assert df["customer_id"].is_unique, "customer_id must be unique"
    assert not df.isna().any().any(), "No missing values expected"
    assert df["age"].between(21, 70).all(), "age out of range"
    assert (df["annual_income"] > 0).all(), "income must be positive"
    assert df["debt_to_income_ratio"].between(0, 1).all(), "DTI out of range"
