"""Helper functions for the Streamlit dashboard (app.py).

Everything that is not layout lives here: loading artefacts, scoring,
the decision rule, and data preparation for charts. Keeping this logic
out of app.py makes it unit-testable without launching Streamlit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd

from src import config

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

APPROVE_BELOW = 0.25      # probability < 25%            -> Approved
REJECT_ABOVE = 0.50       # probability > 50%            -> Rejected
                          # 25% to 50% inclusive         -> Manual Review

LOAN_TYPE_LABELS = {
    "credit_card": "Credit card",
    "personal_loan": "Personal loan",
    "auto_loan": "Auto loan",
    "mortgage": "Mortgage",
}

FEATURE_LABELS = {
    "age": "Age",
    "annual_income": "Annual income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "macro_interest_rate": "Central bank interest rate",
    "credit_limit": "Credit limit",
    "months_on_book": "Months on book",
    "past_default_count": "Past defaults",
    "avg_revolving_balance_last_6m": "Avg revolving balance (6m)",
    "credit_utilisation_last_6m": "Credit utilisation (6m)",
    **{f"loan_type_{k}": f"Loan type: {v.lower()}" for k, v in LOAN_TYPE_LABELS.items()},
}


@dataclass(frozen=True)
class Decision:
    label: str        # Approved / Manual Review / Rejected
    colour: str       # green / yellow / red
    icon: str


DECISIONS = {
    "Approved": Decision("Approved", "green", "✅"),
    "Manual Review": Decision("Manual Review", "yellow", "⚠️"),
    "Rejected": Decision("Rejected", "red", "⛔"),
}


# ---------------------------------------------------------------------------
# Loading artefacts
# ---------------------------------------------------------------------------
def load_model():
    """Load the champion pipeline (preprocessing + model) saved by Task 3."""
    if not config.CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.CHAMPION_MODEL_PATH} not found. Run `python run_task3.py` first."
        )
    return joblib.load(config.CHAMPION_MODEL_PATH)


def load_metadata() -> dict:
    return json.loads(config.MODEL_METADATA_PATH.read_text())


def load_portfolio() -> pd.DataFrame:
    """Load the processed applicant feature matrix from Task 2."""
    return pd.read_parquet(config.FEATURE_MATRIX_PARQUET)


# ---------------------------------------------------------------------------
# Scoring + decision rule
# ---------------------------------------------------------------------------
def classify_decision(probability: float) -> Decision:
    """Apply the underwriting rule to a probability of default (0 to 1)."""
    if probability < APPROVE_BELOW:
        return DECISIONS["Approved"]
    if probability > REJECT_ABOVE:
        return DECISIONS["Rejected"]
    return DECISIONS["Manual Review"]


def predict_default_probability(model, applicants: pd.DataFrame) -> pd.Series:
    """Return P(default) for each row. Columns are selected and ordered here."""
    proba = model.predict_proba(applicants[FEATURES])[:, 1]
    return pd.Series(proba, index=applicants.index, name="default_probability")


def score_portfolio(model, portfolio: pd.DataFrame) -> pd.DataFrame:
    scored = portfolio.copy()
    scored["default_probability"] = predict_default_probability(model, scored)
    scored["decision"] = scored["default_probability"].map(
        lambda p: classify_decision(p).label
    )
    return scored


def build_applicant_frame(
    *,
    age: int,
    annual_income: float,
    debt_to_income_ratio: float,
    macro_interest_rate: float,
    loan_type: str,
    credit_limit: float,
    avg_revolving_balance_last_6m: float,
    months_on_book: int,
    past_default_count: int,
) -> pd.DataFrame:
    """Turn widget values into the one-row dataframe the model expects.

    Credit utilisation is derived exactly as in the Spark pipeline
    (balance / limit) so the officer cannot enter an inconsistent value.
    """
    utilisation = (
        avg_revolving_balance_last_6m / credit_limit if credit_limit > 0 else 0.0
    )
    return pd.DataFrame(
        [{
            "age": age,
            "annual_income": annual_income,
            "debt_to_income_ratio": debt_to_income_ratio,
            "macro_interest_rate": macro_interest_rate,
            "credit_limit": credit_limit,
            "months_on_book": months_on_book,
            "past_default_count": past_default_count,
            "avg_revolving_balance_last_6m": avg_revolving_balance_last_6m,
            "credit_utilisation_last_6m": round(utilisation, 4),
            "loan_type": loan_type,
        }]
    )[FEATURES]


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------
def feature_importance_frame(metadata: dict) -> pd.DataFrame:
    """Champion feature importances as a tidy frame, largest first."""
    items = metadata["feature_importance"].items()
    df = pd.DataFrame(items, columns=["feature", "importance"])
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def filter_scored_applicants(
    scored: pd.DataFrame,
    *,
    decisions: list[str] | None = None,
    loan_types: list[str] | None = None,
    min_probability: float = 0.0,
    max_probability: float = 1.0,
    customer_id_query: str = "",
) -> pd.DataFrame:
    """Filter the scored portfolio for the applicant table.

    None means "no filter" for the list arguments; an empty list matches
    nothing (the officer has deselected every option). The probability
    range is inclusive at both ends. The ID search is a case-insensitive
    substring match, so "19" finds CUST000019, CUST000190, ...
    Results are sorted riskiest first.
    """
    mask = scored["default_probability"].between(min_probability, max_probability)
    if decisions is not None:
        mask &= scored["decision"].isin(decisions)
    if loan_types is not None:
        mask &= scored["loan_type"].isin(loan_types)
    query = customer_id_query.strip()
    if query:
        mask &= scored["customer_id"].str.contains(query, case=False, regex=False)
    return scored[mask].sort_values("default_probability", ascending=False)


def decision_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Count and share of applicants in each decision band, in fixed order."""
    counts = scored["decision"].value_counts().reindex(list(DECISIONS), fill_value=0)
    return pd.DataFrame({
        "decision": counts.index,
        "applicants": counts.to_numpy(),
        "share": (counts / max(len(scored), 1)).to_numpy(),
    })
