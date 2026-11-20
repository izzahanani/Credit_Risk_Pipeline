"""Tests for the dashboard decision rule and scoring helpers."""
import pytest

import utils


@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, "Approved"), (0.2499, "Approved"), (0.25, "Manual Review"),
     (0.40, "Manual Review"), (0.50, "Manual Review"), (0.5001, "Rejected"),
     (1.0, "Rejected")],
)
def test_decision_rule_boundaries(probability, expected):
    assert utils.classify_decision(probability).label == expected


def test_riskier_applicant_scores_higher():
    model = utils.load_model()
    common = dict(macro_interest_rate=2.75, loan_type="credit_card",
                  credit_limit=15_000, months_on_book=12)
    safe = utils.build_applicant_frame(
        age=45, annual_income=150_000, debt_to_income_ratio=0.10,
        avg_revolving_balance_last_6m=1_500, past_default_count=0, **common)
    risky = utils.build_applicant_frame(
        age=24, annual_income=24_000, debt_to_income_ratio=0.85,
        avg_revolving_balance_last_6m=14_500, past_default_count=3, **common)
    p_safe = utils.predict_default_probability(model, safe).iloc[0]
    p_risky = utils.predict_default_probability(model, risky).iloc[0]
    assert p_safe < 0.25 < 0.50 < p_risky


def _toy_scored():
    import pandas as pd
    return pd.DataFrame({
        "customer_id": ["CUST000001", "CUST000019", "CUST000190", "CUST000500"],
        "loan_type": ["credit_card", "mortgage", "credit_card", "auto_loan"],
        "default_probability": [0.05, 0.30, 0.70, 0.25],
        "decision": ["Approved", "Manual Review", "Rejected", "Manual Review"],
    })


def test_filter_defaults_return_everything_riskiest_first():
    out = utils.filter_scored_applicants(_toy_scored())
    assert list(out["customer_id"]) == ["CUST000190", "CUST000019", "CUST000500", "CUST000001"]


def test_filter_combines_conditions():
    out = utils.filter_scored_applicants(
        _toy_scored(), decisions=["Manual Review", "Rejected"],
        loan_types=["credit_card", "mortgage"], min_probability=0.25)
    assert list(out["customer_id"]) == ["CUST000190", "CUST000019"]


def test_filter_probability_range_is_inclusive():
    out = utils.filter_scored_applicants(_toy_scored(), min_probability=0.25, max_probability=0.30)
    assert set(out["customer_id"]) == {"CUST000019", "CUST000500"}


def test_filter_id_search_is_case_insensitive_substring():
    toy = _toy_scored()
    # substring: "019" appears in CUST000019 and CUST000190
    out = utils.filter_scored_applicants(toy, customer_id_query=" 019 ")
    assert set(out["customer_id"]) == {"CUST000019", "CUST000190"}
    # case-insensitive: lower-case search finds the upper-case ID
    out = utils.filter_scored_applicants(toy, customer_id_query="cust000500")
    assert list(out["customer_id"]) == ["CUST000500"]


def test_filter_empty_selection_matches_nothing():
    assert utils.filter_scored_applicants(_toy_scored(), decisions=[]).empty
