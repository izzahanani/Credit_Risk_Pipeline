"""Credit Risk Dashboard for credit officers.

Run from the project root:
    streamlit run app.py
"""
import altair as alt
import streamlit as st

import utils

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="🏦", layout="wide")

BLUE = "#2a78d6"
DECISION_COLOURS = {"Approved": "#0ca30c", "Manual Review": "#fab219", "Rejected": "#d03b3b"}


# ---------------------------------------------------------------------------
# Cached loaders: the model is read from disk once, not on every interaction
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading champion model...")
def get_model():
    return utils.load_model()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return utils.load_metadata()


@st.cache_data(show_spinner="Scoring portfolio...")
def get_scored_portfolio():
    return utils.score_portfolio(get_model(), utils.load_portfolio())


# ---------------------------------------------------------------------------
# Section 1: batch metrics
# ---------------------------------------------------------------------------
def importance_chart(metadata: dict) -> alt.Chart:
    data = utils.feature_importance_frame(metadata)
    return (
        alt.Chart(data)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("importance:Q", axis=alt.Axis(format=".0%", title="Share of model importance")),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".1%"),
            ],
        )
        .properties(height=24 * len(data) + 40)
    )


def decision_chart(scored) -> alt.Chart:
    data = utils.decision_summary(scored)
    order = list(utils.DECISIONS)
    base = alt.Chart(data).encode(
        y=alt.Y("decision:N", sort=order, title=None),
        x=alt.X("applicants:Q", title="Applicants"),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color(
            "decision:N",
            scale=alt.Scale(domain=order, range=[DECISION_COLOURS[d] for d in order]),
            legend=None,                      # the axis labels already name each band
        ),
        tooltip=[
            alt.Tooltip("decision:N", title="Decision"),
            alt.Tooltip("applicants:Q", title="Applicants", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    labels = base.mark_text(align="left", dx=6).encode(text=alt.Text("applicants:Q", format=","))
    return (bars + labels).properties(height=150)


def render_batch_metrics() -> None:
    scored = get_scored_portfolio()
    metadata = get_metadata()
    summary = utils.decision_summary(scored).set_index("decision")

    st.subheader("Portfolio overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total applicants processed", f"{len(scored):,}")
    k2.metric("Average default probability", f"{scored['default_probability'].mean():.1%}")
    k3.metric("Approval rate", f"{summary.loc['Approved', 'share']:.1%}")
    k4.metric("Observed default rate", f"{scored['is_default'].mean():.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Feature importance ({metadata['champion']})**")
        st.altair_chart(importance_chart(metadata), width="stretch")
    with right:
        st.markdown("**Applicants by decision**")
        st.altair_chart(decision_chart(scored), width="stretch")
        st.caption(
            "Approved: probability below 25%. Manual Review: 25% to 50%. "
            "Rejected: above 50%."
        )

    with st.expander("View scored applicants (table)"):
        render_applicant_table(scored)


def render_applicant_table(scored) -> None:
    """Scored applicants with a row of filters above the table."""
    decisions = list(utils.DECISIONS)
    loan_types = sorted(scored["loan_type"].unique())

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.5])
    chosen_decisions = f1.multiselect("Decision", decisions, default=decisions)
    chosen_loans = f2.multiselect(
        "Loan type", loan_types, default=loan_types,
        format_func=lambda v: utils.LOAN_TYPE_LABELS.get(v, v))
    prob_low, prob_high = f3.slider(
        "Default probability (%)", 0, 100, (0, 100), step=1)
    search = f4.text_input("Customer ID", placeholder="e.g. CUST000019")

    filtered = utils.filter_scored_applicants(
        scored,
        decisions=chosen_decisions,
        loan_types=chosen_loans,
        min_probability=prob_low / 100,
        max_probability=prob_high / 100,
        customer_id_query=search,
    )

    st.caption(f"Showing {len(filtered):,} of {len(scored):,} applicants, "
               "highest default probability first.")
    if filtered.empty:
        st.info("No applicants match these filters.")
        return

    table = filtered[["customer_id", "age", "annual_income", "debt_to_income_ratio",
                      "loan_type", "default_probability", "decision"]]
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer ID"),
            "age": st.column_config.NumberColumn("Age"),
            "annual_income": st.column_config.NumberColumn("Annual income (MYR)", format="%.0f"),
            "debt_to_income_ratio": st.column_config.NumberColumn("Debt-to-income", format="%.2f"),
            "loan_type": st.column_config.TextColumn("Loan type"),
            "default_probability": st.column_config.ProgressColumn(
                "Default probability", format="percent", min_value=0, max_value=1),
            "decision": st.column_config.TextColumn("Decision"),
        },
    )
    st.download_button(
        "Download filtered list (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="scored_applicants_filtered.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Section 2: real-time underwriting
# ---------------------------------------------------------------------------
def render_underwriting() -> None:
    portfolio = utils.load_portfolio()
    default_rate = float(portfolio["macro_interest_rate"].iloc[0])

    st.subheader("Evaluate a new application")
    with st.form("underwriting_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Applicant**")
            age = st.slider("Age", 21, 70, 35)
            annual_income = st.number_input(
                "Annual income (MYR)", min_value=12_000, max_value=1_000_000,
                value=60_000, step=1_000)
            dti = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.30, 0.01,
                            help="Total monthly debt repayments divided by monthly income.")
        with c2:
            st.markdown("**Facility**")
            loan_type = st.selectbox(
                "Loan type", list(utils.LOAN_TYPE_LABELS),
                format_func=utils.LOAN_TYPE_LABELS.get)
            credit_limit = st.number_input(
                "Credit limit (MYR)", min_value=1_000, max_value=500_000,
                value=15_000, step=500)
            avg_balance = st.number_input(
                "Average revolving balance, last 6 months (MYR)",
                min_value=0, max_value=600_000, value=6_000, step=500)
        with c3:
            st.markdown("**History and environment**")
            months_on_book = st.slider("Months on book", 0, 24, 12)
            past_defaults = st.number_input("Past defaults", min_value=0, max_value=10, value=0)
            macro_rate = st.number_input(
                "Central bank interest rate (%)", min_value=0.0, max_value=20.0,
                value=default_rate, step=0.25,
                help="Pre-filled with the rate scraped in Task 1.")

        submitted = st.form_submit_button("Evaluate Application", type="primary")

    if not submitted:
        st.info("Enter the applicant's details and select Evaluate Application.")
        return

    applicant = utils.build_applicant_frame(
        age=age, annual_income=float(annual_income), debt_to_income_ratio=dti,
        macro_interest_rate=macro_rate, loan_type=loan_type,
        credit_limit=float(credit_limit),
        avg_revolving_balance_last_6m=float(avg_balance),
        months_on_book=months_on_book, past_default_count=int(past_defaults),
    )
    model = get_model()                      # champion loaded here (cached)
    probability = float(utils.predict_default_probability(model, applicant).iloc[0])
    decision = utils.classify_decision(probability)

    r1, r2 = st.columns([1, 2])
    r1.metric("Predicted probability of default", f"{probability:.1%}")
    r1.progress(min(probability, 1.0))
    message = f"{decision.icon} **{decision.label}**"
    with r2:
        if decision.label == "Approved":
            st.success(f"{message}: probability is below 25%.")
        elif decision.label == "Manual Review":
            st.warning(f"{message}: probability is between 25% and 50%. Refer to a credit officer.")
        else:
            st.error(f"{message}: probability is above 50%.")
        st.caption(
            f"Derived credit utilisation: "
            f"{applicant['credit_utilisation_last_6m'].iloc[0]:.0%} of limit."
        )

    with st.expander("Inputs sent to the model"):
        st.dataframe(applicant, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🏦 Credit Risk Dashboard")
    try:
        metadata = get_metadata()
        get_model()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()

    champion = metadata["champion"]
    with st.sidebar:
        st.header("Champion model")
        st.metric("Algorithm", champion)
        st.metric("Test ROC-AUC", f"{metadata['results'][champion]['test_roc_auc']:.3f}")
        st.metric("Test F1 (tuned threshold)",
                  f"{metadata['results'][champion]['test_f1_at_tuned_threshold']:.3f}")
        st.caption(f"Trained {metadata['trained_at_utc'][:10]} on "
                   f"{metadata['n_train']:,} customers.")
        st.caption("Prototype built on synthetic data. Not for real lending decisions.")

    tab_batch, tab_underwrite = st.tabs(["Batch metrics", "Application underwriting"])
    with tab_batch:
        render_batch_metrics()
    with tab_underwrite:
        render_underwriting()


main()
