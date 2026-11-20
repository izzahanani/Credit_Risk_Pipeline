"""Task 3: benchmark XGBoost vs CatBoost and save the champion.

Flow:
    load parquet -> stratified split -> preprocessing ColumnTransformer
    -> tune each model with cross-validated random search (ROC-AUC)
    -> evaluate once on the untouched test set -> save best by ROC-AUC
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config

logger = logging.getLogger(__name__)
SEED = config.RANDOM_SEED


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(config.FEATURE_MATRIX_PARQUET)
    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    return df[features], df[config.TARGET]


def split_data(X: pd.DataFrame, y: pd.Series):
    """80/20 split, stratified so both sets keep the ~11% default rate."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=SEED
    )


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical ones.

    Tree models do not need scaling, but the brief asks for it and it keeps
    the pipeline reusable for linear models. handle_unknown="ignore" means
    an unseen loan type at scoring time will not crash the dashboard.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# 3. Candidate models + search spaces
# ---------------------------------------------------------------------------
def candidate_models() -> dict[str, tuple[object, dict]]:
    """Return {name: (estimator, hyper-parameter search space)}.

    Class weights are deliberately NOT used: re-weighting inflates predicted
    probabilities, and the dashboard's decision rule (25% / 50%) relies on
    probabilities that reflect the true default rate. ROC-AUC is unaffected
    by imbalance, and stratified folds handle it during validation.
    """
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
    )
    xgb_space = {
        "model__n_estimators": randint(150, 600),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(0.1, 10),
        "model__gamma": uniform(0, 2),
    }

    cat = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=SEED,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )
    cat_space = {
        "model__iterations": randint(150, 600),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__depth": randint(2, 7),
        "model__l2_leaf_reg": loguniform(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__random_strength": uniform(0, 3),
    }
    return {"XGBoost": (xgb, xgb_space), "CatBoost": (cat, cat_space)}


# ---------------------------------------------------------------------------
# 4. Tune + evaluate
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    best_params: dict
    cv_roc_auc: float
    test_roc_auc: float
    test_f1_at_050: float
    tuned_threshold: float
    test_f1_at_tuned: float
    report: str
    feature_importance: dict = field(default_factory=dict)


def _best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Threshold maximising F1, chosen on out-of-fold TRAIN predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def _feature_importance(pipeline: Pipeline) -> dict[str, float]:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    scores = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    scores = scores / scores.sum()                 # normalise to sum to 1
    ranked = sorted(zip(names, scores), key=lambda kv: kv[1], reverse=True)
    return {name: round(float(score), 5) for name, score in ranked}


def tune_and_evaluate(name, estimator, space, X_train, X_test, y_train, y_test) -> ModelResult:
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=config.N_SEARCH_ITERATIONS,
        scoring="roc_auc",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,                                # refit best on all train data
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    logger.info("%s: best CV ROC-AUC %.4f", name, search.best_score_)

    # F1 depends on the cut-off. Pick one on the training data only.
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = _best_f1_threshold(y_train, oof)

    proba = best.predict_proba(X_test)[:, 1]
    pred_050 = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= threshold).astype(int)

    return ModelResult(
        name=name,
        pipeline=best,
        best_params={k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                     for k, v in search.best_params_.items()},
        cv_roc_auc=float(search.best_score_),
        test_roc_auc=float(roc_auc_score(y_test, proba)),
        test_f1_at_050=float(f1_score(y_test, pred_050)),
        tuned_threshold=threshold,
        test_f1_at_tuned=float(f1_score(y_test, pred_tuned)),
        report=classification_report(
            y_test, pred_tuned, target_names=["non-default", "default"], digits=3
        ),
        feature_importance=_feature_importance(best),
    )


# ---------------------------------------------------------------------------
# 5. Compare + persist
# ---------------------------------------------------------------------------
def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [r.name for r in results],
            "cv_roc_auc": [r.cv_roc_auc for r in results],
            "test_roc_auc": [r.test_roc_auc for r in results],
            "test_f1@0.50": [r.test_f1_at_050 for r in results],
            "f1_threshold": [r.tuned_threshold for r in results],
            "test_f1@tuned": [r.test_f1_at_tuned for r in results],
        }
    ).round(4)


def save_champion(results: list[ModelResult], n_train: int, n_test: int) -> ModelResult:
    """Persist the model with the highest test ROC-AUC, plus metadata."""
    champion = max(results, key=lambda r: r.test_roc_auc)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # The whole Pipeline (preprocessing + model) is saved, so the dashboard
    # can pass raw applicant data straight to predict_proba.
    joblib.dump(champion.pipeline, config.CHAMPION_MODEL_PATH)

    metadata = {
        "champion": champion.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_metric": "test_roc_auc",
        "n_train": n_train,
        "n_test": n_test,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "results": {
            r.name: {
                "cv_roc_auc": round(r.cv_roc_auc, 4),
                "test_roc_auc": round(r.test_roc_auc, 4),
                "test_f1_at_0.50": round(r.test_f1_at_050, 4),
                "f1_threshold": round(r.tuned_threshold, 4),
                "test_f1_at_tuned_threshold": round(r.test_f1_at_tuned, 4),
                "best_params": r.best_params,
            }
            for r in results
        },
        "feature_importance": champion.feature_importance,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return champion


def write_report(results: list[ModelResult], champion: ModelResult) -> str:
    lines = ["MODEL BENCHMARK: XGBoost vs CatBoost", "=" * 60, "",
             comparison_table(results).to_string(index=False), ""]
    for r in results:
        lines += [
            "-" * 60,
            f"{r.name} - classification report on test set "
            f"(threshold = {r.tuned_threshold:.3f})",
            r.report,
            f"Best parameters: {json.dumps(r.best_params, default=str)}",
            "",
        ]
    lines += ["=" * 60,
              f"CHAMPION (highest test ROC-AUC): {champion.name} "
              f"({champion.test_roc_auc:.4f})"]
    text = "\n".join(lines)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.BENCHMARK_REPORT_PATH.write_text(text)
    return text


def run_benchmark() -> tuple[list[ModelResult], ModelResult, str]:
    X, y = load_feature_matrix()
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Train %d rows (%.2f%% default) | Test %d rows (%.2f%% default)",
                len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean())

    results = [
        tune_and_evaluate(name, est, space, X_train, X_test, y_train, y_test)
        for name, (est, space) in candidate_models().items()
    ]
    champion = save_champion(results, len(X_train), len(X_test))
    return results, champion, write_report(results, champion)
