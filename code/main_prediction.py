from __future__ import annotations

import importlib.metadata
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT / "data" / "sim_data" / "simulated_dataset.csv"
OUTPUT = PROJECT / "output"
FIGURES = OUTPUT / "figures"
PREDICTIONS = OUTPUT / "predictions"
SEED = 2026
BOOTSTRAP_N = 2000
FOLDS = 5
PRIMARY = ["AGE", "SEX", "HCT", "RDW", "RBC", "MCV", "MCH", "MCHC", "HGB"]
CBC = ["HCT", "RDW", "RBC", "MCV", "MCH", "MCHC", "HGB"]
NO_RBC = ["AGE", "SEX", "HCT", "RDW", "MCV", "MCH", "MCHC", "HGB"]
TASK1_MODELS = [
    "TabPFN", "XGBoost", "LogisticRegression", "NPJ-CatBoost-ZeroShot",
    "NPJ-CatBoost-Retrained", "CatBoost", "RandomForest", "LightGBM",
    "GradientBoosting",
]
TASK2_MODELS = [
    "TabPFN", "XGBoost", "NPJ-CatBoost-Retrained", "CatBoost", "RandomForest",
    "LogisticRegression", "LightGBM", "NPJ-CatBoost-ZeroShot",
    "Formula-Shine_Lal", "Formula-Green_King", "Formula-Mentzer",
    "Formula-Sirdah", "Formula-England_Fraser",
]
COLORS = {
    "TabPFN": "#0F4D92",
    "XGBoost": "#3775BA",
    "LogisticRegression": "#8BCF8B",
    "CatBoost": "#42949E",
    "RandomForest": "#9A4D8E",
    "LightGBM": "#767676",
    "NPJ-CatBoost-ZeroShot": "#B64342",
    "NPJ-CatBoost-Retrained": "#EA84DD",
    "Formula-Shine_Lal": "#FFD700",
    "Formula-Mentzer": "#E9A6A1",
    "Formula-Green_King": "#6E9F3F",
    "Formula-Sirdah": "#B84E4E",
    "Formula-England_Fraser": "#4C7B8F",
    "GradientBoosting": "#7D6BA8",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.2,
    "legend.frameon": False,
})

def save_figure(fig: plt.Figure, name: str, pdf: bool | None = None) -> None:
    if pdf is None:
        pdf = name.startswith("Fig") or name.startswith("Figure_R")
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    if pdf:
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def split_data(data: pd.DataFrame, task: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_column = f"TASK{task}_SPLIT"
    if task == 1:
        source = data.copy()
        source["target"] = source["TASK1_TARGET"]
    else:
        source = data[data["TASK2_TARGET"].isin(["alpha", "beta"])].copy()
        source["target"] = source["TASK2_TARGET"].map({"alpha": 0, "beta": 1}).astype(int)
        if not {"development_train", "development_internal_test", "temporal_external"}.issubset(source[split_column].dropna().unique()):
            source[split_column] = source["TASK1_SPLIT"]
    train = source[source[split_column] == "development_train"].reset_index(drop=True)
    internal = source[source[split_column] == "development_internal_test"].reset_index(drop=True)
    temporal = source[source[split_column] == "temporal_external"].reset_index(drop=True)
    if train.empty or internal.empty or temporal.empty:
        raise ValueError(f"Missing split for task {task}")
    return train, internal, temporal


def numeric(frame: pd.DataFrame, features: list[str], fill: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    fill = values.median() if fill is None else fill
    return values.fillna(fill), fill


def published(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame({
        "0": frame["SEX"].map({1.0: 0.0, 2.0: 1.0}).fillna(0.0),
        "1": frame["AGE"],
        "2": frame["RBC"],
        "3": frame["HGB"],
        "4": frame["HCT"],
        "5": frame["MCV"],
        "6": frame["MCH"],
        "7": frame["MCHC"],
        "8": pd.to_numeric(frame["RDW"], errors="coerce") / 100,
        "9": 40.5,
    })
    return values.apply(pd.to_numeric, errors="coerce").fillna(values.median(numeric_only=True))


def model(name: str, x: pd.DataFrame, y: np.ndarray, task: int) -> object:
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier

        estimator = TabPFNClassifier(
            n_estimators=8,
            random_state=SEED if task == 1 else 0,
            device="cpu",
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
        )
    elif name == "XGBoost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=500,
            learning_rate=.03,
            max_depth=3,
            subsample=.8,
            colsample_bytree=.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
        )
    elif name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=500,
            learning_rate=.03,
            num_leaves=15,
            class_weight="balanced",
            random_state=SEED,
            verbosity=-1,
            n_jobs=1,
        )
    elif name == "CatBoost":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=500,
            learning_rate=.03,
            depth=4,
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
            thread_count=1,
        )
    elif name == "NPJ-CatBoost-Retrained":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=1000,
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
            thread_count=1,
        )
    elif name == "GradientBoosting":
        estimator = GradientBoostingClassifier(random_state=SEED)
    elif name == "LogisticRegression":
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000 if task == 1 else 3000,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])
    else:
        raise ValueError(name)
    if name == "GradientBoosting":
        counts = np.bincount(y)
        weights = np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0]))
        estimator.fit(x, y, sample_weight=weights)
    else:
        estimator.fit(x, y)
    return estimator


def load_external(task: int) -> object | None:
    from catboost import CatBoostClassifier

    filename = "catboost_model_smote_thalassemia.cbm" if task == 1 else "catboost_model_smote_ab.cbm"
    path = PROJECT / "data" / "external_models" / filename
    if not path.exists():
        return None
    fitted = CatBoostClassifier()
    fitted.load_model(str(path))
    return fitted


def formula(name: str, frame: pd.DataFrame) -> np.ndarray:
    mcv = frame["MCV"].to_numpy(float)
    rbc = frame["RBC"].to_numpy(float)
    mch = frame["MCH"].to_numpy(float)
    rdw = frame["RDW"].to_numpy(float)
    hgb = frame["HGB"].to_numpy(float) / 10
    raw = {
        "Formula-Mentzer": mcv / rbc,
        "Formula-Shine_Lal": mcv**2 * mch / 100,
        "Formula-England_Fraser": mcv - rbc - 5 * hgb - 3.4,
        "Formula-Green_King": mcv**2 * rdw / (100 * hgb),
        "Formula-Sirdah": mcv - rbc - 3 * hgb,
    }
    cutoff = {
        "Formula-Mentzer": 13,
        "Formula-Shine_Lal": 1530,
        "Formula-England_Fraser": 0,
        "Formula-Green_King": 72,
        "Formula-Sirdah": 27,
    }
    return cutoff[name] - raw[name]


def task_predictions(task: int, train: pd.DataFrame, internal: pd.DataFrame, temporal: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    target = train["target"].to_numpy(int)
    from tabpfn import TabPFNClassifier

    transformer = ColumnTransformer(
        [("features", SimpleImputer(strategy="median"), PRIMARY)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    primary_model = TabPFNClassifier(
        n_estimators=8,
        random_state=SEED if task == 1 else 0,
        device="cpu",
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        n_preprocessing_jobs=1,
    )
    primary = Pipeline([
        ("features", transformer),
        ("model", primary_model),
    ])
    primary.fit(train, target)
    dump(primary, OUTPUT / f"tabpfn_task{task}_primary.joblib")
    outputs: dict[str, dict[str, np.ndarray]] = {}
    formulas = {name for name in (TASK1_MODELS + TASK2_MODELS) if name.startswith("Formula-")}
    for name in (TASK1_MODELS if task == 1 else TASK2_MODELS):
        if name in formulas:
            internal_scores = formula(name, internal)
            temporal_scores = formula(name, temporal)
            outputs[name] = {
                "development_internal_test": internal_scores,
                "temporal_external": temporal_scores,
            }
            continue
        if name == "NPJ-CatBoost-ZeroShot":
            fitted = load_external(task)
            if fitted is None:
                continue
            internal_scores = fitted.predict_proba(published(internal))[:, 1]
            temporal_scores = fitted.predict_proba(published(temporal))[:, 1]
            outputs[name] = {
                "development_internal_test": internal_scores,
                "temporal_external": temporal_scores,
            }
            continue
        if name == "TabPFN":
            internal_scores = primary.predict_proba(internal[PRIMARY])[:, 1]
            temporal_scores = primary.predict_proba(temporal[PRIMARY])[:, 1]
            outputs[name] = {
                "development_internal_test": internal_scores,
                "temporal_external": temporal_scores,
            }
            continue
        if name == "NPJ-CatBoost-Retrained":
            x_train, x_internal, x_temporal = published(train), published(internal), published(temporal)
        else:
            x_train, fill = numeric(train, PRIMARY)
            x_internal, _ = numeric(internal, PRIMARY, fill)
            x_temporal, _ = numeric(temporal, PRIMARY, fill)
        fitted = model(name, x_train, target, task)
        internal_scores = fitted.predict_proba(x_internal)[:, 1]
        temporal_scores = fitted.predict_proba(x_temporal)[:, 1]
        outputs[name] = {
            "development_internal_test": internal_scores,
            "temporal_external": temporal_scores,
        }
    if task == 1:
        for suffix, features in (("cbc", CBC), ("no_rbc", NO_RBC)):
            for name in ("TabPFN", "XGBoost"):
                x_train, fill = numeric(train, features)
                fitted = model(name, x_train, target, task)
                internal_x, _ = numeric(internal, features, fill)
                temporal_x, _ = numeric(temporal, features, fill)
                outputs[f"{name}_{suffix}"] = {
                    "development_internal_test": fitted.predict_proba(internal_x)[:, 1],
                    "temporal_external": fitted.predict_proba(temporal_x)[:, 1],
                }
    return outputs


def oof_predictions(train: pd.DataFrame) -> np.ndarray:
    target = train["target"].to_numpy(int)
    scores = np.full(len(train), np.nan)
    for fit_index, validation_index in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(train, target):
        from tabpfn import TabPFNClassifier

        fitted = TabPFNClassifier(
            n_estimators=8,
            random_state=SEED,
            device="cpu",
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
        )
        x_fit, fill = numeric(train.iloc[fit_index], PRIMARY)
        x_valid, _ = numeric(train.iloc[validation_index], PRIMARY, fill)
        fitted.fit(x_fit, target[fit_index])
        scores[validation_index] = fitted.predict_proba(x_valid)[:, 1]
    return scores


def metrics(target: np.ndarray, scores: np.ndarray, threshold: float = .5) -> dict[str, float]:
    target, scores = np.asarray(target, int), np.asarray(scores, float)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    binary = len(np.unique(target)) == 2
    return {
        "n": len(target),
        "positive_n": int(target.sum()),
        "accuracy": accuracy_score(target, predicted),
        "roc_auc": roc_auc_score(target, scores) if binary else np.nan,
        "pr_auc": average_precision_score(target, scores) if binary else np.nan,
        "sensitivity": recall_score(target, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "ppv": precision_score(target, predicted, zero_division=0),
        "f1": f1_score(target, predicted, zero_division=0),
        "brier": float(np.mean((scores - target) ** 2)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "model_positive_n": int(predicted.sum()),
    }


def bootstrap_values(target: np.ndarray, scores: np.ndarray, threshold: float, rng: np.random.Generator, index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    target, scores = np.asarray(target, int), np.asarray(scores, float)
    index = rng.integers(0, len(target), (BOOTSTRAP_N, len(target))) if index is None else index
    target, scores = target[index], scores[index]
    predicted = scores >= threshold
    tp = np.sum((target == 1) & predicted, axis=1)
    tn = np.sum((target == 0) & ~predicted, axis=1)
    fp = np.sum((target == 0) & predicted, axis=1)
    fn = np.sum((target == 1) & ~predicted, axis=1)
    positives, negatives = np.sum(target, axis=1), np.sum(target == 0, axis=1)
    order = np.argsort(-scores, axis=1)
    sorted_target = np.take_along_axis(target, order, axis=1)
    rank = np.arange(1, target.shape[1] + 1, dtype=float)
    precision_area = np.divide(
        np.sum(np.cumsum(sorted_target, axis=1) / rank * sorted_target, axis=1),
        positives,
        out=np.full(len(index), np.nan),
        where=positives > 0,
    )
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1) + 1
    roc_area = np.divide(
        np.sum(ranks * target, axis=1) - positives * (positives + 1) / 2,
        positives * negatives,
        out=np.full(len(index), np.nan),
        where=(positives > 0) & (negatives > 0),
    )
    return {
        "roc_auc": roc_area,
        "pr_auc": precision_area,
        "sensitivity": np.divide(tp, tp + fn, out=np.zeros(len(index)), where=tp + fn > 0),
        "specificity": np.divide(tn, tn + fp, out=np.full(len(index), np.nan), where=tn + fp > 0),
        "npv": np.divide(tn, tn + fn, out=np.full(len(index), np.nan), where=tn + fn > 0),
        "ppv": np.divide(tp, tp + fp, out=np.zeros(len(index)), where=tp + fp > 0),
        "f1": np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(index)), where=2 * tp + fp + fn > 0),
        "brier": np.mean((scores - target) ** 2, axis=1),
    }


def confidence_intervals(target: np.ndarray, scores: np.ndarray, threshold: float, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    return {name: tuple(np.nanpercentile(values, [2.5, 97.5])) for name, values in bootstrap_values(target, scores, threshold, rng).items()}


def metric_table(task: int, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], predictions: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for dataset, frame in zip(("development_internal_test", "temporal_external"), frames[1:]):
        target = frame["target"].to_numpy(int)
        names = [name for name in (TASK1_MODELS if task == 1 else TASK2_MODELS) if name in predictions]
        for name in names:
            threshold = 0.0 if name.startswith("Formula-") else .5
            point = metrics(target, predictions[name][dataset], threshold)
            intervals = confidence_intervals(target, predictions[name][dataset], threshold, np.random.default_rng(SEED + task * 100 + len(rows)))
            row = {
                "task": task,
                "dataset": dataset,
                "model": name,
                "threshold": threshold,
                **point,
            }
            row.update({f"{key}_ci_low": value[0] for key, value in intervals.items()})
            row.update({f"{key}_ci_high": value[1] for key, value in intervals.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_table(train: pd.DataFrame, temporal: pd.DataFrame, oof: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    target = train["target"].to_numpy(int)
    positive, negative = oof[target == 1], oof[target == 0]
    rules = [
        ("default_0.5", .5),
        ("oof_sensitivity_0.90", float(np.quantile(positive, .10, method="lower"))),
        ("oof_sensitivity_0.95", float(np.quantile(positive, .05, method="lower"))),
        ("oof_specificity_0.98", float(np.quantile(negative, .98, method="higher"))),
    ]
    target_temporal = temporal["target"].to_numpy(int)
    rows = []
    for rule, threshold in rules:
        point = metrics(target_temporal, score, threshold)
        intervals = confidence_intervals(target_temporal, score, threshold, np.random.default_rng(SEED + len(rows)))
        row = {
            "model": "TabPFN",
            "feature_set": "CBC+age+sex",
            "dataset": "temporal_external",
            "threshold_rule": rule,
            "rule": rule,
            "locked_threshold": threshold,
            "threshold": threshold,
            **point,
        }
        row.update({f"{key}_ci_low": value[0] for key, value in intervals.items()})
        row.update({f"{key}_ci_high": value[1] for key, value in intervals.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    default_fn = int(result.loc[result["threshold_rule"] == "default_0.5", "fn"].iloc[0])
    result["referral_rate"] = result["model_positive_n"] / len(temporal)
    result["false_negative_reduction_vs_default"] = default_fn - result["fn"]
    return result


def save_prediction_files(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], predictions: dict[str, dict[str, np.ndarray]], task: int) -> None:
    normalized, raw = [], []
    for dataset, frame in zip(("development_internal_test", "temporal_external"), frames[1:]):
        for name, values in predictions.items():
            if name.endswith("_no_rbc"):
                continue
            scores = np.asarray(values[dataset], float)
            threshold = 0.0 if name.startswith("Formula-") else .5
            predicted = (scores >= threshold).astype(int)
            if task == 1:
                raw.append(pd.DataFrame({
                    "task": "task1_mutation_carrier",
                    "dataset": dataset,
                    "model": name,
                    "INPATIENT_ID": frame["INPATIENT_ID"].to_numpy(),
                    "INSPECTION_DATE": frame["INSPECTION_DATE"].to_numpy(),
                    "y_true": frame["target"].to_numpy(int),
                    "y_score": scores,
                    "y_pred": predicted,
                }))
            else:
                raw.append(pd.DataFrame({
                    "task": "task2_alpha_vs_beta",
                    "dataset": dataset,
                    "method_type": "Formula" if name.startswith("Formula-") else "ML",
                    "method": name,
                    "INPATIENT_ID": frame["INPATIENT_ID"].to_numpy(),
                    "INSPECTION_DATE": frame["INSPECTION_DATE"].to_numpy(),
                    "y_true_beta": frame["target"].to_numpy(int),
                    "score_beta": scores,
                    "y_pred_beta": predicted,
                }))
            normalized.append(pd.DataFrame({
                "task": task,
                "model": name,
                "dataset": dataset,
                "INPATIENT_ID": frame["INPATIENT_ID"].to_numpy(),
                "target": frame["target"].to_numpy(int),
                "score": scores,
            }))
    pd.concat(normalized, ignore_index=True).to_csv(PREDICTIONS / f"task{task}_predictions.csv", index=False, encoding="utf-8-sig")
    raw_name = "task1_predictions_long.csv" if task == 1 else "task2_alpha_beta_predictions_long.csv"
    pd.concat(raw, ignore_index=True).to_csv(OUTPUT / raw_name, index=False, encoding="utf-8-sig")


def plot_metrics(table: pd.DataFrame, task: int) -> None:
    names = TASK1_MODELS[:6] if task == 1 else TASK2_MODELS
    table = table[table["dataset"].eq("temporal_external") & table["model"].isin(names)].copy()
    table["order"] = table["model"].map({name: i for i, name in enumerate(names)})
    table = table.sort_values("order")
    specs = [
        ("pr_auc", "PR-AUC"),
        ("sensitivity", "Sensitivity" if task == 1 else "Beta recall"),
        ("specificity", "Specificity" if task == 1 else "Alpha specificity"),
        ("npv", "NPV" if task == 1 else "Alpha NPV"),
        ("f1", "F1" if task == 1 else "Beta F1"),
    ]
    fig, axis = plt.subplots(figsize=(10.5 if task == 1 else 11, 5))
    x, width = np.arange(len(table)), .16
    for i, (metric, label) in enumerate(specs):
        axis.bar(x + (i - 2) * width, table[metric], width, color=["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E"][i], edgecolor="black", linewidth=.6, label=label)
    values = table[[metric for metric, _ in specs]].to_numpy(float)
    axis.set_ylim(max(0, np.nanmin(values) - .04), min(1.04, np.nanmax(values) + .04))
    axis.set_ylabel("Metric value", fontsize=12)
    axis.set_xticks(x)
    axis.set_xticklabels(table["model"], rotation=35, ha="right", fontsize=11.5)
    axis.set_title(f"Task {task} temporal validation metrics", fontsize=13.5)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.18), fontsize=10)
    fig.tight_layout(pad=2)
    save_figure(fig, "Fig2_task1_metrics")


def plot_task1_performance(predictions: dict[str, dict[str, np.ndarray]], frame: pd.DataFrame) -> None:
    names = [name for name in TASK1_MODELS[:6] if name in predictions]
    target = frame["target"].to_numpy(int)
    thresholds = np.linspace(.01, .50, 80)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    for name in names:
        score = predictions[name]["temporal_external"]
        fpr, tpr, _ = roc_curve(target, score)
        axes[0, 0].plot(fpr, tpr, lw=1.8, color=COLORS[name], label=f"{name} ({auc(fpr, tpr):.3f})")
        precision, recall, _ = precision_recall_curve(target, score)
        axes[0, 1].plot(recall, precision, lw=1.8, color=COLORS[name], label=f"{name} ({average_precision_score(target, score):.3f})")
        fraction, mean_probability = calibration_curve(target, score, n_bins=10, strategy="quantile")
        axes[1, 0].plot(mean_probability, fraction, marker="o", ms=3, lw=1.5, color=COLORS[name], label=name)
        benefit = [np.mean((score >= threshold) & (target == 1)) - np.mean((score >= threshold) & (target == 0)) * threshold / (1 - threshold) for threshold in thresholds]
        axes[1, 1].plot(thresholds, benefit, lw=1.8, color=COLORS[name], label=name)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#767676")
    axes[0, 1].axhline(target.mean(), color="#767676", linestyle="--", label="Prevalence")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#767676")
    axes[1, 1].plot(thresholds, np.zeros_like(thresholds), "--", color="#767676", label="Treat none")
    axes[1, 1].plot(thresholds, target.mean() - (1 - target.mean()) * thresholds / (1 - thresholds), ":", color="#4D4D4D", label="Treat all")
    panels = [
        (axes[0, 0], "ROC curves", "False positive rate", "True positive rate"),
        (axes[0, 1], "Precision-recall curves", "Recall", "Precision"),
        (axes[1, 0], "Calibration", "Mean predicted probability", "Observed carrier rate"),
        (axes[1, 1], "Decision curve analysis", "Threshold probability", "Net benefit"),
    ]
    for axis, title, xlabel, ylabel in panels:
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(color="#D9D9D9", linewidth=.6)
        axis.legend(fontsize=7.5, loc="best")
    fig.suptitle("Task 1 discrimination, calibration, and decision analysis", y=1.01, fontsize=15)
    fig.tight_layout()
    save_figure(fig, "Fig3_task1_performance")


def plot_task2_performance(table: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]], frame: pd.DataFrame) -> None:
    names = [name for name in TASK2_MODELS if name in predictions]
    table = table[table["dataset"].eq("temporal_external") & table["model"].isin(names)].copy()
    table["order"] = table["model"].map({name: i for i, name in enumerate(names)})
    table = table.sort_values("order")
    target = frame["target"].to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    specs = [
        ("pr_auc", "PR-AUC"),
        ("sensitivity", "Beta recall"),
        ("specificity", "Alpha specificity"),
        ("npv", "Alpha NPV"),
        ("f1", "Beta F1"),
    ]
    x, width = np.arange(len(table)), .15
    for index, (metric, label) in enumerate(specs):
        colors = ["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E"]
        axes[0].bar(
            x + (index - 2) * width,
            table[metric],
            width,
            color=colors[index],
            edgecolor="black",
            linewidth=.5,
            label=label,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(table["model"], rotation=50, ha="right", fontsize=8)
    axes[0].set_ylim(max(0, table[[metric for metric, _ in specs]].to_numpy(float).min() - .04), 1.04)
    axes[0].set(title="Summary metrics", ylabel="Metric value")
    axes[0].legend(fontsize=7, ncol=2)
    for name in names:
        score = predictions[name]["temporal_external"]
        fpr, tpr, _ = roc_curve(target, score)
        precision, recall, _ = precision_recall_curve(target, score)
        axes[1].plot(fpr, tpr, lw=1.6, color=COLORS[name], label=f"{name} ({auc(fpr, tpr):.3f})")
        axes[2].plot(recall, precision, lw=1.6, color=COLORS[name], label=f"{name} ({average_precision_score(target, score):.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="#767676")
    axes[2].axhline(target.mean(), color="#767676", linestyle="--", label="Prevalence")
    axes[1].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[2].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for axis in axes[1:]:
        axis.grid(color="#D9D9D9", linewidth=.6)
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("Task 2 temporal validation performance", y=1.02, fontsize=15)
    fig.tight_layout()
    save_figure(fig, "Fig7_task2_performance")


def plot_confusion(predictions: dict[str, dict[str, np.ndarray]], frame: pd.DataFrame, task: int) -> None:
    names = [name for name in (TASK1_MODELS if task == 1 else TASK2_MODELS) if name in predictions]
    columns, rows = (3, (len(names) + 2) // 3) if task == 1 else (4, (len(names) + 3) // 4)
    target = frame["target"].to_numpy(int)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    labels = ["Non-carrier", "Carrier"] if task == 1 else ["Alpha", "Beta"]
    for index, name in enumerate(names):
        threshold = 0.0 if name.startswith("Formula-") else .5
        matrix = confusion_matrix(target, predictions[name]["temporal_external"] >= threshold, labels=[0, 1]).astype(float)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        axes[index].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axes[index].set_title(name, fontsize=9)
        axes[index].set_xticks([0, 1])
        axes[index].set_xticklabels(labels, rotation=30 if task == 1 else 0, ha="right" if task == 1 else "center")
        axes[index].set_yticks([0, 1])
        axes[index].set_yticklabels(labels)
        for row in range(2):
            for column in range(2):
                text_color = "white" if normalized[row, column] > .55 else "black"
                axes[index].text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n{normalized[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
    for axis in axes[len(names):]:
        axis.set_visible(False)
    fig.tight_layout(pad=2)
    save_figure(fig, "Fig5_task1_confusion_matrices" if task == 1 else "Fig8_task2_confusion_matrices")


def plot_ablation(predictions: dict[str, dict[str, np.ndarray]], frame: pd.DataFrame) -> None:
    target = frame["target"].to_numpy(int)
    rows = []
    for name in ("TabPFN", "XGBoost"):
        for feature_set, key in (("CBC-only", f"{name}_cbc"), ("CBC+age+sex", name)):
            point = metrics(target, predictions[key]["temporal_external"])
            intervals = confidence_intervals(target, predictions[key]["temporal_external"], .5, np.random.default_rng(SEED + len(rows)))
            row = {
                "model": name,
                "feature_set": feature_set,
                **point,
            }
            row.update({f"{metric}_ci_low": bounds[0] for metric, bounds in intervals.items()})
            row.update({f"{metric}_ci_high": bounds[1] for metric, bounds in intervals.items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    specs = [("TabPFN", "CBC-only"), ("TabPFN", "CBC+age+sex"), ("XGBoost", "CBC-only"), ("XGBoost", "CBC+age+sex")]
    labels = ["TabPFN\nCBC", "TabPFN\nCBC + age + sex", "XGBoost\nCBC", "XGBoost\nCBC + age + sex"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharex=True)
    for axis, (metric, title) in zip(axes, (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier score"))):
        selected = [table[(table["model"] == model) & (table["feature_set"] == feature_set)].iloc[0] for model, feature_set in specs]
        values = np.array([row[metric] for row in selected], float)
        low = values - np.array([row[f"{metric}_ci_low"] for row in selected], float)
        high = np.array([row[f"{metric}_ci_high"] for row in selected], float) - values
        bars = axis.bar(np.arange(4), values, color=[COLORS["TabPFN"], COLORS["TabPFN"], COLORS["XGBoost"], COLORS["XGBoost"]], edgecolor="black", linewidth=.6)
        for bar, alpha in zip(bars, (.45, 1, .45, 1)):
            bar.set_alpha(alpha)
        axis.errorbar(np.arange(4), values, yerr=[low, high], fmt="none", ecolor="black", elinewidth=1, capsize=2)
        axis.set(title=title, xticks=np.arange(4), xticklabels=labels, ylim=(0, 1.05 if metric != "brier" else max(.12, values.max() * 1.35)))
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", color="#D9D9D9", linewidth=.6)
    axes[0].set_ylabel("Temporal validation value")
    fig.suptitle("Feature-set ablation in temporal validation", y=1.03)
    fig.tight_layout()
    save_figure(fig, "Fig4_feature_set_ablation")


def plot_thresholds(table: pd.DataFrame) -> None:
    order = ["default_0.5", "oof_sensitivity_0.90", "oof_sensitivity_0.95", "oof_specificity_0.98"]
    table = table.set_index("threshold_rule").reindex(order).reset_index()
    labels = ["0.5", "Sens 0.90", "Sens 0.95", "Spec 0.98"]
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    sensitivity_error = [
        table["sensitivity"] - table["sensitivity_ci_low"],
        table["sensitivity_ci_high"] - table["sensitivity"],
    ]
    specificity_error = [
        table["specificity"] - table["specificity_ci_low"],
        table["specificity_ci_high"] - table["specificity"],
    ]
    axes[0].errorbar(
        x,
        table["sensitivity"],
        yerr=sensitivity_error,
        marker="o", lw=1.8, capsize=3,
        color="#B64342", label="Sensitivity",
    )
    axes[0].errorbar(
        x,
        table["specificity"],
        yerr=specificity_error,
        marker="o", lw=1.8, capsize=3,
        color="#42949E", label="Specificity",
    )
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1.04), ylabel="Temporal validation metric", title="Locked operating points")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=.6)
    axes[1].plot(x, table["referral_rate"], marker="o", lw=1.8, color="#6A9BCB", label="Referral rate")
    right = axes[1].twinx()
    right.plot(x, table["false_negative_reduction_vs_default"], marker="s", lw=1.8, color="#B64342", label="FN reduction")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Referral rate", title="Predicted referral workload")
    axes[1].tick_params(axis="x", rotation=25)
    right.set_ylabel("False-negative reduction")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=.6)
    lines = axes[1].get_lines() + right.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], fontsize=8)
    fig.suptitle("OOF-derived thresholds: CBC + age + sex TabPFN", y=1.03)
    fig.tight_layout()
    save_figure(fig, "Fig6_oof_threshold_tradeoff")


print("Step 1/8: prepare output", flush=True)
for folder in (OUTPUT, FIGURES, PREDICTIONS):
    folder.mkdir(parents=True, exist_ok=True)

print("Step 2/8: load and split data", flush=True)
data = pd.read_csv(DATA_FILE, dtype=str, encoding="utf-8-sig")
for column in PRIMARY + ["TASK1_TARGET"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data["SEX"] = data["SEX"].fillna(data["SEX"].mode().iloc[0]).astype(float)
data["TASK1_TARGET"] = data["TASK1_TARGET"].astype(int)
task1, task2 = split_data(data, 1), split_data(data, 2)
print(f"data={DATA_FILE} n={len(data)}", flush=True)

print("Step 3/8: fit models and generate predictions", flush=True)
predictions1, predictions2 = task_predictions(1, *task1), task_predictions(2, *task2)

print("Step 4/8: calculate locked thresholds", flush=True)
oof = oof_predictions(task1[0])
thresholds = threshold_table(task1[0], task1[2], oof, predictions1["TabPFN"]["temporal_external"])

print("Step 5/8: prepare plot data", flush=True)
task1_metrics = metric_table(1, task1, predictions1)
task2_metrics = metric_table(2, task2, predictions2)

print("Step 6/8: save model predictions", flush=True)
thresholds.to_csv(OUTPUT / "thresholds.csv", index=False, encoding="utf-8-sig")
save_prediction_files(task1, predictions1, 1)
save_prediction_files(task2, predictions2, 2)

print("Step 7/8: generate figures", flush=True)
plot_metrics(task1_metrics, 1)
plot_task1_performance(predictions1, task1[2])
plot_task2_performance(task2_metrics, predictions2, task2[2])
plot_confusion(predictions1, task1[2], 1)
plot_confusion(predictions2, task2[2], 2)
plot_ablation(predictions1, task1[2])
plot_thresholds(thresholds)

print("Step 8/8: save run metadata", flush=True)
software = {}
for package in ("numpy", "pandas", "scikit-learn", "tabpfn", "torch", "xgboost", "lightgbm", "catboost", "shap", "PyALE", "scipy", "matplotlib"):
    try:
        software[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
metadata = {
    "data_file": DATA_FILE.relative_to(PROJECT).as_posix(),
    "seed": SEED,
    "bootstrap_n": BOOTSTRAP_N,
    "folds": FOLDS,
    "saved_models": [
        "output/tabpfn_task1_primary.joblib",
        "output/tabpfn_task2_primary.joblib",
    ],
    "software": software,
}
(OUTPUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"saved={OUTPUT}", flush=True)
