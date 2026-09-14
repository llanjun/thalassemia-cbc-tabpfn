from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from PyALE import ale
import shap
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "sim_data" / "simulated_dataset.csv"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TASK1_PREDICTIONS = OUTPUT / "task1_predictions_long.csv"
TASK2_PREDICTIONS = OUTPUT / "task2_alpha_beta_predictions_long.csv"
THRESHOLDS = OUTPUT / "thresholds.csv"
FEATURES = ["AGE", "SEX", "HGB", "RBC", "MCV", "MCH", "MCHC", "RDW", "HCT"]
PRIMARY = ["AGE", "SEX", "HCT", "RDW", "RBC", "MCV", "MCH", "MCHC", "HGB"]
PCA_FEATURES = ["HGB", "RBC", "MCV", "MCH", "MCHC", "RDW", "HCT"]
BOOTSTRAP_N = 2000
SEED = 2026
PROFILE_RULES = ("default_0.5", "oof_sensitivity_0.90", "oof_sensitivity_0.95")
GENOTYPE_COLORS = {"non_carrier": "#BDBDBD", "alpha_only": "#0F4D92", "beta_only": "#B64342", "mixed_or_other": "#6A6A6A"}
GENOTYPE_MARKERS = {"non_carrier": ".", "alpha_only": "o", "beta_only": "^", "mixed_or_other": "s"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.2,
    "legend.frameon": False,
})
def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    if name.startswith("Figure_R"):
        fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = [DATA_FILE, TASK1_PREDICTIONS, TASK2_PREDICTIONS, THRESHOLDS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run main_prediction.py first:\n" + "\n".join(missing))
    data = pd.read_csv(DATA_FILE, dtype=str, encoding="utf-8-sig")
    data["INPATIENT_ID"] = data["INPATIENT_ID"].astype(str).str.strip()
    for column in FEATURES + ["TASK1_TARGET"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["TASK1_TARGET"] = data["TASK1_TARGET"].astype(int)
    task1 = pd.read_csv(TASK1_PREDICTIONS, encoding="utf-8-sig")
    task2 = pd.read_csv(TASK2_PREDICTIONS, encoding="utf-8-sig")
    threshold = pd.read_csv(THRESHOLDS, encoding="utf-8-sig")
    task1["INPATIENT_ID"] = task1["INPATIENT_ID"].astype(str).str.strip()
    task2["INPATIENT_ID"] = task2["INPATIENT_ID"].astype(str).str.strip()
    if "score" not in task1 and "y_score" in task1:
        task1["score"] = task1["y_score"]
    task1 = task1[(task1["model"] == "TabPFN") & (task1["dataset"] == "temporal_external")].copy()
    task2 = task2[(task2["method"] == "TabPFN") & (task2["dataset"] == "temporal_external")].copy()
    rows = threshold[(threshold["model"] == "TabPFN") & (threshold["dataset"] == "temporal_external")]
    rows = rows[rows["threshold_rule"].isin(PROFILE_RULES)]
    if task1.empty or task2.empty or rows.empty:
        raise ValueError("TabPFN temporal predictions or thresholds are incomplete")
    thresholds = rows.drop_duplicates("threshold_rule").set_index("threshold_rule")["threshold"].astype(float).to_dict()
    missing = [rule for rule in PROFILE_RULES if rule not in thresholds]
    if missing:
        raise ValueError(f"Missing thresholds: {', '.join(missing)}")
    return data, task1, task2, thresholds


def numeric(frame: pd.DataFrame, features: list[str], fill: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    fill = values.median() if fill is None else fill
    return values.fillna(fill), fill


def interpretability(task: int, train: pd.DataFrame, temporal: pd.DataFrame, fitted: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x, fill = numeric(train, PRIMARY)
    temporal_x, _ = numeric(temporal, PRIMARY, fill)
    rng = np.random.default_rng(SEED)
    background = train_x.iloc[rng.choice(len(train_x), min(50, len(train_x)), replace=False)]
    explained_index = rng.choice(len(temporal_x), min(100, len(temporal_x)), replace=False)
    explained = temporal_x.iloc[explained_index]
    explainer = shap.PermutationExplainer(fitted.predict_proba, background)
    explanation = explainer(explained, max_evals=2 * len(PRIMARY) + 1, batch_size=32)
    values = explanation.values
    values = values[:, :, 1] if values.ndim == 3 else values
    prefix = f"task{task}"
    summary = pd.DataFrame({
        "task": prefix,
        "dataset": "temporal_external",
        "model": "TabPFN",
        "feature": PRIMARY,
        "mean_abs_shap": np.abs(values).mean(0),
        "mean_shap": values.mean(0),
        "std_abs_shap": np.abs(values).std(0),
        "n_explained": values.shape[0],
    }).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(OUTPUT / f"{prefix}_tabpfn_shap_summary_temporal_external.csv", index=False, encoding="utf-8-sig")
    scores = fitted.predict_proba(temporal_x)[:, 1]
    rows = []
    for index, feature in enumerate(PRIMARY):
        rows.append(pd.DataFrame({
            "task": prefix,
            "dataset": "temporal_external",
            "model": "TabPFN",
            "INPATIENT_ID": temporal.iloc[explained_index]["INPATIENT_ID"].to_numpy(),
            "INSPECTION_DATE": temporal.iloc[explained_index]["INSPECTION_DATE"].to_numpy(),
            "feature": feature,
            "feature_value": explained[feature].to_numpy(),
            "shap_value": values[:, index],
            "y_true": temporal.iloc[explained_index]["target"].to_numpy(int),
            "y_score": scores[explained_index],
        }))
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / f"{prefix}_tabpfn_shap_values_temporal_external.csv", index=False, encoding="utf-8-sig")
    ale_x = temporal_x.sample(min(300 if task == 1 else 155, len(temporal_x)), random_state=SEED)
    wrapped = type("Model", (), {"predict": lambda self, values: fitted.predict_proba(pd.DataFrame(values, columns=PRIMARY))[:, 1]})()
    ale_rows = []
    for feature in PRIMARY:
        result = ale(
            X=ale_x,
            model=wrapped,
            feature=[feature],
            grid_size=20,
            include_CI=True,
            plot=False,
        )
        result = result.reset_index().rename(columns={feature: "feature_value"})
        result["task"], result["dataset"], result["model"], result["feature"] = prefix, "temporal_external", "TabPFN", feature
        ale_rows.append(result)
    ale_data = pd.concat(ale_rows, ignore_index=True)
    ale_data.to_csv(OUTPUT / f"{prefix}_tabpfn_ale_temporal_external.csv", index=False, encoding="utf-8-sig")
    return summary, ale_data


def interpretability_figures(shap_data: tuple[pd.DataFrame, pd.DataFrame], ale_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=False)
    for axis, summary, title in zip(axes, shap_data, ("Task 1 carrier screening", "Task 2 alpha-versus-beta classification")):
        current = summary.sort_values("mean_abs_shap", ascending=True)
        axis.barh(current["feature"], current["mean_abs_shap"], color="#0F4D92", edgecolor="black", linewidth=.6)
        axis.set(title=title, xlabel="Mean absolute SHAP value")
    fig.suptitle("TabPFN SHAP feature importance", y=1.02, fontsize=15)
    fig.tight_layout()
    save_figure(fig, "Fig9_tabpfn_shap_feature_importance")
    for task, current in enumerate(ale_data, 1):
        features = current["feature"].drop_duplicates().tolist()
        columns = min(len(features), 3)
        rows = (len(features) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.3 * rows))
        axes = np.atleast_1d(axes).ravel()
        for axis, feature in zip(axes, features):
            values = current[current["feature"] == feature].sort_values("feature_value")
            values = values[values["size"] > 0]
            axis.plot(values["feature_value"], values["eff"], color="#0F4D92", lw=2)
            mask = values["lowerCI_95%"].notna() & values["upperCI_95%"].notna()
            if mask.any():
                axis.fill_between(values.loc[mask, "feature_value"], values.loc[mask, "lowerCI_95%"], values.loc[mask, "upperCI_95%"], color="#0F4D92", alpha=.15)
            axis.axhline(0, color="#767676", lw=1, linestyle="--")
            axis.set(title=feature, xlabel=feature, ylabel="ALE effect")
        for axis in axes[len(features):]:
            axis.set_visible(False)
        fig.tight_layout(pad=2)
        save_figure(fig, f"Fig{9 + task}_task{task}_tabpfn_ale")


def binary_metrics(y_true: object, y_score: object, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = y_score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_score)
        result["pr_auc"] = average_precision_score(y_true, y_score)
    return result


def bootstrap_metrics(y_true: object, y_score: object, threshold: float, seed: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    point = binary_metrics(y_true, y_score, threshold)
    names = ["roc_auc", "pr_auc", "sensitivity", "specificity", "npv", "ppv", "f1"]
    values = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(BOOTSTRAP_N):
        index = rng.integers(0, len(y_true), len(y_true))
        sample = binary_metrics(y_true[index], y_score[index], threshold)
        for name in names:
            if np.isfinite(sample[name]):
                values[name].append(sample[name])
    for name, sample in values.items():
        sample = np.asarray(sample, dtype=float)
        point[f"{name}_ci_low"] = np.nanpercentile(sample, 2.5) if len(sample) else np.nan
        point[f"{name}_ci_high"] = np.nanpercentile(sample, 97.5) if len(sample) else np.nan
    return point


def group_labels(y_true: object, y_score: object, threshold: float) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_score, dtype=float) >= threshold
    return np.select([(y_true == 1) & ~predicted, (y_true == 1) & predicted, (y_true == 0) & predicted], ["FN", "TP", "FP"], default="TN")


def false_negative_analysis(data: pd.DataFrame, task1: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, object]:
    train = data[data["TASK1_SPLIT"] == "development_train"].copy()
    train_rbc_q75 = train.groupby("SEX")["RBC"].quantile(.75).to_dict()
    columns = ["INPATIENT_ID", "TASK1_SPLIT", "TASK2_TARGET", "TASK1_TARGET"] + FEATURES
    task1 = task1.merge(data[columns], on="INPATIENT_ID", how="left", suffixes=("", "_raw"))
    for column in FEATURES:
        task1[column] = pd.to_numeric(task1[column], errors="coerce")
    for rule in PROFILE_RULES:
        task1[f"group_{rule}"] = group_labels(task1["y_true"], task1["y_score"], thresholds[rule])
    return {"eval": task1, "train": train, "train_rbc_q75": train_rbc_q75}
def phenotype_analysis(data: pd.DataFrame, task2: pd.DataFrame) -> dict[str, object]:
    columns = ["INPATIENT_ID", "COHORT", "TASK1_SPLIT", "TASK2_SPLIT", "TASK1_TARGET", "TASK2_TARGET"] + FEATURES
    pheno = data[columns].copy()
    for column in FEATURES:
        pheno[column] = pd.to_numeric(pheno[column], errors="coerce")
    conditions = [
        pheno["TASK1_TARGET"].eq(0),
        pheno["TASK2_TARGET"].eq("alpha"),
        pheno["TASK2_TARGET"].eq("beta"),
        pheno["TASK2_TARGET"].eq("mixed_or_other"),
    ]
    labels = ["non_carrier", "alpha_only", "beta_only", "mixed_or_other"]
    pheno["genotype_group"] = np.select(conditions, labels, default="unknown")
    short = task2[["INPATIENT_ID", "y_true_beta", "score_beta", "y_pred_beta"]]
    pheno = pheno.merge(short, on="INPATIENT_ID", how="left")
    pheno["task2_error"] = pheno["y_true_beta"].notna() & pheno["y_true_beta"].ne(pheno["y_pred_beta"])
    temporal_carriers = pheno[
        (pheno["TASK1_SPLIT"] == "temporal_external")
        & pheno["genotype_group"].isin(["alpha_only", "beta_only"])
    ].dropna(subset=PCA_FEATURES).copy()
    scaler = StandardScaler().fit(temporal_carriers[PCA_FEATURES])
    pca = PCA(n_components=2, random_state=SEED).fit(scaler.transform(temporal_carriers[PCA_FEATURES]))
    pheno[["PC1", "PC2"]] = np.nan
    complete = pheno.dropna(subset=PCA_FEATURES)
    pheno.loc[complete.index, ["PC1", "PC2"]] = pca.transform(scaler.transform(complete[PCA_FEATURES]))
    temporal = pheno[pheno["TASK1_SPLIT"] == "temporal_external"].copy()
    carriers = temporal[temporal["genotype_group"].isin(["alpha_only", "beta_only"])].dropna(subset=["PC1", "PC2"]).copy()
    if len(carriers) >= 10:
        neighbors = NearestNeighbors(n_neighbors=min(16, len(carriers))).fit(carriers[["PC1", "PC2"]])
        _, indices = neighbors.kneighbors(carriers[["PC1", "PC2"]])
        labels = carriers["genotype_group"].to_numpy()
        carriers["overlap_region"] = [np.mean(labels[index[1:]] != labels[row]) >= .30 for row, index in enumerate(indices)]
    else:
        carriers["overlap_region"] = False
    overlap = carriers[carriers["overlap_region"]]
    return {
        "all": pheno,
        "temporal": temporal,
        "overlap": overlap,
        "variance": pca.explained_variance_ratio_ * 100,
    }


def subgroup_masks(frame: pd.DataFrame, train_rbc_q75: dict[float, float]) -> list[tuple[str, str, pd.Series]]:
    anemia = np.where(((frame["SEX"] == 1) & (frame["HGB"] < 130)) | ((frame["SEX"] == 2) & (frame["HGB"] < 120)), "anemic", "non_anemic")
    rbc_high = pd.Series([bool(row.RBC > train_rbc_q75.get(row.SEX, frame["RBC"].quantile(.75))) for row in frame.itertuples()], index=frame.index)
    frame["anemia"] = anemia
    frame["rbc_high"] = rbc_high
    return [
        ("sex", "Male", frame["SEX"] == 1),
        ("sex", "Female", frame["SEX"] == 2),
        ("age", "Age <30", frame["AGE"] < 30),
        ("age", "Age 30-45", (frame["AGE"] >= 30) & (frame["AGE"] < 45)),
        ("age", "Age 45-60", (frame["AGE"] >= 45) & (frame["AGE"] < 60)),
        ("age", "Age >=60", frame["AGE"] >= 60),
        ("anemia", "Anemic", frame["anemia"] == "anemic"),
        ("anemia", "Non-anemic", frame["anemia"] == "non_anemic"),
        ("microcytosis", "MCV <80", frame["MCV"] < 80),
        ("microcytosis", "MCV >=80", frame["MCV"] >= 80),
        ("hypochromia", "MCH <27", frame["MCH"] < 27),
        ("hypochromia", "MCH >=27", frame["MCH"] >= 27),
        ("rbc", "RBC high", frame["rbc_high"]),
        ("rbc", "RBC normal/low", ~frame["rbc_high"]),
    ]


def subgroup_analysis(task1: pd.DataFrame, task2: pd.DataFrame, train_rbc_q75: dict[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    task1_rows = []
    for index, (variable, label, mask) in enumerate(subgroup_masks(task1, train_rbc_q75)):
        subset = task1[mask.fillna(False)]
        if len(subset) < 5:
            continue
        row = bootstrap_metrics(subset["y_true"], subset["y_score"], .5, SEED + index)
        row.update({
            "task": "Task 1 carrier detection",
            "variable": variable,
            "subgroup": label,
            "threshold": .5,
            "estimate_flag": "exploratory" if row["n"] < 30 or row["n_positive"] < 10 else "stable",
        })
        task1_rows.append(row)

    task2_rows = []
    for index, (variable, label, mask) in enumerate(subgroup_masks(task2, train_rbc_q75)):
        subset = task2[mask.fillna(False)]
        if len(subset) < 5:
            continue
        row = bootstrap_metrics(subset["y_true_beta"], subset["score_beta"], .5, SEED + 500 + index)
        row.update({
            "task": "Task 2 alpha-vs-beta",
            "variable": variable,
            "subgroup": label,
            "threshold": .5,
            "estimate_flag": "exploratory" if row["n"] < 30 or row["n_positive"] < 10 or row["n_negative"] < 10 else "stable",
            "positive_class": "beta",
        })
        task2_rows.append(row)
    return pd.DataFrame(task1_rows), pd.DataFrame(task2_rows)
def limits(series: object, low: float, high: float) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 5:
        return low, high
    q_low, q_high = np.nanpercentile(values, [1, 99])
    span = max(q_high - q_low, 1.0)
    return max(low, q_low - .08 * span), min(high, q_high + .08 * span)


def phenotype_figure(result: dict[str, object]) -> None:
    pheno, temporal, overlap, variance = result["all"], result["temporal"], result["overlap"], result["variance"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.4))
    xlim, ylim = limits(pheno["MCV"], 45, 115), limits(pheno["MCH"], 14, 36)
    overlap_ids = set(overlap["INPATIENT_ID"].astype(str))

    def points(axis, frame, x, y):
        for group in ("non_carrier", "alpha_only", "beta_only", "mixed_or_other"):
            subset = frame[frame["genotype_group"] == group]
            if not subset.empty:
                size = 12 if group == "non_carrier" else 34
                alpha = .18 if group == "non_carrier" else .74
                axis.scatter(
                    subset[x], subset[y],
                    s=size,
                    marker=GENOTYPE_MARKERS[group],
                    color=GENOTYPE_COLORS[group],
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                )
        errors = frame[frame["task2_error"]]
        if not errors.empty:
            axis.scatter(errors[x], errors[y], s=82, facecolors="none", edgecolors="#000000", linewidths=1.2, rasterized=True)
        if overlap_ids:
            marked = frame[frame["INPATIENT_ID"].astype(str).isin(overlap_ids)]
            if not marked.empty:
                axis.scatter(marked[x], marked[y], s=96, facecolors="none", edgecolors="#D55E00", linewidths=1.1, rasterized=True)

    def cbc(axis, frame, title):
        points(axis, frame, "MCV", "MCH")
        axis.axvline(80, color="#767676", linestyle="--", linewidth=1.1)
        axis.axhline(27, color="#767676", linestyle="--", linewidth=1.1)
        axis.set(xlim=xlim, ylim=ylim, xlabel="MCV (fL)", ylabel="MCH (pg)", title=title)
        axis.grid(color="#E6E6E6", linewidth=.55)

    def pca(axis, frame, title, marked=None):
        points(axis, frame.dropna(subset=["PC1", "PC2"]), "PC1", "PC2")
        if marked is not None and len(marked) >= 3:
            center = marked[["PC1", "PC2"]].mean().to_numpy()
            width = max(marked["PC1"].std(ddof=1) * 3, .2)
            height = max(marked["PC2"].std(ddof=1) * 3, .2)
            axis.add_patch(Ellipse(center, width, height, fill=False, edgecolor="#D55E00", linewidth=1.5, linestyle="--"))
        axis.set(xlabel=f"PC1 ({variance[0]:.1f}% variance)", ylabel=f"PC2 ({variance[1]:.1f}% variance)", title=title)
        axis.grid(color="#E6E6E6", linewidth=.55)

    cbc(axes[0, 0], temporal, "Temporal validation: CBC thresholds")
    pca(axes[0, 1], temporal, "Temporal validation: PCA phenotype space", overlap)
    cbc(axes[1, 0], pheno, "Full cohort: CBC thresholds")
    pca(axes[1, 1], pheno, "Full cohort: PCA phenotype space")
    for axis, label in zip(axes.ravel(), "ABCD"):
        axis.tick_params(axis="both", labelsize=14.5)
        axis.xaxis.label.set_size(14.5)
        axis.yaxis.label.set_size(14.5)
        axis.text(-.12, 1.06, label, transform=axis.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")
    legend_items = [
        ("o", GENOTYPE_COLORS["non_carrier"], "Non-carrier"),
        ("o", GENOTYPE_COLORS["alpha_only"], "Alpha-only"),
        ("^", GENOTYPE_COLORS["beta_only"], "Beta-only"),
        ("s", GENOTYPE_COLORS["mixed_or_other"], "Mixed/other"),
    ]
    handles = [Line2D(
        [0], [0], marker=marker, linestyle="", markerfacecolor=color,
        markeredgecolor="none", markersize=8, alpha=.45, label=label,
    ) for marker, color, label in legend_items]
    handles += [Line2D(
        [0], [0], marker="o", linestyle="", markerfacecolor="none",
        markeredgecolor=color, markersize=9, label=label,
    ) for color, label in (("#000000", "Task 2 error"), ("#D55E00", "Temporal overlap region"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02), ncol=6, fontsize=13.5, columnspacing=1.2, handletextpad=.5)
    fig.tight_layout(rect=[0, .09, 1, 1])
    save_figure(fig, "Fig12_phenotype_overlap")


def subgroup_figure(task1: pd.DataFrame, task2: pd.DataFrame) -> None:
    metrics = [("sensitivity", "Sensitivity", "#B64342", "o", -.22), ("npv", "NPV", "#0F4D92", "s", 0), ("f1", "F1", "#42949E", "^", .22)]
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 14.2), sharex=True)

    def panel(axis, table, title, label, xlabel):
        if table.empty:
            axis.axis("off")
            return
        table = table.reset_index(drop=True).copy()
        table["label"] = table["variable"] + ": " + table["subgroup"] + "\n" + "n=" + table["n"].astype(int).astype(str) + ", pos=" + table["n_positive"].astype(int).astype(str)
        table.loc[table["estimate_flag"].eq("exploratory"), "label"] += " *"
        y = np.arange(len(table), dtype=float)
        for metric, metric_label, color, marker, offset in metrics:
            values = table[metric].to_numpy(float)
            low = table[f"{metric}_ci_low"].to_numpy(float)
            high = table[f"{metric}_ci_high"].to_numpy(float)
            finite = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
            for flag, face in (("stable", color), ("exploratory", "white")):
                mask = finite & table["estimate_flag"].eq(flag).to_numpy()
                if mask.any():
                    axis.errorbar(
                        values[mask],
                        y[mask] + offset,
                        xerr=[values[mask] - low[mask], high[mask] - values[mask]],
                        fmt=marker,
                        color=color,
                        markerfacecolor=face,
                        markeredgecolor=color,
                        markeredgewidth=1.2,
                        elinewidth=1.15,
                        capsize=2.8,
                        markersize=5.8,
                        label=metric_label if flag == "stable" else None,
                    )
            if (~finite).any():
                axis.scatter(np.repeat(.5, int((~finite).sum())), y[~finite] + offset, marker="x", color="#A0A0A0", s=22)
        axis.set(yticks=y, yticklabels=table["label"], xlim=(0, 1.02), xlabel="Metric value with 95% bootstrap CI" if xlabel else "", title=title)
        axis.invert_yaxis()
        axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
        axis.axvline(.80, color="#BDBDBD", linestyle="--", linewidth=.8)
        axis.axvline(.95, color="#BDBDBD", linestyle=":", linewidth=.8)
        axis.text(-.12, 1.06, label, transform=axis.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")

    panel(axes[0], task1, "Task 1: any carrier detection", "A", False)
    panel(axes[1], task2, "Task 2: beta versus alpha among carriers", "B", True)
    handles = [Line2D([0], [0], marker=m, linestyle="", markerfacecolor=c, markeredgecolor=c, markersize=7, label=l) for _, l, c, m, _ in metrics]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#767676", markersize=7, label="Exploratory subgroup"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=4, fontsize=10.5, columnspacing=1.5, handletextpad=.5)
    fig.tight_layout(rect=[0, .04, 1, 1])
    save_figure(fig, "Fig13_subgroup_performance")

def calibration_figures(data: pd.DataFrame, prediction: pd.DataFrame) -> None:
    data["INPATIENT_ID"] = data["INPATIENT_ID"].astype(str)
    prediction["INPATIENT_ID"] = prediction["INPATIENT_ID"].astype(str)
    temporal = data[data["TASK1_SPLIT"] == "temporal_external"].copy()
    required_models = {"TabPFN_cbc", "TabPFN"}
    missing = required_models - set(prediction["model"].unique())
    if missing:
        raise ValueError(f"Missing required TabPFN predictions: {', '.join(sorted(missing))}")
    temporal_predictions = prediction[
        (prediction["dataset"] == "temporal_external")
        & prediction["model"].isin(required_models)
    ].merge(temporal[["INPATIENT_ID", "SEX"]], on="INPATIENT_ID")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for axis, subgroup, title in zip(axes, ("overall", "male", "female"), ("Overall", "Male", "Female")):
        for model, color, label in (("TabPFN_cbc", "#4E9F75", "CBC-only"), ("TabPFN", "#0F4D92", "CBC + age + sex")):
            current = temporal_predictions[temporal_predictions["model"] == model]
            group = current if subgroup == "overall" else current[current["SEX"] == (1 if subgroup == "male" else 2)]
            if group.empty:
                continue
            fraction, mean_probability = calibration_curve(group["target"], group["score"], n_bins=10, strategy="quantile")
            axis.plot(mean_probability, fraction, marker="o", ms=3.5, lw=1.8, color=color, label=label)
        axis.plot([0, 1], [0, 1], color="#767676", lw=1, linestyle="--")
        axis.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel="Mean predicted probability")
        axis.grid(color="#D9D9D9", linewidth=.6, alpha=.7)
    axes[0].set_ylabel("Observed carrier rate")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("TabPFN calibration in temporal validation", y=1.03)
    fig.tight_layout()
    save_figure(fig, "Figure_R2_tabpfn_calibration")

    sex_rows = []
    for model, label in (("TabPFN_cbc", "CBC-only"), ("TabPFN", "CBC + age + sex")):
        current = temporal_predictions[temporal_predictions["model"] == model]
        for subgroup, code in (("male", 1), ("female", 2)):
            group = current[current["SEX"] == code]
            if group.empty:
                continue
            sex_rows.append({"feature_set": label, "subgroup": subgroup, **bootstrap_metrics(group["target"], group["score"], .5, SEED + len(sex_rows))})
    sex_table = pd.DataFrame(sex_rows)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.0), sharey=True)
    for axis, metric, title in zip(axes, ("roc_auc", "sensitivity", "specificity", "npv"), ("ROC-AUC", "Sensitivity", "Specificity", "NPV")):
        for sex_index, subgroup in enumerate(("male", "female")):
            for variant_index, variant in enumerate(("CBC-only", "CBC + age + sex")):
                row = sex_table[(sex_table["subgroup"] == subgroup) & (sex_table["feature_set"] == variant)]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = sex_index * 2 + variant_index
                value = float(row[metric])
                error = [[value - float(row[f"{metric}_ci_low"])], [float(row[f"{metric}_ci_high"]) - value]]
                color = "#4E9F75" if variant == "CBC-only" else "#0F4D92"
                axis.errorbar(
                    x, value, yerr=error, fmt="o", capsize=3,
                    color=color, markeredgecolor="black", markersize=5,
                )
        axis.set(xticks=[.5, 2.5], xticklabels=["Male", "Female"], ylim=(0, 1.05), title=title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=.6)
    axes[0].set_ylabel("Temporal validation value")
    legend = [
        Line2D([0], [0], marker="o", color="#4E9F75", label="CBC-only", linestyle=""),
        Line2D([0], [0], marker="o", color="#0F4D92", label="CBC + age + sex", linestyle=""),
    ]
    axes[-1].legend(handles=legend, fontsize=8, loc="lower right")
    fig.suptitle("Sex-stratified TabPFN performance", y=1.03)
    fig.tight_layout()
    save_figure(fig, "Figure_R4_sex_stratified_performance")


print("Step 1/7: prepare output", flush=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("Step 2/7: load generated predictions", flush=True)
data, task1, task2, thresholds = load_inputs()

print("Step 3/7: generate SHAP and ALE", flush=True)
task1_train = data[data["TASK1_SPLIT"] == "development_train"].copy()
task1_temporal = data[data["TASK1_SPLIT"] == "temporal_external"].copy()
task1_train["target"] = task1_train["TASK1_TARGET"]
task1_temporal["target"] = task1_temporal["TASK1_TARGET"]
task2_source = data[data["TASK2_TARGET"].isin(["alpha", "beta"])].copy()
task2_split = "TASK2_SPLIT" if {"development_train", "development_internal_test", "temporal_external"}.issubset(set(task2_source["TASK2_SPLIT"].dropna())) else "TASK1_SPLIT"
task2_train = task2_source[task2_source[task2_split] == "development_train"].copy()
task2_temporal = task2_source[task2_source[task2_split] == "temporal_external"].copy()
task2_train["target"] = task2_train["TASK2_TARGET"].map({"alpha": 0, "beta": 1}).astype(int)
task2_temporal["target"] = task2_temporal["TASK2_TARGET"].map({"alpha": 0, "beta": 1}).astype(int)
shap1, ale1 = interpretability(1, task1_train, task1_temporal, load(OUTPUT / "tabpfn_task1_primary.joblib"))
shap2, ale2 = interpretability(2, task2_train, task2_temporal, load(OUTPUT / "tabpfn_task2_primary.joblib"))
interpretability_figures((shap1, shap2), (ale1, ale2))

print("Step 4/7: analyze threshold and phenotype errors", flush=True)
fn = false_negative_analysis(data, task1, thresholds)
task1_eval = fn["eval"]
task1_eval["anemia"] = np.where(((task1_eval["SEX"] == 1) & (task1_eval["HGB"] < 130)) | ((task1_eval["SEX"] == 2) & (task1_eval["HGB"] < 120)), "anemic", "non_anemic")
task1_eval["rbc_high"] = [bool(row.RBC > fn["train_rbc_q75"].get(row.SEX, fn["train"]["RBC"].quantile(.75))) for row in task1_eval.itertuples()]
task2_eval = task2.merge(data[["INPATIENT_ID"] + FEATURES], on="INPATIENT_ID", how="left")
for column in FEATURES:
    task2_eval[column] = pd.to_numeric(task2_eval[column], errors="coerce")
pheno = phenotype_analysis(data, task2)
task1_table, task2_table = subgroup_analysis(task1_eval, task2_eval, fn["train_rbc_q75"])

print("Step 5/7: generate phenotype and subgroup figures", flush=True)
phenotype_figure(pheno)
subgroup_figure(task1_table, task2_table)

print("Step 6/7: generate calibration figures", flush=True)
prediction = pd.read_csv(OUTPUT / "predictions" / "task1_predictions.csv", encoding="utf-8-sig")
calibration_figures(data, prediction)

print("Step 7/7: finish", flush=True)
print(f"Figures: {FIGURES}", flush=True)
