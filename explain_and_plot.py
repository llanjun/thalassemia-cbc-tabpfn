import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import torch
from tabpfn import TabPFNClassifier
import shap
from PyALE import ale

SEED = 2026
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "synthetic_data.csv"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "figure.dpi": 300
})

FEATURES = ["AGE", "SEX", "HCT", "RDW", "RBC", "MCV", "MCH", "MCHC", "HGB"]

def get_models():
    return {
        "TabPFN": TabPFNClassifier(device=DEVICE, inference_precision=torch.float32, n_estimators=8, random_state=SEED, ignore_pretraining_limits=True),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.035, subsample=0.70, colsample_bytree=0.80, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=200, depth=5, learning_rate=0.03, l2_leaf_reg=10.0, auto_class_weights="SqrtBalanced", random_seed=100, thread_count=-1, verbose=False, allow_writing_files=False),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED)
    }

def plot_roc_and_pr_curves(models, X_val, y_val):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    colors = {
        "TabPFN": "#1f77b4",
        "LightGBM": "#2ca02c",
        "XGBoost": "#ff7f0e",
        "CatBoost": "#d62728",
        "RandomForest": "#9467bd",
        "LogisticRegression": "#8c564b"
    }
    
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc_val = roc_auc_score(y_val, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, "black"), lw=1.8)
        
        prec, rec, _ = precision_recall_curve(y_val, probs)
        pr_val = average_precision_score(y_val, probs)
        axes[1].plot(rec, prec, label=f"{name} (AP = {pr_val:.3f})", color=colors.get(name, "black"), lw=1.8)
        
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("1 - Specificity (False Positive Rate)")
    axes[0].set_ylabel("Sensitivity (True Positive Rate)")
    axes[0].set_title("A: Receiver Operating Characteristic", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    
    no_skill = np.sum(y_val) / len(y_val)
    axes[1].plot([0, 1], [no_skill, no_skill], "k--", lw=1.0, alpha=0.6)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_title("B: Precision-Recall Curve", loc="left", fontweight="bold")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.5)
    
    plt.tight_layout()
    out_path = FIG_DIR / "roc_pr_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_shap_feature_importance(model, X_train, X_val):
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_val)
    
    if isinstance(shap_raw, list):
        vals = shap_raw[1]
    elif len(shap_raw.shape) == 3:
        vals = shap_raw[:, :, 1]
    else:
        vals = shap_raw
        
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values(by="MeanAbsSHAP", ascending=True)
    
    importance_df.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    
    plt.figure(figsize=(7, 5))
    bars = plt.barh(importance_df["Feature"], importance_df["MeanAbsSHAP"], color="#2c7bb6", height=0.65)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    plt.title("SHAP Feature Importance (Task 1 Carrier Screening)", loc="left", fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02 * max(importance_df["MeanAbsSHAP"]), bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", ha="left", fontsize=8.5)
                 
    plt.xlim(0, max(importance_df["MeanAbsSHAP"]) * 1.15)
    plt.tight_layout()
    out_path = FIG_DIR / "shap_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_ale_effects(model, X_val):
    target_features = ["MCV", "MCH", "RBC", "RDW"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    
    records = []
    
    for idx, feat in enumerate(target_features):
        ax = axes[idx]
        eff = ale(X=X_val, model=model, feature=[feat], grid_size=15, include_CI=True, plot=False)
        
        val_x = eff.index.values
        eff_y = eff["eff"].values
        ci_lower = eff["lowerCI_95%"].values
        ci_upper = eff["upperCI_95%"].values
        
        valid = ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
        
        ax.plot(val_x, eff_y, color="#d7191c", lw=2.0, label="Main Effect")
        if np.any(valid):
            ax.fill_between(val_x[valid], ci_lower[valid], ci_upper[valid], color="#d7191c", alpha=0.2, label="95% CI")
            
        ax.axhline(0, color="gray", linestyle="--", lw=0.9, alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel("Accumulated Local Effect")
        ax.set_title(f"ALE for {feat}", loc="left", fontweight="bold")
        
        for x, y, lo, hi in zip(val_x, eff_y, ci_lower, ci_upper):
            records.append({
                "Feature": feat,
                "Value": round(float(x), 2),
                "ALE_Effect": round(float(y), 4),
                "CI_Lower": round(float(lo), 4) if not np.isnan(lo) else None,
                "CI_Upper": round(float(hi), 4) if not np.isnan(hi) else None
            })
            
    pd.DataFrame(records).to_csv(OUT_DIR / "ale_summary.csv", index=False)
    
    plt.tight_layout()
    out_path = FIG_DIR / "ale_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_calibration_curves(models, X_val, y_val):
    plt.figure(figsize=(6, 5))
    
    colors = {
        "TabPFN": "#1f77b4",
        "LightGBM": "#2ca02c",
        "XGBoost": "#ff7f0e",
        "CatBoost": "#d62728"
    }
    
    for name in ["TabPFN", "LightGBM", "XGBoost", "CatBoost"]:
        if name in models:
            probs = models[name].predict_proba(X_val)[:, 1]
            fraction_pos, mean_pred = calibration_curve(y_val, probs, n_bins=8, strategy="uniform")
            plt.plot(mean_pred, fraction_pos, "s-", label=name, color=colors.get(name, "black"), lw=1.6, markersize=4.5)
            
    plt.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Calibration Curves (Task 1)", loc="left", fontweight="bold")
    plt.legend(loc="upper left", frameon=False, fontsize=8.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.tight_layout()
    out_path = FIG_DIR / "calibration_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def run_all():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)
    train_df = df[df["split"] == "development_train"].copy()
    val_df = df[df["split"] == "temporal_validation"].copy()
    
    X_train = train_df[FEATURES]
    y_train = train_df["TASK1_CARRIER"].values
    X_val = val_df[FEATURES]
    y_val = val_df["TASK1_CARRIER"].values
    
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    plot_roc_and_pr_curves(models, X_val, y_val)
    plot_shap_feature_importance(models["LightGBM"], X_train, X_val)
    plot_ale_effects(models["LightGBM"], X_val)
    plot_calibration_curves(models, X_val, y_val)

if __name__ == "__main__":
    run_all()

