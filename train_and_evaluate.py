import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

SEED = 2026
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "synthetic_data.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["AGE", "SEX", "HCT", "RDW", "RBC", "MCV", "MCH", "MCHC", "HGB"]

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "Threshold": threshold,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }

def get_models(task):
    if task == 1:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=-1),
            "CatBoost": CatBoostClassifier(iterations=200, depth=5, learning_rate=0.03, l2_leaf_reg=10.0, auto_class_weights="SqrtBalanced", random_seed=100, thread_count=-1, verbose=False, allow_writing_files=False),
            "XGBoost": XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.035, subsample=0.70, colsample_bytree=0.80, eval_metric="logloss", random_state=SEED, n_jobs=-1),
            "TabPFN": TabPFNClassifier(device=DEVICE, inference_precision=torch.float32, n_estimators=8, random_state=SEED, ignore_pretraining_limits=True)
        }
    else:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=-1),
            "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_seed=SEED, thread_count=-1, verbose=False, allow_writing_files=False),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, random_state=SEED, n_jobs=-1),
            "TabPFN": TabPFNClassifier(device=DEVICE, inference_precision=torch.float32, n_estimators=8, random_state=SEED, ignore_pretraining_limits=True)
        }

def run_pipeline():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)
    
    train_df = df[df["split"] == "development_train"].copy()
    val_df = df[df["split"] == "temporal_validation"].copy()
    
    results = []
    
    X_train = train_df[FEATURES]
    y_train = train_df["TASK1_CARRIER"].values
    X_val = val_df[FEATURES]
    y_val = val_df["TASK1_CARRIER"].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(train_df))
    for t_idx, v_idx in skf.split(X_train, y_train):
        fold_model = TabPFNClassifier(device=DEVICE, inference_precision=torch.float32, n_estimators=8, random_state=SEED, ignore_pretraining_limits=True)
        fold_model.fit(X_train.iloc[t_idx], y_train[t_idx])
        oof_probs[v_idx] = fold_model.predict_proba(X_train.iloc[v_idx])[:, 1]
        
    carrier_oof = oof_probs[y_train == 1]
    locked_threshold = float(np.percentile(carrier_oof, 10))
    
    models_t1 = get_models(task=1)
    for name, model in models_t1.items():
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        roc_auc = roc_auc_score(y_val, val_probs)
        pr_auc = average_precision_score(y_val, val_probs)
        m_def = compute_metrics(y_val, val_probs, threshold=0.5)
        m_lock = compute_metrics(y_val, val_probs, threshold=locked_threshold)
        
        results.append({
            "Task": "Task 1 (Carrier)",
            "Model": name,
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC": round(pr_auc, 3),
            "Threshold": 0.5,
            "Sensitivity": round(m_def["Sensitivity"], 3),
            "Specificity": round(m_def["Specificity"], 3),
            "NPV": round(m_def["NPV"], 3),
            "PPV": round(m_def["PPV"], 3),
            "Accuracy": round((m_def["TP"] + m_def["TN"]) / len(y_val), 3)
        })
        results.append({
            "Task": "Task 1 (Carrier)",
            "Model": name,
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC": round(pr_auc, 3),
            "Threshold": round(locked_threshold, 4),
            "Sensitivity": round(m_lock["Sensitivity"], 3),
            "Specificity": round(m_lock["Specificity"], 3),
            "NPV": round(m_lock["NPV"], 3),
            "PPV": round(m_lock["PPV"], 3),
            "Accuracy": round((m_lock["TP"] + m_lock["TN"]) / len(y_val), 3)
        })
        
    train_t2 = train_df[train_df["TASK2_TYPE"].isin(["alpha", "beta"])].copy()
    val_t2 = val_df[val_df["TASK2_TYPE"].isin(["alpha", "beta"])].copy()
    
    X_train_t2 = train_t2[FEATURES]
    y_train_t2 = (train_t2["TASK2_TYPE"] == "beta").astype(int).values
    X_val_t2 = val_t2[FEATURES]
    y_val_t2 = (val_t2["TASK2_TYPE"] == "beta").astype(int).values
    
    models_t2 = get_models(task=2)
    for name, model in models_t2.items():
        model.fit(X_train_t2, y_train_t2)
        val_probs_t2 = model.predict_proba(X_val_t2)[:, 1]
        
        roc_t2 = roc_auc_score(y_val_t2, val_probs_t2)
        pr_t2 = average_precision_score(y_val_t2, val_probs_t2)
        y_pred_t2 = (val_probs_t2 >= 0.5).astype(int)
        acc_t2 = accuracy_score(y_val_t2, y_pred_t2)
        
        tn, fp, fn, tp = confusion_matrix(y_val_t2, y_pred_t2, labels=[0, 1]).ravel()
        beta_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        alpha_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results.append({
            "Task": "Task 2 (Alpha vs Beta)",
            "Model": name,
            "ROC-AUC": round(roc_t2, 3),
            "PR-AUC": round(pr_t2, 3),
            "Threshold": 0.5,
            "Sensitivity": round(beta_sens, 3),
            "Specificity": round(alpha_spec, 3),
            "NPV": round(tn / (tn + fn) if (tn + fn) > 0 else 0.0, 3),
            "PPV": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 3),
            "Accuracy": round(acc_t2, 3)
        })
        
    res_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "evaluation_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
