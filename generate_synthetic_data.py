import os
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 2026
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "synthetic_data.csv"

def generate_cohort():
    n_train = 1000
    n_test = 250
    n_val = 250
    total_n = n_train + n_test + n_val
    
    splits = (
        ["development_train"] * n_train +
        ["development_internal_test"] * n_test +
        ["temporal_validation"] * n_val
    )
    
    carrier_probs = [0.90, 0.10]
    task1_carrier = np.random.choice([0, 1], size=total_n, p=carrier_probs)
    
    task2_type = []
    for c in task1_carrier:
        if c == 1:
            task2_type.append(np.random.choice(["alpha", "beta"], p=[0.55, 0.45]))
        else:
            task2_type.append("non_carrier")
            
    sex = np.random.choice([1, 2], size=total_n, p=[0.40, 0.60])
    age = np.random.normal(loc=32, scale=9, size=total_n).clip(18, 75).round().astype(int)
    
    mcv = np.zeros(total_n)
    mch = np.zeros(total_n)
    hgb = np.zeros(total_n)
    rbc = np.zeros(total_n)
    rdw = np.zeros(total_n)
    
    for i in range(total_n):
        is_male = (sex[i] == 1)
        c_status = task1_carrier[i]
        subtype = task2_type[i]
        
        if c_status == 0:
            has_ida = (np.random.rand() < 0.08)
            if has_ida:
                base_mcv = np.random.normal(76.0, 4.0)
                base_mch = np.random.normal(25.0, 1.8)
                base_hgb = np.random.normal(120.0 if is_male else 105.0, 10.0)
                base_rbc = np.random.normal(4.4 if is_male else 3.8, 0.4)
                base_rdw = np.random.normal(16.5, 2.0)
            else:
                base_mcv = np.random.normal(90.5, 4.2)
                base_mch = np.random.normal(30.2, 1.6)
                base_hgb = np.random.normal(150.0 if is_male else 130.0, 10.0)
                base_rbc = np.random.normal(5.0 if is_male else 4.3, 0.35)
                base_rdw = np.random.normal(12.5, 0.7)
        else:
            is_silent = (np.random.rand() < 0.10)
            if is_silent:
                base_mcv = np.random.normal(81.0, 3.5)
                base_mch = np.random.normal(27.0, 1.5)
                base_hgb = np.random.normal(140.0 if is_male else 122.0, 10.0)
                base_rbc = np.random.normal(5.4 if is_male else 4.7, 0.4)
                base_rdw = np.random.normal(13.8, 1.2)
            elif subtype == "alpha":
                base_mcv = np.random.normal(71.5, 4.5)
                base_mch = np.random.normal(23.2, 1.8)
                base_hgb = np.random.normal(135.0 if is_male else 115.0, 12.0)
                base_rbc = np.random.normal(5.8 if is_male else 5.1, 0.45)
                base_rdw = np.random.normal(14.8, 1.5)
            else:
                base_mcv = np.random.normal(65.5, 4.0)
                base_mch = np.random.normal(20.8, 1.5)
                base_hgb = np.random.normal(128.0 if is_male else 108.0, 11.0)
                base_rbc = np.random.normal(6.1 if is_male else 5.3, 0.45)
                base_rdw = np.random.normal(15.5, 1.6)
                
        mcv[i] = np.clip(base_mcv, 50.0, 105.0)
        mch[i] = np.clip(base_mch, 15.0, 36.0)
        hgb[i] = np.clip(base_hgb, 60.0, 185.0)
        rbc[i] = np.clip(base_rbc, 2.5, 7.5)
        rdw[i] = np.clip(base_rdw, 10.5, 26.0)
        
    hct = (rbc * mcv / 1000.0).clip(0.18, 0.58)
    mchc = (hgb / hct).clip(280.0, 370.0)
    
    df = pd.DataFrame({
        "split": splits,
        "AGE": age,
        "SEX": sex,
        "HCT": hct.round(3),
        "RDW": rdw.round(1),
        "RBC": rbc.round(2),
        "MCV": mcv.round(1),
        "MCH": mch.round(1),
        "MCHC": mchc.round(1),
        "HGB": hgb.round(1),
        "TASK1_CARRIER": task1_carrier,
        "TASK2_TYPE": task2_type
    })
    
    return df

if __name__ == "__main__":
    df_synthetic = generate_cohort()
    df_synthetic.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
