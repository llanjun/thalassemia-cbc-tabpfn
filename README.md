# Interpretable TabPFN for Thalassemia Carrier Prediction from Complete Blood Counts: Threshold Trade-Offs and Phenotype-Specific Errors

Reproducible reference code and synthetic benchmark data for the manuscript:
*Interpretable TabPFN for thalassemia carrier prediction from complete blood counts: threshold trade-offs and phenotype-specific errors*.

## Data Notice
Actual patient records are protected under clinical ethics regulations and cannot be publicly shared. A synthetic dataset (`data/synthetic_data.csv`) with the identical predictor schema (Age, Sex, HCT, RDW, RBC, MCV, MCH, MCHC, HGB) is provided for code verification and pipeline reproduction.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Model training, out-of-fold threshold locking, and temporal validation:
```bash
python train_and_evaluate.py
```

2. Model interpretation (SHAP, ALE) and figure generation:
```bash
python explain_and_plot.py
```

3. (Optional) Re-generate the synthetic dataset:
```bash
python generate_synthetic_data.py
```

Generated metrics are saved in `outputs/` and rendered plots in `figures/`.

## License
MIT License.
