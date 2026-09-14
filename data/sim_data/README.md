# Synthetic data

`simulated_dataset.csv` is a fixed synthetic CBC dataset for public testing and examples. The values use seed `2026` with measurement variation and overlapping phenotypes.

## Fields

| Field | Meaning |
| --- | --- |
| `INPATIENT_ID` | Synthetic identifier with the `SIM_` prefix. |
| `INSPECTION_DATE` | Synthetic CBC inspection date. |
| `GENE_DATE` | Synthetic genetic-test reference date. |
| `COHORT` | `development` or `temporal`. |
| `TASK1_SPLIT` | `development_train`, `development_internal_test`, or `temporal_external`. |
| `TASK2_SPLIT` | `not_applicable` in this public dataset; the code uses `TASK1_SPLIT` for eligible alpha/beta records. |
| `TASK2_ELIGIBLE` | Eligibility flag retained for schema compatibility; it is `False` for all public synthetic records. |
| `AGE` | Age in years. |
| `HCT` | Haematocrit as a fraction (L/L). |
| `RDW` | Red-cell distribution width, percent. |
| `RBC` | Red blood cell count. |
| `MCV` | Mean corpuscular volume, fL. |
| `MCH` | Mean corpuscular haemoglobin, pg. |
| `MCHC` | Mean corpuscular haemoglobin concentration, g/L. |
| `HGB` | Haemoglobin, g/L. |
| `SEX` | Synthetic sex code: `1` male, `2` female. |
| `TASK1_TARGET` | `0` non-carrier, `1` carrier. |
| `TASK2_TARGET` | `non_carrier`, `alpha`, `beta`, or `mixed_or_other`. |

Task 2 model evaluation uses only `alpha` and `beta` records. The other labels are retained for phenotype visualisation.
