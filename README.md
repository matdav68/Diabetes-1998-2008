# Diabetes-1998-2008
# Predicting 30-Day Hospital Readmissions in Diabetic Patients

A machine learning project analyzing clinical and demographic factors that drive 30-day hospital readmissions in diabetic patients, using the UCI Diabetes 130-US Hospitals dataset.

> **Final model AUC: 0.690** — outperforms the published peer-reviewed benchmark of 0.667 on the same dataset (PMC, 2025)

---

## Research Question

> What clinical and demographic factors drive 30-day hospital readmissions in diabetic patients, and can these factors be used to predict high-risk patients before discharge?

---

## Results

The final tuned XGBoost model achieved an **AUC of 0.690**, outperforming the published peer-reviewed benchmark of 0.667 on the same dataset (PMC, 2025). The top three drivers of readmission risk were:

1. **Prior inpatient hospitalization history** — patients with 8+ prior visits had a 44% readmission rate vs. 8.5% for first-time admissions
2. **Discharge destination** — where a patient is sent after discharge is the second strongest predictor
3. **Total prior healthcare utilization** — combined inpatient, emergency, and outpatient visit history

---

## Dataset

**UCI Diabetes 130-US Hospitals for Years 1999–2008**

| Attribute | Value |
|-----------|-------|
| Total encounters | 101,766 |
| Hospitals | 130 US hospitals |
| Time period | 1999–2008 |
| Features | 50 original / 47 after engineering |
| Target | 30-day readmission (binary) |
| Positive rate | 11.2% |

The raw data files are included in the `data/` folder:

- `data/diabetic_data.csv` — main encounters dataset (101,766 rows)
- `data/IDS_mapping.csv` — mapping of discharge disposition, admission type, and admission source IDs to clinical labels

Original source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

> The dataset is made available under CC BY 4.0. See [LICENSE.txt](LICENSE.txt) for attribution requirements.

---

## Project Structure

```
Diabetes/
│
├── README.md
├── LICENSE.txt
├── diabetes_db.sql                 # MySQL schema setup and exploratory queries
├── best_params.json                # Saved optimal XGBoost hyperparameters
│
├── src/                            # Python analysis pipeline
│   ├── diabetes.py                 # MySQL connection and data loading module
│   ├── eda.py                      # Exploratory data analysis and visualizations
│   ├── features.py                 # Feature engineering pipeline
│   ├── tune.py                     # XGBoost hyperparameter tuning
│   ├── model.py                    # Baseline Logistic Regression + Random Forest
│   └── model_final.py              # Final tuned XGBoost model
│
├── data/                           # Raw data files
│   ├── diabetic_data.csv           # Main dataset (101,766 encounters)
│   └── IDS_mapping.csv             # ID-to-label mapping for categorical columns
│
├── outputs/                        # Generated charts
│   ├── age_readmission.png         # Readmission rate by age group
│   ├── chi2_results.png            # Chi-square significance results
│   ├── confusion_matrix.png        # Final model confusion matrix
│   ├── correlation.png             # Numeric feature correlations with target
│   ├── feature_importance.png      # XGBoost feature importances
│   ├── inpatient_readmission.png   # Readmission rate by prior inpatient visits
│   ├── insulin_readmission.png     # Readmission rate by insulin status
│   ├── precision_recall.png        # Precision-recall curve
│   ├── roc_curve.png               # ROC curve comparing all models
│   ├── tuning_results.png          # AUC across all 50 hyperparameter candidates
│   └── tuning_results.csv          # Full cross-validated results for all 50 candidates
│
└── deliverables/                   # Final compiled outputs
    ├── Diabetes_Readmission_Report.docx
    ├── Diabetes_Readmission_Report.pdf
    └── Diabetes_Analysis_Charts.pdf
```

---

## File Descriptions

### Source files (`src/`)

| File | Description |
|------|-------------|
| `diabetes.py` | Establishes the MySQL database connection using SQLAlchemy and exposes a `load_data()` function imported by all other modules |
| `eda.py` | Generates all exploratory charts — correlation bar chart, age group readmission rates, prior inpatient visit rates, insulin status breakdown, and chi-square significance tests for categorical features |
| `features.py` | Full feature engineering pipeline: drops high-missingness columns, maps ICD-9 diagnosis codes to clinical categories, encodes discharge/admission ID columns, creates binary flags, ordinal-encodes age brackets, label-encodes categorical variables, and engineers interaction terms |
| `tune.py` | Runs RandomizedSearchCV with 5-fold stratified cross-validation across 50 hyperparameter combinations and saves the best parameters to `best_params.json` |
| `model.py` | Baseline comparison models — Logistic Regression and Random Forest — used to benchmark against the final XGBoost model |
| `model_final.py` | Final model — tuned XGBoost with optimal hyperparameters loaded from `best_params.json`, generates ROC curve, confusion matrix, feature importance chart, and precision-recall threshold tuning |

### Other files

| File | Description |
|------|-------------|
| `diabetes_db.sql` | MySQL database and table setup, binary readmission target creation, and all SQL-based exploratory queries (readmission rates by age, prior visits, insulin, and medical specialty) |
| `best_params.json` | Serialized output from `tune.py` containing the optimal XGBoost hyperparameters |

---

## Model Performance

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.659 |
| Random Forest | 0.676 |
| XGBoost (default) | 0.685 |
| **XGBoost (tuned) — Final** | **0.690** |
| Published benchmark (PMC 2025) | 0.667 |

---

## Setup and Usage

### Prerequisites

- Python 3.8+
- MySQL 8.0+

### 1. Load the data into MySQL

Run `diabetes_db.sql` in MySQL Workbench or the MySQL CLI to set up the schema and load the data.

### 2. Install Python dependencies

```bash
pip install pandas sqlalchemy pymysql cryptography scikit-learn xgboost matplotlib seaborn scipy
```

### 3. Configure the database connection

Open `src/diabetes.py` and update the connection string with your credentials:

```python
engine = create_engine("mysql+pymysql://your_user:your_password@localhost:your_port/diabetes")
```

### 4. Run the pipeline in order

```bash
# 1. Exploratory data analysis
python src/eda.py

# 2. Hyperparameter tuning (optional — best_params.json already saved)
python src/tune.py

# 3. Baseline model comparison
python src/model.py

# 4. Final model training and evaluation
python src/model_final.py
```

All charts save to `outputs/` automatically.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Strongest predictor | `number_inpatient` — prior hospitalization count dominates all feature selection methods |
| Most actionable | `discharge_disposition_id` — discharge destination is directly controllable by clinical staff |
| Best interaction | `number_inpatient × A1C_tested` (r = 0.165, p < 0.001) |
| Only dropped demographic | `gender` — only feature with non-significant chi-square (p = 0.539) |
| AUC ceiling reason | Dataset lacks lab values, discharge text, and comorbidity severity scores present in richer clinical EHR datasets |

---

## Limitations

- Dataset covers 1999–2008; clinical practices have evolved significantly since then
- `weight` (97% missing) and `max_glu_serum` (95% missing) had to be dropped entirely
- Class imbalance (88/12) keeps readmission precision low at ~0.19
- Dataset contains encounters rather than unique patients, which may slightly inflate performance estimates

---

## References

- Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*. https://doi.org/10.1155/2014/781670
- PMC benchmark study (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/
- UCI dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

## License

MIT License
