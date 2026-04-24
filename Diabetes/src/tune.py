# tune.py — XGBoost hyperparameter tuning
from diabetes import load_data
from features import build_features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ── 1. Load features ─────────────────────────────────────────────────────────
X, y = build_features()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

# ── 2. Define search space ────────────────────────────────────────────────────
param_dist = {
    'n_estimators':      [300, 500, 700, 1000],
    'max_depth':         [3, 4, 5, 6, 7, 8],
    'learning_rate':     [0.01, 0.05, 0.1, 0.15],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'min_child_weight':  [1, 3, 5, 7],
    'gamma':             [0, 0.1, 0.2, 0.3],
    'reg_alpha':         [0, 0.1, 0.5, 1.0],    # L1 regularization
    'reg_lambda':        [1.0, 1.5, 2.0, 3.0],  # L2 regularization
}

# ── 3. Base model ─────────────────────────────────────────────────────────────
xgb = XGBClassifier(
    scale_pos_weight=spw,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# ── 4. Randomized search with cross-validation ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,              # tries 50 random combinations
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter search — this will take several minutes...")
search.fit(X_train, y_train)

# ── 5. Results ────────────────────────────────────────────────────────────────
print(f"\nBest cross-validated AUC: {search.best_score_:.4f}")
print("\nBest parameters:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

# ── 6. Evaluate best model on held-out test set ───────────────────────────────
from sklearn.metrics import roc_auc_score, classification_report

best_model = search.best_estimator_
test_probs = best_model.predict_proba(X_test)[:, 1]
test_preds = best_model.predict(X_test)
test_auc   = roc_auc_score(y_test, test_probs)

print(f"\nTest set AUC with best params: {test_auc:.4f}")
print(classification_report(y_test, test_preds,
      target_names=['Not readmitted', 'Readmitted']))

# ── 7. Save best params for use in model.py ───────────────────────────────────
import json
with open('best_params.json', 'w') as f:
    json.dump(search.best_params_, f, indent=2)
print("\nBest parameters saved to best_params.json")