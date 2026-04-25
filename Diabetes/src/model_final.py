# model.py — XGBoost + tuning
from diabetes import load_data
from features import build_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
from xgboost import XGBClassifier

OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load features ─────────────────────────────────────────────────────────
X, y = build_features()

# ── 2. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

# ── 3. Scale for Logistic Regression ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Logistic Regression baseline ──────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train_scaled, y_train)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_preds = lr.predict(X_test_scaled)
lr_auc   = roc_auc_score(y_test, lr_probs)
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(classification_report(y_test, lr_preds,
      target_names=['Not readmitted', 'Readmitted']))

# ── 5. Random Forest ──────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_probs)
print(f"Random Forest AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_preds,
      target_names=['Not readmitted', 'Readmitted']))

# ── 6. XGBoost ────────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")

# scale_pos_weight handles class imbalance natively
# ratio of negative to positive class
neg  = (y_train == 0).sum()
pos  = (y_train == 1).sum()
spw  = neg / pos
print(f"scale_pos_weight: {spw:.2f}")

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

# Use a validation set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50          # prints every 50 rounds
)

xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_preds = xgb.predict(X_test)
xgb_auc   = roc_auc_score(y_test, xgb_probs)
print(f"\nXGBoost AUC: {xgb_auc:.4f}")
print(classification_report(y_test, xgb_preds,
      target_names=['Not readmitted', 'Readmitted']))

# ── 7. ROC curve — all three models ──────────────────────────────────────────
lr_fpr,  lr_tpr,  _ = roc_curve(y_test, lr_probs)
rf_fpr,  rf_tpr,  _ = roc_curve(y_test, rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr,  lr_tpr,  label=f'Logistic Regression (AUC = {lr_auc:.3f})',  color='steelblue')
plt.plot(rf_fpr,  rf_tpr,  label=f'Random Forest      (AUC = {rf_auc:.3f})',  color='coral')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost            (AUC = {xgb_auc:.3f})', color='seagreen')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve — 30-day readmission prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve.png', dpi=150)
plt.show()

# ── 8. Confusion matrix — best model ─────────────────────────────────────────
aucs       = {'Logistic Regression': lr_auc, 'Random Forest': rf_auc, 'XGBoost': xgb_auc}
best_name  = max(aucs, key=aucs.get)
best_preds = {'Logistic Regression': lr_preds,
              'Random Forest': rf_preds,
              'XGBoost': xgb_preds}[best_name]

cm   = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Not readmitted', 'Readmitted'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion matrix — {best_name}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.show()

# ── 9. XGBoost feature importance ────────────────────────────────────────────
importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(20)

plt.figure(figsize=(8, 7))
importances.plot(kind='barh', color='seagreen')
plt.xlabel('Feature importance')
plt.title('Top 20 feature importances — XGBoost')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150)
plt.show()

# ── 10. Threshold tuning on XGBoost ──────────────────────────────────────────
precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)

target_recall = 0.60
idx = next(i for i, r in enumerate(recall) if r <= target_recall)
best_threshold = thresholds[idx]
print(f"\nTuned threshold: {best_threshold:.3f}")

xgb_preds_tuned = (xgb_probs >= best_threshold).astype(int)
print("\nTuned XGBoost results:")
print(classification_report(y_test, xgb_preds_tuned,
      target_names=['Not readmitted', 'Readmitted']))

# Precision-recall curve
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='seagreen')
plt.axvline(x=recall[idx], color='gray', linestyle='--',
            label=f'Threshold = {best_threshold:.2f}  |  Recall = {recall[idx]:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-recall curve — XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall.png', dpi=150)
plt.show()

# ── 11. Summary table ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'AUC':   [round(lr_auc, 4), round(rf_auc, 4), round(xgb_auc, 4)]
}).sort_values('AUC', ascending=False)
print(summary.to_string(index=False))