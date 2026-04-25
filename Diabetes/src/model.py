# model.py — model training and evaluation
from diabetes import load_data
from features import build_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load features ─────────────────────────────────────────────────────────
X, y = build_features()

# ── 2. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

# Scale features for Logistic Regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 3. Logistic Regression baseline ─────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # handles the 88/12 imbalance
    random_state=42
)
lr.fit(X_train_scaled, y_train)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_preds = lr.predict(X_test_scaled)
lr_auc   = roc_auc_score(y_test, lr_probs)
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(classification_report(y_test, lr_preds, target_names=['Not readmitted', 'Readmitted']))

# ── 4. Random Forest ─────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1             # uses all CPU cores to speed up training
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_probs)
print(f"Random Forest AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_preds, target_names=['Not readmitted', 'Readmitted']))

# ── 5. ROC curve comparison ──────────────────────────────────────────────────
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', color='steelblue')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', color='coral')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve — 30-day readmission prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve.png', dpi=150)
plt.show()

# ── 6. Confusion matrix for best model ──────────────────────────────────────
best_preds = rf_preds if rf_auc > lr_auc else lr_preds
best_name  = 'Random Forest' if rf_auc > lr_auc else 'Logistic Regression'

cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Not readmitted', 'Readmitted'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion matrix — {best_name}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.show()

# ── 7. Feature importance (Random Forest) ───────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(20)

plt.figure(figsize=(8, 7))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('Feature importance')
plt.title('Top 20 feature importances — Random Forest')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150)
plt.show()

# ── 8. Print top 10 features ─────────────────────────────────────────────────
print("\nTop 10 most important features:")
print(importances.tail(10).sort_values(ascending=False).to_string())