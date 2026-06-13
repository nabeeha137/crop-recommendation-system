"""
Crop Recommendation System - Model Training
============================================
Trains Random Forest and KNN classifiers on the crop recommendation dataset.
Saves trained models, scaler, and label encoder for use in the Streamlit app.

Run:  python train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET   = 'label'


# ── 1. Load & Explore ────────────────────────────────────────────────────────
print("=" * 55)
print("  CROP RECOMMENDATION SYSTEM — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Crops ({df[TARGET].nunique()}): {sorted(df[TARGET].unique())}")
print(f"\n    Missing values:\n{df.isnull().sum()}")
print(f"\n    Summary stats:\n{df[FEATURES].describe().round(2)}")


# ── 2. EDA Plots ─────────────────────────────────────────────────────────────
print("\n[2] Generating EDA plots …")

# Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, feat in zip(axes.flatten()[:7], FEATURES):
    ax.hist(df[feat], bins=30, color='#1D9E75', edgecolor='white', alpha=0.85)
    ax.set_title(feat, fontsize=12)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')
axes.flatten()[-1].axis('off')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_dist.png'), dpi=120, bbox_inches='tight')
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation.png'), dpi=120, bbox_inches='tight')
plt.close()

# Crop sample counts
fig, ax = plt.subplots(figsize=(10, 5))
df[TARGET].value_counts().plot(kind='bar', ax=ax, color='#185FA5', edgecolor='white')
ax.set_title('Samples per Crop', fontsize=13)
ax.set_xlabel('Crop'); ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'crop_counts.png'), dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: feature_dist.png, correlation.png, crop_counts.png")


# ── 3. Preprocessing ─────────────────────────────────────────────────────────
print("\n[3] Preprocessing …")
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df[TARGET])

X = df[FEATURES].values
y = df['label_enc'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")


# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("\n[4] Training Random Forest …")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)
acc_rf    = accuracy_score(y_test, y_pred_rf)

cv_rf = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring='accuracy')
print(f"    Test Accuracy:  {acc_rf:.4f}  ({acc_rf:.2%})")
print(f"    5-Fold CV:      {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")


# ── 5. Train KNN ──────────────────────────────────────────────────────────────
print("\n[5] Training KNN …")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)
y_pred_knn = knn.predict(X_test_sc)
acc_knn    = accuracy_score(y_test, y_pred_knn)

cv_knn = cross_val_score(knn, X_train_sc, y_train, cv=5, scoring='accuracy')
print(f"    Test Accuracy:  {acc_knn:.4f}  ({acc_knn:.2%})")
print(f"    5-Fold CV:      {cv_knn.mean():.4f} ± {cv_knn.std():.4f}")


# ── 6. Evaluation Plots ───────────────────────────────────────────────────────
print("\n[6] Generating evaluation plots …")

# Confusion matrix — RF
fig, ax = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.3)
ax.set_title(f'Random Forest — Confusion Matrix (Acc: {acc_rf:.2%})', fontsize=13)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix_rf.png'), dpi=120, bbox_inches='tight')
plt.close()

# Confusion matrix — KNN
fig, ax = plt.subplots(figsize=(14, 12))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.3)
ax.set_title(f'KNN — Confusion Matrix (Acc: {acc_knn:.2%})', fontsize=13)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix_knn.png'), dpi=120, bbox_inches='tight')
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
importances.plot(kind='barh', ax=ax, color='#1D9E75', edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png'), dpi=120, bbox_inches='tight')
plt.close()

# Model comparison
fig, ax = plt.subplots(figsize=(5, 4))
models = ['Random Forest', 'KNN']
accs   = [acc_rf, acc_knn]
bars   = ax.bar(models, [a * 100 for a in accs],
                color=['#1D9E75', '#185FA5'], edgecolor='white', width=0.4)
ax.set_ylim(70, 100); ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Comparison', fontsize=13)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{a:.2%}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'model_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()

print("    Saved: confusion_matrix_rf.png, confusion_matrix_knn.png,")
print("           feature_importance.png, model_comparison.png")


# ── 7. Classification Report ──────────────────────────────────────────────────
report = classification_report(y_test, y_pred_rf, target_names=le.classes_)
with open(os.path.join(BASE_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"Random Forest Classification Report\n")
    f.write(f"Accuracy: {acc_rf:.4f}\n\n")
    f.write(report)
print("\n[7] classification_report.txt saved.")


# ── 8. Save Models ────────────────────────────────────────────────────────────
joblib.dump(rf,     os.path.join(MODEL_DIR, 'rf_model.pkl'))
joblib.dump(knn,    os.path.join(MODEL_DIR, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("[8] Models saved:")
print(f"    models/rf_model.pkl       — Random Forest ({acc_rf:.2%})")
print(f"    models/knn_model.pkl      — KNN ({acc_knn:.2%})")
print(f"    models/scaler.pkl         — StandardScaler")
print(f"    models/label_encoder.pkl  — LabelEncoder ({len(le.classes_)} classes)")

print("\n" + "=" * 55)
print("  Training complete!")
print(f"  Best model: Random Forest  {acc_rf:.2%}")
print("=" * 55)
