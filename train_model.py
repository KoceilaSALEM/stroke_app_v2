"""
train_model.py
──────────────
Script à exécuter UNE SEULE FOIS en local pour entraîner le modèle
et générer les fichiers dans models/.

Usage :
    python train_model.py

Fichiers générés dans models/ :
    model_lr.pkl    — Logistic Regression entraînée
    scaler.pkl      — StandardScaler (fitté sur train)
    encoders.pkl    — dict des LabelEncoders par colonne catégorielle
    metrics.json    — métriques de performance (pour affichage dans l'app)
    idx_test.npy    — indices du jeu de test (pour reconstruire df_test)
    y_test.npy      — labels réels du test set
    y_pred.npy      — prédictions (seuil 0.30)
    y_prob.npy      — probabilités prédites
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

from utils.data import load_data, FEATURE_LABELS

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status",
]
CAT_COLS = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status",
]
DECISION_THRESHOLD = 0.30
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── Chargement ────────────────────────────────────────────────────────────────
print("📂 Chargement des données...")
df = load_data()
print(f"   Shape : {df.shape} | AVC : {df['stroke'].sum()} ({df['stroke'].mean()*100:.1f}%)")

# ── Encodage ──────────────────────────────────────────────────────────────────
print("\n🔧 Encodage des variables catégorielles...")
df_enc = df.copy()
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le
    print(f"   {col:20s} → {list(le.classes_)}")

X = df_enc[FEATURES].values
y = df_enc["stroke"].values

# ── Split stratifié ───────────────────────────────────────────────────────────
print("\n✂️  Split stratifié 80/20...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train : {len(X_train):,} | Test : {len(X_test):,}")

# ── SMOTE ────────────────────────────────────────────────────────────────────
print("\n⚖️  SMOTE sur le train set...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"   Avant  : {y_train.sum()} positifs / {(y_train==0).sum()} négatifs")
print(f"   Après  : {y_res.sum()} positifs / {(y_res==0).sum()} négatifs")

# ── Normalisation ────────────────────────────────────────────────────────────
print("\n📐 StandardScaler...")
scaler = StandardScaler()
X_res_s  = scaler.fit_transform(X_res)
X_test_s = scaler.transform(X_test)

# ── Modèle ────────────────────────────────────────────────────────────────────
print("\n🤖 Entraînement Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_res_s, y_res)
print("   ✅ Entraînement terminé")

# ── Prédictions ───────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test_s)[:, 1]
y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

# ── Métriques ────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

metrics = {
    "model":       "Logistic Regression",
    "accuracy":    float(accuracy_score(y_test, y_pred)),
    "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
    "recall":      float(recall_score(y_test, y_pred, zero_division=0)),
    "f1":          float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc":     float(roc_auc_score(y_test, y_prob)),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "threshold":   DECISION_THRESHOLD,
    "n_train":     len(X_train),
    "n_test":      len(X_test),
    "n_smote_pos": int(y_res.sum()),
    "n_smote_neg": int((y_res == 0).sum()),
}

print(f"\n📊 Résultats (seuil={DECISION_THRESHOLD}) :")
print(f"   Accuracy  : {metrics['accuracy']:.3f}")
print(f"   Precision : {metrics['precision']:.3f}")
print(f"   Recall    : {metrics['recall']:.3f}  ← priorité médicale")
print(f"   F1        : {metrics['f1']:.3f}")
print(f"   ROC-AUC   : {metrics['roc_auc']:.3f}")
print(f"   CM        : TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
print(f"\n💾 Sauvegarde dans {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model,    os.path.join(OUTPUT_DIR, "model_lr.pkl"))
joblib.dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(OUTPUT_DIR, "encoders.pkl"))

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"), idx_test.to_numpy())
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),   y_test)
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"),   y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_prob.npy"),   y_prob)

print("\n✅ Fichiers générés :")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"   {fname:25s}  {size:>8,} bytes")

print("\n🎉 Modèle prêt. Lance l'app avec : streamlit run app.py")
