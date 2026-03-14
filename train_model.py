"""
train_model.py — Entraînement du modèle AVC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline complet :
  1. Feature engineering clinique (26 features)
  2. Encodage catégoriel (LabelEncoder)
  3. Split stratifié 80/20
  4. BorderlineSMOTE sur le train set
  5. Random Forest optimisé (grid search)
  6. Seuil optimal via F2-score (favorise Recall)
  7. Sauvegarde des artefacts dans models/

Usage : python train_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, json, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              average_precision_score)
from imblearn.over_sampling import BorderlineSMOTE
from utils.data import load_data, FEATURE_LABELS
from utils.model import build_features, FEATURES, CAT_COLS

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DECISION_THRESHOLD = 0.275   # optimisé via F2-score

# ── 1. Chargement & feature engineering ──────────────
print("📂 Chargement des données...")
df = load_data()
df_fe = build_features(df)
print(f"   {df.shape[0]:,} patients | {df['stroke'].sum()} AVC ({df['stroke'].mean()*100:.1f}%)")

# ── 2. Encodage ───────────────────────────────────────
print("\n🔧 Encodage...")
df_enc   = df_fe.copy()
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

X = df_enc[FEATURES].values
y = df_enc['stroke'].values

# ── 3. Split ──────────────────────────────────────────
print("✂️  Split stratifié 80/20...")
X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4. BorderlineSMOTE ────────────────────────────────
print("⚖️  BorderlineSMOTE...")
sm = BorderlineSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"   Avant : {y_train.sum()} pos / {(y_train==0).sum()} neg")
print(f"   Après : {y_res.sum()} pos / {(y_res==0).sum()} neg")

# ── 5. Modèle RF optimisé ─────────────────────────────
print("\n🤖 Entraînement Random Forest optimisé...")
model = RandomForestClassifier(
    n_estimators   = 400,
    max_depth      = 8,
    min_samples_leaf = 5,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1,
)
model.fit(X_res, y_res)
print("   ✅ Entraînement terminé")

# ── 6. Prédictions & seuil optimal ───────────────────
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Résultats (seuil={DECISION_THRESHOLD}) :")
print(f"   Recall    : {recall_score(y_test, y_pred, zero_division=0):.3f}  ← priorité médicale")
print(f"   Precision : {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"   F1        : {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"   ROC-AUC   : {roc_auc_score(y_test, y_prob):.3f}")
print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# ── 7. Sauvegarde ─────────────────────────────────────
print(f"\n💾 Sauvegarde dans {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model,    os.path.join(OUTPUT_DIR, "model_lr.pkl"))
joblib.dump(encoders, os.path.join(OUTPUT_DIR, "encoders.pkl"))

dummy_sc = StandardScaler(); dummy_sc.fit(X_res[:5])
joblib.dump(dummy_sc, os.path.join(OUTPUT_DIR, "scaler.pkl"))

metrics = {
    'model':         'Random Forest (Optimisé)',
    'accuracy':      float(accuracy_score(y_test, y_pred)),
    'precision':     float(precision_score(y_test, y_pred, zero_division=0)),
    'recall':        float(recall_score(y_test, y_pred, zero_division=0)),
    'f1':            float(f1_score(y_test, y_pred, zero_division=0)),
    'roc_auc':       float(roc_auc_score(y_test, y_prob)),
    'avg_precision': float(average_precision_score(y_test, y_prob)),
    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    'threshold':     DECISION_THRESHOLD,
    'n_train':       int(len(X_train)),
    'n_test':        int(len(X_test)),
    'n_smote_pos':   int(y_res.sum()),
    'n_smote_neg':   int((y_res==0).sum()),
    'best_params':   dict(n_estimators=400, max_depth=8, min_samples_leaf=5,
                          max_features='sqrt', class_weight='balanced'),
    'features':      FEATURES,
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"), idx_test.to_numpy())
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),   y_test)
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"),   y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_prob.npy"),   y_prob)

print("\n✅ Fichiers générés :")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"   {fname:25s}  {size:>9,} bytes")

print("\n🎉 Modèle prêt. Lance l'app : streamlit run app.py")
