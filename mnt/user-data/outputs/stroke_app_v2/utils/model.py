"""
utils/model.py
──────────────
Pipeline de modélisation :
  1. Encodage des variables catégorielles
  2. Split train/test stratifié (80/20)
  3. SMOTE sur le train set (rééquilibrage ~5% → 50/50)
  4. Entraînement Random Forest ou Logistic Regression
  5. Prédiction avec seuil abaissé (0.30) pour maximiser le Recall
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE


FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status",
]

CAT_COLS = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status",
]

# Seuil abaissé : en médecine un faux négatif (AVC non détecté) est plus
# coûteux qu'un faux positif → on favorise le Recall.
DECISION_THRESHOLD = 0.30


@st.cache_data(show_spinner=False)
def train_and_evaluate(model_type: str = "Random Forest") -> dict:
    """
    Entraîne le modèle et retourne toutes les métriques + données de test.

    Retourne un dict avec :
      - metrics       : dict des métriques globales
      - df_test       : DataFrame de test avec y_true, y_pred, y_prob
      - feature_imp   : DataFrame importance des variables (RF seulement)
      - encoders      : dict des LabelEncoders par colonne
      - smote_counts  : répartition après SMOTE
    """
    from utils.data import load_data
    df = load_data()

    # ── Encodage ──────────────────────────────────────────────────────────────
    df_enc = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURES].values
    y = df_enc["stroke"].values

    # ── Split stratifié ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE (train seulement) ───────────────────────────────────────────────
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # ── Modèle ────────────────────────────────────────────────────────────────
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, n_jobs=-1,
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_res, y_res)

    # ── Prédictions avec seuil abaissé ────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    # ── DataFrame de test enrichi ─────────────────────────────────────────────
    df_test = df.loc[idx_test].copy().reset_index(drop=True)
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["y_prob"] = y_prob

    # ── Métriques globales ────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":   float(accuracy_score(y_test, y_pred)),
        "precision":  float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_test, y_prob)),
        "cm":         cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold":  DECISION_THRESHOLD,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_smote_pos": int(y_res.sum()),
        "n_smote_neg": int((y_res == 0).sum()),
    }

    # ── Importance des variables ──────────────────────────────────────────────
    feature_imp = None
    if model_type == "Random Forest":
        from utils.data import FEATURE_LABELS
        feature_imp = pd.DataFrame({
            "feature":    FEATURES,
            "label":      [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True).reset_index(drop=True)

    return {
        "metrics":      metrics,
        "df_test":      df_test,
        "feature_imp":  feature_imp,
        "encoders":     encoders,
        "model_type":   model_type,
    }
