"""
utils/model.py
──────────────
Chargement du modèle pré-entraîné (fichiers .pkl) et inférence.
Le modèle est entraîné une seule fois via train_model.py.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from utils.data import load_data, FEATURE_LABELS

# ── Chemins ───────────────────────────────────────────────────────────────────
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(_ROOT, "models")

FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status",
]
CAT_COLS = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status",
]


def _check_models():
    """Vérifie que les fichiers pkl existent, sinon affiche un message clair."""
    required = ["model_lr.pkl", "scaler.pkl", "encoders.pkl",
                "metrics.json", "idx_test.npy", "y_test.npy",
                "y_pred.npy", "y_prob.npy"]
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        st.error(
            f"❌ Fichiers manquants dans `models/` : {missing}\n\n"
            "Lance d'abord : `python train_model.py`"
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Charge les artefacts du modèle (mis en cache pour toute la session)."""
    _check_models()
    model    = joblib.load(os.path.join(MODELS_DIR, "model_lr.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, scaler, encoders, metrics


@st.cache_data(show_spinner=False)
def get_results() -> dict:
    """
    Charge les résultats pré-calculés du test set et reconstruit df_test.

    Retourne un dict avec :
      - metrics      : dict des métriques globales
      - df_test      : DataFrame avec y_true, y_pred, y_prob + colonnes originales
      - feature_imp  : DataFrame coefficients LR triés par importance absolue
      - model_name   : str
    """
    _check_models()
    model, scaler, encoders, metrics = load_model_artifacts()

    # Reconstruire df_test depuis les artefacts sauvegardés
    df       = load_data()
    idx_test = np.load(os.path.join(MODELS_DIR, "idx_test.npy"))
    y_test   = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    y_pred   = np.load(os.path.join(MODELS_DIR, "y_pred.npy"))
    y_prob   = np.load(os.path.join(MODELS_DIR, "y_prob.npy"))

    df_test = df.loc[idx_test].copy().reset_index(drop=True)
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["y_prob"] = y_prob

    # Importance des variables = |coefficients| de la LR
    coefs = np.abs(model.coef_[0])
    feature_imp = pd.DataFrame({
        "feature":    FEATURES,
        "label":      [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "importance": coefs,
    }).sort_values("importance", ascending=True).reset_index(drop=True)

    # Ajouter numpy arrays à metrics pour la confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "cm": cm,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    })

    return {
        "metrics":    metrics,
        "df_test":    df_test,
        "feature_imp": feature_imp,
        "model_name": "Logistic Regression",
    }
