"""
utils/model.py
──────────────
Chargement du modèle pré-entraîné (Random Forest optimisé) et inférence.
Entraîné via train_model.py — ne fait que charger les .pkl ici.
"""

import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from utils.data import load_data, FEATURE_LABELS

_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(_ROOT, "models")

CAT_COLS = ['gender','ever_married','work_type','Residence_type','smoking_status']

FEATURES = [
    'gender','age','hypertension','heart_disease','ever_married',
    'work_type','Residence_type','avg_glucose_level','bmi','smoking_status',
    'age_hypertension','age_heart','age_glucose','hypertension_heart',
    'glucose_bmi','comorbidity_score','metabolic_risk','age_sq',
    'age_over_60','age_over_70','diabetes_proxy','obese','overweight',
    'clinical_risk_score','glucose_bmi_ratio','bmi_missing',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering clinique — même pipeline que train_model.py."""
    d = df.copy()
    d['age_hypertension']    = d['age'] * d['hypertension']
    d['age_heart']           = d['age'] * d['heart_disease']
    d['age_glucose']         = d['age'] * d['avg_glucose_level']
    d['hypertension_heart']  = d['hypertension'] * d['heart_disease']
    d['glucose_bmi']         = d['avg_glucose_level'] * d['bmi']
    d['comorbidity_score']   = d['hypertension'] + d['heart_disease']
    d['metabolic_risk']      = ((d['avg_glucose_level'] > 140).astype(int) +
                                (d['bmi'] > 30).astype(int))
    d['age_sq']              = d['age'] ** 2
    d['age_over_60']         = (d['age'] > 60).astype(int)
    d['age_over_70']         = (d['age'] > 70).astype(int)
    d['diabetes_proxy']      = (d['avg_glucose_level'] > 126).astype(int)
    d['obese']               = (d['bmi'] > 30).astype(int)
    d['overweight']          = (d['bmi'].between(25, 30)).astype(int)
    d['clinical_risk_score'] = (d['age_over_60'] * 2 + d['hypertension'] * 2 +
                                d['heart_disease'] * 2 + d['diabetes_proxy'] +
                                d['obese'])
    d['glucose_bmi_ratio']   = d['avg_glucose_level'] / (d['bmi'] + 1e-5)
    return d


def _check_models():
    required = ["model_lr.pkl", "encoders.pkl", "metrics.json",
                "idx_test.npy", "y_test.npy", "y_pred.npy", "y_prob.npy"]
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        st.error(f"❌ Fichiers manquants dans `models/` : {missing}\n\n"
                 "Lance d'abord : `python train_model.py`")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    _check_models()
    model    = joblib.load(os.path.join(MODELS_DIR, "model_lr.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, encoders, metrics


@st.cache_data(show_spinner=False)
def get_results() -> dict:
    _check_models()
    model, encoders, metrics = load_model_artifacts()

    df       = load_data()
    idx_test = np.load(os.path.join(MODELS_DIR, "idx_test.npy"))
    y_test   = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    y_pred   = np.load(os.path.join(MODELS_DIR, "y_pred.npy"))
    y_prob   = np.load(os.path.join(MODELS_DIR, "y_prob.npy"))

    df_test = df.loc[idx_test].copy().reset_index(drop=True)
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["y_prob"] = y_prob

    # Importance des variables (RF)
    feature_imp = pd.DataFrame({
        "feature":    FEATURES,
        "label":      [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).reset_index(drop=True)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"cm": cm,
                    "tn": int(tn), "fp": int(fp),
                    "fn": int(fn), "tp": int(tp)})

    return {
        "metrics":     metrics,
        "df_test":     df_test,
        "feature_imp": feature_imp,
        "model_name":  metrics.get("model", "Random Forest Optimisé"),
    }
