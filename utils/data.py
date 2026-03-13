"""
utils/data.py
─────────────
Chargement, nettoyage et feature engineering du dataset Stroke Prediction.
Toutes les pages importent exclusivement depuis ce module.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st


# ── Constantes ────────────────────────────────────────────────────────────────
# Chemin absolu vers le CSV — fonctionne en local ET sur Streamlit Cloud
# peu importe le répertoire de travail courant
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(_ROOT, "healthcare-dataset-stroke-data.csv")

AGE_BINS   = [0, 18, 40, 60, 80, 120]
AGE_LABELS = ["0-18", "18-40", "40-60", "60-80", "80+"]

GLUCOSE_BINS   = [0, 100, 125, 200, 500]
GLUCOSE_LABELS = ["Normal (<100)", "Pré-diabète (100-125)", "Diabète (125-200)", "Très élevé (>200)"]

SENSITIVE_ATTRS = {
    "Genre":             "gender",
    "Zone géographique": "Residence_type",
}

FEATURE_LABELS = {
    "gender":             "Genre",
    "age":                "Âge",
    "hypertension":       "Hypertension",
    "heart_disease":      "Maladie cardiaque",
    "ever_married":       "Marié(e)",
    "work_type":          "Type d'emploi",
    "Residence_type":     "Zone de résidence",
    "avg_glucose_level":  "Glycémie moyenne",
    "bmi":                "IMC (BMI)",
    "smoking_status":     "Statut tabagique",
}

WORK_LABELS = {
    "Private":      "Secteur privé",
    "Self-employed":"Indépendant",
    "Govt_job":     "Fonctionnaire",
    "children":     "Enfant",
    "Never_worked": "Jamais travaillé",
}

SMOKING_LABELS = {
    "never smoked":    "Jamais fumé",
    "formerly smoked": "Ex-fumeur",
    "smokes":          "Fumeur actif",
    "Unknown":         "Inconnu",
}


# ── Loader (avec cache Streamlit) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Charge et prépare le dataset.
    Opérations appliquées :
      1. Suppression de l'individu 'Other' (genre unique, non significatif)
      2. Imputation des 201 valeurs manquantes de BMI par la médiane
      3. Création de variables catégorielles enrichies (tranches d'âge, glycémie)
      4. Labels lisibles pour work_type et smoking_status
    """
    df = pd.read_csv(DATA_PATH)

    # 1. Nettoyage gender
    df = df[df["gender"] != "Other"].copy()
    df = df.reset_index(drop=True)

    # 2. Imputation BMI
    bmi_median = df["bmi"].median()
    df["bmi_missing"] = df["bmi"].isna().astype(int)   # flag avant imputation
    df["bmi"] = df["bmi"].fillna(bmi_median)

    # 3. Tranches d'âge
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["age_group"] = df["age_group"].astype(str)

    # 4. Catégories glycémie
    df["glucose_cat"] = pd.cut(
        df["avg_glucose_level"],
        bins=GLUCOSE_BINS,
        labels=GLUCOSE_LABELS,
        right=True,
    )
    df["glucose_cat"] = df["glucose_cat"].astype(str)

    # 5. Labels lisibles
    df["work_type_label"]    = df["work_type"].map(WORK_LABELS).fillna(df["work_type"])
    df["smoking_label"]      = df["smoking_status"].map(SMOKING_LABELS).fillna(df["smoking_status"])

    return df


@st.cache_data(show_spinner=False)
def get_summary(df: pd.DataFrame) -> dict:
    """Retourne un dictionnaire de statistiques globales pour les KPIs."""
    return {
        "n_total":          len(df),
        "n_stroke":         int(df["stroke"].sum()),
        "n_no_stroke":      int((df["stroke"] == 0).sum()),
        "stroke_rate":      df["stroke"].mean(),
        "n_female":         int((df["gender"] == "Female").sum()),
        "n_male":           int((df["gender"] == "Male").sum()),
        "n_urban":          int((df["Residence_type"] == "Urban").sum()),
        "n_rural":          int((df["Residence_type"] == "Rural").sum()),
        "age_mean":         df["age"].mean(),
        "age_median":       df["age"].median(),
        "bmi_mean":         df["bmi"].mean(),
        "bmi_median":       df["bmi"].median(),
        "glucose_mean":     df["avg_glucose_level"].mean(),
        "n_bmi_imputed":    int(df["bmi_missing"].sum()),
        "n_hypertension":   int(df["hypertension"].sum()),
        "n_heart_disease":  int(df["heart_disease"].sum()),
        "stroke_age_mean":  df[df["stroke"] == 1]["age"].mean(),
    }
