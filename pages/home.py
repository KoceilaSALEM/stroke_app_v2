"""pages/home.py — Page d'accueil"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from utils.data import load_data, get_summary
from utils.plots import stroke_donut


def render():
    df = load_data()
    s  = get_summary(df)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-hero">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;">🧠</div>
        <h1>Prédiction du Risque d'AVC</h1>
        <p class="subtitle">DÉTECTION DE BIAIS ALGORITHMIQUE · GENRE & ZONE GÉOGRAPHIQUE</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Vue d\'ensemble</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Patients", f"{s['n_total']:,}")
    k2.metric("🧠 Cas d'AVC", f"{s['n_stroke']:,}",
              f"{s['stroke_rate']*100:.2f}% de prévalence")
    k3.metric("📅 Âge moyen", f"{s['age_mean']:.1f} ans",
              f"AVC : {s['stroke_age_mean']:.0f} ans en moy.")
    k4.metric("💊 Hypertension", f"{s['n_hypertension']:,}",
              f"{s['n_hypertension']/s['n_total']*100:.1f}% des patients")

    st.divider()

    # ── Contexte ──────────────────────────────────────────────────────────────
    col_txt, col_vis = st.columns([3, 2], gap="large")

    with col_txt:
        st.markdown("## Contexte & Problématique")
        st.markdown("""
        L'**accident vasculaire cérébral (AVC)** est la **2ᵉ cause de mortalité mondiale** 
        et la **1ʳᵉ cause de handicap acquis** chez l'adulte. Selon l'OMS, 15 millions de 
        personnes en sont victimes chaque année. Chaque minute sans traitement, un patient 
        perd en moyenne 1,9 million de neurones — l'urgence de la détection précoce est absolue.

        Ce projet analyse le **Stroke Prediction Dataset** (Kaggle, fedesoriano) : 5 110 patients 
        avec leurs caractéristiques médicales et démographiques. L'objectif est double : construire 
        un modèle prédictif du risque d'AVC, et examiner les **biais algorithmiques** potentiels 
        pouvant mener à des prédictions inéquitables.

        Nous ciblons spécifiquement deux **attributs sensibles** : le **genre** (Male/Female) et 
        la **zone de résidence** (Urban/Rural). Un biais sur ces dimensions pourrait priver certains 
        groupes de patients d'alertes médicales essentielles, aggravant des inégalités déjà existantes 
        dans l'accès aux soins.
        """)

        st.markdown("""
        <div class="card-warning">
            <h4>⚠️ Note éthique</h4>
            <p>Ce projet est réalisé dans un cadre académique. Les biais détectés illustrent 
            l'importance cruciale de l'équité algorithmique en santé. Un modèle biaisé peut 
            avoir des conséquences directes sur la qualité des soins prodigués à certaines 
            populations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_vis:
        st.markdown("### Répartition AVC / Non-AVC")
        st.plotly_chart(stroke_donut(s["n_stroke"], s["n_total"]),
                        use_container_width=True)

        st.markdown("""
        <div class="card-info">
            <h4>🔍 Déséquilibre de classes</h4>
            <p>Seulement <b style="color:white;">4.87%</b> des patients ont subi un AVC.
            Ce déséquilibre extrême nécessite une stratégie de rééquilibrage 
            (SMOTE) lors de la modélisation pour éviter un modèle biaisé vers 
            la classe majoritaire.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Description dataset ───────────────────────────────────────────────────
    st.markdown("## 📋 Description du Dataset")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card-neutral">
            <div class="icon">📊</div>
            <h4>Structure</h4>
            <p>5 110 lignes · 12 colonnes<br>
            1 variable cible (stroke)<br>
            10 variables prédictives<br>
            201 valeurs manquantes (BMI)</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card-neutral">
            <div class="icon">👥</div>
            <h4>Démographie</h4>
            <p>58.7% Femmes · 41.3% Hommes<br>
            50.8% zone Urbaine<br>
            49.2% zone Rurale<br>
            Âge : 0 – 82 ans</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card-neutral">
            <div class="icon">🩺</div>
            <h4>Facteurs de risque</h4>
            <p>Hypertension · Maladie cardiaque<br>
            Glycémie · IMC · Tabagisme<br>
            Statut marital · Type d'emploi<br>
            Zone de résidence</p>
        </div>
        """, unsafe_allow_html=True)

    # Variables table
    st.markdown("### Variables du dataset")
    variables = {
        "id":                "Identifiant unique du patient",
        "gender":            "Genre : Male / Female (1 'Other' retiré)",
        "age":               "Âge (0 – 82 ans, valeur flottante)",
        "hypertension":      "Hypertension artérielle : 0 = Non, 1 = Oui",
        "heart_disease":     "Maladie cardiaque : 0 = Non, 1 = Oui",
        "ever_married":      "A déjà été marié(e) : Yes / No",
        "work_type":         "Type d'emploi : Private, Govt_job, Self-employed, children, Never_worked",
        "Residence_type":    "Zone de résidence : Urban / Rural",
        "avg_glucose_level": "Glycémie moyenne en mg/dL (55 – 272)",
        "bmi":               "Indice de Masse Corporelle — 201 valeurs manquantes imputées",
        "smoking_status":    "Statut tabagique : 4 catégories dont 'Unknown'",
        "stroke":            "🎯 Variable cible : 0 = Pas d'AVC, 1 = AVC",
    }
    rows = [f"| `{k}` | {v} |" for k, v in variables.items()]
    st.markdown(
        "| Variable | Description |\n|---|---|\n" + "\n".join(rows)
    )

    st.divider()

    # ── Axes d'analyse ────────────────────────────────────────────────────────
    st.markdown("## 🗺️ Axes d'Analyse")
    a1, a2, a3, a4 = st.columns(4)

    for col, icon, title, desc, color in [
        (a1, "📊", "Exploration", "KPIs, distributions, corrélations, visualisations interactives", "#0ea5e9"),
        (a2, "⚠️", "Biais", "DPD, Disparate Impact, Equal Opportunity — Genre & Résidence", "#f97316"),
        (a3, "🤖", "Modélisation", "Random Forest + SMOTE, seuil optimisé pour le Recall médical", "#10b981"),
        (a4, "📋", "Fairness", "Performances comparées par groupe, confusion matrices croisées", "#8b5cf6"),
    ]:
        col.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-top:3px solid {color};
                    border-radius:12px; padding:1.2rem; text-align:center; height:100%;">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-family:Syne,sans-serif; font-weight:700;
                        color:#0a1628; font-size:1rem; margin:0.3rem 0;">{title}</div>
            <div style="color:#64748b; font-size:0.82rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
