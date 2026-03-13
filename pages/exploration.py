"""pages/exploration.py — Exploration des données"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data import load_data, get_summary
from utils.plots import (
    age_histogram, grouped_bar_comparison, correlation_heatmap,
    glucose_boxplot, work_stroke_bar, PAL,
)


def render():
    df = load_data()
    s  = get_summary(df)

    st.markdown("# 📊 Exploration des Données")
    st.markdown("*Analyse exploratoire · Stroke Prediction Dataset*")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Métriques clés</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lignes totales",        f"{s['n_total']:,}",
              "1 patient 'Other' retiré")
    k2.metric("Taux d'AVC",            f"{s['stroke_rate']*100:.2f}%",
              f"{s['n_stroke']} cas positifs")
    k3.metric("BMI médian",            f"{s['bmi_median']:.1f}",
              f"{s['n_bmi_imputed']} valeurs imputées")
    k4.metric("Glycémie moyenne",      f"{s['glucose_mean']:.1f} mg/dL",
              "Normal < 100 mg/dL")

    st.divider()

    # ── VIZ 1 : Distribution variable cible ──────────────────────────────────
    st.markdown('<p class="section-label">Visualisation 1 · Variable cible</p>',
                unsafe_allow_html=True)
    st.markdown("### Distribution de la Variable Cible (AVC)")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        # Histogram AVC vs non-AVC
        counts = df["stroke"].value_counts().reset_index()
        counts.columns = ["stroke", "count"]
        counts["label"] = counts["stroke"].map({0: "Pas d'AVC", 1: "AVC"})
        counts["pct"]   = (counts["count"] / s["n_total"] * 100).round(2)

        fig = go.Figure()
        for _, row in counts.iterrows():
            color = PAL["stroke_yes"] if row["stroke"] == 1 else PAL["stroke_no"]
            fig.add_trace(go.Bar(
                x=[row["label"]], y=[row["count"]],
                marker_color=color,
                text=f"{row['count']:,}<br>({row['pct']:.1f}%)",
                textposition="inside", textfont=dict(color="white", size=14),
                width=0.4,
            ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=360,
            font=dict(family="Inter"),
            yaxis_gridcolor="#f1f5f9",
            yaxis_title="Nombre de patients",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <div class="card-danger" style="margin-top:1.5rem;">
            <h4>⚠️ Déséquilibre extrême</h4>
            <p><b>95.1%</b> des patients n'ont pas subi d'AVC contre seulement 
            <b>4.9%</b> qui en ont eu un.</p>
            <p>Ce déséquilibre sévère est un défi majeur pour la modélisation : 
            un modèle naïf qui prédit toujours "Pas d'AVC" atteindrait 95% 
            d'accuracy sans aucune utilité médicale.</p>
            <p>→ Solution : <b>SMOTE</b> (Synthetic Minority Oversampling) 
            appliqué sur le train set + seuil de décision optimisé.</p>
        </div>
        """, unsafe_allow_html=True)

        # Glucose cat distribution
        gcat = df["glucose_cat"].value_counts().reset_index()
        gcat.columns = ["cat", "count"]
        fig2 = px.pie(gcat, names="cat", values="count",
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.4)
        fig2.update_layout(height=220, margin=dict(t=10, b=0, l=0, r=0),
                           paper_bgcolor="white",
                           legend=dict(font_size=10, orientation="v"))
        fig2.update_traces(textinfo="percent", textfont_size=11)
        st.caption("Répartition des catégories glycémiques")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── VIZ 2 : Comparaison par attribut sensible ─────────────────────────────
    st.markdown('<p class="section-label">Visualisation 2 · Attributs sensibles</p>',
                unsafe_allow_html=True)
    st.markdown("### Comparaison par Attribut Sensible")

    tab_genre, tab_resid = st.tabs(["👤 Genre", "🏘️ Zone de résidence"])

    with tab_genre:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = grouped_bar_comparison(df, "gender",
                                          {"Male": PAL["male"], "Female": PAL["female"]})
            fig.update_layout(title="Taux d'AVC par Genre (%)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Age + genre heatmap
            pivot = (df.groupby(["age_group", "gender"])["stroke"]
                       .mean().unstack() * 100).round(2)
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}", textfont=dict(size=13),
                colorbar=dict(title="Taux AVC%", len=0.8),
            ))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title="Taux d'AVC (%) · Âge × Genre",
                xaxis_title="Genre", yaxis_title="Tranche d'âge",
                height=360, font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_resid:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = grouped_bar_comparison(df, "Residence_type",
                                          {"Urban": PAL["urban"], "Rural": PAL["rural"]})
            fig.update_layout(title="Taux d'AVC par Zone de Résidence (%)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Résidence × type emploi
            pivot2 = (df.groupby(["Residence_type", "work_type_label"])["stroke"]
                        .mean().unstack() * 100).round(2)
            fig = px.bar(
                df.groupby(["Residence_type", "work_type_label"])["stroke"]
                  .mean().reset_index().assign(pct=lambda x: x["stroke"]*100),
                x="work_type_label", y="pct", color="Residence_type",
                barmode="group",
                color_discrete_map={"Urban": PAL["urban"], "Rural": PAL["rural"]},
                labels={"work_type_label": "Type d'emploi",
                        "pct": "Taux AVC (%)", "Residence_type": "Zone"},
                title="Taux d'AVC : Zone × Type d'emploi",
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               yaxis_gridcolor="#f1f5f9",
                               font=dict(family="Inter"), height=360)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── VIZ 3 : Variables numériques ─────────────────────────────────────────
    st.markdown('<p class="section-label">Visualisation 3 · Variables numériques</p>',
                unsafe_allow_html=True)
    st.markdown("### Distribution des Variables Numériques")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(age_histogram(df), use_container_width=True)
    with c2:
        st.plotly_chart(glucose_boxplot(df), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(work_stroke_bar(df), use_container_width=True)
    with c4:
        # BMI scatter
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="bmi", y="avg_glucose_level",
            color=df.sample(min(1500, len(df)), random_state=42)["stroke"].map(
                {0: "Pas d'AVC", 1: "AVC"}),
            color_discrete_map={"AVC": PAL["stroke_yes"], "Pas d'AVC": PAL["stroke_no"]},
            opacity=0.6,
            labels={"bmi": "IMC (BMI)",
                    "avg_glucose_level": "Glycémie (mg/dL)",
                    "color": "Statut"},
            title="IMC vs Glycémie (échantillon 1 500 pts)",
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_gridcolor="#f1f5f9", xaxis_gridcolor="#f1f5f9",
                           font=dict(family="Inter"), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── VIZ 4 : Heatmap corrélations ─────────────────────────────────────────
    st.markdown('<p class="section-label">Visualisation 4 · Corrélations</p>',
                unsafe_allow_html=True)
    st.markdown("### Matrice de Corrélations")
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    st.divider()

    # ── Aperçu interactif ────────────────────────────────────────────────────
    st.markdown("### 🗂️ Aperçu Interactif du Dataset")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_stroke = st.selectbox("Filtrer par statut AVC",
                                 ["Tous", "AVC (1)", "Pas d'AVC (0)"])
    with fc2:
        f_gender = st.selectbox("Filtrer par genre",
                                 ["Tous", "Female", "Male"])
    with fc3:
        f_res = st.selectbox("Filtrer par zone",
                              ["Toutes", "Urban", "Rural"])

    dff = df.copy()
    if f_stroke == "AVC (1)":      dff = dff[dff["stroke"] == 1]
    elif f_stroke == "Pas d'AVC (0)": dff = dff[dff["stroke"] == 0]
    if f_gender != "Tous":         dff = dff[dff["gender"] == f_gender]
    if f_res != "Toutes":          dff = dff[dff["Residence_type"] == f_res]

    display_cols = ["gender", "age", "hypertension", "heart_disease",
                    "ever_married", "work_type_label", "Residence_type",
                    "avg_glucose_level", "bmi", "smoking_label", "stroke"]

    st.caption(f"{len(dff):,} lignes affichées sur {len(df):,} total")
    st.dataframe(
        dff[display_cols].rename(columns={
            "gender": "Genre", "age": "Âge",
            "hypertension": "HTA", "heart_disease": "Card.",
            "ever_married": "Marié(e)",
            "work_type_label": "Emploi",
            "Residence_type": "Zone",
            "avg_glucose_level": "Glycémie",
            "bmi": "IMC",
            "smoking_label": "Tabac",
            "stroke": "AVC",
        }).head(100),
        use_container_width=True,
        hide_index=True,
    )
