"""pages/bias.py — Détection de biais algorithmique"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data import load_data, get_summary
from utils.plots import fairness_bar, age_stroke_line, PAL
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    predictive_parity,
    interpret_dpd,
    interpret_di,
)


def _badge(level: str, color: str) -> str:
    cls = {
        "Excellent":  "badge-green",
        "Acceptable": "badge-yellow",
        "Acceptable (règle 4/5 OK)": "badge-yellow",
        "Préoccupant": "badge-orange",
    }.get(level, "badge-red")
    return f'<span class="badge {cls}">{level}</span>'


def _metric_card(label: str, value: str, badge_level: str,
                 badge_color: str, detail: str = "") -> str:
    return f"""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.2rem; text-align:center;">
        <div style="font-size:0.75rem; font-weight:600; color:#64748b;
                    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.4rem;">
            {label}
        </div>
        <div style="font-family:Syne,sans-serif; font-size:2rem; font-weight:800;
                    color:#0a1628; line-height:1; margin-bottom:0.4rem;">{value}</div>
        {_badge(badge_level, badge_color)}
        <div style="font-size:0.8rem; color:#64748b; margin-top:0.4rem;">{detail}</div>
    </div>
    """


def _run_fairness(df, sensitive_col, unpriv, priv, label):
    """Calcule les 4 métriques de fairness pour un attribut sensible."""
    y = df["stroke"].values
    s = df[sensitive_col].values
    return {
        "dpd": demographic_parity_difference(y, y, s),
        "di":  disparate_impact_ratio(y, y, s, unpriv, priv),
        "eod": equal_opportunity_difference(y, y, s),
        "pp":  predictive_parity(y, y, s),
        "overall": df["stroke"].mean(),
    }


def render():
    df = load_data()
    s  = get_summary(df)

    st.markdown("# ⚠️ Détection de Biais")
    st.markdown("*Analyse de l'équité algorithmique — Genre & Zone Géographique*")

    # ── Intro ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-info">
        <h4>🔍 Qu'est-ce qu'un biais algorithmique en santé ?</h4>
        <p>Un biais algorithmique survient quand un modèle de prédiction produit des résultats 
        systématiquement différents selon l'appartenance à un groupe (genre, zone géographique, 
        ethnie…). En médecine, cela peut mener à un <b style="color:white;">sous-diagnostic</b> 
        pour certaines populations, réduisant leurs chances de recevoir un traitement à temps.
        Nous analysons ici les données brutes — la page <b style="color:white;">Modélisation</b> 
        étend cette analyse aux prédictions du modèle.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # TABS : Genre / Zone / Vue croisée
    # ══════════════════════════════════════════════════════════════════════════
    tab_genre, tab_resid, tab_cross = st.tabs([
        "👤 Biais de Genre",
        "🏘️ Biais Géographique",
        "🔀 Vue Combinée",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 : GENRE
    # ──────────────────────────────────────────────────────────────────────────
    with tab_genre:
        g = _run_fairness(df, "gender", "Female", "Male", "Genre")
        dpd, di_, eod = g["dpd"], g["di"], g["eod"]

        st.markdown("## Analyse du Biais de Genre")

        st.markdown("""
        **Attribut sensible :** Genre (Male / Female)

        **Pourquoi c'est problématique ?** Un modèle de prédiction d'AVC qui sous-détecte 
        le risque chez un genre — à facteurs de risque médicaux identiques — est inéquitable. 
        Cela pourrait retarder le diagnostic et le traitement pour les patients concernés, 
        avec des conséquences graves car l'AVC est une urgence neurologique nécessitant 
        une intervention dans les **4h30** pour limiter les séquelles.
        """)

        # ── Métriques ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-label">Métriques de fairness</p>',
                    unsafe_allow_html=True)

        dpd_level, dpd_color = interpret_dpd(dpd["difference"])
        di_level,  di_color  = interpret_di(di_["ratio"])

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(_metric_card(
            "Parité Démographique (DPD)",
            f"{dpd['difference']:.4f}",
            dpd_level, dpd_color,
            "Idéal = 0"
        ), unsafe_allow_html=True)
        m2.markdown(_metric_card(
            "Impact Disparate (DI)",
            f"{di_['ratio']:.4f}",
            di_level, di_color,
            "Règle 4/5 : ≥ 0.80"
        ), unsafe_allow_html=True)
        m3.markdown(_metric_card(
            "Equal Opportunity (EOD)",
            f"{eod['difference']:.4f}",
            interpret_dpd(eod["difference"])[0],
            interpret_dpd(eod["difference"])[1],
            "Diff. TPR entre groupes"
        ), unsafe_allow_html=True)

        male_rate   = dpd["rates"].get("Male", 0)
        female_rate = dpd["rates"].get("Female", 0)
        ratio_mf    = female_rate / male_rate if male_rate > 0 else 0
        m4.markdown(_metric_card(
            "Taux F/M",
            f"{ratio_mf:.3f}",
            "Acceptable" if 0.85 <= ratio_mf <= 1.15 else "Préoccupant",
            "#f59e0b",
            f"F={female_rate:.3f} · M={male_rate:.3f}"
        ), unsafe_allow_html=True)

        # ── Détail par groupe ─────────────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:1.5rem;">Visualisations</p>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = fairness_bar(
                dpd["rates"],
                "Taux d'AVC réel par Genre",
                color_map={"Male": PAL["male"], "Female": PAL["female"]},
                overall=g["overall"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.plotly_chart(age_stroke_line(df), use_container_width=True)

        # ── Interprétation ────────────────────────────────────────────────────
        higher = "les Hommes" if male_rate > female_rate else "les Femmes"
        lower  = "les Femmes" if male_rate > female_rate else "les Hommes"
        h_rate = max(male_rate, female_rate)
        l_rate = min(male_rate, female_rate)

        st.markdown(f"""
        <div class="card-danger">
            <h4>📋 Interprétation — Biais de Genre</h4>
            <p>La <b>Différence de Parité Démographique</b> est de <b>{dpd['difference']:.4f}</b> 
            ({dpd_level}). {higher} ont un taux d'AVC de <b>{h_rate:.2%}</b> contre 
            <b>{l_rate:.2%}</b> pour {lower} — soit une différence absolue de 
            <b>{abs(h_rate - l_rate)*100:.2f} points de pourcentage</b>.</p>
            <p>Le <b>Ratio d'Impact Disparate</b> est de <b>{di_['ratio']:.4f}</b>. 
            {'La règle des 4/5 est respectée (DI ≥ 0.80), indiquant une absence de discrimination algorithmique grave selon ce critère.' if di_['rule_4_5_ok'] else '⚠️ La règle des 4/5 est VIOLÉE (DI < 0.80) — signal fort de discrimination potentielle.'}</p>
            <p>La courbe âge × genre montre que <b>le risque augmente fortement après 60 ans pour les deux genres</b>, 
            avec des variations inter-groupes à surveiller dans le modèle prédictif.</p>
            <p><b>Recommandations :</b> Stratifier les évaluations du modèle par genre, 
            surveiller les faux négatifs séparément, et envisager des seuils de décision 
            différenciés si les taux de vrais positifs divergent significativement.</p>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 : ZONE GÉOGRAPHIQUE
    # ──────────────────────────────────────────────────────────────────────────
    with tab_resid:
        r = _run_fairness(df, "Residence_type", "Rural", "Urban", "Zone")
        dpd_r, di_r, eod_r = r["dpd"], r["di"], r["eod"]

        st.markdown("## Analyse du Biais Géographique")

        st.markdown("""
        **Attribut sensible :** Zone de résidence (Urban / Rural)

        **Pourquoi c'est problématique ?** Les populations rurales ont généralement un accès 
        plus limité aux soins primaires et aux structures d'urgence neurologiques. Si un modèle 
        sous-prédit le risque d'AVC en zone rurale, il prive ces patients d'alertes précoces, 
        aggravant des inégalités de santé structurelles déjà importantes.
        """)

        dpd_r_level, dpd_r_color = interpret_dpd(dpd_r["difference"])
        di_r_level,  di_r_color  = interpret_di(di_r["ratio"])

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(_metric_card(
            "Parité Démographique (DPD)",
            f"{dpd_r['difference']:.4f}",
            dpd_r_level, dpd_r_color, "Idéal = 0"
        ), unsafe_allow_html=True)
        m2.markdown(_metric_card(
            "Impact Disparate (DI)",
            f"{di_r['ratio']:.4f}",
            di_r_level, di_r_color, "Règle 4/5 : ≥ 0.80"
        ), unsafe_allow_html=True)
        m3.markdown(_metric_card(
            "Equal Opportunity (EOD)",
            f"{eod_r['difference']:.4f}",
            interpret_dpd(eod_r["difference"])[0],
            interpret_dpd(eod_r["difference"])[1],
            "Diff. TPR entre groupes"
        ), unsafe_allow_html=True)

        rural_rate = dpd_r["rates"].get("Rural", 0)
        urban_rate = dpd_r["rates"].get("Urban", 0)
        m4.markdown(_metric_card(
            "Taux Rural/Urban",
            f"{rural_rate/urban_rate:.3f}" if urban_rate > 0 else "N/A",
            "Acceptable" if abs(rural_rate - urban_rate) < 0.01 else "Préoccupant",
            "#f59e0b",
            f"Rural={rural_rate:.3f} · Urban={urban_rate:.3f}"
        ), unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = fairness_bar(
                dpd_r["rates"],
                "Taux d'AVC réel par Zone de Résidence",
                color_map={"Urban": PAL["urban"], "Rural": PAL["rural"]},
                overall=r["overall"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Zone × Age line
            grp = (df.groupby(["age_group", "Residence_type"])["stroke"]
                     .mean().reset_index())
            grp["pct"] = (grp["stroke"] * 100).round(2)
            fig = px.line(
                grp, x="age_group", y="pct", color="Residence_type",
                color_discrete_map={"Urban": PAL["urban"], "Rural": PAL["rural"]},
                markers=True,
                labels={"age_group": "Tranche d'âge",
                        "pct": "Taux d'AVC (%)",
                        "Residence_type": "Zone"},
            )
            fig.update_traces(line_width=2.5, marker_size=8)
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title="Taux d'AVC par Tranche d'âge × Zone",
                yaxis_gridcolor="#f1f5f9",
                font=dict(family="Inter"), height=340,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card-warning">
            <h4>📋 Interprétation — Biais Géographique</h4>
            <p>La différence de parité entre zones Rurale et Urbaine est de 
            <b>{dpd_r['difference']:.4f}</b> ({dpd_r_level}). 
            Le taux d'AVC est de <b>{rural_rate:.2%}</b> en zone rurale vs 
            <b>{urban_rate:.2%}</b> en zone urbaine.</p>
            <p>Le ratio DI Rural/Urban est de <b>{di_r['ratio']:.4f}</b>. 
            {'La règle des 4/5 est respectée — pas de discrimination algorithmique majeure.' if di_r['rule_4_5_ok'] else '⚠️ Règle des 4/5 violée — discrimination potentielle envers les résidents ruraux.'}</p>
            <p>Le groupe potentiellement défavorisé par un biais algorithmique est celui 
            des <b>résidents ruraux</b> : déjà désavantagés par l'accès aux soins, 
            un modèle qui sous-prédirait leur risque aggraverait les inégalités existantes.</p>
            <p><b>Recommandations :</b> Collecter davantage de données spécifiques aux zones 
            rurales (distance aux urgences, accès aux médicaments antihypertenseurs), 
            et valider le modèle séparément sur chaque zone de résidence.</p>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 : VUE COMBINÉE
    # ──────────────────────────────────────────────────────────────────────────
    with tab_cross:
        st.markdown("## Vue Combinée : Genre × Zone")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            pivot = (df.groupby(["gender", "Residence_type"])["stroke"]
                       .mean().unstack() * 100).round(2)
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                text=[[f"{v:.2f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}", textfont=dict(size=16, color="white"),
                colorbar=dict(title="Taux AVC (%)"),
            ))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title="Taux d'AVC (%) : Genre × Zone de Résidence",
                font=dict(family="Inter"), height=340,
                xaxis_title="Zone", yaxis_title="Genre",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Bubble chart : effectifs et taux
            grp = (df.groupby(["gender", "Residence_type"])
                     .agg(n=("stroke", "count"), stroke_rate=("stroke", "mean"))
                     .reset_index())
            grp["pct"] = (grp["stroke_rate"] * 100).round(2)
            grp["label"] = grp["gender"] + " · " + grp["Residence_type"]

            fig = px.scatter(
                grp, x="Residence_type", y="gender",
                size="pct", color="pct",
                color_continuous_scale="YlOrRd",
                text="label",
                size_max=60,
                labels={"Residence_type": "Zone", "gender": "Genre",
                        "pct": "Taux AVC (%)"},
                title="Taux d'AVC (%) par croisement (taille ∝ taux)",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               font=dict(family="Inter"), height=340)
            st.plotly_chart(fig, use_container_width=True)

        # Tableau récapitulatif
        st.markdown("### 📋 Tableau Récapitulatif des Métriques de Fairness")

        g  = _run_fairness(df, "gender", "Female", "Male", "Genre")
        r_ = _run_fairness(df, "Residence_type", "Rural", "Urban", "Zone")

        summary = pd.DataFrame([
            {
                "Attribut":             "Genre (Male / Female)",
                "DPD":                  f"{g['dpd']['difference']:.4f}",
                "DI (F/M)":             f"{g['di']['ratio']:.4f}",
                "Règle 4/5":            "✅ Respectée" if g['di']['rule_4_5_ok'] else "❌ Violée",
                "EOD":                  f"{g['eod']['difference']:.4f}",
                "Niveau DPD":           interpret_dpd(g['dpd']['difference'])[0],
                "Niveau DI":            interpret_di(g['di']['ratio'])[0],
            },
            {
                "Attribut":             "Zone (Urban / Rural)",
                "DPD":                  f"{r_['dpd']['difference']:.4f}",
                "DI (Rural/Urban)":     f"{r_['di']['ratio']:.4f}",
                "Règle 4/5":            "✅ Respectée" if r_['di']['rule_4_5_ok'] else "❌ Violée",
                "EOD":                  f"{r_['eod']['difference']:.4f}",
                "Niveau DPD":           interpret_dpd(r_['dpd']['difference'])[0],
                "Niveau DI":            interpret_di(r_['di']['ratio'])[0],
            },
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="card-success">
            <h4>✅ Conclusion sur les données brutes</h4>
            <p>Sur ce dataset, les biais dans les <b>données brutes</b> sont relativement 
            modérés pour les deux attributs sensibles analysés. Les DPD sont faibles et les 
            ratios DI respectent la règle des 4/5.</p>
            <p>Cependant, ces résultats sur les données brutes ne garantissent pas l'absence 
            de biais dans les <b>prédictions du modèle</b> — voir la page Modélisation pour 
            l'analyse de fairness sur les prédictions.</p>
        </div>
        """, unsafe_allow_html=True)
