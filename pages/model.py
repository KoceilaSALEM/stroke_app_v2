"""pages/model.py — Modélisation & Fairness sur prédictions"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from utils.model import get_results
from utils.plots import (
    confusion_matrix_plot, feature_importance_bar,
    fairness_bar, PAL,
)
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    interpret_dpd, interpret_di,
)


def _group_metrics(df_test: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for group in sorted(df_test[col].unique()):
        mask = df_test[col] == group
        y_t = df_test.loc[mask, "y_true"]
        y_p = df_test.loc[mask, "y_pred"]
        rows.append({
            "Groupe":    group,
            "N":         int(mask.sum()),
            "AVC réels": int(y_t.sum()),
            "Accuracy":  round(accuracy_score(y_t, y_p), 3),
            "Precision": round(precision_score(y_t, y_p, zero_division=0), 3),
            "Recall":    round(recall_score(y_t, y_p, zero_division=0), 3),
            "F1":        round(f1_score(y_t, y_p, zero_division=0), 3),
        })
    return pd.DataFrame(rows)


def render():
    st.markdown("# 🤖 Modélisation & Fairness")
    st.markdown("*Régression Logistique pré-entraînée — évaluation et analyse d'équité*")

    # ── Chargement des résultats ──────────────────────────────────────────────
    with st.spinner("📦 Chargement du modèle..."):
        result = get_results()

    m   = result["metrics"]
    dft = result["df_test"]
    fi  = result["feature_imp"]

    # ── Bandeau info pipeline ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">Pipeline</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card-info">
        <h4>🔧 Pipeline de modélisation</h4>
        <p>
        <b style="color:white;">Modèle :</b> Régression Logistique (C=1.0, max_iter=1000)<br>
        <b style="color:white;">Split :</b> 80% train / 20% test — stratifié sur la variable cible<br>
        <b style="color:white;">SMOTE :</b> {m['n_smote_pos']:,} exemples positifs synthétiques générés sur le train set<br>
        <b style="color:white;">Seuil de décision :</b> {m['threshold']} — abaissé pour maximiser le Recall
        (un faux négatif = AVC non détecté est plus coûteux qu'un faux positif)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Badges résumé
    s1, s2, s3, s4 = st.columns(4)
    s1.success(f"✅ Train set : {m['n_train']:,} exemples après SMOTE")
    s2.info(f"🧪 Test set : {m['n_test']:,} exemples")
    s3.info(f"🔄 {m['n_smote_pos']:,} cas AVC synthétiques")
    s4.warning(f"⚖️ Seuil = {m['threshold']}")

    st.divider()

    # ── Performances globales ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">Performances globales</p>',
                unsafe_allow_html=True)
    st.markdown("### 📈 Métriques sur le Test Set")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy",  f"{m['accuracy']:.3f}")
    k2.metric("Precision", f"{m['precision']:.3f}")
    k3.metric("Recall",    f"{m['recall']:.3f}", "↑ Priorité médicale")
    k4.metric("F1-Score",  f"{m['f1']:.3f}")
    k5.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(confusion_matrix_plot(m["cm"]), use_container_width=True)
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:.5rem;">
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:.6rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:#15803d;">{tp}</div>
                <div style="font-size:.75rem;color:#166534;">Vrais Positifs</div>
            </div>
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:.6rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:#dc2626;">{fn}</div>
                <div style="font-size:.75rem;color:#991b1b;">Faux Négatifs ⚠️</div>
            </div>
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:.6rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:#c2410c;">{fp}</div>
                <div style="font-size:.75rem;color:#9a3412;">Faux Positifs</div>
            </div>
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:.6rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:#0369a1;">{tn}</div>
                <div style="font-size:.75rem;color:#075985;">Vrais Négatifs</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.plotly_chart(feature_importance_bar(fi), use_container_width=True)
        st.caption("Importance = valeur absolue des coefficients de la Régression Logistique")

    st.divider()

    # ── Fairness sur prédictions ──────────────────────────────────────────────
    st.markdown('<p class="section-label">Fairness sur les prédictions</p>',
                unsafe_allow_html=True)
    st.markdown("### ⚖️ Analyse d'Équité")
    st.markdown("""
    > **Note :** Cette section évalue si le modèle introduit ou amplifie des biais
    > dans ses prédictions, indépendamment des biais présents dans les données brutes.
    """)

    tab_g, tab_r = st.tabs(["👤 Genre", "🏘️ Zone Géographique"])

    for tab, col, unpriv, priv, colors in [
        (tab_g, "gender",         "Female", "Male",
         {"Male": PAL["male"], "Female": PAL["female"]}),
        (tab_r, "Residence_type", "Rural",  "Urban",
         {"Urban": PAL["urban"], "Rural": PAL["rural"]}),
    ]:
        with tab:
            y_true = dft["y_true"].values
            y_pred = dft["y_pred"].values
            s_attr = dft[col].values

            dpd = demographic_parity_difference(y_true, y_pred, s_attr)
            di  = disparate_impact_ratio(y_true, y_pred, s_attr, unpriv, priv)
            eod = equal_opportunity_difference(y_true, y_pred, s_attr)

            dpd_l, _ = interpret_dpd(dpd["difference"])
            di_l, _  = interpret_di(di["ratio"])

            m1, m2, m3 = st.columns(3)
            m1.metric("DPD", f"{dpd['difference']:.4f}", f"Niveau : {dpd_l}")
            m2.metric(f"DI ({unpriv}/{priv})", f"{di['ratio']:.4f}",
                      "✅ Règle 4/5 OK" if di["rule_4_5_ok"] else "❌ Règle 4/5 violée")
            m3.metric("EOD", f"{eod['difference']:.4f}")

            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = fairness_bar(
                    dpd["rates"],
                    f"Taux de prédiction positive — {col}",
                    color_map=colors,
                    overall=dft["y_pred"].mean(),
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                gm = _group_metrics(dft, col)
                fig = px.bar(
                    gm.melt(id_vars="Groupe",
                            value_vars=["Accuracy", "Precision", "Recall", "F1"]),
                    x="variable", y="value", color="Groupe",
                    barmode="group", color_discrete_map=colors,
                    title=f"Métriques par groupe — {col}",
                    labels={"variable": "Métrique", "value": "Score", "Groupe": "Groupe"},
                )
                fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
                fig.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis_gridcolor="#f1f5f9", font=dict(family="Inter"),
                    height=350, yaxis_range=[0, 1.15],
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Matrices de confusion par groupe — {col}**")
            groups = sorted(dft[col].unique())
            cols_cm = st.columns(len(groups))
            for i, group in enumerate(groups):
                mask = dft[col] == group
                cm_g = confusion_matrix(dft.loc[mask, "y_true"],
                                        dft.loc[mask, "y_pred"])
                rec  = recall_score(dft.loc[mask, "y_true"],
                                    dft.loc[mask, "y_pred"], zero_division=0)
                with cols_cm[i]:
                    st.plotly_chart(
                        confusion_matrix_plot(
                            cm_g,
                            title=f"{group}<br><sup>Recall={rec:.3f}</sup>",
                            height=280,
                        ),
                        use_container_width=True,
                    )

            st.markdown("**Performances détaillées par groupe**")
            st.dataframe(_group_metrics(dft, col), use_container_width=True,
                         hide_index=True)

    st.divider()

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.markdown("### 📝 Conclusion")
    recall_g = _group_metrics(dft, "gender")
    recall_r = _group_metrics(dft, "Residence_type")
    min_rg, max_rg = recall_g["Recall"].min(), recall_g["Recall"].max()
    min_rr, max_rr = recall_r["Recall"].min(), recall_r["Recall"].max()

    st.markdown(f"""
    <div class="card-success">
        <h4>✅ Synthèse</h4>
        <p>La <b>Régression Logistique</b> entraînée avec SMOTE et seuil {m['threshold']}
        atteint un <b>Recall de {m['recall']:.3f}</b> et un <b>ROC-AUC de {m['roc_auc']:.3f}</b>.</p>
        <p>Écart de Recall par genre : <b>{max_rg - min_rg:.3f}</b>
        ({min_rg:.3f} → {max_rg:.3f}).
        Par zone géographique : <b>{max_rr - min_rr:.3f}</b>
        ({min_rr:.3f} → {max_rr:.3f}).</p>
        <p>Un faible écart de Recall entre groupes indique que le modèle traite
        équitablement les différentes populations — objectif clé en contexte médical.</p>
    </div>
    """, unsafe_allow_html=True)
