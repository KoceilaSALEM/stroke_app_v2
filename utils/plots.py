"""
utils/plots.py
──────────────
Fonctions réutilisables pour générer des graphiques Plotly cohérents.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Palette cohérente ─────────────────────────────────────────────────────────
PAL = {
    "stroke_yes": "#ef4444",
    "stroke_no":  "#94a3b8",
    "male":       "#3b82f6",
    "female":     "#ec4899",
    "urban":      "#0ea5e9",
    "rural":      "#10b981",
    "primary":    "#0ea5e9",
    "navy":       "#0a1628",
    "amber":      "#f59e0b",
}

LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    margin=dict(t=50, b=40, l=40, r=20),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _base(**kwargs) -> dict:
    d = dict(LAYOUT_BASE)
    d.update(kwargs)
    return d


# ── Graphiques ────────────────────────────────────────────────────────────────
def stroke_donut(n_stroke: int, n_total: int) -> go.Figure:
    """Donut chart : répartition AVC / Non-AVC."""
    n_no = n_total - n_stroke
    rate = n_stroke / n_total * 100
    fig = go.Figure(go.Pie(
        labels=["Pas d'AVC", "AVC"],
        values=[n_no, n_stroke],
        marker_colors=[PAL["stroke_no"], PAL["stroke_yes"]],
        hole=0.62,
        textinfo="percent",
        textfont_size=13,
        direction="clockwise",
        sort=False,
    ))
    fig.add_annotation(
        text=f"<b>{rate:.1f}%</b><br><span style='font-size:10px'>AVC</span>",
        x=0.5, y=0.5, showarrow=False, font_size=18,
        font_color=PAL["stroke_yes"],
    )
    fig.update_layout(
        **_base(height=280, margin=dict(t=20, b=10, l=10, r=10)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    )
    return fig


def age_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogramme de l'âge coloré par statut AVC."""
    fig = go.Figure()
    for val, label, color in [(0, "Pas d'AVC", PAL["stroke_no"]),
                               (1, "AVC",       PAL["stroke_yes"])]:
        sub = df[df["stroke"] == val]["age"]
        fig.add_trace(go.Histogram(
            x=sub, name=label, nbinsx=40,
            marker_color=color, opacity=0.75,
        ))
    fig.update_layout(
        **_base(title="Distribution de l'âge par statut AVC",
                height=350, barmode="overlay"),
        xaxis_title="Âge", yaxis_title="Effectif",
        yaxis_gridcolor="#f1f5f9",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def group_bar(df: pd.DataFrame, group_col: str, title: str,
              color_map: dict) -> go.Figure:
    """Bar chart : taux d'AVC (%) par groupe."""
    grp = (
        df.groupby(group_col)["stroke"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    grp["pct"] = (grp["rate"] * 100).round(2)
    overall = df["stroke"].mean() * 100

    fig = go.Figure()
    for _, row in grp.iterrows():
        fig.add_trace(go.Bar(
            x=[row[group_col]], y=[row["pct"]],
            name=str(row[group_col]),
            marker_color=color_map.get(row[group_col], PAL["primary"]),
            text=f"{row['pct']:.1f}%",
            textposition="outside",
            width=0.45,
        ))
    fig.add_hline(y=overall, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Moyenne : {overall:.1f}%",
                  annotation_position="top right")
    fig.update_layout(
        **_base(title=title, height=360),
        xaxis_title="", yaxis_title="Taux d'AVC (%)",
        yaxis_gridcolor="#f1f5f9", showlegend=False,
        yaxis_range=[0, grp["pct"].max() * 1.3],
    )
    return fig


def grouped_bar_comparison(df: pd.DataFrame, group_col: str,
                            color_map: dict) -> go.Figure:
    """Bar chart groupé AVC/Non-AVC par groupe."""
    grp = (df.groupby([group_col, "stroke"])
             .size().reset_index(name="count"))
    total = df.groupby(group_col).size().reset_index(name="total")
    grp = grp.merge(total, on=group_col)
    grp["pct"] = (grp["count"] / grp["total"] * 100).round(2)
    grp["label"] = grp["stroke"].map({0: "Pas d'AVC", 1: "AVC"})

    fig = px.bar(
        grp, x=group_col, y="pct", color="label",
        color_discrete_map={"AVC": PAL["stroke_yes"], "Pas d'AVC": PAL["stroke_no"]},
        barmode="group",
        text=grp["pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **_base(height=370),
        xaxis_title="", yaxis_title="% de patients",
        yaxis_gridcolor="#f1f5f9",
        legend=dict(title="", orientation="h", y=1.1),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap de corrélation des variables numériques."""
    cols = ["age", "avg_glucose_level", "bmi",
            "hypertension", "heart_disease", "stroke"]
    labels = ["Âge", "Glycémie", "IMC",
              "Hypertension", "Maladie cardiaque", "AVC"]
    corr = df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hoverongaps=False,
        colorbar=dict(title="r", len=0.8),
    ))
    fig.update_layout(
        **_base(title="Corrélations entre variables numériques", height=400),
    )
    return fig


def fairness_bar(rates: dict, title: str,
                 color_map: dict = None, overall: float = None) -> go.Figure:
    """Bar chart des taux par groupe pour les métriques de fairness."""
    groups = list(rates.keys())
    values = [rates[g] * 100 for g in groups]

    colors = [color_map.get(g, PAL["primary"]) if color_map else PAL["primary"]
              for g in groups]

    fig = go.Figure(go.Bar(
        x=groups, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        width=0.45,
    ))
    if overall is not None:
        fig.add_hline(y=overall * 100, line_dash="dash", line_color="#94a3b8",
                      annotation_text=f"Global : {overall*100:.1f}%")
    fig.update_layout(
        **_base(title=title, height=340),
        xaxis_title="", yaxis_title="Taux (%)",
        yaxis_gridcolor="#f1f5f9", showlegend=False,
        yaxis_range=[0, max(values) * 1.35 if values else 10],
    )
    return fig


def confusion_matrix_plot(cm: np.ndarray, title: str = "Matrice de Confusion",
                           height: int = 320) -> go.Figure:
    """Heatmap d'une matrice de confusion."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Prédit : Non-AVC", "Prédit : AVC"],
        y=["Réel : Non-AVC", "Réel : AVC"],
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        showscale=False,
    ))
    fig.update_layout(
        **_base(title=title, height=height,
                margin=dict(t=55, b=20, l=20, r=20)),
    )
    return fig


def feature_importance_bar(df_imp: pd.DataFrame) -> go.Figure:
    """Bar chart horizontal : importance des variables."""
    fig = px.bar(
        df_imp, x="importance", y="label", orientation="h",
        color="importance",
        color_continuous_scale=[[0, "#e0f2fe"], [0.5, "#38bdf8"], [1, "#0369a1"]],
        text=df_imp["importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **_base(title="Importance des Variables (Random Forest)", height=380,
                margin=dict(t=50, b=30, l=160, r=60)),
        xaxis_title="Importance", yaxis_title="",
        coloraxis_showscale=False,
        xaxis_gridcolor="#f1f5f9",
    )
    return fig


def age_stroke_line(df: pd.DataFrame) -> go.Figure:
    """Taux d'AVC par tranche d'âge et genre."""
    grp = df.groupby(["age_group", "gender"])["stroke"].mean().reset_index()
    grp["pct"] = (grp["stroke"] * 100).round(2)
    color_map = {"Male": PAL["male"], "Female": PAL["female"]}

    fig = px.line(
        grp, x="age_group", y="pct", color="gender",
        color_discrete_map=color_map,
        markers=True,
        labels={"age_group": "Tranche d'âge",
                "pct": "Taux d'AVC (%)", "gender": "Genre"},
    )
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(
        **_base(title="Taux d'AVC par tranche d'âge et genre", height=360),
        xaxis_title="Tranche d'âge", yaxis_title="Taux d'AVC (%)",
        yaxis_gridcolor="#f1f5f9",
        legend=dict(title="Genre", orientation="h", y=1.1),
    )
    return fig


def glucose_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plot glycémie selon le statut AVC."""
    fig = go.Figure()
    for val, label, color in [(0, "Pas d'AVC", PAL["stroke_no"]),
                               (1, "AVC",       PAL["stroke_yes"])]:
        fig.add_trace(go.Box(
            y=df[df["stroke"] == val]["avg_glucose_level"],
            name=label, marker_color=color,
            boxmean=True, width=0.4,
        ))
    fig.update_layout(
        **_base(title="Glycémie moyenne selon le statut AVC", height=360),
        yaxis_title="Glycémie (mg/dL)",
        yaxis_gridcolor="#f1f5f9", showlegend=True,
    )
    return fig


def work_stroke_bar(df: pd.DataFrame) -> go.Figure:
    """Taux d'AVC par type d'emploi."""
    grp = df.groupby("work_type_label")["stroke"].mean().reset_index()
    grp["pct"] = (grp["stroke"] * 100).round(2)
    grp = grp.sort_values("pct", ascending=False)

    fig = px.bar(
        grp, x="work_type_label", y="pct",
        color="pct",
        color_continuous_scale=[[0, "#e0f2fe"], [1, "#0369a1"]],
        text=grp["pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **_base(title="Taux d'AVC par type d'emploi", height=360),
        xaxis_title="", yaxis_title="Taux d'AVC (%)",
        yaxis_gridcolor="#f1f5f9", coloraxis_showscale=False,
    )
    return fig
