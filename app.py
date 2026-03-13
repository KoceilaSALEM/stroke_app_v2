"""
app.py — Point d'entrée principal
══════════════════════════════════
Application Streamlit — Parcours A : Détection de Biais
Dataset : Stroke Prediction (Kaggle / fedesoriano)
"""

# ── Fix chemin Python (DOIT être en PREMIER, avant tout import utils) ─────────
# Ajoute la racine du projet dans sys.path pour que `import utils.*`
# fonctionne peu importe le répertoire de travail courant de Streamlit.
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
from utils.styles import inject_css

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AVC · Détection de Biais",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:1.2rem 0 0.5rem; text-align:center;'>
    <div style='font-size:2.8rem; line-height:1;'>🧠</div>
    <div style='font-family:Syne,sans-serif; font-size:1.15rem;
                color:#7dd3fc; margin-top:0.4rem; font-weight:700;'>
        AVC Risk Analysis
    </div>
    <div style='font-size:0.7rem; color:#64748b; letter-spacing:0.12em;
                margin-top:0.1rem; text-transform:uppercase;'>
        Détection de Biais · Parcours A
    </div>
</div>
<hr style='border-color:#1e3a5f; margin:0.8rem 0 1rem 0;'>
""", unsafe_allow_html=True)

PAGES = {
    "🏠  Accueil":                 "home",
    "📊  Exploration des Données": "exploration",
    "⚠️  Détection de Biais":      "bias",
    "🤖  Modélisation":            "model",
}

selection = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown("""
<hr style='border-color:#1e3a5f; margin:1rem 0;'>
<div style='font-size:0.78rem; color:#475569; line-height:1.7; padding:0 0.2rem;'>
    <b style='color:#7dd3fc;'>Dataset</b><br>
    Stroke Prediction<br>
    <span style='color:#94a3b8;'>Kaggle — fedesoriano</span><br><br>
    <b style='color:#7dd3fc;'>5 109 patients</b><br>
    <span style='color:#94a3b8;'>249 cas d'AVC (4.87 %)</span><br><br>
    <b style='color:#7dd3fc;'>Biais analysés</b><br>
    <span style='color:#94a3b8;'>Genre · Zone géographique</span>
</div>
""", unsafe_allow_html=True)

# ── Routing ───────────────────────────────────────────────────────────────────
page = PAGES[selection]

if page == "home":
    from pages.home import render
elif page == "exploration":
    from pages.exploration import render
elif page == "bias":
    from pages.bias import render
elif page == "model":
    from pages.model import render

render()
