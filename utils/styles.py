"""
utils/styles.py
───────────────
CSS global partagé entre toutes les pages.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Variables de couleur ────────────────────────────────────────────────── */
:root {
    --navy:      #0a1628;
    --navy-mid:  #1e3a5f;
    --sky:       #0ea5e9;
    --sky-light: #7dd3fc;
    --slate:     #64748b;
    --danger:    #ef4444;
    --warning:   #f97316;
    --success:   #10b981;
    --amber:     #f59e0b;
    --bg:        #ffffff;
    --bg2:       #f8fafc;
    --border:    #e2e8f0;
    --text:      #0f172a;
}

/* ── Typographie ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
h1, h2 {
    font-family: 'Syne', sans-serif !important;
}
h1 { color: var(--navy) !important; font-weight: 800 !important; font-size: 2.2rem !important; }
h2 { color: var(--navy-mid) !important; font-weight: 700 !important; font-size: 1.5rem !important; }
h3 { color: var(--navy-mid) !important; font-weight: 600 !important; }

/* ── Layout ──────────────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1300px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0f2042 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
section[data-testid="stSidebar"] .stRadio label { padding: 4px 0; font-size: 0.9rem; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-top: 3px solid var(--sky);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: var(--navy);
}

/* ── Custom cards ────────────────────────────────────────────────────────── */
.card-hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #0369a1 100%);
    color: white;
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.card-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(14,165,233,0.12);
    border-radius: 50%;
}
.card-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 100px;
    width: 160px; height: 160px;
    background: rgba(125,211,252,0.07);
    border-radius: 50%;
}
.card-hero h1 {
    color: white !important;
    font-size: 2.6rem !important;
    line-height: 1.15 !important;
    margin: 0.3rem 0 0.6rem 0 !important;
}
.card-hero .subtitle {
    color: var(--sky-light);
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.card-info {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
    color: white;
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    margin: 0.5rem 0;
}
.card-info h4 { color: var(--sky-light) !important; margin: 0 0 0.5rem 0; font-family: 'Syne', sans-serif; }
.card-info p  { color: #cbd5e1; font-size: 0.88rem; margin: 0; line-height: 1.6; }

.card-danger {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid var(--danger);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.card-danger h4 { color: #dc2626 !important; margin: 0 0 0.4rem 0; }
.card-danger p  { color: #475569; font-size: 0.9rem; margin: 0.3rem 0; line-height: 1.6; }

.card-warning {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-left: 4px solid var(--warning);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.card-warning h4 { color: #c2410c !important; margin: 0 0 0.4rem 0; }
.card-warning p  { color: #475569; font-size: 0.9rem; margin: 0.3rem 0; line-height: 1.6; }

.card-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid var(--success);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.card-success h4 { color: #15803d !important; margin: 0 0 0.4rem 0; }
.card-success p  { color: #475569; font-size: 0.9rem; margin: 0.3rem 0; line-height: 1.6; }

.card-neutral {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
}
.card-neutral .icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.card-neutral h4 { color: var(--navy-mid) !important; margin: 0 0 0.4rem 0; }
.card-neutral p  { color: var(--slate); font-size: 0.88rem; margin: 0; }

/* ── Badge fairness ──────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-green  { background: #dcfce7; color: #15803d; }
.badge-yellow { background: #fef9c3; color: #854d0e; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.badge-red    { background: #fee2e2; color: #b91c1c; }

/* ── Plotly chart borders ─────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    padding: 4px;
}

/* ── Section separator ───────────────────────────────────────────────────── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sky);
    margin-bottom: 0.3rem;
}

hr { border: none; border-top: 2px solid var(--border); margin: 1.8rem 0; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { border-bottom: 2px solid var(--border); gap: 4px; }
.stTabs [data-baseweb="tab"]       { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500; }

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }

/* ── Selectbox / radio ───────────────────────────────────────────────────── */
[data-baseweb="select"] > div { border-radius: 8px !important; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
