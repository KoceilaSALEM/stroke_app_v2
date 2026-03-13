# 🧠 AVC Risk Analysis — Stroke Prediction Bias Detection

Application Streamlit · **Parcours A : Détection de Biais**  
Dataset : [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — Kaggle / fedesoriano

---

## 📋 Présentation

Analyse du risque d'AVC sur 5 110 patients avec détection de biais algorithmique sur :
- **Genre** (Male / Female)
- **Zone géographique** (Urban / Rural)

---

## 🗂️ Structure du Projet

```
stroke_app/
├── app.py                         # Point d'entrée · navigation · routing
├── requirements.txt               # Dépendances Python
├── README.md
├── healthcare-dataset-stroke-data.csv
│
├── .streamlit/
│   └── config.toml                # Thème Streamlit
│
├── pages/                         # Une page = un module
│   ├── home.py                    # 🏠 Accueil
│   ├── exploration.py             # 📊 Exploration des données
│   ├── bias.py                    # ⚠️ Détection de biais
│   └── model.py                   # 🤖 Modélisation & Fairness
│
└── utils/                         # Modules réutilisables
    ├── data.py                    # Chargement & preprocessing (cache)
    ├── fairness.py                # Métriques : DPD, DI, EOD, PP
    ├── model.py                   # Pipeline SMOTE + Random Forest
    ├── plots.py                   # Graphiques Plotly réutilisables
    └── styles.py                  # CSS global partagé
```

---

## 📊 Pages

| Page | Contenu |
|---|---|
| 🏠 Accueil | Hero, KPIs, contexte, description du dataset |
| 📊 Exploration | 4 sections visuelles, filtres interactifs, aperçu dataset |
| ⚠️ Détection de Biais | DPD · DI · EOD · Parité Prédictive — Genre & Zone |
| 🤖 Modélisation | SMOTE · Random Forest · Métriques par groupe · Fairness |

---

## 📐 Métriques de Fairness

| Métrique | Description | Seuil idéal |
|---|---|---|
| **DPD** — Demographic Parity Difference | Diff. des taux de prédiction positive | ≤ 0.05 |
| **DI** — Disparate Impact Ratio | Ratio des taux (règle des 4/5) | ≥ 0.80 |
| **EOD** — Equal Opportunity Difference | Diff. des taux de vrais positifs | ≤ 0.05 |
| **PP** — Predictive Parity | Diff. de précision entre groupes | ≤ 0.05 |

---

## 🚀 Lancement Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Déploiement Streamlit Cloud

```bash
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

Puis sur [share.streamlit.io](https://share.streamlit.io) :
- **New app** → sélectionner le repo
- Main file path : `app.py`
- **Deploy**

---

*Projet académique — Orange · Parcours A · Détection de Biais Algorithmique*
