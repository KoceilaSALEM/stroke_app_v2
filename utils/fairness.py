"""
utils/fairness.py
─────────────────
Métriques de fairness algorithmique pour l'analyse de biais.

Métriques implémentées :
  - Demographic Parity Difference (DPD)
  - Disparate Impact Ratio (DI)
  - Equal Opportunity Difference (EOD)
  - Predictive Parity (PP)
"""

import numpy as np
from typing import Union


ArrayLike = Union[np.ndarray, list]


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


# ── Métrique 1 : Parité Démographique ────────────────────────────────────────
def demographic_parity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attribute: ArrayLike,
) -> dict:
    """
    Différence de parité démographique.
    DPD = |P(Ŷ=1 | S=a) − P(Ŷ=1 | S=b)|
    Idéal : DPD = 0
    """
    y_pred = _to_array(y_pred)
    s      = _to_array(sensitive_attribute)
    groups = np.unique(s)

    rates = {g: float(y_pred[s == g].mean()) for g in groups}

    vals = list(rates.values())
    max_diff = max(vals) - min(vals)

    return {
        "rates":      rates,
        "difference": float(max_diff),
        "groups":     list(groups),
        "favored":    max(rates, key=rates.get),
        "unfavored":  min(rates, key=rates.get),
    }


# ── Métrique 2 : Impact Disparate ────────────────────────────────────────────
def disparate_impact_ratio(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attribute: ArrayLike,
    unprivileged_value: str,
    privileged_value: str,
) -> dict:
    """
    Ratio d'impact disparate.
    DI = P(Ŷ=1 | S=unprivileged) / P(Ŷ=1 | S=privileged)
    Règle des 4/5 : DI ≥ 0.80 pour absence de discrimination.
    """
    y_pred = _to_array(y_pred)
    s      = _to_array(sensitive_attribute)

    mask_u = s == unprivileged_value
    mask_p = s == privileged_value

    rate_u = float(y_pred[mask_u].mean()) if mask_u.sum() > 0 else 0.0
    rate_p = float(y_pred[mask_p].mean()) if mask_p.sum() > 0 else 1.0

    ratio  = rate_u / rate_p if rate_p > 0 else float("inf")

    return {
        "ratio":              float(ratio),
        "rate_unprivileged":  rate_u,
        "rate_privileged":    rate_p,
        "unprivileged_value": unprivileged_value,
        "privileged_value":   privileged_value,
        "rule_4_5_ok":        ratio >= 0.8,
    }


# ── Métrique 3 : Égalité des Chances (Equal Opportunity) ─────────────────────
def equal_opportunity_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attribute: ArrayLike,
) -> dict:
    """
    Différence de taux de vrais positifs (TPR) entre groupes.
    EOD = |TPR_a − TPR_b|
    Idéal : EOD = 0
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    s      = _to_array(sensitive_attribute)
    groups = np.unique(s)

    tpr = {}
    for g in groups:
        mask = (s == g) & (y_true == 1)
        tpr[g] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0

    vals = list(tpr.values())
    diff = max(vals) - min(vals) if len(vals) >= 2 else 0.0

    return {
        "tpr":        tpr,
        "difference": float(diff),
        "groups":     list(groups),
    }


# ── Métrique 4 : Parité Prédictive ───────────────────────────────────────────
def predictive_parity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attribute: ArrayLike,
) -> dict:
    """
    Différence de précision (PPV) entre groupes.
    PP = |P(Y=1 | Ŷ=1, S=a) − P(Y=1 | Ŷ=1, S=b)|
    Idéal : PP = 0
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    s      = _to_array(sensitive_attribute)
    groups = np.unique(s)

    ppv = {}
    for g in groups:
        mask = (s == g) & (y_pred == 1)
        ppv[g] = float(y_true[mask].mean()) if mask.sum() > 0 else 0.0

    vals = list(ppv.values())
    diff = max(vals) - min(vals) if len(vals) >= 2 else 0.0

    return {
        "ppv":        ppv,
        "difference": float(diff),
        "groups":     list(groups),
    }


# ── Helpers d'interprétation ─────────────────────────────────────────────────
def interpret_dpd(diff: float) -> tuple[str, str]:
    """Retourne (niveau, couleur_hex) selon la valeur de DPD."""
    if diff <= 0.03:
        return "Excellent", "#10b981"
    elif diff <= 0.07:
        return "Acceptable", "#f59e0b"
    elif diff <= 0.10:
        return "Préoccupant", "#f97316"
    else:
        return "Critique", "#ef4444"


def interpret_di(ratio: float) -> tuple[str, str]:
    """Retourne (niveau, couleur_hex) selon la valeur de DI."""
    if ratio >= 0.90:
        return "Excellent", "#10b981"
    elif ratio >= 0.80:
        return "Acceptable (règle 4/5 OK)", "#f59e0b"
    elif ratio >= 0.60:
        return "Préoccupant", "#f97316"
    else:
        return "Critique — Discrimination potentielle", "#ef4444"
