from __future__ import annotations

import os
import math
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Persistencia de grupos
# =========================
GROUPS_CSV = "grupos_guardados.csv"

def load_groups() -> Dict[str, List[str]]:
    if not os.path.exists(GROUPS_CSV):
        return {}
    try:
        df = pd.read_csv(GROUPS_CSV)
        if "Grupo" in df.columns and "Alojamiento" in df.columns:
            out: Dict[str, List[str]] = {}
            for g, sub in df.groupby("Grupo"):
                out[str(g)] = sorted(list(sub["Alojamiento"].dropna().astype(str).unique()))
            return out
    except Exception:
        pass
    return {}

def save_group_csv(name: str, props: List[str]) -> None:
    name = str(name).strip()
    if not name or not props:
        return
    cur = pd.DataFrame({"Grupo": [name] * len(props), "Alojamiento": list(map(str, props))})
    if os.path.exists(GROUPS_CSV):
        try:
            prev = pd.read_csv(GROUPS_CSV)
            prev = prev[prev["Grupo"] != name]
            cur = pd.concat([prev, cur], ignore_index=True)
        except Exception:
            pass
    cur.to_csv(GROUPS_CSV, index=False, encoding="utf-8-sig")

def delete_group_csv(name: str) -> None:
    if not os.path.exists(GROUPS_CSV):
        return
    try:
        df = pd.read_csv(GROUPS_CSV)
        df = df[df["Grupo"] != name]
        df.to_csv(GROUPS_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def group_selector(label: str, all_props: List[str], key_prefix: str, default: Optional[List[str]] = None) -> List[str]:
    return st.multiselect(label, options=sorted(all_props), default=default or [], key=f"{key_prefix}_selector")

# =========================
# Helpers de UI
# =========================
def period_inputs(label_start: str, label_end: str, default_start: pd.Timestamp, default_end: pd.Timestamp, key_prefix: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    c1, c2 = st.columns(2)
    d1 = c1.date_input(label_start, value=pd.to_datetime(default_start).date(), key=f"{key_prefix}_start")
    d2 = c2.date_input(label_end, value=pd.to_datetime(default_end).date(), key=f"{key_prefix}_end")
    return pd.to_datetime(d1), pd.to_datetime(d2)

def help_block(txt: str):
    st.info(txt)

# =========================
# Parsing de fechas
# =========================
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# =========================
# Núcleo KPIs
# =========================
def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _prep_df(
    df_all: pd.DataFrame,
    filter_props: Optional[List[str]] = None,
) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"])
    df = df_all.copy()
    # Columnas mínimas
    req = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan
    df = parse_dates(df)
    # Filtrado por propiedades
    if filter_props:
        df = df[df["Alojamiento"].astype(str).isin([str(p) for p in filter_props])]
    # Limpieza básica
    df = df.dropna(subset=["Fecha entrada", "Fecha salida"])
    df = df[df["Fecha salida"] > df["Fecha entrada"]]
    # Noche media (ADR por reserva)
    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["rent"] = pd.to_numeric(df["Alquiler con IVA (€)"], errors="coerce").fillna(0.0).astype(float)
    df["adr_reserva"] = df["rent"] / df["los"]
    # Normalizamos fechas a medianoche
    for c in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    return df

def _overlap_nights(entry: pd.Timestamp, exit_: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> int:
    # Noches superpuestas entre [entry, exit_) y [start, end] (end inclusive como noche de salida end+1)
    start_n = pd.to_datetime(start).normalize()
    end_n = pd.to_datetime(end).normalize() + pd.Timedelta(days=1)  # fin inclusivo
    a = max(pd.to_datetime(entry).normalize(), start_n)
    b = min(pd.to_datetime(exit_).normalize(), end_n)
    return max((b - a).days, 0)

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    KPIs del periodo con reservas cuya Fecha alta <= cutoff.
    - Ocupación = noches ocupadas / (inventario * noches_periodo) * 100
    - ADR = ingresos / noches ocupadas
    - Ingresos = suma prorrateada por noches en el periodo
    """
    df = _prep_df(df_all, filter_props)
    if df.empty:
        return pd.DataFrame(columns=["Alojamiento","noches","ingresos","adr","ocupacion_pct"]), {
            "noches_ocupadas": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "ocupacion_pct": 0.0,
        }

    # Filtra por cutoff (OTB a fecha de corte)
    df = df[(df["Fecha alta"].notna()) & (df["Fecha alta"] <= pd.to_datetime(cutoff).normalize())]

    if df.empty:
        nights_period = max((pd.to_datetime(period_end).normalize() - pd.to_datetime(period_start).normalize()).days + 1, 1)
        inv = int(inventory_override) if inventory_override and int(inventory_override) > 0 else df_all["Alojamiento"].astype(str).nunique()
        denom = max(inv * nights_period, 1)
        return pd.DataFrame(columns=["Alojamiento","noches","ingresos","adr","ocupacion_pct"]), {
            "noches_ocupadas": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "ocupacion_pct": 0.0,
        }

    # Noches solapadas con el periodo y prorrateo de ingresos
    ps, pe = pd.to_datetime(period_start), pd.to_datetime(period_end)
    df["nights_in_range"] = df.apply(lambda r: _overlap_nights(r["Fecha entrada"], r["Fecha salida"], ps, pe), axis=1)
    df = df[df["nights_in_range"] > 0].copy()
    if df.empty:
        nights_period = max((pe.normalize() - ps.normalize()).days + 1, 1)
        inv = int(inventory_override) if inventory_override and int(inventory_override) > 0 else df_all["Alojamiento"].astype(str).nunique()
        denom = max(inv * nights_period, 1)
        return pd.DataFrame(columns=["Alojamiento","noches","ingresos","adr","ocupacion_pct"]), {
            "noches_ocupadas": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "ocupacion_pct": 0.0,
        }

    df["ingresos_prorr"] = df["adr_reserva"] * df["nights_in_range"]

    by_prop = df.groupby("Alojamiento", as_index=False).agg(
        noches=("nights_in_range", "sum"),
        ingresos=("ingresos_prorr", "sum"),
    )
    by_prop["adr"] = by_prop.apply(lambda r: r["ingresos"] / r["noches"] if r["noches"] > 0 else 0.0, axis=1)

    # Ocupación
    nights_period = max((pe.normalize() - ps.normalize()).days + 1, 1)
    inv = int(inventory_override) if inventory_override and int(inventory_override) > 0 else df_all["Alojamiento"].astype(str).nunique()
    denom = max(inv * nights_period, 1)
    noches_total = float(by_prop["noches"].sum())
    ingresos_total = float(by_prop["ingresos"].sum())
    adr_total = ingresos_total / noches_total if noches_total > 0 else 0.0
    occ_pct = noches_total / denom * 100.0 if denom > 0 else 0.0

    by_prop["ocupacion_pct"] = by_prop["noches"] / nights_period * 100.0 if inv <= 1 else by_prop["noches"] / (nights_period) * (100.0 / inv)

    totals = {
        "noches_ocupadas": noches_total,
        "ingresos": ingresos_total,
        "adr": adr_total,
        "ocupacion_pct": occ_pct,
    }
    return by_prop.sort_values("ingresos", ascending=False), totals

# =========================
# Pace y Forecast P50
# =========================
def pace_series(
    df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    d_max: int = 180,
    props: Optional[List[str]] = None,
    inv_override: Optional[int] = None,  # no se usa, para compatibilidad
) -> pd.DataFrame:
    """
    Serie Pace: noches confirmadas por D (días entre alta y entrada) dentro del periodo.
    """
    dfx = _prep_df(df, props)
    if dfx.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx = dfx[(dfx["Fecha entrada"] >= pd.to_datetime(period_start)) & (dfx["Fecha entrada"] <= pd.to_datetime(period_end))]
    if dfx.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx["D"] = (dfx["Fecha entrada"] - dfx["Fecha alta"]).dt.days.clip(lower=0, upper=int(d_max))
    dfx["los"] = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    out = dfx.groupby("D", as_index=False)["los"].sum().rename(columns={"los": "noches"}).sort_values("D")
    return out

def pace_forecast_month(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    ref_years: int = 2,
    dmax: int = 180,
    props: Optional[List[str]] = None,
    inv_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Forecast P50 sencillo usando pickup típico de los últimos ref_years.
    Devuelve claves estándar: nights_otb, nights_p50, pickup_typ_p50, pickup_needed_p50, adr_tail_p50, revenue_final_p50
    """
    if df is None or df.empty:
        return {}

    # Actual OTB
    _, tot_now = compute_kpis(
        df_all=df,
        cutoff=pd.to_datetime(cutoff),
        period_start=pd.to_datetime(period_start),
        period_end=pd.to_datetime(period_end),
        inventory_override=int(inv_override) if (inv_override is not None and int(inv_override) > 0) else None,
        filter_props=props if props else None,
    )
    n_otb = float(tot_now.get("noches_ocupadas", 0.0))
    ingresos_otb = float(tot_now.get("ingresos", 0.0))
    adr_now = float(tot_now.get("adr", 0.0))

    pickups = []
    adr_tails = []
    for y in range(1, int(ref_years) + 1):
        p_start_y = pd.to_datetime(period_start) - pd.DateOffset(years=y)
        p_end_y   = pd.to_datetime(period_end) - pd.DateOffset(years=y)
        cut_y     = pd.to_datetime(cutoff) - pd.DateOffset(years=y)

        # OTB a ese corte LY-y
        _, tot_cut_y = compute_kpis(
            df_all=df, cutoff=cut_y, period_start=p_start_y, period_end=p_end_y,
            inventory_override=None, filter_props=props if props else None,
        )
        # Final LY-y (corte = fin del periodo)
        _, tot_final_y = compute_kpis(
            df_all=df, cutoff=p_end_y, period_start=p_start_y, period_end=p_end_y,
            inventory_override=None, filter_props=props if props else None,
        )
        pick_y = float(tot_final_y.get("noches_ocupadas", 0.0) - tot_cut_y.get("noches_ocupadas", 0.0))
        pickups.append(max(pick_y, 0.0))
        adr_tails.append(float(tot_final_y.get("adr", 0.0)))

    def p50(arr: List[float]) -> float:
        arr = [float(x) for x in arr if np.isfinite(x)]
        if not arr:
            return 0.0
        return float(np.percentile(arr, 50))

    pickup_typ_p50 = p50(pickups)
    adr_tail_p50 = p50(adr_tails) if p50(adr_tails) > 0 else (adr_now if adr_now > 0 else 0.0)

    nights_p50 = n_otb + pickup_typ_p50
    pickup_needed_p50 = max(nights_p50 - n_otb, 0.0)
    revenue_final_p50 = ingresos_otb + pickup_typ_p50 * adr_tail_p50

    return {
        "nights_otb": n_otb,
        "nights_p50": nights_p50,
        "pickup_typ_p50": pickup_typ_p50,
        "pickup_needed_p50": pickup_needed_p50,
        "adr_tail_p50": adr_tail_p50,
        "revenue_final_p50": revenue_final_p50,
    }

# =========================
# Análisis PRO (semáforo)
# =========================
def _pace_get(d: dict, keys: List[str], default=0.0):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _kai_cdm_pro_analysis(
    tot_now: dict,
    tot_ly_cut: dict,
    tot_ly_final: dict,
    pace: dict,
    price_ref_p50: float = None
) -> str:
    """
    Semáforo y análisis robusto (acepta alias de claves en pace).
    """
    n_otb = float(_pace_get(pace, ["nights_otb", "otb_nights", "otb", "noches_otb"], 0.0))
    n_p50 = float(_pace_get(pace, ["nights_p50", "forecast_nights_p50", "p50_nights"], 0.0))
    pick_typ50 = float(_pace_get(pace, ["pickup_typ_p50", "p50_pickup_typ", "pickup_typical_p50"], 0.0))
    pick_need = float(_pace_get(pace, ["pickup_needed_p50", "p50_pickup_needed", "pickup_need_p50"], 0.0))
    adr_tail_p50 = float(_pace_get(pace, ["adr_tail_p50", "p50_adr_tail", "adr_typ_tail_p50"], 0.0))
    rev_final_p50 = float(_pace_get(pace, ["revenue_final_p50", "rev_final_p50", "p50_revenue_final"], 0.0))

    # Estado pace
    pace_state = "—"
    expected_otb_typ = max(n_p50 - pick_typ50, 0.0)
    if expected_otb_typ > 0 and n_otb > 0:
        ratio = n_otb / expected_otb_typ
        if ratio >= 1.10:
            pace_state = "🟢 Adelantado"
        elif ratio <= 0.90:
            pace_state = "🔴 Retrasado"
        else:
            pace_state = "🟠 En línea"

    msg = ""
    if pace_state == "🟢 Adelantado":
        msg += "### 🟢 Adelantado\n"
        msg += "Buen ritmo de reservas respecto a años anteriores. Mantén estrategia y monitoriza pickup restante.\n"
        if pick_need > pick_typ50 * 1.2:
            msg += "- Aún queda pickup elevado. Refuerza ventas para asegurar el cierre.\n"
    elif pace_state == "🟠 En línea":
        msg += "### 🟠 En línea\n"
        msg += "Ritmo en línea con años anteriores. Revisa pickup pendiente y ADR para microajustes.\n"
        if adr_tail_p50 < float(tot_ly_cut.get("adr", 0.0)) * 0.95:
            msg += "- ADR previsto por debajo de LY. Revisa precios.\n"
    elif pace_state == "🔴 Retrasado":
        msg += "### 🔴 Retrasado\n"
        msg += "Ritmo retrasado. Considera promos, campañas y ajustes de precios.\n"
        if pick_need > pick_typ50:
            msg += "- Pickup pendiente elevado. Refuerza captación y canales.\n"
        if adr_tail_p50 < float(tot_ly_cut.get("adr", 0.0)) * 0.95:
            msg += "- ADR previsto por debajo de LY. Ajusta precio/ofertas.\n"
    else:
        msg += "No hay suficiente información para evaluar el ritmo de reservas.\n"

    # Resumen
    msg += f"\n**Estado actual:** {pace_state}\n"
    msg += f"- Pickup pendiente objetivo: **{pick_need:,.0f} noches**\n"
    msg += f"- ADR previsto (P50): **{adr_tail_p50:.2f} €**\n"
    msg += f"- Forecast ingresos (P50): **{rev_final_p50:.2f} €**\n"
    msg += "\n**KPIs actuales:**\n"
    msg += f"- Ocupación actual: **{float(tot_now.get('ocupacion_pct', 0.0)):.2f}%**\n"
    msg += f"- ADR actual: **{float(tot_now.get('adr', 0.0)):.2f} €**\n"
    msg += f"- Ingresos actuales: **{float(tot_now.get('ingresos', 0.0)):.2f} €**\n"
    return msg

# =========================
# Explicación ejecutiva
# =========================
def _pct_delta(cur: float, ref: float) -> float:
    cur, ref = _safe_float(cur), _safe_float(ref)
    if ref == 0:
        return 0.0
    return (cur - ref) / ref * 100.0

def _pp_delta(cur_pct: float, ref_pct: float) -> float:
    return _safe_float(cur_pct) - _safe_float(ref_pct)

def pro_exec_summary(
    tot_now: Dict[str, float],
    tot_ly_cut: Dict[str, float],
    tot_ly_final: Dict[str, float],
    pace: Dict[str, Any],
) -> Dict[str, str]:
    occ_now = _safe_float(tot_now.get("ocupacion_pct", 0))
    adr_now = _safe_float(tot_now.get("adr", 0))
    rev_now = _safe_float(tot_now.get("ingresos", 0))
    occ_ly = _safe_float(tot_ly_cut.get("ocupacion_pct", 0))
    adr_ly = _safe_float(tot_ly_cut.get("adr", 0))
    rev_ly = _safe_float(tot_ly_cut.get("ingresos", 0))
    rev_ly_final = _safe_float(tot_ly_final.get("ingresos", 0))

    revpar_now = adr_now * occ_now / 100.0
    revpar_ly = adr_ly * occ_ly / 100.0

    d_occ_pp = _pp_delta(occ_now, occ_ly)
    d_adr_pct = _pct_delta(adr_now, adr_ly)
    d_revpar_pct = _pct_delta(revpar_now, revpar_ly)
    d_revenue_pct = _pct_delta(rev_now, rev_ly)

    def g(d, keys, default=0.0):
        if not isinstance(d, dict): return default
        for k in keys:
            if k in d and d[k] is not None: return _safe_float(d[k], default)
        return default

    rev_final_p50 = g(pace, ["revenue_final_p50", "rev_final_p50", "p50_revenue_final"], rev_now)
    pick_typ50 = g(pace, ["pickup_typ_p50", "p50_pickup_typ", "pickup_typical_p50"], 0.0)
    adr_tail_p50 = g(pace, ["adr_tail_p50", "p50_adr_tail", "adr_typ_tail_p50"], adr_now)

    gap_rev = rev_ly_final - rev_final_p50
    cobertura_pct = _pct_delta(rev_final_p50, rev_ly_final) + 100 if rev_ly_final > 0 else 0.0
    gap_txt = f"Faltan {gap_rev:,.0f} €" if gap_rev > 0 else f"Superas LY final en {abs(gap_rev):,.0f} €"
    gap_txt = gap_txt.replace(",", ".")

    if d_adr_pct < -3 and d_occ_pp > 2:
        verdict = "Estamos comprando volumen barato"
    elif d_adr_pct > 3 and d_occ_pp < -2:
        verdict = "Estamos vendiendo caro, falta demanda"
    elif d_adr_pct > 0 and d_occ_pp > 0:
        verdict = "Ejecución sólida: sube precio y volumen"
    elif d_adr_pct < 0 and d_occ_pp < 0:
        verdict = "Alerta: caen precio y ocupación"
    else:
        verdict = "Rendimiento mixto"

    if gap_rev <= 0:
        viab = "Gap cubierto con el forecast P50."
    else:
        est_cob = pick_typ50 * adr_tail_p50
        ratio = est_cob / gap_rev if gap_rev > 0 else 1.0
        if ratio >= 1.0:
            viab = "Con P50 (pickup × ADR tail) se cubriría el gap."
        elif ratio >= 0.7:
            viab = "Cobertura estimada ≈ alta (≥70%). Requiere ejecutar bien el pickup."
        else:
            viab = "Cobertura estimada insuficiente. Hay que activar demanda y/o ajustar precios."

    acciones = []
    if d_adr_pct < -3:
        acciones.append("Revisar y retirar descuentos de baja conversión.")
        acciones.append("Micro-rebajas quirúrgicas en días valle (LT corto).")
    if d_occ_pp < 0:
        acciones.append("Boost de demanda: visibilidad OTAs, campañas directas, partners.")
    if d_adr_pct > 3 and d_occ_pp < 0:
        acciones.append("Mantener precios en picos, test A/B en días flojos.")
    if not acciones:
        acciones = [
            "Monitorizar pickup semanal y mantener pricing en fines/ eventos.",
            "Reasignar presupuesto a canales con mejor conversión.",
        ]

    headline = (
        "🌸 Explicación ejecutiva (narrada)\n\n"
        f"• Veredicto general: {verdict}\n\n"
        f"• Evolución vs LY (a este corte) → Ocupación {d_occ_pp:+.1f} p.p., ADR {d_adr_pct:+.1f}%, "
        f"RevPAR {d_revpar_pct:+.1f}%, Ingresos {d_revenue_pct:+.1f}%.\n"
        f"• Viabilidad de cierre del gap → {gap_txt} · Cobertura estimada P50 ≈ {cobertura_pct:.0f}%."
    )
    detail = (
        "### 👉 Ver análisis detallado\n"
        f"- Ocupación: {'🟢' if d_occ_pp>=0 else '🔴'} {d_occ_pp:+.1f} p.p.\n"
        f"- ADR: {'🟢' if d_adr_pct>=0 else '🔴'} {d_adr_pct:+.1f}%\n"
        f"- RevPAR: {'🟢' if d_revpar_pct>=0 else '🔴'} {d_revpar_pct:+.1f}%\n"
        f"- Ingresos: {'🟢' if d_revenue_pct>=0 else '🔴'} {d_revenue_pct:+.1f}%\n\n"
        "#### Qué explica el resultado (atribución RevPAR)\n"
        f"- Ocupación: {d_occ_pp:+.1f} p.p.\n"
        f"- ADR: {d_adr_pct:+.1f}% (precio medio)\n\n"
        "#### Viabilidad de cierre del gap\n"
        f"- " + viab + "\n"
        f"- " + gap_txt + f" · Cobertura estimada ≈ {cobertura_pct:.0f}%.\n\n"
        "#### Plan de acción (siguiente quincena)\n"
        + "".join([f"- {a}\n" for a in acciones])
    )
    return {"headline": headline, "detail": detail}
