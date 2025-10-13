from __future__ import annotations

import os
import math
from typing import Optional, List, Dict, Any, Tuple
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# ===== Grupos de alojamientos (CSV reutilizable) =====
GROUPS_CSV = "grupos_guardados.csv"  # en la raíz del repo

def save_group_csv(group_name: str, props_list: List[str]) -> None:
    if os.path.exists(GROUPS_CSV):
        df = pd.read_csv(GROUPS_CSV)
    else:
        df = pd.DataFrame(columns=["Grupo", "Alojamiento"])
    df = df[df["Grupo"] != group_name]
    new_rows = [{"Grupo": group_name, "Alojamiento": str(prop)} for prop in props_list]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(GROUPS_CSV, index=False, encoding="utf-8-sig")

def load_groups() -> Dict[str, List[str]]:
    if not os.path.exists(GROUPS_CSV):
        return {}
    df = pd.read_csv(GROUPS_CSV)
    groups: Dict[str, List[str]] = {}
    for group_name in df["Grupo"].astype(str).unique():
        props = df[df["Grupo"].astype(str) == str(group_name)]["Alojamiento"].astype(str).tolist()
        groups[str(group_name)] = props
    return groups

def group_selector(label: str, all_props: List[str], key_prefix: str, default: Optional[List[str]] = None) -> List[str]:
    return st.multiselect(label, options=sorted(all_props), default=default or [], key=f"{key_prefix}_selector")

# ===== Parsing y normalización =====
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parsea fechas (incluye seriales Excel) y normaliza Alojamiento y Alquiler con IVA (€)."""
    if df is None or df.empty:
        return df
    # Alojamiento a str
    if "Alojamiento" in df.columns:
        df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    # Fechas
    for col in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        if col in df.columns:
            s = df[col]
            try:
                if pd.api.types.is_numeric_dtype(s):
                    df[col] = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
                else:
                    df[col] = pd.to_datetime(s, errors="coerce")
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    # Precio
    if "Alquiler con IVA (€)" in df.columns:
        df["Alquiler con IVA (€)"] = pd.to_numeric(df["Alquiler con IVA (€)"], errors="coerce").fillna(0.0)
    else:
        df["Alquiler con IVA (€)"] = 0.0
    return df

@st.cache_data(show_spinner=False)
def get_inventory(df: pd.DataFrame, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return int(override)
    if df is None or df.empty or "Alojamiento" not in df.columns:
        return 0
    return int(df["Alojamiento"].astype(str).nunique())

def help_block(kind: str):
    if kind == "Consulta normal":
        st.info("Consulta normal: KPIs totales y por alojamiento para el periodo y corte seleccionados.")

def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> Tuple[date, date]:
    col1, col2 = st.columns(2)
    start = col1.date_input(label_start, value=default_start, key=f"{key_prefix}_start")
    end = col2.date_input(label_end, value=default_end, key=f"{key_prefix}_end")
    return start, end

# ===== Núcleo KPIs =====
def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _prep_df(df_all: pd.DataFrame, filter_props: Optional[List[str]] = None) -> pd.DataFrame:
    """Normaliza, filtra por props, limpia fechas invertidas y calcula LOS."""
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["Alojamiento","Fecha alta","Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    df = parse_dates(df_all.copy())
    if filter_props:
        props_set = set(map(str, filter_props))
        df = df[df["Alojamiento"].astype(str).isin(props_set)]
    df = df.dropna(subset=["Fecha entrada", "Fecha salida"])
    df = df[df["Fecha salida"] > df["Fecha entrada"]].copy()
    df["los"] = (df["Fecha salida"] - df["Fecha entrada"]).dt.days.clip(lower=1)
    return df

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    KPIs con prorrateo por noches dentro del periodo y OTB a 'cutoff'.
    """
    df = _prep_df(df_all, filter_props)
    if df.empty:
        empty = pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"])
        return empty, {"noches_ocupadas": 0, "noches_disponibles": 0, "ocupacion_pct": 0.0, "ingresos": 0.0, "adr": 0.0, "revpar": 0.0}

    cutoff = pd.to_datetime(cutoff).normalize()
    ps = pd.to_datetime(period_start).normalize()
    pe = pd.to_datetime(period_end).normalize()

    # OTB a corte
    df = df[df["Fecha alta"].notna() & (df["Fecha alta"] <= cutoff)]
    if df.empty:
        days = max((pe - ps).days + 1, 0)
        inv_eff = len(set(map(str, filter_props))) if filter_props else df_all["Alojamiento"].astype(str).nunique()
        noches_disponibles = int(inv_eff) * days if days > 0 else 0
        empty = pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"])
        return empty, {"noches_ocupadas": 0, "noches_disponibles": noches_disponibles, "ocupacion_pct": 0.0, "ingresos": 0.0, "adr": 0.0, "revpar": 0.0}

    # Overlap de noches con el periodo (end inclusivo)
    one_day = np.timedelta64(1, "D")
    start_ns = np.datetime64(ps)
    end_excl_ns = np.datetime64(pe + pd.Timedelta(days=1))
    arr_e = df["Fecha entrada"].values.astype("datetime64[ns]")
    arr_s = df["Fecha salida"].values.astype("datetime64[ns]")

    total_nights = ((arr_s - arr_e) / one_day).astype("int64")
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype("int64")
    ov_days = np.clip(ov_days, 0, None)

    # Ingresos prorrateados
    price = df["Alquiler con IVA (€)"].astype("float64").values
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Ingresos", ascending=False)

    noches_ocupadas = float(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = ingresos / noches_ocupadas if noches_ocupadas > 0 else 0.0

    days = max((pe - ps).days + 1, 0)
    inv_detected = len(set(map(str, filter_props))) if filter_props else df_all["Alojamiento"].astype(str).nunique()
    inv_eff = int(inventory_override) if (inventory_override is not None and int(inventory_override) > 0) else int(inv_detected)
    noches_disponibles = inv_eff * days if days > 0 else 0

    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100.0) if noches_disponibles > 0 else 0.0
    revpar = (ingresos / noches_disponibles) if noches_disponibles > 0 else 0.0

    # Asegura columnas
    for col in ["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]:
        if col not in by_prop.columns:
            by_prop[col] = 0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }
    return by_prop, tot

# ===== Mix por portal (simple) =====
def compute_portal_share(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    filter_props: Optional[List[str]] = None,
    portal_col: str = "Agente/Intermediario",
) -> Optional[pd.DataFrame]:
    if portal_col not in df_all.columns:
        return None
    df = _prep_df(df_all, filter_props)
    if df.empty:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"])
    df = df[(df["Fecha alta"] <= pd.to_datetime(cutoff)) &
            (df["Fecha entrada"] <= pd.to_datetime(period_end)) &
            (df["Fecha salida"] >= pd.to_datetime(period_start))]
    if df.empty:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"])
    portal_counts = df[portal_col].astype(str).value_counts().reset_index()
    portal_counts.columns = ["Portal", "Reservas"]
    portal_counts["% Reservas"] = portal_counts["Reservas"] / max(portal_counts["Reservas"].sum(), 1) * 100.0
    return portal_counts

# ===== Pace y Forecast =====
def _pace_get(d: dict, keys: list, default=0.0):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def pace_series(
    df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    d_max: int = 180,
    props: Optional[List[str]] = None,
    inv_override: Optional[int] = None,
) -> pd.DataFrame:
    """Serie Pace: noches por D (días entre alta y entrada) dentro del periodo."""
    dfx = _prep_df(df, props)
    if dfx.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx = dfx[(dfx["Fecha entrada"] >= pd.to_datetime(period_start)) & (dfx["Fecha entrada"] <= pd.to_datetime(period_end))]
    if dfx.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx["D"] = (dfx["Fecha entrada"] - dfx["Fecha alta"]).dt.days.clip(lower=0, upper=int(d_max))
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
    """Forecast P50: pickup típico de últimos 'ref_years' y ADR tail aproximado."""
    if df is None or df.empty:
        return {}

    # OTB actual
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

    pickups, adr_tails = [], []
    for y in range(1, int(ref_years) + 1):
        p_start_y = pd.to_datetime(period_start) - pd.DateOffset(years=y)
        p_end_y   = pd.to_datetime(period_end) - pd.DateOffset(years=y)
        cut_y     = pd.to_datetime(cutoff) - pd.DateOffset(years=y)

        _, tot_cut_y = compute_kpis(df_all=df, cutoff=cut_y, period_start=p_start_y, period_end=p_end_y, inventory_override=None, filter_props=props if props else None)
        _, tot_final_y = compute_kpis(df_all=df, cutoff=p_end_y, period_start=p_start_y, period_end=p_end_y, inventory_override=None, filter_props=props if props else None)

        pick_y = float(tot_final_y.get("noches_ocupadas", 0.0) - tot_cut_y.get("noches_ocupadas", 0.0))
        pickups.append(max(pick_y, 0.0))
        adr_tails.append(float(tot_final_y.get("adr", 0.0)))

    def p50(arr):
        arr = [x for x in arr if np.isfinite(x)]
        if not arr:
            return 0.0
        return float(np.percentile(arr, 50))

    pickup_typ_p50 = p50(pickups)
    adr_tail_p50 = p50(adr_tails) if np.isfinite(p50(adr_tails)) and p50(adr_tails) > 0 else (adr_now if adr_now > 0 else 0.0)

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

# ===== Análisis PRO y explicación ejecutiva =====
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

    actions = []
    if d_adr_pct < -3:
        actions.append("Revisar y retirar descuentos de baja conversión.")
        actions.append("Micro-rebajas quirúrgicas en días valle (LT corto).")
    if d_occ_pp < 0:
        actions.append("Boost de demanda: visibilidad OTAs, campañas directas, partners.")
    if d_adr_pct > 3 and d_occ_pp < 0:
        actions.append("Mantener precios en picos, test A/B en días flojos.")
    if not actions:
        actions = [
            "Monitorizar pickup semanal y mantener pricing en fines/eventos.",
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
        f"- " + ("Gap cubierto con el forecast P50." if gap_rev <= 0 else "Se requiere activar demanda y/o ajustar precios.") + "\n"
        f"- " + gap_txt + f" · Cobertura estimada ≈ {cobertura_pct:.0f}%.\n\n"
        "#### Plan de acción (siguiente quincena)\n"
        + "".join([f"- {a}\n" for a in actions])
    )
    return {"headline": headline, "detail": detail}
