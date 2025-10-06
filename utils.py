import pandas as pd
import streamlit as st
from datetime import date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import math
import numpy as np
from pathlib import Path
import os

# ===== Grupos de alojamientos (CSV reutilizable) =====

GROUPS_CSV = "grupos_guardados.csv"  # Se guarda en la raíz del repositorio

def save_group_csv(group_name, props_list):
    if os.path.exists(GROUPS_CSV):
        df = pd.read_csv(GROUPS_CSV)
    else:
        df = pd.DataFrame(columns=["Grupo", "Alojamiento"])
    # Elimina grupo si ya existe (evita duplicados)
    df = df[df["Grupo"] != group_name]
    new_rows = [{"Grupo": group_name, "Alojamiento": prop} for prop in props_list]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(GROUPS_CSV, index=False)

def load_groups():
    if not os.path.exists(GROUPS_CSV):
        return {}
    df = pd.read_csv(GROUPS_CSV)
    groups = {}
    for group_name in df["Grupo"].unique():
        props = df[df["Grupo"] == group_name]["Alojamiento"].tolist()
        groups[group_name] = props
    return groups

def group_selector(label: str, all_props: list[str], key_prefix: str, default: Optional[list[str]] = None) -> list[str]:
    selected = st.multiselect(label, options=all_props, default=default or [], key=f"{key_prefix}_selector")
    return selected

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def get_inventory(df: pd.DataFrame, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return int(override)
    return df["Alojamiento"].nunique()

def help_block(kind: str):
    if kind == "Consulta normal":
        st.info("Consulta normal: muestra KPIs totales y por alojamiento para el periodo y corte seleccionados.")
    # Puedes añadir más tipos si quieres

def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> Tuple[date, date]:
    col1, col2 = st.columns(2)
    start = col1.date_input(label_start, value=default_start, key=f"{key_prefix}_start")
    end = col2.date_input(label_end, value=default_end, key=f"{key_prefix}_end")
    return start, end

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    inv_detected = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    inv_eff = int(inventory_override) if (inventory_override is not None and int(inventory_override) > 0) else int(inv_detected)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv_eff * days if days > 0 else 0

    if df_cut.empty:
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": noches_disponibles,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    arr_e = df_cut["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df_cut["Fecha salida"].values.astype('datetime64[ns]')

    total_nights = ((arr_s - arr_e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    ov_days = np.clip(ov_days, 0, None)

    price = df_cut["Alquiler con IVA (€)"].values.astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df_cut["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }
    # Asegura que las columnas existen en el DataFrame de salida
    for col in ["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]:
        if col not in by_prop.columns:
            by_prop[col] = 0

    return by_prop, tot

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
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()
    df_cut = df_cut[
        (df_cut["Fecha entrada"] <= period_end) & (df_cut["Fecha salida"] >= period_start)
    ]
    if df_cut.empty:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"])
    portal_counts = df_cut[portal_col].value_counts().reset_index()
    portal_counts.columns = ["Portal", "Reservas"]
    portal_counts["% Reservas"] = portal_counts["Reservas"] / portal_counts["Reservas"].sum() * 100

    return portal_counts

def _pace_get(d: dict, keys: list, default=0.0):
    """Devuelve d[k] usando alias posibles."""
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
    """
    Serie base de Pace: noches confirmadas por D (días entre alta y entrada).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx = df.copy()
    dfx = dfx.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"])
    if props:
        dfx = dfx[dfx["Alojamiento"].isin(props)]
    # Filtrar reservas que impactan en el periodo (por fecha de entrada)
    dfx = dfx[(dfx["Fecha entrada"] >= pd.to_datetime(period_start)) & (dfx["Fecha entrada"] <= pd.to_datetime(period_end))]
    if dfx.empty:
        return pd.DataFrame(columns=["D", "noches"])
    dfx["D"] = (dfx["Fecha entrada"].dt.normalize() - dfx["Fecha alta"].dt.normalize()).dt.days
    dfx["D"] = dfx["D"].clip(lower=0, upper=int(d_max))
    dfx["los"] = (dfx["Fecha salida"].dt.normalize() - dfx["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
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
    Forecast simple estilo P50: usa años anteriores para estimar pickup típico pendiente.
    """
    if df is None or df.empty:
        return {}

    # Actual OTB en el corte
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

    # Construye pickup típico usando ref_years años anteriores
    pickups = []
    adr_tails = []
    for y in range(1, int(ref_years) + 1):
        p_start_y = pd.to_datetime(period_start) - pd.DateOffset(years=y)
        p_end_y   = pd.to_datetime(period_end) - pd.DateOffset(years=y)
        cut_y     = pd.to_datetime(cutoff) - pd.DateOffset(years=y)
        # OTB a ese corte en LY-y
        _, tot_cut_y = compute_kpis(
            df_all=df,
            cutoff=cut_y,
            period_start=p_start_y,
            period_end=p_end_y,
            inventory_override=None,
            filter_props=props if props else None,
        )
        # Final LY-y (corte = fin de periodo)
        _, tot_final_y = compute_kpis(
            df_all=df,
            cutoff=p_end_y,  # para cierre LY, usamos corte = fin del periodo LY
            period_start=p_start_y,
            period_end=p_end_y,
            inventory_override=None,
            filter_props=props if props else None,
        )
        pick_y = float(tot_final_y.get("noches_ocupadas", 0.0) - tot_cut_y.get("noches_ocupadas", 0.0))
        pickups.append(max(pick_y, 0.0))
        # ADR tail aproximado: usamos ADR LY final como referencia
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

def _kai_cdm_pro_analysis(
    tot_now: dict,
    tot_ly_cut: dict,
    tot_ly_final: dict,
    pace: dict,
    price_ref_p50: float = None
) -> str:
    """
    Semáforo y análisis (robusto, acepta alias de claves y no falla si faltan datos).
    """
    # Leer pace con alias de claves
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

    # Mensaje
    msg = ""
    if pace_state == "🟢 Adelantado":
        msg += "### 🟢 Adelantado\n"
        msg += "Buen ritmo de reservas respecto a años anteriores. Mantén la estrategia y monitoriza el pickup restante.\n"
        if pick_need > pick_typ50 * 1.2:
            msg += "- Aún queda pickup elevado. Refuerza acciones de venta para asegurar el cierre.\n"
    elif pace_state == "🟠 En línea":
        msg += "### 🟠 En línea\n"
        msg += "Ritmo de reservas en línea con años anteriores. Revisa pickup y ADR por si necesitas ajustes.\n"
        if adr_tail_p50 < float(tot_ly_cut.get("adr", 0.0)) * 0.95:
            msg += "- ADR previsto por debajo de LY. Considera revisar precios.\n"
    elif pace_state == "🔴 Retrasado":
        msg += "### 🔴 Retrasado\n"
        msg += "Ritmo retrasado. Considera acciones urgentes: promos, campañas o ajustes de precios.\n"
        if pick_need > pick_typ50:
            msg += "- Pickup pendiente elevado. Refuerza captación y canales.\n"
        if adr_tail_p50 < float(tot_ly_cut.get("adr", 0.0)) * 0.95:
            msg += "- ADR previsto por debajo de LY. Considera bajar precios/ofertas.\n"
    else:
        msg += "No hay suficiente información para evaluar el ritmo de reservas.\n"

    # Resumen visual y KPIs
    msg += f"\n**Estado actual:** {pace_state}\n"
    msg += f"- Pickup pendiente objetivo: **{pick_need:,.0f} noches**\n"
    msg += f"- ADR previsto (P50): **{adr_tail_p50:.2f} €**\n"
    msg += f"- Forecast ingresos (P50): **{rev_final_p50:.2f} €**\n"
    msg += "\n**KPIs actuales:**\n"
    msg += f"- Ocupación actual: **{float(tot_now.get('ocupacion_pct', 0.0)):.2f}%**\n"
    msg += f"- ADR actual: **{float(tot_now.get('adr', 0.0)):.2f} €**\n"
    msg += f"- Ingresos actuales: **{float(tot_now.get('ingresos', 0.0)):.2f} €**\n"
    return msg

def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def _pct_delta(cur: float, ref: float) -> float:
    cur, ref = _safe(cur), _safe(ref)
    if ref == 0:
        return 0.0
    return (cur - ref) / ref * 100.0

def _pp_delta(cur_pct: float, ref_pct: float) -> float:
    return _safe(cur_pct) - _safe(ref_pct)

def pro_exec_summary(
    tot_now: Dict[str, float],
    tot_ly_cut: Dict[str, float],
    tot_ly_final: Dict[str, float],
    pace: Dict[str, Any],
) -> Dict[str, str]:
    """
    Devuelve {'headline': ..., 'detail': ...} con un análisis ejecutivo y plan de acción.
    - tot_now / tot_ly_*: dicts de compute_kpis
    - pace: dict de pace_forecast_month (usa alias si faltan claves)
    """
    # Lecturas básicas
    occ_now = _safe(tot_now.get("ocupacion_pct", 0))
    adr_now = _safe(tot_now.get("adr", 0))
    rev_now = _safe(tot_now.get("ingresos", 0))

    occ_ly = _safe(tot_ly_cut.get("ocupacion_pct", 0))
    adr_ly = _safe(tot_ly_cut.get("adr", 0))
    rev_ly = _safe(tot_ly_cut.get("ingresos", 0))

    rev_ly_final = _safe(tot_ly_final.get("ingresos", 0))

    # RevPAR aproximado
    revpar_now = adr_now * occ_now / 100.0
    revpar_ly = adr_ly * occ_ly / 100.0

    # Deltas
    d_occ_pp = _pp_delta(occ_now, occ_ly)
    d_adr_pct = _pct_delta(adr_now, adr_ly)
    d_revpar_pct = _pct_delta(revpar_now, revpar_ly)
    d_revenue_pct = _pct_delta(rev_now, rev_ly)

    # Pace (alias seguros)
    def g(d, keys, default=0.0):
        if not isinstance(d, dict): return default
        for k in keys:
            if k in d and d[k] is not None: return _safe(d[k], default)
        return default

    rev_final_p50 = g(pace, ["revenue_final_p50", "rev_final_p50", "p50_revenue_final"], rev_now)
    pick_typ50 = g(pace, ["pickup_typ_p50", "p50_pickup_typ", "pickup_typical_p50"], 0.0)
    adr_tail_p50 = g(pace, ["adr_tail_p50", "p50_adr_tail", "adr_typ_tail_p50"], adr_now)

    # Gap de cierre vs LY final usando forecast P50
    gap_rev = rev_ly_final - rev_final_p50
    cobertura_pct = _pct_delta(rev_final_p50, rev_ly_final) + 100 if rev_ly_final > 0 else 0.0
    gap_txt = f"Faltan {gap_rev:,.0f} €" if gap_rev > 0 else f"Superas LY final en {abs(gap_rev):,.0f} €"
    gap_txt = gap_txt.replace(",", ".")

    # Veredicto en función de d_occ y d_adr
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

    # Atribución simple del RevPAR
    atrib_occ_pp = d_occ_pp
    atrib_adr_pp = d_adr_pct  # mostramos como p.p. de precio relativo, se explica en texto

    # Viabilidad de cierre (heurística usando pickup y ADR tail)
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

    # Plan de acción
    acciones = []
    if d_adr_pct < -3:
        acciones.append("Revisar y retirar descuentos de baja conversión.")
        acciones.append("Micro-rebajas quirúrgicas en días valle (LT corto).")
    if d_occ_pp < 0:
        acciones.append("Boost de demanda: visibilidad OTAs, campañas directas, partners.")
    if d_adr_pct > 3 and d_occ_pp < 0:
        acciones.append("Mantener precios en picos, test A/B de precio en días flojos.")
    if not acciones:
        acciones = [
            "Monitorizar pickup semanal y mantener pricing en fines de semana/eventos.",
            "Reasignar presupuesto a canales con mejor conversión."
        ]

    # Build headline + detail
    headline = f"🌸 Explicación ejecutiva (narrada)\n\n" \
               f"• Veredicto general: {verdict}\n\n" \
               f"• Evolución vs LY (a este corte) → Ocupación {d_occ_pp:+.1f} p.p., ADR {d_adr_pct:+.1f}%, RevPAR {d_revpar_pct:+.1f}%, Ingresos {d_revenue_pct:+.1f}%.\n" \
               f"• Viabilidad de cierre del gap → {gap_txt} · Cobertura estimada P50 ≈ {cobertura_pct:.0f}%."

    detail = (
        "### 👉 Ver análisis detallado\n"
        f"- Ocupación: {'🟢' if d_occ_pp>=0 else '🔴'} {d_occ_pp:+.1f} p.p.\n"
        f"- ADR: {'🟢' if d_adr_pct>=0 else '🔴'} {d_adr_pct:+.1f}%\n"
        f"- RevPAR: {'🟢' if d_revpar_pct>=0 else '🔴'} {d_revpar_pct:+.1f}%\n"
        f"- Ingresos: {'🟢' if d_revenue_pct>=0 else '🔴'} {d_revenue_pct:+.1f}%\n\n"
        "#### Qué explica el resultado (atribución RevPAR)\n"
        f"- Ocupación: {atrib_occ_pp:+.1f} p.p.\n"
        f"- ADR: {atrib_adr_pp:+.1f}% (precio medio)\n\n"
        "#### Viabilidad de cierre del gap\n"
        f"- " + viab + "\n"
        f"- " + gap_txt + f" · Cobertura estimada ≈ {cobertura_pct:.0f}%.\n\n"
        "#### Plan de acción (siguiente quincena)\n"
        + "".join([f"- {a}\n" for a in acciones])
    )

    return {"headline": headline, "detail": detail}
