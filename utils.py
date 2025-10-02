import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

# ===== Grupos de alojamientos (CSV reutilizable) =====
GROUPS_PATH = Path("data/grupos.csv")

@st.cache_data(show_spinner=False)
def save_group_csv(name: str, props: list[str], path: str | Path = GROUPS_PATH):
    df = pd.DataFrame({"Grupo": [name]*len(props), "Alojamiento": props})
    df.to_csv(path, index=False)

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
) -> Optional[pd.DataFrame]:
    if "Portal" not in df_all.columns:
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
    portal_counts = df_cut["Portal"].value_counts().reset_index()
    portal_counts.columns = ["Portal", "Reservas"]
    portal_counts["% Reservas"] = portal_counts["Reservas"] / portal_counts["Reservas"].sum() * 100

    return portal_counts

def pace_series(*args, **kwargs):
    # TODO: Implementa la lógica real aquí
    return pd.DataFrame()

def pace_forecast_month(*args, **kwargs):
    # TODO: Implementa la lógica real aquí
    return {}
