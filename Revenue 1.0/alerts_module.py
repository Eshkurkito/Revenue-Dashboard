from __future__ import annotations

from datetime import date, datetime
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any, Tuple

from utils import save_group_csv, load_groups, group_selector


# ==============================
# Helpers: columnas y ventanas
# ==============================

def _get_config(config: Optional[dict]) -> dict:
    """Obtiene el mapeo de columnas desde config o st.session_state.config_cols."""
    if config is None:
        config = st.session_state.get("config_cols", {})
    return config or {}


def _reservas_to_alert_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte reservas en filas diarias para alertas:
    - unidad = Alojamiento
    - fecha_llegada = cada día entre Fecha entrada y Fecha salida-1
    - ocupacion = 100 (% por unidad ocupada)
    - adr = Alquiler con IVA (€) / LOS (precio medio por noche)
    """
    req = ["Alojamiento", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]
    if not all(c in df.columns for c in req):
        return pd.DataFrame(columns=["unidad", "fecha_llegada", "ocupacion", "adr"])

    dfx = df.dropna(subset=["Alojamiento", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]).copy()
    dfx["Fecha entrada"] = pd.to_datetime(dfx["Fecha entrada"], errors="coerce").dt.normalize()
    dfx["Fecha salida"] = pd.to_datetime(dfx["Fecha salida"], errors="coerce").dt.normalize()
    dfx = dfx[dfx["Fecha salida"] > dfx["Fecha entrada"]]

    dfx["los"] = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days
    dfx["adr_reserva"] = pd.to_numeric(dfx["Alquiler con IVA (€)"], errors="coerce") / dfx["los"].replace(0, np.nan)

    # Genera lista de días de estancia y expande
    dfx["stay_dates"] = dfx.apply(
        lambda r: pd.date_range(r["Fecha entrada"], r["Fecha salida"] - pd.Timedelta(days=1), freq="D"), axis=1
    )
    out = dfx.loc[dfx["stay_dates"].str.len() > 0, ["Alojamiento", "stay_dates", "adr_reserva"]].explode("stay_dates")
    out = out.rename(columns={"Alojamiento": "unidad", "stay_dates": "fecha_llegada"})
    out["fecha_llegada"] = pd.to_datetime(out["fecha_llegada"]).dt.normalize()
    out["ocupacion"] = 100.0
    out["adr"] = out["adr_reserva"].astype(float)
    return out[["unidad", "fecha_llegada", "ocupacion", "adr"]]


def normalize_columns(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normaliza nombres/tipos y devuelve (df_norm, used_map).
    Si faltan campos mínimos pero existen columnas de reservas
    (Alojamiento, Fecha entrada/Fecha salida, Alquiler con IVA (€)),
    construye automáticamente las columnas para alertas.
    """
    # Intento de mapeo directo (si ya vienen las columnas mínimas)
    cols = {c.lower().strip(): c for c in df.columns}
    have_min = all(k in cols for k in ["unidad", "fecha_llegada", "ocupacion", "adr"])
    if have_min:
        out = pd.DataFrame()
        out["unidad"] = df[cols["unidad"]].astype(str).str.strip()
        out["fecha_llegada"] = pd.to_datetime(df[cols["fecha_llegada"]], errors="coerce").dt.normalize()
        occ = pd.to_numeric(df[cols["ocupacion"]], errors="coerce")
        if np.nanmax(occ.values) <= 1.5:
            occ = occ * 100.0
        out["ocupacion"] = occ.astype(float)
        out["adr"] = pd.to_numeric(df[cols["adr"]], errors="coerce").astype(float)
        return out, {"unidad": cols["unidad"], "fecha_llegada": cols["fecha_llegada"], "ocupacion": cols["ocupacion"], "adr": cols["adr"]}

    # Fallback: construir desde tus columnas de reservas
    if all(c in df.columns for c in ["Alojamiento", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]):
        built = _reservas_to_alert_rows(df)
        if built.empty:
            st.error("No se han podido construir filas diarias para alertas a partir de las reservas.")
            raise ValueError("alerts: empty after build from reservas")
        used = {
            "unidad": "Alojamiento",
            "fecha_llegada": "Fecha entrada…Fecha salida (expandido)",
            "ocupacion": "100% por unidad ocupada",
            "adr": "Alquiler con IVA (€)/LOS",
        }
        return built, used

    # Si no hay ni mapeo ni columnas de reservas, avisar
    st.error(
        "Faltan columnas mínimas para el módulo de alertas: unidad, fecha_llegada, ocupacion, adr. "
        "O bien proporciona el mapping o asegúrate de tener: Alojamiento, Fecha entrada, Fecha salida, Alquiler con IVA (€)."
    )
    raise ValueError("alerts: column mapping missing")


def compute_windows(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Añade days_to_arrival, window, mes_tab ('Mes actual'|'Mes siguiente'|'Otro')."""
    out = df.copy()
    base_day = pd.to_datetime(pd.Timestamp(today).normalize())
    out["days_to_arrival"] = (out["fecha_llegada"] - base_day).dt.days

    # Ventanas
    def _win(d: float) -> str:
        if pd.isna(d):
            return "fuera"
        d = int(d)
        if 0 <= d <= 7:
            return "0-7"
        if 8 <= d <= 30:
            return "8-30"
        if 31 <= d <= 60:
            return "31-60"
        return "fuera"

    out["window"] = out["days_to_arrival"].apply(_win)

    # Tabs por mes de llegada
    this_month = base_day.to_period("M")
    next_month = (base_day + pd.DateOffset(months=1)).to_period("M")
    month_period = out["fecha_llegada"].dt.to_period("M")
    out["mes_tab"] = np.where(
        month_period == this_month, "Mes actual",
        np.where(month_period == next_month, "Mes siguiente", "Otro")
    )
    return out


# ==============================
# Reglas, acciones y KPIs
# ==============================

def _pickup_threshold_p25(series: pd.Series) -> float:
    """Umbral de pickup bajo: P25; fallback media - 1*std; fallback 0."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    try:
        return float(np.percentile(s.values, 25))
    except Exception:
        m, sd = float(np.nanmean(s)), float(np.nanstd(s))
        return max(m - sd, 0.0)


def suggest_action(row: pd.Series) -> str:
    """Devuelve texto breve de acción sugerida basado en ventana y color + reglas delta."""
    win = str(row.get("window", ""))
    color = str(row.get("semaforo", ""))
    delta_adr = float(row.get("delta_adr", np.nan)) if pd.notna(row.get("delta_adr", np.nan)) else np.nan
    pickup_low = bool(row.get("pickup_low", False))

    base = ""
    if win == "0-7":
        if color == "Rojo":
            base = "Bajar ADR 5–12 %, abrir 1N/gap-filler 24–48 h, late-deal móvil."
        elif color == "Verde":
            base = "Subir ADR 5–8 %, considerar min-stay 2N en fin de semana."
        else:
            base = "Ajuste fino de precio; revisar min-stay y disponibilidad."
    elif win == "8-30":
        if color == "Rojo":
            base = "Ajuste -5–8 % ADR, early pick-up (NR blanda), revisar min-stay."
        elif color == "Verde":
            base = "+5–10 % ADR, reinstaurar min-stay fin de semana, limitar promos."
        else:
            base = "Microajustes y control de ritmo; vigilar conversión."
    elif win == "31-60":
        if color == "Rojo":
            base = "Micro-promos (-3–5 %) selectivas y revisión contenido OTA."
        elif color == "Verde":
            base = "Mantener estrategia; test A/B de precio en días valle."
        else:
            base = "Ajuste suave y optimización de contenido."

    extra = []
    if pd.notna(delta_adr) and delta_adr > 10 and pickup_low:
        extra.append("Rebajar a +2–5€ sobre compset.")
    if win == "0-7" and float(row.get("ocupacion", 0.0)) < 92:
        extra.append("Abrir 1N con suplemento/gap-filler.")

    txt = base
    if extra:
        txt += " " + " ".join(extra)
    return txt


def _target_occ_for_window(row: pd.Series) -> float:
    """Objetivo de ocupación por ventana si no hay objetivo explícito."""
    if pd.notna(row.get("objetivo_ocupacion", np.nan)) and row.get("objetivo_ocupacion", 0) > 0:
        return float(row["objetivo_ocupacion"])
    win = row.get("window", "")
    if win == "0-7":
        return 90.0
    if win == "8-30":
        return 85.0
    if win == "31-60":
        return 80.0
    return 80.0


def compute_alerts(df: pd.DataFrame, config: Optional[dict] = None, today: Optional[date] = None) -> pd.DataFrame:
    """Calcula semáforos y columnas de salida para el panel."""
    df0, used_map = normalize_columns(df, config)
    today = today or date.today()
    dfx = compute_windows(df0, today)

    # Delta ADR vs compset
    dfx["delta_adr"] = np.where(dfx["adr"].notna() & dfx["adr_compset"].notna(), dfx["adr"] - dfx["adr_compset"], np.nan)

    # Pickup bajo umbral global
    p25 = _pickup_threshold_p25(dfx["pickup_7d"])
    dfx["pickup_low"] = np.where(dfx["pickup_7d"].notna(), dfx["pickup_7d"] <= p25, False)

    # Objetivo de ocupación por fila
    dfx["target_occ"] = dfx.apply(_target_occ_for_window, axis=1)

    # Reglas de semáforo
    def _semaforo(row: pd.Series) -> str:
        win = row["window"]
        occ = float(row.get("ocupacion", np.nan))
        pace = float(row.get("pace_vs_ly", np.nan)) if pd.notna(row.get("pace_vs_ly", np.nan)) else np.nan
        delta = float(row.get("delta_adr", np.nan)) if pd.notna(row.get("delta_adr", np.nan)) else np.nan
        pk_low = bool(row.get("pickup_low", False))
        tgt = float(row.get("target_occ", 0.0))

        # 0–7
        if win == "0-7":
            if pd.notna(occ) and occ < 85 and (row.get("pickup_7d", np.nan) is np.nan or pk_low or np.isnan(row.get("pickup_7d", np.nan))):
                return "Rojo"
            if pd.notna(occ) and 85 <= occ < 92:
                return "Ámbar"
            if pd.notna(occ) and occ >= 92:
                return "Verde"
            # Fallback por objetivo
            if pd.notna(occ) and occ < max(0.0, tgt - 5):
                return "Ámbar"
            return "Ámbar"

        # 8–30
        if win == "8-30":
            if pd.notna(pace):
                if pace < 80 or (pd.notna(delta) and delta > 10 and pk_low):
                    return "Rojo"
                if (80 <= pace < 95) or (pd.notna(delta) and 5 < delta <= 10):
                    return "Ámbar"
                if pace >= 95 and (not pd.notna(delta) or abs(delta) <= 5):
                    return "Verde"
                return "Ámbar"
            # Fallback por objetivo de ocupación
            if pd.notna(occ):
                if occ < max(0.0, tgt - 10):
                    return "Rojo"
                if occ < tgt:
                    return "Ámbar"
                return "Verde"
            return "Ámbar"

        # 31–60
        if win == "31-60":
            if pd.notna(pace):
                if pace < 90:
                    return "Rojo"
                if 90 <= pace < 100:
                    return "Ámbar"
                return "Verde"
            # Fallback por objetivo
            if pd.notna(occ):
                if occ < max(0.0, tgt - 10):
                    return "Rojo"
                if occ < tgt:
                    return "Ámbar"
                return "Verde"
            return "Ámbar"

        return "—"

    dfx["semaforo"] = dfx.apply(_semaforo, axis=1)

    # Acción sugerida
    dfx["accion_sugerida"] = dfx.apply(suggest_action, axis=1)

    # Columnas de salida
    out = dfx.copy()
    out["Fecha llegada"] = out["fecha_llegada"].dt.strftime("%Y-%m-%d")
    out["Días a llegada"] = out["days_to_arrival"].astype("Int64")
    out["Ocupación %"] = out["ocupacion"].round(2)
    out["Pace vs LY %"] = out["pace_vs_ly"].round(2)
    out["ΔADR vs compset €"] = out["delta_adr"].round(2)

    return out


def kpi_counts(df_alerts: pd.DataFrame) -> Dict[str, float]:
    """Porcentaje de filas por color."""
    if df_alerts.empty or "semaforo" not in df_alerts.columns:
        return {"Rojo": 0.0, "Ámbar": 0.0, "Verde": 0.0}
    total = max(len(df_alerts), 1)
    pct = {
        k: float((df_alerts["semaforo"] == k).sum()) / total * 100.0
        for k in ["Rojo", "Ámbar", "Verde"]
    }
    return pct


def style_alerts(df_alerts: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styler con fondos suaves por color."""
    if df_alerts.empty:
        return df_alerts.style

    COLORS = {
        "Rojo": "#f8d7da",
        "Ámbar": "#fff3cd",
        "Verde": "#d4edda",
    }
    def _row_style(r: pd.Series):
        color = COLORS.get(str(r.get("semaforo", "")), "")
        return [f"background-color: {color};" if color else "" for _ in r]

    num_fmt = {
        "Ocupación %": "{:.2f}",
        "Pace vs LY %": "{:.2f}",
        "ΔADR vs compset €": "{:.2f}",
    }
    cols = [c for c in ["Unidad","Fecha llegada","Días a llegada","Ocupación %","Pace vs LY %","ΔADR vs compset €","Semáforo","Acción sugerida"] if c in df_alerts.columns]
    view = df_alerts.rename(columns={
        "unidad": "Unidad",
        "semaforo": "Semáforo",
        "accion_sugerida": "Acción sugerida",
    })[cols]
    styler = (
        view.style
        .apply(_row_style, axis=1)
        .format(num_fmt)
    )
    return styler


# ==============================
# Render principal del módulo
# ==============================

def _severity_key(row: pd.Series) -> Tuple[int, float, float]:
    """Orden: Rojo(0) > Ámbar(1) > Verde(2). Dentro: menor ocupación / peor pace primero."""
    sev = {"Rojo": 0, "Ámbar": 1, "Verde": 2}.get(str(row.get("semaforo", "")), 3)
    occ = float(row.get("ocupacion", np.nan))
    pace = float(row.get("pace_vs_ly", np.nan))
    return (sev, occ if pd.notna(occ) else 9999.0, -pace if pd.notna(pace) else 0.0)


def _filter_current_scope(df_alerts: pd.DataFrame, tab_label: str, days_bucket: str, zona_sel: Optional[str], search: str) -> pd.DataFrame:
    """Aplica filtros del UI."""
    d = df_alerts.copy()
    if tab_label in ("Mes actual", "Mes siguiente"):
        d = d[d["mes_tab"] == tab_label]
    if days_bucket in ("0-7", "8-30", "31-60"):
        d = d[d["window"] == days_bucket]
    if zona_sel and "zona" in d.columns and zona_sel != "(todas)":
        d = d[d["zona"] == zona_sel]
    if search:
        s = search.strip().lower()
        d = d[d["unidad"].str.lower().str.contains(s)]
    return d


def render_alerts_module(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    show_debug: bool = True,
) -> None:
    """
    Render sencillo del panel de alertas:
    - Normaliza/convierte reservas -> filas diarias (unidad-fecha).
    - Muestra KPIs, preview y alertas por unidad.
    """
    df_norm, used = normalize_columns(df, config)

    if df_norm.empty:
        st.warning("Alertas: no hay filas tras normalizar datos.")
        return

    # Rango de fechas
    min_d, max_d = df_norm["fecha_llegada"].min(), df_norm["fecha_llegada"].max()
    start = pd.to_datetime(start) if start is not None else min_d
    end = pd.to_datetime(end) if end is not None else max_d
    mask = (df_norm["fecha_llegada"] >= start) & (df_norm["fecha_llegada"] <= end)
    df_f = df_norm.loc[mask].copy()

    if show_debug:
        st.caption(
            f"Alertas: {len(df_f):,} filas · {start.date()} → {end.date()} · mapeo: {used}"
            .replace(",", ".")
        )
        st.dataframe(df_f.head(20), use_container_width=True)

    if df_f.empty:
        st.info("No hay datos en el rango seleccionado para alertas.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades", f"{df_f['unidad'].nunique():,}".replace(",", "."))
    c2.metric("Días", f"{df_f['fecha_llegada'].nunique():,}".replace(",", "."))
    c3.metric("Ocupación media", f"{df_f['ocupacion'].mean():.1f}%")
    c4.metric("ADR medio", f"{df_f['adr'].mean():.2f} €")

    # Alertas por unidad: occ baja o ADR bajo vs mediana propia
    med_adr = df_f.groupby("unidad")["adr"].median().rename("adr_med")
    agg = df_f.groupby("unidad").agg(
        occ=("ocupacion", "mean"),
        adr=("adr", "mean"),
        dias=("fecha_llegada", "nunique"),
    ).reset_index()
    agg = agg.merge(med_adr, on="unidad", how="left")
    agg["flag"] = np.where(
        agg["occ"] < 40,
        "⚠️ Occ baja",
        np.where(agg["adr"] < 0.90 * agg["adr_med"], "⚠️ ADR bajo", "✅"),
    )
    st.subheader("Alertas por unidad")
    st.dataframe(
        agg.sort_values(["flag", "occ"]).reset_index(drop=True),
        use_container_width=True,
    )

    # Preview detallada: tabla completa filtrada
    with st.expander("Ver detalle de alertas por unidad", expanded=True):
        st.dataframe(style_alerts(df_f), use_container_width=True)


    # Sidebar: selección y guardado de grupo
    groups = load_groups()
    group_names = ["Ninguno"] + sorted(list(groups.keys()))
    selected_group = st.selectbox("Grupo guardado", group_names)

    if selected_group and selected_group != "Ninguno":
        props_rc = groups[selected_group]
        # Botón para eliminar grupo
        if st.button(f"Eliminar grupo '{selected_group}'"):
            import pandas as pd
            from utils import GROUPS_PATH
            df_groups = pd.read_csv(GROUPS_PATH)
            df_groups = df_groups[df_groups["Grupo"] != selected_group]
            df_groups.to_csv(GROUPS_PATH, index=False)
            st.success(f"Grupo '{selected_group}' eliminado.")
            st.experimental_rerun()
    else:
        props_rc = group_selector(
            "Filtrar alojamientos (opcional)",
            sorted([str(x) for x in df["Alojamiento"].dropna().unique()]),
            key_prefix="props_rc",
            default=[]
        )

    group_name = st.text_input("Nombre del grupo para guardar")
    if st.button("Guardar grupo de pisos") and group_name and props_rc:
        save_group_csv(group_name, props_rc)
        st.success(f"Grupo '{group_name}' guardado.")

    # Filtra el DataFrame por el grupo seleccionado
    if props_rc:
        df = df[df["Alojamiento"].isin(props_rc)]