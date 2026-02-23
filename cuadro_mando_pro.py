from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import date

from utils import (
    compute_kpis, period_inputs, group_selector, help_block,
    pace_series, pace_forecast_month, save_group_csv, load_groups,
    _kai_cdm_pro_analysis,
)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    norm = {c: str(c).lower() for c in df.columns}
    def find(*cands):
        for col, n in norm.items():
            for c in cands:
                if n == c or c in n:
                    return col
        return None
    mapping = {}
    col_aloj = find("alojamiento", "propiedad", "property", "listing", "unidad", "apartamento", "room", "unit", "nombre alojamiento")
    if col_aloj: mapping[col_aloj] = "Alojamiento"
    col_fa = find("fecha alta", "fecha de alta", "booking date", "fecha reserva", "creado", "created", "booked")
    if col_fa: mapping[col_fa] = "Fecha alta"
    col_fe = find("fecha entrada", "check in", "entrada", "arrival")
    if col_fe: mapping[col_fe] = "Fecha entrada"
    col_fs = find("fecha salida", "check out", "salida", "departure")
    if col_fs: mapping[col_fs] = "Fecha salida"
    col_rev = find("alquiler con iva (€)", "alquiler con iva", "ingresos", "revenue", "importe", "total", "precio total", "alquiler con tasas")
    if col_rev: mapping[col_rev] = "Alquiler con IVA (€)"
    return df.rename(columns=mapping) if mapping else df

def _load_saved_groups(props_all: list[str]) -> dict[str, list[str]]:
    mod_dir = Path(__file__).resolve().parent
    candidates = [
        mod_dir / "grupos_guardados.csv",
        mod_dir / "assets" / "grupos_guardados.csv",
        Path.cwd() / "grupos_guardados.csv",
        Path.cwd() / "assets" / "grupos_guardados.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return {}

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            dfg = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            dfg = None
    if dfg is None or dfg.empty:
        return {}

    dfg.columns = [str(c).strip().lower() for c in dfg.columns]
    if not {"grupo", "alojamiento"}.issubset(dfg.columns):
        return {}

    dfg = dfg.dropna(subset=["grupo", "alojamiento"]).astype(str)
    dfg["grupo"] = dfg["grupo"].str.strip()
    dfg["alojamiento"] = dfg["alojamiento"].str.strip()

    props_map = {str(p).strip().upper(): str(p) for p in props_all}
    def map_prop(p: str) -> str | None:
        return props_map.get(str(p).strip().upper())

    groups: dict[str, list[str]] = {}
    for g, sub in dfg.groupby("grupo"):
        mapped = [mp for p in sub["alojamiento"].tolist() if (mp := map_prop(p))]
        if mapped:
            groups[g] = sorted(dict.fromkeys(mapped))
    return groups

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs=(0.1, 0.5, 0.9)) -> dict:
    if values.size == 0 or np.nansum(weights) <= 0:
        return {"P10": 0.0, "Mediana": 0.0, "P90": 0.0}
    m = ~np.isnan(values)
    v = values[m]
    w = weights[m]
    if v.size == 0 or np.nansum(w) <= 0:
        return {"P10": 0.0, "Mediana": 0.0, "P90": 0.0}
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    res = {}
    for q, name in zip(qs, ["P10", "Mediana", "P90"]):
        idx = np.searchsorted(cw, q, side="left")
        idx = min(idx, v.size - 1)
        res[name] = float(v[idx])
    return res

def _compute_adr_bands_period_prorate(df: pd.DataFrame, period_start, period_end, cutoff) -> pd.Series:
    """
    Bandas ADR (P10/Mediana/P90) para TODO el periodo usando noches prorrateadas:
    - Solo reservas con Fecha alta <= cutoff.
    - Se prorratea el ingreso por noche y se cuentan solo las noches que caen dentro del periodo.
    """
    if df is None or df.empty:
        return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})

    start_dt = pd.to_datetime(period_start)
    end_dt = pd.to_datetime(period_end)
    cut_dt = pd.to_datetime(cutoff)
    end_inclusive = end_dt + pd.Timedelta(days=1)  # para incluir la noche del último día

    cols_needed = {"Fecha alta", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"}
    if not cols_needed.issubset(set(df.columns)):
        return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})

    dfx = (
        df[(df["Fecha alta"] <= cut_dt)]
        .dropna(subset=["Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"])
        .copy()
    )
    if dfx.empty:
        return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})

    # Normaliza a días completos
    s_all = dfx["Fecha entrada"].dt.normalize()
    e_all = dfx["Fecha salida"].dt.normalize()

    los_total = (e_all - s_all).dt.days.clip(lower=1)
    nightly_rate = (dfx["Alquiler con IVA (€)"] / los_total).astype(float)

    # Tramo de noches dentro del periodo
    seg_start = np.maximum(s_all.values.astype("datetime64[D]"), start_dt.normalize().to_datetime64())
    seg_end = np.minimum(e_all.values.astype("datetime64[D]"), end_inclusive.normalize().to_datetime64())
    nights_in = (seg_end - seg_start).astype("timedelta64[D]").astype(int)

    mask = nights_in > 0
    if not np.any(mask):
        return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})

    values = nightly_rate.values[mask]
    weights = nights_in[mask].astype(float)

    q = _weighted_quantile(values, weights, qs=(0.1, 0.5, 0.9))
    return pd.Series(q)

def _load_forecast_table(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tabla de forecast a formato largo con columnas:
    'Alojamiento' (str), 'mes_num' (1..12) y 'Forecast' (float).
    Admite encabezados tipo 'Enero', 'enero', '2024-01' o '2024/01'.
    """
    if df_forecast is None or df_forecast.empty:
        return pd.DataFrame(columns=["Alojamiento", "mes_num", "Forecast"])

    # Renombra columna alojamiento si aparece como 'Apartamento' u otras variantes
    cols = {c.lower().strip(): c for c in df_forecast.columns}
    if "alojamiento" not in cols and "apartamento" in cols:
        df_forecast = df_forecast.rename(columns={cols["apartamento"]: "Alojamiento"})
    elif "alojamiento" not in cols and "apartamento" not in cols:
        # intenta encontrar col primera como alojamiento
        first = df_forecast.columns[0]
        df_forecast = df_forecast.rename(columns={first: "Alojamiento"})

    if "Alojamiento" not in df_forecast.columns:
        return pd.DataFrame(columns=["Alojamiento", "mes_num", "Forecast"])

    # Mapeo de meses (español / inglés)
    meses_map = {
        "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
        "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12,
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    }

    month_cols = [c for c in df_forecast.columns if c != "Alojamiento"]
    rows = []
    for col in month_cols:
        col_key = str(col).strip()
        lc = col_key.lower().strip()
        mes_num = None
        # formato YYYY-MM o YYYY/MM
        try:
            if "-" in lc or "/" in lc:
                p = pd.to_datetime(lc, errors="coerce")
                if not pd.isna(p):
                    mes_num = int(p.month)
        except Exception:
            mes_num = None
        if mes_num is None and lc in meses_map:
            mes_num = meses_map[lc]
        # si no pudo, intenta extraer número
        if mes_num is None:
            import re
            m = re.search(r"(\b0?[1-9]\b|\b1[0-2]\b)", lc)
            if m:
                mes_num = int(m.group(1))
        if mes_num is None:
            continue
        # convierte valores numéricos (quita miles/€, comas)
        # limpiar: quitar todo menos dígitos, comas, puntos y signos; eliminar separador de miles '.'; convertir ','->'.'
        ser_raw = df_forecast[col].astype(str).str.replace(r"[^\d\-,\.]", "", regex=True)
        ser_clean = ser_raw.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        ser = pd.to_numeric(ser_clean, errors="coerce").fillna(0.0)
        for apt, val in zip(df_forecast["Alojamiento"].astype(str), ser):
            rows.append({"Alojamiento": apt.strip(), "mes_num": int(mes_num), "Forecast": float(val)})

    if not rows:
        return pd.DataFrame(columns=["Alojamiento", "mes_num", "Forecast"])
    return pd.DataFrame(rows)

def _render_forecast_vs_actual(df: pd.DataFrame, forecast: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, props: list | None, cutoff, inv_override) -> pd.DataFrame:
    """
    Construye y muestra comparativa Forecast vs Actual por mes.
    - forecast: salida de _load_forecast_table (mes_num 1..12)
    - Actual ahora se filtra SOLO por alojamientos presentes en el forecast (y, si hay selección, por la intersección).
    """
    # Determina alojamientos válidos para la comparación
    forecast_props = []
    if forecast is not None and not forecast.empty and "Alojamiento" in forecast.columns:
        forecast_props = (
            forecast["Alojamiento"].astype(str).str.strip().dropna().unique().tolist()
        )

    # props_use: si hay selección del usuario, usa la intersección; si no, usa todo el set del forecast
    if props:
        props_use = [p for p in props if p in forecast_props] if forecast_props else props
    else:
        props_use = forecast_props

    months = pd.period_range(pd.to_datetime(start).to_period("M"), pd.to_datetime(end).to_period("M"), freq="M")
    rows = []
    for p in months:
        s = p.to_timestamp(how="start")
        e = p.to_timestamp(how="end")

        # Si no hay ningún alojamiento que coincida con el forecast, los actual = 0
        if props_use is not None and len(props_use) == 0:
            actual_ing = 0.0
        else:
            _, tot = compute_kpis(
                df,
                pd.to_datetime(cutoff),
                pd.to_datetime(s),
                pd.to_datetime(e),
                inventory_override=int(inv_override) if (inv_override is not None and int(inv_override) > 0) else None,
                filter_props=props_use if props_use else None,  # ← filtro restringido al forecast
            )
            actual_ing = float(tot.get("ingresos", 0.0))

        # Suma forecast del mes restringido a los props_use
        if forecast is not None and not forecast.empty:
            fmask = forecast["mes_num"] == int(p.month)
            if props_use:
                fmask &= forecast["Alojamiento"].isin(props_use)
            forecast_ing = float(forecast.loc[fmask, "Forecast"].sum()) if fmask.any() else 0.0
        else:
            forecast_ing = 0.0

        diff = actual_ing - forecast_ing
        pct = (diff / forecast_ing * 100.0) if forecast_ing != 0 else np.nan
        rows.append({"Mes": p.strftime("%Y-%m"), "Forecast": forecast_ing, "Actual": actual_ing, "Diff": diff, "DiffPct": pct})

    df_cmp = pd.DataFrame(rows)
    st.subheader("📊 Comparativa Forecast vs Actual (mensual)")

    for col in ["Forecast", "Actual", "Diff", "DiffPct"]:
        if col in df_cmp.columns:
            df_cmp[col] = pd.to_numeric(df_cmp[col], errors="coerce")

    sty = df_cmp.style.format(
        {"Forecast": "{:.2f}", "Actual": "{:.2f}", "Diff": "{:.2f}", "DiffPct": "{:.1f}%"},
        na_rep=""
    )
    st.dataframe(sty, use_container_width=True)
    if not df_cmp.empty:
        plot = df_cmp.melt(id_vars=["Mes"], value_vars=["Forecast","Actual"], var_name="Serie", value_name="Ingresos")
        chart = (
            alt.Chart(plot)
            .mark_bar()
            .encode(x=alt.X("Mes:N", sort=list(df_cmp["Mes"])), y=alt.Y("Ingresos:Q"), color="Serie:N", tooltip=["Mes","Serie","Ingresos"])
        )
        st.altair_chart(chart, use_container_width=True)
    return df_cmp

def _count_props_with_data(df: pd.DataFrame, period_start, period_end, cutoff) -> int:
    """Cuenta alojamientos con reservas que intersectan el periodo y con Fecha alta ≤ corte."""
    if df is None or df.empty:
        return 0
    needed = {"Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida"}
    if not needed.issubset(set(df.columns)):
        return 0

    d = df.copy()
    for c in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d.dropna(subset=["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida"])

    cut_dt = pd.to_datetime(cutoff).normalize()
    start_dt = pd.to_datetime(period_start).normalize()
    end_dt = pd.to_datetime(period_end).normalize()
    end_inclusive = end_dt + pd.Timedelta(days=1)

    d = d[d["Fecha alta"] <= cut_dt]
    overlap = (d["Fecha entrada"] < end_inclusive) & (d["Fecha salida"] > start_dt)
    return int(d.loc[overlap, "Alojamiento"].astype(str).nunique())

def _count_props_active_adjacent(df: pd.DataFrame, period_start, period_end, cutoff, months_window: int = 1) -> int:
    """
    Cuenta alojamientos 'activos' si tienen alguna reserva con Fecha alta ≤ corte
    que intersecta el periodo extendido ±N meses respecto al periodo visible.
    """
    if df is None or df.empty:
        return 0
    needed = {"Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida"}
    if not needed.issubset(set(df.columns)):
        return 0

    d = df.copy()
    for c in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d.dropna(subset=["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida"])

    cut_dt = pd.to_datetime(cutoff).normalize()
    # Limita a reservas confirmadas al corte
    d = d[d["Fecha alta"] <= cut_dt]

    # Ventana extendida: desde el inicio del mes anterior al periodo hasta el fin del mes siguiente
    p_start = pd.to_datetime(period_start)
    p_end = pd.to_datetime(period_end)
    start_m = pd.Period(p_start, freq="M")
    end_m = pd.Period(p_end, freq="M")
    ext_start_m = start_m - months_window
    ext_end_m = end_m + months_window
    ext_start = ext_start_m.start_time.normalize()
    ext_end = ext_end_m.end_time.normalize()
    ext_end_inclusive = ext_end + pd.Timedelta(days=1)

    overlap_ext = (d["Fecha entrada"] < ext_end_inclusive) & (d["Fecha salida"] > ext_start)
    return int(d.loc[overlap_ext, "Alojamiento"].astype(str).nunique())

def _count_props_active_by_first_booking(df: pd.DataFrame, cutoff) -> int:
    """Cuenta apartamentos activos al corte según la primera 'Fecha alta' (primer booking) por alojamiento."""
    if df is None or df.empty:
        return 0
    if not {"Alojamiento", "Fecha alta"}.issubset(df.columns):
        return 0
    d = df.copy()
    d["Alojamiento"] = d["Alojamiento"].astype(str)
    d["Fecha alta"] = pd.to_datetime(d["Fecha alta"], errors="coerce")
    d = d.dropna(subset=["Alojamiento", "Fecha alta"])
    first_booking = d.groupby("Alojamiento", as_index=False)["Fecha alta"].min()
    cut_dt = pd.to_datetime(cutoff).normalize()
    return int(first_booking.loc[first_booking["Fecha alta"] <= cut_dt, "Alojamiento"].nunique())

def render_cuadro_mando_pro(raw: pd.DataFrame | None = None):
    # --- validación ---
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("No hay datos cargados. Sube un Excel/CSV en la barra lateral para usar PRO.")
        st.stop()

    df = _standardize_columns(raw.copy())  # ← asegura nombres canónicos
    if "Alojamiento" not in df.columns:
        st.warning("No se encuentra la columna ‘Alojamiento’.")
        st.stop()

    df["Alojamiento"] = df["Alojamiento"].astype(str)
    props_all = sorted(df["Alojamiento"].dropna().unique())

    with st.sidebar:
        st.subheader("Parámetros · PRO")

        c1, c2 = st.columns(2)
        pro_start = c1.date_input("Inicio periodo", value=date.today().replace(day=1), key="pro_start")
        pro_end   = c2.date_input("Fin periodo", value=date.today(), key="pro_end")
        pro_cut   = st.date_input("Fecha de corte", value=date.today(), key="pro_cut")

        # --- Grupos guardados (siempre visible) ---
        groups = _load_saved_groups(props_all)
        group_names = ["(Sin grupo)"] + sorted(groups.keys())

        # Estado inicial y saneo si el grupo guardado ya no existe
        if "pro_group" not in st.session_state:
            st.session_state.pro_group = "(Sin grupo)"
        if "pro_props" not in st.session_state:
            st.session_state.pro_props = []
        if st.session_state.pro_group not in group_names:
            st.session_state.pro_group = "(Sin grupo)"
            st.session_state.pro_props = []

        # Al cambiar de grupo, actualiza los alojamientos seleccionados
        def _on_group_change():
            g = st.session_state.get("pro_group")
            st.session_state["pro_props"] = (
                groups.get(g, []) if g and g != "(Sin grupo)" else []
            )
            # No llames a st.rerun() aquí; Streamlit ya re‑ejecuta automáticamente

        st.selectbox("Grupo guardado", group_names, key="pro_group", on_change=_on_group_change)

        # Multiselect toma su valor de session_state["pro_props"]
        props_pro = st.multiselect("Alojamientos", options=props_all, key="pro_props")

        c3, c4 = st.columns(2)
        inv_pro    = c3.number_input("Inventario actual", min_value=0, value=0, step=1, key="inv_pro")
        inv_pro_ly = c4.number_input("Inventario LY",     min_value=0, value=0, step=1, key="inv_pro_ly")
        ref_years_pro = st.selectbox("Años de referencia (Pace)", options=[1, 2, 3], index=0, key="ref_years_pro")

    # --- Filtro por alojamientos seleccionados ---
    selected_props = st.session_state.get("pro_props") or []
    if selected_props:
        df = df[df["Alojamiento"].isin(selected_props)]

    # ========= KPIs (usa las variables definidas arriba) =========
    by_prop_now, tot_now = compute_kpis(
        df,
        pd.to_datetime(pro_cut),
        pd.to_datetime(pro_start),
        pd.to_datetime(pro_end),
        int(inv_pro) if inv_pro > 0 else None,
        props_pro,
    )
    _, tot_ly_cut = compute_kpis(
        df,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=1),
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro,
    )
    cutoff_ly_final = pd.to_datetime(pro_end) - pd.DateOffset(years=1)
    _, tot_ly_final = compute_kpis(
        df,
        cutoff_ly_final,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro,
    )

    # NUEVO: Ingresos LY-2 (a este corte) y LY-2 final
    _, tot_ly2_cut_ing = compute_kpis(
        df,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=2),
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro,
    )
    cutoff_ly2_final = pd.to_datetime(pro_end) - pd.DateOffset(years=2)
    _, tot_ly2_final_ing = compute_kpis(
        df,
        cutoff_ly2_final,
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro,
    )

    # ====== Ingresos ======
    # Calcula Rev par (ingreso por apartamento) para mostrar como métricas principales
    n_props_act_res = _count_props_with_data(df, pro_start, pro_end, pro_cut)
    n_props_ly_res  = _count_props_with_data(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        cutoff_ly_final,
    )
    n_props_ly2_res = _count_props_with_data(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        cutoff_ly2_final,
    )

    avg_act = float(tot_now.get("ingresos", 0.0)) / n_props_act_res if n_props_act_res > 0 else 0.0
    avg_ly  = float(tot_ly_final.get("ingresos", 0.0)) / n_props_ly_res if n_props_ly_res > 0 else 0.0
    avg_ly2 = float(tot_ly2_final_ing.get("ingresos", 0.0)) / n_props_ly2_res if n_props_ly2_res > 0 else 0.0

    st.subheader("💶 Ingresos (periodo seleccionado)")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Ingresos actuales (€)", f"{tot_now['ingresos']:.2f}")
    g2.metric("Ingresos LY a este corte (€)", f"{tot_ly_cut['ingresos']:.2f}")
    g3.metric("Ingresos LY final (€)", f"{tot_ly_final['ingresos']:.2f}")
    g4.metric("Ingresos LY-2 a este corte (€)", f"{tot_ly2_cut_ing['ingresos']:.2f}")
    g5.metric("Ingresos LY-2 final (€)", f"{tot_ly2_final_ing['ingresos']:.2f}")

    # RevPAR: usa inventario cuando esté disponible; si no, ADR * ocupación (fracción)
    days_period = (pd.to_datetime(pro_end) - pd.to_datetime(pro_start)).days + 1
    def _compute_revpar(total_ing, adr, occ_pct, inv):
        if isinstance(inv, (int, float)) and int(inv) > 0:
            return float(total_ing) / (int(inv) * max(int(days_period), 1))
        # fallback a ADR * ocupación (ocupacion en % -> fracción)
        return float(adr) * (float(occ_pct or 0.0) / 100.0)

    revpar_act = _compute_revpar(tot_now.get("ingresos", 0.0), tot_now.get("adr", 0.0), tot_now.get("ocupacion_pct", 0.0), inv_pro)
    revpar_ly  = _compute_revpar(tot_ly_final.get("ingresos", 0.0), tot_ly_final.get("adr", 0.0), tot_ly_final.get("ocupacion_pct", 0.0), inv_pro_ly)
    revpar_ly2 = _compute_revpar(tot_ly2_final_ing.get("ingresos", 0.0), tot_ly2_final_ing.get("adr", 0.0), tot_ly2_final_ing.get("ocupacion_pct", 0.0), None)

    rp1, rp2, rp3 = st.columns(3)
    rp1.metric("RevPAR actual (€)", f"{revpar_act:,.2f}".replace(",","."))
    rp2.metric("RevPAR LY final (€)", f"{revpar_ly:,.2f}".replace(",","."))
    rp3.metric("RevPAR LY-2 final (€)", f"{revpar_ly2:,.2f}".replace(",","."))

    st.caption(
        f"Apartamentos con reservas en el periodo: Act {n_props_act_res} · LY {n_props_ly_res} · LY-2 {n_props_ly2_res} · "
        f"RevPAR estimado: Act €{revpar_act:,.2f} · LY final €{revpar_ly:,.2f} · LY-2 final €{revpar_ly2:,.2f}"
    )
    st.caption(f"Apartamentos activos (±1 mes del periodo): Act {act_adj_act} · LY {act_adj_ly} · LY-2 {act_adj_ly2}")

    # Nuevo: conteo de apartamentos activos al corte y con reservas en el periodo (Act, LY y LY-2)
    actives_act = _count_props_active_by_first_booking(df, pro_cut)
    actives_ly  = _count_props_active_by_first_booking(df, pd.to_datetime(pro_cut) - pd.DateOffset(years=1))
    actives_ly2 = _count_props_active_by_first_booking(df, pd.to_datetime(pro_cut) - pd.DateOffset(years=2))

    n_props_act_res = _count_props_with_data(df, pro_start, pro_end, pro_cut)
    # Usar para LY/LY-2 el corte "final" (fin del periodo del año correspondiente)
    n_props_ly_res  = _count_props_with_data(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        cutoff_ly_final,
    )
    n_props_ly2_res = _count_props_with_data(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        cutoff_ly2_final,
    )

    # Activos por actividad en meses adyacentes (±1 mes del periodo)
    act_adj_act = _count_props_active_adjacent(df, pro_start, pro_end, pro_cut, months_window=1)
    act_adj_ly  = _count_props_active_adjacent(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        pd.to_datetime(pro_cut) - pd.DateOffset(years=1),
        months_window=1,
    )
    act_adj_ly2 = _count_props_active_adjacent(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        pd.to_datetime(pro_cut) - pd.DateOffset(years=2),
        months_window=1,
    )

    # Ingreso medio por piso = ingresos del periodo / número de pisos con reservas (proteger división por 0)
    avg_act = float(tot_now.get("ingresos", 0.0)) / n_props_act_res if n_props_act_res > 0 else 0.0
    # Para años anteriores dividir los ingresos "finales" (no a fecha de corte)
    avg_ly  = float(tot_ly_final.get("ingresos", 0.0)) / n_props_ly_res if n_props_ly_res > 0 else 0.0
    avg_ly2 = float(tot_ly2_final_ing.get("ingresos", 0.0)) / n_props_ly2_res if n_props_ly2_res > 0 else 0.0
    st.caption(
        f"Apartamentos con reservas en el periodo: Act {n_props_act_res} · LY {n_props_ly_res} · LY-2 {n_props_ly2_res} · "
        f"Ingreso medio por piso: Act €{avg_act:,.2f} · LY final €{avg_ly:,.2f} · LY-2 final €{avg_ly2:,.2f}"
    )
    st.caption(f"Apartamentos activos (±1 mes del periodo): Act {act_adj_act} · LY {act_adj_ly} · LY-2 {act_adj_ly2}")

    # ---- Forecast: carga desde data/forecast_db.csv por defecto (sin subir) ----
    with st.expander("Forecast mensual (opcional): usar data/forecast_db.csv o subir archivo para reemplazar", expanded=False):
        # intenta cargar automáticamente el forecast DB
        forecast_df = _find_and_read_forecast_db()

        # Si el usuario sube un archivo, lo usa para reemplazar la DB en memoria (no sobrescribe fichero en disco)
        uploaded = st.file_uploader("Opcional: sube CSV/Excel para reemplazar el forecast cargado", type=["csv","xlsx","xls"], key="pro_forecast_upload")
        if uploaded is not None:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    # Intentar leer CSV europeo con distintos encodings/separadores
                    for enc in ("cp1252","latin-1","utf-8","utf-8-sig"):
                        try:
                            forecast_df = pd.read_csv(uploaded, sep=";", encoding=enc, engine="python", dtype=str)
                            break
                        except Exception:
                            try:
                                uploaded.seek(0)
                            except Exception:
                                pass
                    if forecast_df.empty:
                        # fallback lectura sin separador semicolon
                        forecast_df = pd.read_csv(uploaded)
                else:
                    forecast_df = pd.read_excel(uploaded, engine="openpyxl", dtype=str)
            except Exception:
                st.error("No se pudo leer el archivo subido. Se usará el forecast local si existe.")

        if not forecast_df.empty:
            fc_long = _load_forecast_table(forecast_df)
            if fc_long.empty:
                st.warning("No se detectó formato válido en el forecast.")
            else:
                # --- Mostrar forecast filtrado por periodo y por pisos seleccionados ---
                months = pd.period_range(pd.to_datetime(pro_start).to_period("M"), pd.to_datetime(pro_end).to_period("M"), freq="M")
                months_map = {p.month: p.strftime("%Y-%m") for p in months}
                month_nums = [p.month for p in months]
                mask = fc_long["mes_num"].isin(month_nums)
                if props_pro:
                    mask &= fc_long["Alojamiento"].isin(props_pro)
                fc_sel = fc_long.loc[mask].copy()
                if not fc_sel.empty:
                    fc_sel["Mes"] = fc_sel["mes_num"].map(months_map).fillna(fc_sel["mes_num"].astype(str))
                    pivot = fc_sel.pivot_table(index="Alojamiento", columns="Mes", values="Forecast", aggfunc="sum", fill_value=0.0)
                    st.subheader("🔎 Forecast por Alojamiento / Mes (periodo seleccionado)")
                    st.dataframe(pivot.reset_index(), use_container_width=True)
                else:
                    st.info("No hay forecast para los alojamientos/meses seleccionados.")
                # --- Comparativa agregada mensual ---
                _render_forecast_vs_actual(df=_standardize_columns(raw.copy()), forecast=fc_long, start=pro_start, end=pro_end, props=props_pro if props_pro else None, cutoff=pro_cut, inv_override=inv_pro)
        else:
            st.info("No hay forecast cargado y no se encontró data/forecast_db.csv.")

    # ====== ADR ======
    st.subheader("🏷️ ADR (a fecha de corte)")
    _, tot_ly2_cut = compute_kpis(
        df,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=2),
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ADR actual (€)", f"{tot_now['adr']:.2f}")
    a2.metric("ADR LY (€)", f"{tot_ly_cut['adr']:.2f}")
    a3.metric("ADR LY final (€)", f"{tot_ly_final['adr']:.2f}")
    a4.metric("ADR LY-2 (€)", f"{tot_ly2_cut['adr']:.2f}")

    # === ADR entre semana vs fin de semana (periodo seleccionado, uplift vie/sáb = 25%) ===
    u = 0.25  # uplift fines de semana (viernes y sábado)
    d_adr = _standardize_columns(df.copy())
    # Tipos de fecha seguros
    for c in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        if c in d_adr.columns:
            d_adr[c] = pd.to_datetime(d_adr[c], errors="coerce")

    # Filtra reservas confirmadas al corte y que intersectan el periodo
    start_dt = pd.to_datetime(pro_start).normalize()
    end_dt   = pd.to_datetime(pro_end).normalize()
    end_inclusive = end_dt + pd.Timedelta(days=1)

    d_adr = d_adr.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"]).copy()
    d_adr = d_adr[d_adr["Fecha alta"] <= pd.to_datetime(pro_cut)]

    overlap = (d_adr["Fecha entrada"] < end_inclusive) & (d_adr["Fecha salida"] > start_dt)
    d_adr = d_adr.loc[overlap].copy()

    if not d_adr.empty:
        d_adr["Fecha entrada"] = d_adr["Fecha entrada"].dt.normalize()
        d_adr["Fecha salida"]  = d_adr["Fecha salida"].dt.normalize()

        # Expande a noches dentro del periodo y calcula tarifa base por reserva
        rows = []
        for _, r in d_adr.iterrows():
            s = r["Fecha entrada"]; e = r["Fecha salida"]
            # Noches totales de la reserva
            los_total = int((e - s).days)
            if los_total <= 0:
                continue
            # Tarifa base entre semana r = T / [W + S*(1+u)] usando noches del segmento
            seg_start = max(s, start_dt)
            seg_end   = min(e, end_inclusive)
            if seg_end <= seg_start:
                continue
            nights = pd.date_range(seg_start, seg_end - pd.Timedelta(days=1), freq="D")
            dows = nights.weekday  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
            S = int(np.isin(dows, [4, 5]).sum())  # viernes y sábado
            W = int(len(nights) - S)
            T_total = float(pd.to_numeric(r["Alquiler con IVA (€)"], errors="coerce") or 0.0)
            # Proporción del importe para el segmento dentro del periodo
            frac = len(nights) / max(los_total, 1)
            T_in = T_total * frac
            denom = W + S * (1.0 + u)
            if denom <= 0:
                continue
            r_week = T_in / denom
            r_weekend = r_week * (1.0 + u)

            # Genera filas diarias con ADR asignado según DOW
            for d, dow in zip(nights, dows):
                is_weekend = dow in (4, 5)  # vie/sáb
                rows.append({"Fecha": d, "grupo": "Finde" if is_weekend else "Semana", "ADR": r_weekend if is_weekend else r_week})

        daily = pd.DataFrame(rows)
        if daily.empty:
            w1, w2 = st.columns(2)
            w1.metric("ADR entre semana (€)", "—")
            w2.metric("ADR fin de semana (€)", "—")
        else:
            # Promedio ponderado por noches (cada fila = 1 noche)
            adr_week = float(daily.loc[daily["grupo"]=="Semana", "ADR"].mean()) if (daily["grupo"]=="Semana").any() else np.nan
            adr_weekend = float(daily.loc[daily["grupo"]=="Finde", "ADR"].mean()) if (daily["grupo"]=="Finde").any() else np.nan

            w1, w2 = st.columns(2)
            w1.metric("ADR entre semana (€)", f"{adr_week:.2f}" if np.isfinite(adr_week) else "—")
            w2.metric("ADR fin de semana (€)", f"{adr_weekend:.2f}" if np.isfinite(adr_weekend) else "—")
    else:
        w1, w2 = st.columns(2)
        w1.metric("ADR entre semana (€)", "—")
        w2.metric("ADR fin de semana (€)", "—")

    # === Bandas ADR (NOCHES PRORRATEADAS) — 3 filas x 3 columnas ===
    bands_act = _compute_adr_bands_period_prorate(df, pro_start, pro_end, pro_cut)
    bands_ly1 = _compute_adr_bands_period_prorate(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        pd.to_datetime(pro_cut) - pd.DateOffset(years=1),
    )
    bands_ly2 = _compute_adr_bands_period_prorate(
        df,
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        pd.to_datetime(pro_cut) - pd.DateOffset(years=2),
    )

    adr_bandas_tbl = pd.DataFrame(
        [bands_act.round(2), bands_ly1.round(2), bands_ly2.round(2)],
        index=["Act", "LY-1", "LY-2"],
    )[["P10", "Mediana", "P90"]]

    st.dataframe(adr_bandas_tbl, use_container_width=True)
    st.download_button(
        "📥 Descargar bandas ADR (CSV)",
        data=adr_bandas_tbl.reset_index(names=["Periodo"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="adr_bandas_prorrateadas_act_ly1_ly2.csv",
        mime="text/csv",
    )

    # ====== Ocupación ======
    st.subheader("🏨 Ocupación (periodo seleccionado)")
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Ocupación actual", f"{tot_now['ocupacion_pct']:.2f}%")
    o2.metric("Ocupación LY (a este corte)", f"{tot_ly_cut['ocupacion_pct']:.2f}%")
    o3.metric("Ocupación LY final", f"{tot_ly_final['ocupacion_pct']:.2f}%")
    o4.metric("Ocupación LY-2 (a este corte)", f"{tot_ly2_cut_ing['ocupacion_pct']:.2f}%")
    o5.metric("Ocupación LY-2 final", f"{tot_ly2_final_ing['ocupacion_pct']:.2f}%")
    st.caption("Actual, LY y LY-2: reservas con Fecha alta ≤ corte. “LY final” y “LY-2 final”: corte = fin del periodo correspondiente.")

    # ====== Ritmo de reservas (Pace) ======
    st.subheader("🏁 Ritmo de reservas (Pace)")
    try:
        pace_res = pace_forecast_month(
            df=df,
            cutoff=pd.to_datetime(pro_cut),
            period_start=pd.to_datetime(pro_start),
            period_end=pd.to_datetime(pro_end),
            ref_years=int(ref_years_pro),
            dmax=180,
            props=props_pro if props_pro else None,
            inv_override=int(inv_pro) if inv_pro > 0 else None,
        ) or {}
    except Exception:
        pace_res = {}
        st.caption("No se pudo calcular Pace (se continúa con KPIs actuales).")

    # Métricas rápidas de Pace
    n_otb = float(pace_res.get("nights_otb", 0.0))
    n_p50 = float(pace_res.get("nights_p50", 0.0))
    rev_p50 = float(pace_res.get("revenue_final_p50", 0.0))
    pick_typ50 = float(pace_res.get("pickup_typ_p50", 0.0))
    adr_tail_p50 = float(pace_res.get("adr_tail_p50", np.nan)) if pace_res else np.nan

    p1, p2, p3 = st.columns(3)
    p1.metric("OTB noches", f"{n_otb:,.0f}".replace(",",".")) 
    p2.metric("Forecast Noches (P50)", f"{n_p50:,.0f}".replace(",",".")) 
    p3.metric("Forecast Ingresos (P50)", f"{rev_p50:,.2f}")
    st.caption(f"Pickup típico (P50): {pick_typ50:,.0f} · ADR tail (P50): {adr_tail_p50:,.2f}".replace(",","."))

    # ====== Pace (YoY) – Noches confirmadas por D ======
    st.subheader("📉 Pace (YoY) – Noches confirmadas por D")
    dmax_y = 180
    p_start_ly = pd.to_datetime(pro_start) - pd.DateOffset(years=1)
    p_end_ly   = pd.to_datetime(pro_end) - pd.DateOffset(years=1)
    base_cur = pace_series(
        df=df,
        period_start=pd.to_datetime(pro_start),
        period_end=pd.to_datetime(pro_end),
        d_max=int(dmax_y),
        props=props_pro if props_pro else None,
        inv_override=int(inv_pro) if inv_pro > 0 else None,
    )
    base_ly = pace_series(
        df=df,
        period_start=p_start_ly,
        period_end=p_end_ly,
        d_max=int(dmax_y),
        props=props_pro if props_pro else None,
        inv_override=int(inv_pro_ly) if inv_pro_ly > 0 else None,
    )
    if base_cur.empty or base_ly.empty:
        st.info("No hay datos suficientes para calcular Pace YoY en el periodo.")
    else:
        D_all = list(range(0, int(max(base_cur["D"].max(), base_ly["D"].max())) + 1))
        df_plot = pd.DataFrame({"D": D_all})
        df_plot = df_plot.merge(base_cur[["D","noches"]].rename(columns={"noches":"Actual"}), on="D", how="left")
        df_plot = df_plot.merge(base_ly[["D","noches"]].rename(columns={"noches":"LY"}), on="D", how="left")
        df_plot = df_plot.fillna(0.0)
        df_long = df_plot.melt(id_vars=["D"], value_vars=["Actual","LY"], var_name="Serie", value_name="Noches")
        pace_colors = {"Actual": "#1f77b4", "LY": "#9e9e9e"}
        base = alt.Chart(df_long).encode(x=alt.X("D:Q", title="Días antes de la estancia"))
        pace_line = base.mark_line(strokeWidth=2).encode(
            y=alt.Y("Noches:Q", title="Noches confirmadas"),
            color=alt.Color("Serie:N",
                            scale=alt.Scale(domain=list(pace_colors.keys()), range=[pace_colors[k] for k in pace_colors]), title=None),
            strokeDash=alt.condition("datum.Serie == 'LY'", alt.value([5,3]), alt.value([0,0])),
            opacity=alt.condition("datum.Serie == 'LY'", alt.value(0.85), alt.value(1.0)),
            tooltip=[alt.Tooltip("D:Q", title="D"), alt.Tooltip("Serie:N"), alt.Tooltip("Noches:Q", title="Valor", format=",.0f")],
        )
        pace_pts = base.mark_circle(size=55).encode(
            y="Noches:Q",
            color=alt.Color("Serie:N",
                            scale=alt.Scale(domain=list(pace_colors.keys()), range=[pace_colors[k] for k in pace_colors]), title=None),
            tooltip=[alt.Tooltip("D:Q", title="D"), alt.Tooltip("Serie:N"), alt.Tooltip("Noches:Q", title="Valor", format=",.0f")],
        )
        st.altair_chart((pace_line + pace_pts).properties(height=300).interactive(bind_y=False), use_container_width=True)

    # ====== Evolución por fecha de corte: Ocupación y ADR ======
    st.subheader("📈 Evolución por fecha de corte: Ocupación (izq) y ADR (dcha)")
    with st.expander("Ver evolución", expanded=True):
        evo_cut_start = st.date_input(
            "Inicio de corte", value=pd.to_datetime(pro_cut).date().replace(day=1), key="evo_cut_start_pro"
        )
        evo_cut_end   = st.date_input("Fin de corte", value=pd.to_datetime(pro_cut).date(), key="evo_cut_end_pro")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=int(inv_pro), step=1, key="inv_evo_pro")
        run_evo = st.button("Calcular evolución (Ocupación y ADR)", type="primary", key="btn_evo_pro")

        if run_evo:
            cstart = pd.to_datetime(evo_cut_start); cend = pd.to_datetime(evo_cut_end)
            if cstart > cend:
                st.error("El inicio del rango de corte no puede ser posterior al fin.")
            else:
                rows = []
                for c in pd.date_range(cstart, cend, freq="D"):
                    _, tot_now_e = compute_kpis(
                        df_all=df,
                        cutoff=c,
                        period_start=pd.to_datetime(pro_start),
                        period_end=pd.to_datetime(pro_end),
                        inventory_override=int(inv_e) if inv_e > 0 else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    _, tot_ly_e = compute_kpis(
                        df_all=df,
                        cutoff=c - pd.DateOffset(years=1),
                        period_start=pd.to_datetime(pro_start) - pd.DateOffset(years=1),
                        period_end=pd.to_datetime(pro_end) - pd.DateOffset(years=1),
                        inventory_override=int(inv_pro_ly) if (isinstance(inv_pro_ly, int) and inv_pro_ly > 0) else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    rows.append({
                        "Corte": c.normalize(),  # <- normaliza fecha (sin hora)
                        "occ_now": float(tot_now_e["ocupacion_pct"]),
                        "adr_now": float(tot_now_e["adr"]),
                        "occ_ly": float(tot_ly_e["ocupacion_pct"]),
                        "adr_ly": float(tot_ly_e["adr"]),
                    })
                evo_df = pd.DataFrame(rows)
                if evo_df.empty:
                    st.info("Sin datos en el rango seleccionado.")
                else:
                    # Asegura dtype datetime sin tz
                    evo_df["Corte"] = pd.to_datetime(evo_df["Corte"]).dt.tz_localize(None)

                    occ_long = evo_df.melt(id_vars=["Corte"], value_vars=["occ_now","occ_ly"],
                                           var_name="serie", value_name="valor")
                    occ_long["serie"] = occ_long["serie"].map({"occ_now": "Ocupación actual", "occ_ly": "Ocupación LY"})
                    adr_long = evo_df.melt(id_vars=["Corte"], value_vars=["adr_now","adr_ly"],
                                           var_name="serie", value_name="valor")
                    adr_long["serie"] = adr_long["serie"].map({"adr_now": "ADR actual (€)", "adr_ly": "ADR LY (€)"})

                    occ_colors = {"Ocupación actual": "#1f77b4", "Ocupación LY": "#6baed6"}
                    adr_colors = {"ADR actual (€)": "#ff7f0e", "ADR LY (€)": "#fdae6b"}

                    # Líneas + puntos SIEMPRE visibles (sin hover), sin interpolación para no “curvar”
                    occ_line = (
                        alt.Chart(occ_long)
                        .mark_line(strokeWidth=2)  # linear (por defecto)
                        .encode(
                            x=alt.X("Corte:T", title="Fecha de corte"),
                            y=alt.Y("valor:Q", axis=alt.Axis(orient="left", title="Ocupación %", tickCount=6, format=".0f")),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None),
                            strokeDash=alt.condition("datum.serie == 'Ocupación LY'", alt.value([5,3]), alt.value([0,0])),
                            opacity=alt.condition("datum.serie == 'Ocupación LY'", alt.value(0.7), alt.value(1.0)),
                        )
                    )
                    occ_points = (
                        alt.Chart(occ_long)
                        .mark_circle(size=60, filled=True)
                        .encode(
                            x="Corte:T",
                            y=alt.Y("valor:Q", axis=None),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None, legend=None),
                            tooltip=[alt.Tooltip("Corte:T", title="Día"),
                                     alt.Tooltip("serie:N", title="Serie"),
                                     alt.Tooltip("valor:Q", title="Valor", format=".2f")],
                        )
                    )

                    adr_line = (
                        alt.Chart(adr_long)
                        .mark_line(strokeWidth=2)  # linear
                        .encode(
                            x=alt.X("Corte:T"),
                            y=alt.Y("valor:Q", axis=alt.Axis(orient="right", title="ADR (€)", tickCount=6, format=",.2f")),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None),
                            strokeDash=alt.condition("datum.serie == 'ADR LY (€)'", alt.value([5,3]), alt.value([0,0])),
                            opacity=alt.condition("datum.serie == 'ADR LY (€)'", alt.value(0.7), alt.value(1.0)),
                        )
                    )
                    adr_points = (
                        alt.Chart(adr_long)
                        .mark_circle(size=60, filled=True)
                        .encode(
                            x="Corte:T",
                            y=alt.Y("valor:Q", axis=None),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None, legend=None),
                            tooltip=[alt.Tooltip("Corte:T", title="Día"),
                                     alt.Tooltip("serie:N", title="Serie"),
                                     alt.Tooltip("valor:Q", title="Valor", format=",.2f")],
                        )
                    )

                    chart = (
                        alt.layer(occ_line, occ_points, adr_line, adr_points)
                        .resolve_scale(y="independent", color="independent")
                        .properties(height=380)
                    )
                    st.altair_chart(chart, use_container_width=True)

    # ====== Semáforos y análisis ======
    st.subheader("🚦 Semáforos y análisis")
    tech_block = _kai_cdm_pro_analysis(
        tot_now=tot_now,
        tot_ly_cut=tot_ly_cut,
        tot_ly_final=tot_ly_final,
        pace=pace_res,
        price_ref_p50=None,
    )
    st.markdown(tech_block)

    # ====== Explicación ejecutiva (narrada) ======
    st.subheader("🧠 Explicación ejecutiva (narrada)")
    from utils import pro_exec_summary
    exec_blocks = pro_exec_summary(tot_now, tot_ly_cut, tot_ly_final, pace_res)
    st.markdown(exec_blocks["headline"])
    with st.expander("Ver análisis detallado", expanded=False):
        st.markdown(exec_blocks["detail"])

def _find_and_read_forecast_db() -> pd.DataFrame:
    """Busca y lee data/forecast_db.csv probando varios paths, encodings y separadores."""
    mod_dir = Path(__file__).resolve().parent
    candidates = [
        mod_dir / "data" / "forecast_db.csv",
        Path.cwd() / "data" / "forecast_db.csv",
    ]
    # incluye cualquier fichero con ese nombre dentro del módulo
    candidates += [p for p in mod_dir.rglob("forecast_db.csv")]
    seen = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if not p.exists():
            continue
        for enc in ("cp1252", "latin-1", "utf-8", "utf-8-sig"):
            for sep in (";", ","):
                try:
                    df = pd.read_csv(p, sep=sep, engine="python", encoding=enc, dtype=str)
                    if not df.empty:
                        return df
                except Exception:
                    continue
            # intento sin forzar separador
            try:
                df = pd.read_csv(p, engine="python", encoding=enc, dtype=str)
                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame()