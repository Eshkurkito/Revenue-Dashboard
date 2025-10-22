import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import re
from datetime import date, timedelta

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm = {c: str(c).strip().lower() for c in df.columns}
    def find(*keys):
        for col, n in norm.items():
            for k in keys:
                if n == k or k in n:
                    return col
        return None
    mapping = {}
    a  = find("alojamiento","propiedad","listing","unidad","apartamento","room","unit")
    fa = find("fecha alta","creado","created","booking","reserva","created at","creation")
    if a:  mapping[a] = "Alojamiento"
    if fa: mapping[fa] = "Fecha alta"
    return df.rename(columns=mapping) if mapping else df

@st.cache_data(ttl=1800, show_spinner=False)
def _daily_bookings(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Cuenta reservas por día (Fecha alta) entre start y end (incl.)."""
    dfx = df.dropna(subset=["Fecha alta"]).copy()
    dfx["Fecha alta"] = pd.to_datetime(dfx["Fecha alta"]).dt.normalize()
    s = pd.to_datetime(start).normalize()
    e = (pd.to_datetime(end)).normalize()
    dfx = dfx[(dfx["Fecha alta"] >= s) & (dfx["Fecha alta"] <= e)]
    if dfx.empty:
        rng = pd.date_range(s, e, freq="D")
        return pd.DataFrame({"Fecha": rng, "Reservas": np.zeros(len(rng), dtype=int)})
    series = dfx.groupby("Fecha alta").size().rename("Reservas")
    ser_full = series.reindex(pd.date_range(s, e, freq="D"), fill_value=0)
    return ser_full.reset_index(names=["Fecha"])

def _align_for_plot(df: pd.DataFrame, shift_years: int, label: str) -> pd.DataFrame:
    out = df.copy()
    out["Serie"] = label
    if shift_years != 0:
        out["FechaPlot"] = (pd.to_datetime(out["Fecha"]) + pd.DateOffset(years=shift_years))
    else:
        out["FechaPlot"] = pd.to_datetime(out["Fecha"])
    return out

def render_reservas_por_dia(raw: pd.DataFrame | None = None):
    st.header("Reservas recibidas por día (Fecha alta)")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("No hay datos cargados. Sube un archivo en la barra lateral.")
        return

    df = _standardize_columns(raw.copy())

    # --- Selección/normalización de 'Fecha alta' ---
    if "Fecha alta" not in df.columns:
        # sugerir columnas candidatas
        cols = list(df.columns)
        cand = [c for c in cols if re.search(r"(fecha|alta|crea|book|reserva|created)", str(c).lower())]
        sel = st.selectbox("Selecciona la columna de 'Fecha alta'", cand, key="rpd_col_fecha_alta")
        if not sel:
            st.warning("Selecciona la columna que contiene la Fecha alta de la reserva.")
            return
        df = df.rename(columns={sel: "Fecha alta"})

    # Asegura tipo datetime
    try:
        df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    except Exception:
        df["Fecha alta"] = pd.to_datetime(df["Fecha alta"].astype(str), errors="coerce")

    if df["Fecha alta"].isna().all():
        st.warning("No se pudieron interpretar las fechas de ‘Fecha alta’.")
        return

    # Parámetros (sidebar)
    with st.sidebar:
        st.subheader("Parámetros · Reservas por día")
        today = date.today()
        default_start = today - timedelta(days=60)
        start = st.date_input("Inicio (Fecha alta)", value=default_start, key="rpd_start")
        end   = st.date_input("Fin (Fecha alta)",    value=today,         key="rpd_end")
        if start > end:
            st.error("La fecha de inicio no puede ser posterior a la de fin.")
            return

        if "Alojamiento" in df.columns:
            props = st.multiselect(
                "Alojamientos (opcional)",
                sorted(df["Alojamiento"].astype(str).dropna().unique()),
                key="rpd_props",
            )
        else:
            props = []

        compare_ly1 = st.checkbox("Comparar con LY-1", value=True, key="rpd_cmp_ly1")
        compare_ly2 = st.checkbox("Comparar con LY-2", value=True, key="rpd_cmp_ly2")

        # ← nuevo: botón para disparar el cálculo
        run_rpd = st.button("Generar", type="primary", use_container_width=True, key="rpd_run")

    # Si no se pulsa, no hacemos cálculos
    if not run_rpd:
        st.info("Elige fechas y pulsa Generar.")
        return

    # Filtro por alojamientos (solo cuando se pulsa Generar)
    if props and "Alojamiento" in df.columns:
        df = df[df["Alojamiento"].astype(str).isin(props)].copy()

    # Series
    act = _daily_bookings(df, start, end)
    ly1 = _daily_bookings(df, pd.to_datetime(start) - pd.DateOffset(years=1),
                             pd.to_datetime(end)   - pd.DateOffset(years=1)) if compare_ly1 else pd.DataFrame()
    ly2 = _daily_bookings(df, pd.to_datetime(start) - pd.DateOffset(years=2),
                             pd.to_datetime(end)   - pd.DateOffset(years=2)) if compare_ly2 else pd.DataFrame()

    if act.empty and (ly1.empty and ly2.empty):
        st.info("No hay reservas en el rango seleccionado.")
        return

    # Alinear para superponer en el eje de fechas del año actual
    act_p = _align_for_plot(act, 0, "Act")
    ly1_p = _align_for_plot(ly1, 1, "LY-1") if not ly1.empty else pd.DataFrame()
    ly2_p = _align_for_plot(ly2, 2, "LY-2") if not ly2.empty else pd.DataFrame()
    plot_df = pd.concat([x for x in [act_p, ly1_p, ly2_p] if not x.empty], ignore_index=True)
    plot_df["FechaPlot"] = pd.to_datetime(plot_df["FechaPlot"])

    # Tabla resumen
    totals = (
        plot_df.groupby("Serie")["Reservas"]
        .sum()
        .reindex(["Act","LY-1","LY-2"])
        .dropna()
        .rename("Total reservas")
        .reset_index()
    )
    st.dataframe(totals, use_container_width=True)

    # Gráfico
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("FechaPlot:T", title="Fecha (alineada)"),
            y=alt.Y("sum(Reservas):Q", title="Reservas por día"),
            color=alt.Color("Serie:N", title="Serie"),
            tooltip=[
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Fecha:T", title="Fecha real"),
                alt.Tooltip("Reservas:Q", title="Reservas", aggregate="sum"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # Descarga
    export = plot_df[["Serie","Fecha","Reservas"]].copy()
    export["Fecha"] = pd.to_datetime(export["Fecha"]).dt.date
    st.download_button(
        "📥 Descargar series (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="reservas_por_dia_act_ly1_ly2.csv",
        mime="text/csv",
    )