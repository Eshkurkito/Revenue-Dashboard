import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta

# Recupera el DF si el router no lo pasó
def _get_raw_session():
    return st.session_state.get("df_active") or st.session_state.get("raw")

def _ensure_fecha_alta(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve una Serie datetime con la columna 'Fecha alta'.
    - Si hay columnas duplicadas con el mismo nombre, usa la primera.
    - Intenta parseo flexible: texto (dayfirst) y serial Excel.
    """
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns]")

    # Tomar la primera columna que se llame exactamente 'Fecha alta'
    mask = (df.columns.astype(str) == "Fecha alta")
    if mask.sum() == 0:
        raise KeyError("El archivo no contiene la columna 'Fecha alta'.")
    serie = df.loc[:, mask].iloc[:, 0] if mask.sum() > 1 else df["Fecha alta"]

    # Intento 1: parseo normal
    s1 = pd.to_datetime(serie, errors="coerce", dayfirst=True, infer_datetime_format=True)

    # Intento 2: si casi todo quedó NaT, prueba como serial Excel
    if s1.isna().mean() > 0.8:
        s_num = pd.to_numeric(serie, errors="coerce")
        s2 = pd.to_datetime(s_num, unit="d", origin="1899-12-30", errors="coerce")
        # Elige el que tenga más válidos
        if s2.notna().sum() > s1.notna().sum():
            s1 = s2

    # Normaliza a medianoche
    return pd.to_datetime(s1.dt.normalize())

@st.cache_data(ttl=1800, show_spinner=False)
def _daily_bookings_from_series(fechas: pd.Series, start: date, end: date, _v: int = 2) -> pd.DataFrame:
    """Cuenta reservas por día (Fecha alta ya parseada) en [start, end]."""
    if fechas is None or fechas.empty:
        return pd.DataFrame(columns=["Fecha", "Reservas"])
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    ser = fechas[(fechas >= s) & (fechas <= e)]
    rng = pd.date_range(s, e, freq="D")
    if ser.empty:
        return pd.DataFrame({"Fecha": rng, "Reservas": np.zeros(len(rng), dtype=int)})
    counts = ser.value_counts().rename_axis("Fecha").sort_index()
    counts = counts.reindex(rng, fill_value=0).rename("Reservas")
    # Asegura nombres de columnas correctos en todas las versiones de pandas
    try:
        dfc = counts.reset_index(names=["Fecha"])
    except TypeError:
        dfc = counts.reset_index().rename(columns={"index": "Fecha"})
    return dfc

def _align_for_plot(df: pd.DataFrame, shift_years: int, label: str) -> pd.DataFrame:
    out = df.copy()
    out["Serie"] = label
    out["FechaPlot"] = pd.to_datetime(out["Fecha"]) + pd.DateOffset(years=shift_years)
    return out

def render_reservas_por_dia(raw: pd.DataFrame | None = None):
    st.header("Reservas recibidas por día (Fecha alta)")

    if raw is None:
        raw = _get_raw_session()
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("Sube un archivo en la barra lateral para continuar.")
        return

    # Asegura 'Fecha alta'
    try:
        fa = _ensure_fecha_alta(raw)
    except KeyError as e:
        st.error(str(e))
        return

    if fa.isna().all():
        st.warning("No se pudieron interpretar las fechas de 'Fecha alta'.")
        return

    # Parámetros en sidebar
    with st.sidebar:
        st.subheader("Parámetros · Reservas por día")
        today = date.today()
        default_end = today
        default_start = default_end - timedelta(days=60)
        start = st.date_input("Inicio (Fecha alta)", value=default_start, key="rpd_start")
        end = st.date_input("Fin (Fecha alta)", value=default_end, key="rpd_end")
        compare_ly1 = st.checkbox("Comparar con LY-1", value=True, key="rpd_cmp_ly1")
        compare_ly2 = st.checkbox("Comparar con LY-2", value=True, key="rpd_cmp_ly2")
        run = st.button("Generar", type="primary", use_container_width=True, key="rpd_run")

    if start > end:
        st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        return

    if not run:
        st.info("Elige fechas y pulsa Generar.")
        return

    # Series
    act = _daily_bookings_from_series(fa, start, end)
    ly1 = _daily_bookings_from_series(fa, pd.to_datetime(start) - pd.DateOffset(years=1),
                                         pd.to_datetime(end) - pd.DateOffset(years=1)) if compare_ly1 else pd.DataFrame()
    ly2 = _daily_bookings_from_series(fa, pd.to_datetime(start) - pd.DateOffset(years=2),
                                         pd.to_datetime(end) - pd.DateOffset(years=2)) if compare_ly2 else pd.DataFrame()

    if act.empty and (ly1.empty and ly2.empty):
        st.info("No hay reservas en el rango seleccionado.")
        return

    # Totales
    totals = []
    totals.append({"Serie": "Act", "Total reservas": int(act["Reservas"].sum())})
    if not ly1.empty:
        totals.append({"Serie": "LY-1", "Total reservas": int(ly1["Reservas"].sum())})
    if not ly2.empty:
        totals.append({"Serie": "LY-2", "Total reservas": int(ly2["Reservas"].sum())})
    st.dataframe(pd.DataFrame(totals), use_container_width=True)

    # Gráfico
    act_p = _align_for_plot(act, 0, "Act")
    frames = [act_p]
    if not ly1.empty:
        frames.append(_align_for_plot(ly1, 1, "LY-1"))
    if not ly2.empty:
        frames.append(_align_for_plot(ly2, 2, "LY-2"))
    plot_df = pd.concat(frames, ignore_index=True)

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("FechaPlot:T", title="Fecha (alineada)"),
            y=alt.Y("Reservas:Q", title="Reservas por día"),
            color=alt.Color("Serie:N", title="Serie"),
            tooltip=[
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Fecha:T", title="Fecha real"),
                alt.Tooltip("Reservas:Q", title="Reservas"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # ===================== NUEVO: Patrones históricos =====================
    st.subheader("🔍 Patrones históricos (mismo periodo en todos los años)")

    def _doy_range(s_d: int, e_d: int) -> list[int]:
        # Excluye 366 (evita problemas de años no bisiestos)
        if s_d <= e_d:
            doys = list(range(s_d, e_d + 1))
        else:
            doys = list(range(s_d, 366)) + list(range(1, e_d + 1))
        return [d for d in doys if d <= 365]

    # Dataset completo de fechas de alta (no solo el rango actual)
    df_all = pd.DataFrame({"Fecha": fa.dropna().dt.normalize()})
    if df_all.empty:
        st.info("Sin fechas de alta para analizar patrones.")
        return
    df_all["Año"] = df_all["Fecha"].dt.year
    df_all["DOY"] = df_all["Fecha"].dt.dayofyear.clip(upper=365)  # 366 -> 365
    years = sorted(df_all["Año"].unique().tolist())

    s_doy = pd.to_datetime(start).dayofyear
    e_doy = pd.to_datetime(end).dayofyear
    window_doys = _doy_range(s_doy, e_doy)

    df_win = df_all[df_all["DOY"].isin(window_doys)].copy()
    if df_win.empty:
        st.info("No hay reservas históricas en este periodo para detectar patrones.")
    else:
        # Matriz Año x DOY con conteos diarios
        mat = (
            df_win.groupby(["Año", "DOY"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=years, columns=window_doys, fill_value=0)
        )

        avg = mat.mean(axis=0)  # media por DOY
        std = mat.std(axis=0)
        n_years = float(len(years))
        avg_df = (
            pd.DataFrame({"DOY": avg.index.astype(int), "Media": avg.values, "Std": std.values})
            .sort_values("DOY")
            .reset_index(drop=True)
        )
        # Fecha "de ejemplo" en el año actual para el eje X
        base_year = date.today().year
        avg_df["FechaPlot"] = pd.to_datetime(base_year * 1000 + avg_df["DOY"], format="%Y%j")
        # Suavizado 7D
        avg_df["Media7D"] = avg_df["Media"].rolling(7, center=True, min_periods=1).mean()
        # NUEVO: constante para tooltip
        avg_df["Anios"] = int(n_years)

        # Gráfico perfil medio
        c1 = alt.Chart(avg_df).mark_line(color="#2e485f").encode(
            x=alt.X("FechaPlot:T", title="Fecha (alineada al año actual)"),
            y=alt.Y("Media:Q", title="Reservas medias por día"),
            tooltip=[
                alt.Tooltip("FechaPlot:T", title="Fecha"),
                alt.Tooltip("Media:Q", format=".2f"),
                alt.Tooltip("Std:Q", format=".2f", title="Desv. típica"),
                alt.Tooltip("Anios:Q", title="Años considerados"),  # ← reemplaza value=
            ],
        )
        c2 = alt.Chart(avg_df).mark_line(color="#7aa6d9").encode(
            x="FechaPlot:T", y=alt.Y("Media7D:Q", title=""),
        )
        st.altair_chart((c1 + c2).properties(height=280), use_container_width=True)

        # Top 10 fechas con mayor media de reservas
        top = avg_df.nlargest(10, "Media").copy()
        top["Fecha"] = top["FechaPlot"].dt.strftime("%d-%b")
        top_tbl = top[["Fecha", "Media", "Std"]].rename(columns={"Media": "Media reservas", "Std": "Desv. típica"})
        st.write("Top 10 fechas recurrentes con más reservas (media histórica en el periodo):")
        st.dataframe(top_tbl.round(2), use_container_width=True)

        # Totales por año en el mismo periodo
        tot_year = (
            df_win.groupby("Año")
            .size()
            .reindex(years, fill_value=0)
            .rename("Reservas")
            .reset_index()
        )
        bar = alt.Chart(tot_year).mark_bar(color="#2e485f").encode(
            x=alt.X("Año:O", title="Año"),
            y=alt.Y("Reservas:Q", title="Total reservas (periodo)"),
            tooltip=["Año:O", "Reservas:Q"],
        ).properties(height=220)
        st.altair_chart(bar, use_container_width=True)

    # ===================== FIN Patrones históricos =====================

    # Descarga
    export = plot_df[["Serie", "Fecha", "Reservas"]].copy()
    export["Fecha"] = pd.to_datetime(export["Fecha"]).dt.date
    st.download_button(
        "📥 Descargar series (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="reservas_por_dia_act_ly1_ly2.csv",
        mime="text/csv",
    )