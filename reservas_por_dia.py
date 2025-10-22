import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from pathlib import Path

# --- Filtros de grupos (utils + fallback) ---
try:
    from utils import load_groups  # dict[str, list[str]]
except Exception:
    def _read_csv_any(path: Path):
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return None
    def load_groups() -> dict[str, list[str]]:
        p = Path(__file__).resolve().parent / "grupos_guardados.csv"
        if not p.exists():
            return {}
        df = _read_csv_any(p)
        if df is None or df.empty:
            return {}
        cols = {c.lower(): c for c in df.columns}
        if not {"grupo", "alojamiento"}.issubset(cols.keys()):
            return {}
        gcol, pcol = cols["grupo"], cols["alojamiento"]
        d = (
            df.dropna(subset=[gcol, pcol])
              .astype({gcol: str, pcol: str})
              .groupby(gcol)[pcol].apply(list).to_dict()
        )
        return {g: list(dict.fromkeys(v)) for g, v in d.items()}

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

    mask = (df.columns.astype(str) == "Fecha alta")
    if mask.sum() == 0:
        raise KeyError("El archivo no contiene la columna 'Fecha alta'.")
    serie = df.loc[:, mask].iloc[:, 0] if mask.sum() > 1 else df["Fecha alta"]

    s1 = pd.to_datetime(serie, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if s1.isna().mean() > 0.8:
        s_num = pd.to_numeric(serie, errors="coerce")
        s2 = pd.to_datetime(s_num, unit="d", origin="1899-12-30", errors="coerce")
        if s2.notna().sum() > s1.notna().sum():
            s1 = s2

    return pd.to_datetime(s1.dt.normalize())

def _ensure_fecha_entrada(df: pd.DataFrame) -> pd.Series:
    """Serie datetime con 'Fecha entrada' (dup names y serial Excel soportados)."""
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns]")
    mask = (df.columns.astype(str) == "Fecha entrada")
    if mask.sum() == 0:
        raise KeyError("El archivo no contiene la columna 'Fecha entrada'.")
    serie = df.loc[:, mask].iloc[:, 0] if mask.sum() > 1 else df["Fecha entrada"]
    s1 = pd.to_datetime(serie, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if s1.isna().mean() > 0.8:
        s_num = pd.to_numeric(serie, errors="coerce")
        s2 = pd.to_datetime(s_num, unit="d", origin="1899-12-30", errors="coerce")
        if s2.notna().sum() > s1.notna().sum():
            s1 = s2
    return pd.to_datetime(s1.dt.normalize())

@st.cache_data(ttl=1800, show_spinner=False)
def _daily_bookings_from_series(fechas: pd.Series, start: date, end: date, _v: int = 3) -> pd.DataFrame:
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

    # --------- Sidebar: filtros y parámetros ----------
    with st.sidebar:
        st.subheader("Parámetros · Reservas por día")

        # Filtro por grupo y/o alojamiento
        props_selected: list[str] = []
        if "Alojamiento" in raw.columns:
            all_props = sorted(raw["Alojamiento"].astype(str).dropna().unique().tolist())
            groups = load_groups()
            group_names = ["Ninguno"] + sorted(groups.keys())
            gsel = st.selectbox("Grupo guardado (opcional)", group_names, index=0, key="rpd_group_sel")
            if gsel != "Ninguno":
                props_selected = groups.get(gsel, [])
                st.caption(f"{len(props_selected)} alojamientos en el grupo seleccionado.")
            props_selected = st.multiselect(
                "Alojamientos (opcional)",
                options=all_props,
                default=props_selected,
                key="rpd_props",
            )

        today = date.today()
        default_end = today
        default_start = default_end - timedelta(days=60)
        start = st.date_input("Inicio (Fecha alta)", value=default_start, key="rpd_start")
        end = st.date_input("Fin (Fecha alta)", value=default_end, key="rpd_end")

        # Filtro por periodo de estancia (Fecha entrada)
        use_stay = st.checkbox("Filtrar por periodo de estancia (Fecha entrada)", value=False, key="rpd_use_stay")
        if use_stay:
            stay_start = st.date_input("Estancia desde (Fecha entrada)", value=today.replace(day=1), key="rpd_stay_start")
            stay_end   = st.date_input("Estancia hasta (Fecha entrada)", value=default_end, key="rpd_stay_end")
        else:
            stay_start, stay_end = None, None

        compare_ly1 = st.checkbox("Comparar con LY-1", value=True, key="rpd_cmp_ly1")
        compare_ly2 = st.checkbox("Comparar con LY-2", value=True, key="rpd_cmp_ly2")
        run = st.button("Generar", type="primary", use_container_width=True, key="rpd_run")

    if start > end:
        st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        return
    if use_stay and (pd.to_datetime(stay_start) > pd.to_datetime(stay_end)):
        st.error("La fecha de estancia 'desde' no puede ser posterior a 'hasta'.")
        return
    if not run:
        st.info("Elige filtros y fechas y pulsa Generar.")
        return

    # Aplica filtro por alojamientos antes de construir series
    df_work = raw.copy()
    if props_selected and "Alojamiento" in df_work.columns:
        df_work = df_work[df_work["Alojamiento"].astype(str).isin(props_selected)].copy()

    # Asegura 'Fecha alta' y, si aplica, 'Fecha entrada'
    try:
        fa = _ensure_fecha_alta(df_work)
    except KeyError as e:
        st.error(str(e))
        return
    if fa.isna().all():
        st.warning("No se pudieron interpretar las fechas de 'Fecha alta'.")
        return

    # Series filtradas por periodo de estancia (si aplica)
    if use_stay:
        try:
            fe_all = _ensure_fecha_entrada(df_work)
        except KeyError as e:
            st.error(str(e))
            return
        s0 = pd.to_datetime(stay_start).normalize()
        e0 = pd.to_datetime(stay_end).normalize()
        m_act = fe_all.notna() & (fe_all >= s0) & (fe_all <= e0)
        m_ly1 = fe_all.notna() & (fe_all >= (s0 - pd.DateOffset(years=1))) & (fe_all <= (e0 - pd.DateOffset(years=1)))
        m_ly2 = fe_all.notna() & (fe_all >= (s0 - pd.DateOffset(years=2))) & (fe_all <= (e0 - pd.DateOffset(years=2)))
        fa_act = fa[m_act].dropna()
        fa_ly1 = fa[m_ly1].dropna()
        fa_ly2 = fa[m_ly2].dropna()
    else:
        fa_act, fa_ly1, fa_ly2 = fa, fa, fa

    # Series (Fecha alta en el rango [start,end])
    act = _daily_bookings_from_series(fa_act, start, end)
    ly1 = _daily_bookings_from_series(fa_ly1, pd.to_datetime(start) - pd.DateOffset(years=1),
                                     pd.to_datetime(end)   - pd.DateOffset(years=1)) if compare_ly1 else pd.DataFrame()
    ly2 = _daily_bookings_from_series(fa_ly2, pd.to_datetime(start) - pd.DateOffset(years=2),
                                     pd.to_datetime(end)   - pd.DateOffset(years=2)) if compare_ly2 else pd.DataFrame()

    if act.empty and (ly1.empty and ly2.empty):
        st.info("No hay reservas en el rango seleccionado.")
        return

    # Totales
    totals = [{"Serie": "Act", "Total reservas": int(act["Reservas"].sum())}]
    if not ly1.empty:
        totals.append({"Serie": "LY-1", "Total reservas": int(ly1["Reservas"].sum())})
    if not ly2.empty:
        totals.append({"Serie": "LY-2", "Total reservas": int(ly2["Reservas"].sum())})
    st.dataframe(pd.DataFrame(totals), use_container_width=True)

    # Gráfico líneas
    frames = [_align_for_plot(act, 0, "Act")]
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

    # ===================== Patrones históricos (respetando filtros) =====================
    st.subheader("🔍 Patrones históricos (mismo periodo en todos los años)")

    df_all = pd.DataFrame({"FechaAlta": fa.dropna().dt.normalize()})
    if df_all.empty:
        st.info("Sin fechas de alta para analizar patrones.")
        return

    if use_stay:
        try:
            fe_all = _ensure_fecha_entrada(df_work)
        except Exception:
            fe_all = None
        if fe_all is not None and not fe_all.dropna().empty:
            def _doy_range(s_d: int, e_d: int) -> list[int]:
                if s_d <= e_d:
                    doys = list(range(s_d, e_d + 1))
                else:
                    doys = list(range(s_d, 366)) + list(range(1, e_d + 1))
                return [d for d in doys if d <= 365]
            stay_s_doy = pd.to_datetime(stay_start).dayofyear
            stay_e_doy = pd.to_datetime(stay_end).dayofyear
            stay_doys = set(_doy_range(stay_s_doy, stay_e_doy))
            tmp = pd.DataFrame({"Alta": fa, "Entrada": fe_all}).dropna(subset=["Alta", "Entrada"])
            tmp["EntradaDOY"] = tmp["Entrada"].dt.dayofyear.clip(upper=365)
            tmp = tmp[tmp["EntradaDOY"].isin(stay_doys)]
            df_all = pd.DataFrame({"FechaAlta": tmp["Alta"].dt.normalize()})

    df_all["Año"] = df_all["FechaAlta"].dt.year
    df_all["DOY"] = df_all["FechaAlta"].dt.dayofyear.clip(upper=365)
    years = sorted(df_all["Año"].unique().tolist())

    s_doy = pd.to_datetime(start).dayofyear
    e_doy = pd.to_datetime(end).dayofyear
    def _doy_range(s_d: int, e_d: int) -> list[int]:
        if s_d <= e_d:
            doys = list(range(s_d, e_d + 1))
        else:
            doys = list(range(s_d, 366)) + list(range(1, e_d + 1))
        return [d for d in doys if d <= 365]
    window_doys = _doy_range(s_doy, e_doy)

    df_win = df_all[df_all["DOY"].isin(window_doys)].copy()
    if df_win.empty:
        st.info("No hay reservas históricas en este periodo para detectar patrones.")
    else:
        mat = (
            df_win.groupby(["Año", "DOY"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=years, columns=window_doys, fill_value=0)
        )
        avg = mat.mean(axis=0)
        std = mat.std(axis=0)
        n_years = float(len(years))
        avg_df = (
            pd.DataFrame({"DOY": avg.index.astype(int), "Media": avg.values, "Std": std.values})
            .sort_values("DOY")
            .reset_index(drop=True)
        )
        base_year = date.today().year
        avg_df["FechaPlot"] = pd.to_datetime(base_year * 1000 + avg_df["DOY"], format="%Y%j")
        avg_df["Media7D"] = avg_df["Media"].rolling(7, center=True, min_periods=1).mean()
        avg_df["Anios"] = int(n_years)
        dias_es = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
        avg_df["wd"] = avg_df["FechaPlot"].dt.dayofweek
        avg_df["DiaSemana"] = avg_df["wd"].map({i:n for i,n in enumerate(dias_es)})

        c1 = alt.Chart(avg_df).mark_line(color="#2e485f").encode(
            x=alt.X("FechaPlot:T", title="Fecha (alineada al año actual)"),
            y=alt.Y("Media:Q", title="Reservas medias por día"),
            tooltip=[
                alt.Tooltip("FechaPlot:T", title="Fecha"),
                alt.Tooltip("DiaSemana:N", title="Día semana"),
                alt.Tooltip("Media:Q", format=".2f"),
                alt.Tooltip("Std:Q", format=".2f", title="Desv. típica"),
                alt.Tooltip("Anios:Q", title="Años considerados"),
            ],
        )
        c2 = alt.Chart(avg_df).mark_line(color="#7aa6d9").encode(
            x="FechaPlot:T", y=alt.Y("Media7D:Q", title=""),
        )
        st.altair_chart((c1 + c2).properties(height=280), use_container_width=True)

        top = avg_df.nlargest(10, "Media").copy()
        top["Fecha"] = top["FechaPlot"].dt.strftime("%d-%b")
        top_tbl = top[["Fecha", "DiaSemana", "Media", "Std"]].rename(
            columns={"DiaSemana":"Día sem.", "Media": "Media reservas", "Std": "Desv. típica"}
        )
        st.write("Top 10 fechas recurrentes con más reservas (media histórica en el periodo):")
        st.dataframe(top_tbl.round(2), use_container_width=True)

        dow_df = (
            avg_df.groupby(["wd","DiaSemana"], sort=False)["Media"]
                  .mean()
                  .reset_index()
                  .sort_values("wd")
        )
        bar_dow = alt.Chart(dow_df).mark_bar(color="#2e485f").encode(
            x=alt.X("DiaSemana:N", sort=dias_es, title="Día semana"),
            y=alt.Y("Media:Q", title="Reservas medias"),
            tooltip=[alt.Tooltip("DiaSemana:N", title="Día"), alt.Tooltip("Media:Q", format=".2f")],
        ).properties(height=220)
        st.altair_chart(bar_dow, use_container_width=True)

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

    # ===== Nueva gráfica: reservas por día alineadas por día de la semana =====
    st.subheader("Alineación por día de la semana (Act / LY-1 / LY-2)")
    dias_es = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]

    def _align_weekday(df: pd.DataFrame, label: str, start_dt) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Slot","SemanaIdx","DiaSemana","Reservas","Serie","Fecha"])
        d = df.copy()
        d["Fecha"] = pd.to_datetime(d["Fecha"])
        d["wd"] = d["Fecha"].dt.weekday
        d["DiaSemana"] = d["wd"].map({i:n for i,n in enumerate(dias_es)})
        delta = (d["Fecha"] - pd.to_datetime(start_dt)).dt.days
        d["SemanaIdx"] = (delta // 7).astype(int) + 1
        d["Slot"] = d["SemanaIdx"] * 7 + d["wd"]
        d["Serie"] = label
        d["SlotLabel"] = "S" + d["SemanaIdx"].astype(str) + "-" + d["DiaSemana"]
        return d[["Slot","SlotLabel","SemanaIdx","DiaSemana","Reservas","Serie","Fecha"]]

    # Construye datasets alineados (usa los mismos rangos que ya calculaste arriba)
    aligned_frames = []
    aligned_frames.append(_align_weekday(act, "Act", start))
    if compare_ly1 and not ly1.empty:
        aligned_frames.append(_align_weekday(ly1, "LY-1", pd.to_datetime(start) - pd.DateOffset(years=1)))
    if compare_ly2 and not ly2.empty:
        aligned_frames.append(_align_weekday(ly2, "LY-2", pd.to_datetime(start) - pd.DateOffset(years=2)))

    aligned_df = pd.concat([x for x in aligned_frames if not x.empty], ignore_index=True)
    if not aligned_df.empty:
        # Orden del eje X (Semana-Día)
        order_labels = aligned_df.sort_values("Slot")["SlotLabel"].unique().tolist()

        chart_wd = (
            alt.Chart(aligned_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("SlotLabel:N", sort=order_labels, title="Semana - Día de la semana"),
                y=alt.Y("Reservas:Q", title="Reservas por día"),
                color=alt.Color("Serie:N", title="Serie"),
                tooltip=[
                    alt.Tooltip("Serie:N"),
                    alt.Tooltip("SemanaIdx:Q", title="Semana"),
                    alt.Tooltip("DiaSemana:N", title="Día"),
                    alt.Tooltip("Fecha:T", title="Fecha real"),
                    alt.Tooltip("Reservas:Q", title="Reservas"),
                ],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart_wd, use_container_width=True)
    else:
        st.info("No hay datos suficientes para la alineación por día de la semana.")

    # Descarga
    export = plot_df[["Serie", "Fecha", "Reservas"]].copy()
    export["Fecha"] = pd.to_datetime(export["Fecha"]).dt.date
    st.download_button(
        "📥 Descargar series (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="reservas_por_dia_act_ly1_ly2.csv",
        mime="text/csv",
    )