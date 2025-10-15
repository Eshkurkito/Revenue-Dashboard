from pathlib import Path
import json
import pandas as pd
import streamlit as st
import numpy as np
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
    col_aloj = find("alojamiento", "propiedad", "property", "listing", "unidad", "apartamento", "room", "unit")
    if col_aloj: mapping[col_aloj] = "Alojamiento"
    col_fa = find("fecha alta", "fecha de alta", "booking date", "fecha reserva", "creado", "created", "booked")
    if col_fa: mapping[col_fa] = "Fecha alta"
    col_fe = find("fecha entrada", "check in", "entrada", "arrival")
    if col_fe: mapping[col_fe] = "Fecha entrada"
    col_fs = find("fecha salida", "check out", "salida", "departure")
    if col_fs: mapping[col_fs] = "Fecha salida"
    col_rev = find("alquiler con iva (€)", "alquiler con iva", "ingresos", "revenue", "importe", "total", "precio total")
    if col_rev: mapping[col_rev] = "Alquiler con IVA (€)"
    return df.rename(columns=mapping) if mapping else df

def _load_saved_groups(props_all: list[str]) -> dict[str, list[str]]:
    base = Path(__file__).resolve().parent
    j = base / "assets" / "groups_guardados.json"
    if j.exists():
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            return {g: [p for p in v if p in props_all] for g, v in data.items()}
        except Exception:
            pass
    c = base / "assets" / "groups_guardados.csv"
    if c.exists():
        try:
            df = pd.read_csv(c)
            if {"Grupo","Alojamiento"}.issubset(df.columns):
                d = (df.dropna(subset=["Grupo","Alojamiento"])
                        .astype(str)
                        .groupby("Grupo")["Alojamiento"].apply(list).to_dict())
                return {g: [p for p in v if p in props_all] for g, v in d.items()}
        except Exception:
            pass
    return {}

def render_cuadro_mando_pro(raw: pd.DataFrame | None = None):
    # --- seguridad ante falta de datos ---
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("No hay datos cargados. Sube un archivo Excel/CSV en la barra lateral para usar el Cuadro de mando PRO.")
        st.stop()

    df = _standardize_columns(raw.copy())

    if "Alojamiento" not in df.columns:
        st.warning("No se encuentra la columna ‘Alojamiento’. Renombra tu archivo o usa un nombre equivalente (p.ej. ‘Propiedad’, ‘Listing’).")
        st.stop()

    # ========= CONTROLES (definen pro_* y otros) =========
    props_all = sorted(df["Alojamiento"].dropna().astype(str).unique())

    with st.sidebar:
        st.subheader("Parámetros · PRO")
        col_dates1, col_dates2 = st.columns(2)
        # Periodo de estancias
        pro_start = col_dates1.date_input("Inicio periodo", value=date.today().replace(day=1), key="pro_start")
        pro_end   = col_dates2.date_input("Fin periodo", value=date.today(), key="pro_end")
        # Fecha de corte
        pro_cut   = st.date_input("Fecha de corte", value=date.today(), key="pro_cut")

        # ⬇️ Selector de grupo (opcional) + preselección de alojamientos
        groups = _load_saved_groups(props_all)
        default_props = []
        if groups:
            names = ["(Sin grupo)"] + sorted(groups.keys())
            grp = st.selectbox("Grupo guardado (opcional)", names, key="pro_grupo_guardado")
            if grp and grp != "(Sin grupo)":
                default_props = groups.get(grp, [])

        props_pro = st.multiselect("Alojamientos", options=props_all,
                                   default=default_props, key="props_pro")

        col_inv1, col_inv2 = st.columns(2)
        inv_pro    = col_inv1.number_input("Inventario actual", min_value=0, value=0, step=1, key="inv_pro")
        inv_pro_ly = col_inv2.number_input("Inventario LY",     min_value=0, value=0, step=1, key="inv_pro_ly")
        ref_years_pro = st.selectbox("Años de referencia (Pace)", options=[1, 2, 3], index=0, key="ref_years_pro")

    # Normaliza: None si no se selecciona nada
    props_pro = props_pro if props_pro else None

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
    st.subheader("💶 Ingresos (periodo seleccionado)")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Ingresos actuales (€)", f"{tot_now['ingresos']:.2f}")
    g2.metric("Ingresos LY a este corte (€)", f"{tot_ly_cut['ingresos']:.2f}")
    g3.metric("Ingresos LY final (€)", f"{tot_ly_final['ingresos']:.2f}")
    g4.metric("Ingresos LY-2 a este corte (€)", f"{tot_ly2_cut_ing['ingresos']:.2f}")
    g5.metric("Ingresos LY-2 final (€)", f"{tot_ly2_final_ing['ingresos']:.2f}")

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
    a1, a2, a3 = st.columns(3)
    a1.metric("ADR actual (€)", f"{tot_now['adr']:.2f}")
    a2.metric("ADR LY (€)", f"{tot_ly_cut['adr']:.2f}")
    a3.metric("ADR LY-2 (€)", f"{tot_ly2_cut['adr']:.2f}")

    # Bandas ADR (P10, P50, P90)
    start_dt = pd.to_datetime(pro_start); end_dt = pd.to_datetime(pro_end)
    dfb = df[(df["Fecha alta"] <= pd.to_datetime(pro_cut))].dropna(
        subset=["Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]
    ).copy()
    if props_pro:
        dfb = dfb[dfb["Alojamiento"].isin(props_pro)]
    mask_b = ~((dfb["Fecha salida"] <= start_dt) | (dfb["Fecha entrada"] >= (end_dt + pd.Timedelta(days=1))))
    dfb = dfb[mask_b]
    if not dfb.empty:
        dfb["los"] = (dfb["Fecha salida"].dt.normalize() - dfb["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        dfb["adr_reserva"] = dfb["Alquiler con IVA (€)"] / dfb["los"]
        dfb["Mes"] = dfb["Fecha entrada"].dt.to_period("M").astype(str)
        def _pct_cols(x):
            arr = x.dropna().values
            if arr.size == 0:
                return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})
            return pd.Series({"P10": np.percentile(arr,10), "Mediana": np.percentile(arr,50), "P90": np.percentile(arr,90)})
        bands = dfb.groupby("Mes")["adr_reserva"].apply(_pct_cols).reset_index()
        bands_wide = bands.pivot(index="Mes", columns="level_1", values="adr_reserva").sort_index()
        st.dataframe(bands_wide[["P10","Mediana","P90"]], use_container_width=True)
        st.download_button(
            "📥 Descargar bandas ADR (CSV)",
            data=bands_wide[["P10","Mediana","P90"]].reset_index().to_csv(index=False).encode("utf-8-sig"),
            file_name="adr_bands_cdmpro.csv", mime="text/csv"
        )
    else:
        st.info("Sin datos suficientes para bandas ADR en el periodo.")

    # ====== Ocupación ======
    st.subheader("🏨 Ocupación (periodo seleccionado)")
    o1, o2, o3 = st.columns(3)
    o1.metric("Ocupación actual", f"{tot_now['ocupacion_pct']:.2f}%")
    o2.metric("Ocupación LY (a este corte)", f"{tot_ly_cut['ocupacion_pct']:.2f}%")
    o3.metric("Ocupación LY final", f"{tot_ly_final['ocupacion_pct']:.2f}%")
    st.caption("Actual y LY: reservas con Fecha alta ≤ corte. LY final: corte = fin del periodo LY.")

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