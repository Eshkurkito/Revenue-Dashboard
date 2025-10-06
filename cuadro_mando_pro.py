import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
from utils import (
    compute_kpis, period_inputs, group_selector, help_block,
    pace_series, pace_forecast_month, save_group_csv, load_groups,
    _kai_cdm_pro_analysis,
)

def render_cuadro_mando_pro(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros – PRO")
        pro_cut = st.date_input("Fecha de corte", value=date.today(), key="pro_cut")
        pro_start, pro_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            date.today(),
            "pro_period"
        )
        inv_pro = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="pro_inv")
        inv_pro_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="pro_inv_ly")
        ref_years_pro = st.slider("Años de referencia Pace", min_value=1, max_value=3, value=2, key="pro_ref_years")

        st.header("Gestión de grupos")
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_pro = groups[selected_group]
            if st.button(f"Eliminar grupo '{selected_group}'"):
                df_groups = pd.read_csv("grupos_guardados.csv")
                df_groups = df_groups[df_groups["Grupo"] != selected_group]
                df_groups.to_csv("grupos_guardados.csv", index=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            props_pro = group_selector(
                "Alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="pro_props",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_pro:
            save_group_csv(group_name, props_pro)
            st.success(f"Grupo '{group_name}' guardado.")

    st.subheader("📊 Cuadro de mando (PRO)")

    # KPIs actuales y LY
    by_prop_now, tot_now = compute_kpis(
        raw,
        pd.to_datetime(pro_cut),
        pd.to_datetime(pro_start),
        pd.to_datetime(pro_end),
        int(inv_pro) if inv_pro > 0 else None,
        props_pro if props_pro else None,
    )
    _, tot_ly_cut = compute_kpis(
        raw,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=1),
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )
    cutoff_ly_final = pd.to_datetime(pro_end) - pd.DateOffset(years=1)
    _, tot_ly_final = compute_kpis(
        raw,
        cutoff_ly_final,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )

    # ====== Ingresos ======
    st.subheader("💶 Ingresos (periodo seleccionado)")
    g1, g2, g3 = st.columns(3)
    g1.metric("Ingresos actuales (€)", f"{tot_now['ingresos']:.2f}")
    g2.metric("Ingresos LY a este corte (€)", f"{tot_ly_cut['ingresos']:.2f}")
    g3.metric("Ingresos LY final (€)", f"{tot_ly_final['ingresos']:.2f}")

    # ====== ADR ======
    st.subheader("🏷️ ADR (a fecha de corte)")
    _, tot_ly2_cut = compute_kpis(
        raw,
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
    dfb = raw[(raw["Fecha alta"] <= pd.to_datetime(pro_cut))].dropna(
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
            df=raw,
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
        df=raw,
        period_start=pd.to_datetime(pro_start),
        period_end=pd.to_datetime(pro_end),
        d_max=int(dmax_y),
        props=props_pro if props_pro else None,
        inv_override=int(inv_pro) if inv_pro > 0 else None,
    )
    base_ly = pace_series(
        df=raw,
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
                        df_all=raw,
                        cutoff=c,
                        period_start=pd.to_datetime(pro_start),
                        period_end=pd.to_datetime(pro_end),
                        inventory_override=int(inv_e) if inv_e > 0 else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    _, tot_ly_e = compute_kpis(
                        df_all=raw,
                        cutoff=c - pd.DateOffset(years=1),
                        period_start=pd.to_datetime(pro_start) - pd.DateOffset(years=1),
                        period_end=pd.to_datetime(pro_end) - pd.DateOffset(years=1),
                        inventory_override=int(inv_pro_ly) if (isinstance(inv_pro_ly, int) and inv_pro_ly > 0) else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    rows.append({
                        "Corte": c.normalize(),
                        "occ_now": float(tot_now_e["ocupacion_pct"]),
                        "adr_now": float(tot_now_e["adr"]),
                        "occ_ly": float(tot_ly_e["ocupacion_pct"]),
                        "adr_ly": float(tot_ly_e["adr"]),
                    })
                evo_df = pd.DataFrame(rows)
                if evo_df.empty:
                    st.info("Sin datos en el rango seleccionado.")
                else:
                    occ_long = evo_df.melt(id_vars=["Corte"], value_vars=["occ_now","occ_ly"],
                                           var_name="serie", value_name="valor")
                    occ_long["serie"] = occ_long["serie"].map({"occ_now": "Ocupación actual", "occ_ly": "Ocupación LY"})
                    adr_long = evo_df.melt(id_vars=["Corte"], value_vars=["adr_now","adr_ly"],
                                           var_name="serie", value_name="valor")
                    adr_long["serie"] = adr_long["serie"].map({"adr_now": "ADR actual (€)", "adr_ly": "ADR LY (€)"})

                    occ_colors = {"Ocupación actual": "#1f77b4", "Ocupación LY": "#6baed6"}
                    adr_colors = {"ADR actual (€)": "#ff7f0e", "ADR LY (€)": "#fdae6b"}

                    # Selección 'nearest' común por fecha de corte
                    hover = alt.selection_single(
                        fields=["Corte"], nearest=True, on="mouseover", empty="none", clear="mouseout", name="evoHover"
                    )

                    # Contenedor invisible para capturar el hover (dataset con todas las fechas)
                    x_ticks = alt.Chart(occ_long[["Corte"]].drop_duplicates()).mark_rule(opacity=0).encode(x="Corte:T").add_selection(hover)

                    # Líneas
                    occ_line = (
                        alt.Chart(occ_long)
                        .mark_line(strokeWidth=2, interpolate="monotone")
                        .encode(
                            x=alt.X("Corte:T", title="Fecha de corte"),
                            y=alt.Y("valor:Q", axis=alt.Axis(orient="left", title="Ocupación %", tickCount=6, format=".0f")),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None),
                            strokeDash=alt.condition("datum.serie == 'Ocupación LY'", alt.value([5,3]), alt.value([0,0])),
                            opacity=alt.condition("datum.serie == 'Ocupación LY'", alt.value(0.7), alt.value(1.0)),
                        )
                    )
                    adr_line = (
                        alt.Chart(adr_long)
                        .mark_line(strokeWidth=2, interpolate="monotone")
                        .encode(
                            x=alt.X("Corte:T"),
                            y=alt.Y("valor:Q", axis=alt.Axis(orient="right", title="ADR (€)", tickCount=6, format=",.2f")),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None),
                            strokeDash=alt.condition("datum.serie == 'ADR LY (€)'", alt.value([5,3]), alt.value([0,0])),
                            opacity=alt.condition("datum.serie == 'ADR LY (€)'", alt.value(0.7), alt.value(1.0)),
                        )
                    )

                    # Puntos solo en el dato más cercano al ratón
                    occ_points = (
                        alt.Chart(occ_long)
                        .mark_circle(size=80, filled=True)
                        .encode(
                            x="Corte:T",
                            y=alt.Y("valor:Q", axis=None),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None, legend=None),
                            tooltip=[alt.Tooltip("Corte:T", title="Día"),
                                     alt.Tooltip("serie:N", title="Serie"),
                                     alt.Tooltip("valor:Q", title="Valor", format=".2f")],
                        )
                        .transform_filter(hover)
                    )
                    adr_points = (
                        alt.Chart(adr_long)
                        .mark_circle(size=80, filled=True)
                        .encode(
                            x="Corte:T",
                            y=alt.Y("valor:Q", axis=None),
                            color=alt.Color("serie:N",
                                scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None, legend=None),
                            tooltip=[alt.Tooltip("Corte:T", title="Día"),
                                     alt.Tooltip("serie:N", title="Serie"),
                                     alt.Tooltip("valor:Q", title="Valor", format=",.2f")],
                        )
                        .transform_filter(hover)
                    )

                    # Regla vertical visible cuando hay hover
                    v_rule = alt.Chart(occ_long).mark_rule(color="#999").encode(x="Corte:T").transform_filter(hover)

                    chart = (
                        alt.layer(
                            occ_line, adr_line,
                            x_ticks,  # activa la selección
                            v_rule,
                            occ_points, adr_points
                        )
                        .resolve_scale(y="independent", color="independent")
                        .properties(height=380)
                    )
                    st.altair_chart(chart, use_container_width=True)

                    out = evo_df.rename(columns={
                        "occ_now":"Ocupación % (Actual)", "occ_ly":"Ocupación % (LY)",
                        "adr_now":"ADR (€) (Actual)", "adr_ly":"ADR (€) (LY)",
                    })
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "📥 Descargar evolución (CSV)",
                        data=out.to_csv(index=False).encode("utf-8-sig"),
                        file_name="evolucion_occ_adr_cdmpro.csv",
                        mime="text/csv"
                    )

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