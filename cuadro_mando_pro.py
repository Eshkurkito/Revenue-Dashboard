import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
from utils import (
    compute_kpis, period_inputs, group_selector, help_block,
    pace_series, pace_forecast_month, save_group_csv, load_groups,
    _kai_cdm_pro_analysis,  # IMPORTANTE: trae el análisis
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

    # Actual y LYs
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

    # Bandas ADR en tabla (P10, P50, P90)
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

    # ...puedes mostrar aquí métricas de pace_res si quieres...

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