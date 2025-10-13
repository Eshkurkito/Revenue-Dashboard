import streamlit as st
import pandas as pd
import altair as alt
from datetime import date

from pages.utils import compute_kpis, pace_series, pace_forecast_month

st.header("Pace y Forecast")

raw = st.session_state.get("raw")
if raw is None or raw.empty:
    st.info("No hay datos cargados. Vuelve a la portada y sube un CSV.")
    st.stop()

props = st.session_state.get("active_props", [])

with st.sidebar:
    st.markdown("—")
    cut = st.date_input("Fecha de corte", value=date.today(), key="pf_cut")
    col1, col2 = st.columns(2)
    start = col1.date_input("Inicio periodo", value=pd.to_datetime(raw["Fecha entrada"].min()).date() if "Fecha entrada" in raw else date.today())
    end   = col2.date_input("Fin periodo", value=date.today())
    ref_years = st.slider("Años de referencia", 1, 3, 2, key="pf_ref")

# Forecast P50
pace_res = pace_forecast_month(
    df=raw,
    cutoff=pd.to_datetime(cut),
    period_start=pd.to_datetime(start),
    period_end=pd.to_datetime(end),
    ref_years=int(ref_years),
    dmax=180,
    props=props if props else None,
    inv_override=None,
) or {}

c1, c2, c3 = st.columns(3)
c1.metric("OTB noches", f"{pace_res.get('nights_otb', 0):,.0f}".replace(",", "."))
c2.metric("Forecast noches (P50)", f"{pace_res.get('nights_p50', 0):,.0f}".replace(",", "."))
c3.metric("Forecast ingresos (P50)", f"{pace_res.get('revenue_final_p50', 0):,.2f}")

# Curva Pace YoY (noches por D)
p_start_ly = pd.to_datetime(start) - pd.DateOffset(years=1)
p_end_ly   = pd.to_datetime(end) - pd.DateOffset(years=1)

cur = pace_series(raw, pd.to_datetime(start), pd.to_datetime(end), d_max=180, props=props if props else None)
ly  = pace_series(raw, p_start_ly, p_end_ly, d_max=180, props=props if props else None)

if cur.empty or ly.empty:
    st.info("No hay datos suficientes para Pace YoY en el periodo.")
else:
    D_all = list(range(0, int(max(cur["D"].max(), ly["D"].max())) + 1))
    plot = pd.DataFrame({"D": D_all}).merge(cur.rename(columns={"noches":"Actual"}), on="D", how="left") \
                                     .merge(ly.rename(columns={"noches":"LY"}), on="D", how="left") \
                                     .fillna(0.0)
    data_melt = plot.melt(id_vars=["D"], value_vars=["Actual","LY"], var_name="Serie", value_name="Noches")
    colors = {"Actual": "#1f77b4", "LY": "#9e9e9e"}
    chart = (
        alt.Chart(data_melt)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("D:Q", title="Días antes de la estancia"),
            y=alt.Y("Noches:Q", title="Noches confirmadas"),
            color=alt.Color("Serie:N", scale=alt.Scale(domain=list(colors.keys()), range=[colors[k] for k in colors]), title=None),
            tooltip=[alt.Tooltip("D:Q"), alt.Tooltip("Serie:N"), alt.Tooltip("Noches:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)