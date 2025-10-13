import streamlit as st
import pandas as pd
from datetime import date

from utils import compute_kpis, period_inputs, pro_exec_summary

st.header("Overview")

raw = st.session_state.get("df_active") or st.session_state.get("raw")
props = st.session_state.get("active_props", [])
if props:
    raw = raw[raw["Alojamiento"].astype(str).isin(props)].copy()

if raw is None or raw.empty:
    st.info("No hay datos cargados. Vuelve a la portada y sube un CSV.")
    st.stop()

with st.sidebar:
    st.markdown("—")
    cut = st.date_input("Fecha de corte", value=date.today(), key="ov_cut")
    start, end = period_inputs("Inicio periodo", "Fin periodo",
                               pd.to_datetime(raw["Fecha entrada"].min() if "Fecha entrada" in raw else date.today()).date(),
                               date.today(), key_prefix="ov")

# KPIs actual y LY/LY final
by_prop_now, tot_now = compute_kpis(raw, pd.to_datetime(cut), pd.to_datetime(start), pd.to_datetime(end), None, props if props else None)
_, tot_ly_cut = compute_kpis(raw, pd.to_datetime(cut) - pd.DateOffset(years=1),
                             pd.to_datetime(start) - pd.DateOffset(years=1),
                             pd.to_datetime(end) - pd.DateOffset(years=1), None, props if props else None)
_, tot_ly_final = compute_kpis(raw, pd.to_datetime(end) - pd.DateOffset(years=1),
                               pd.to_datetime(start) - pd.DateOffset(years=1),
                               pd.to_datetime(end) - pd.DateOffset(years=1), None, props if props else None)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ingresos actuales (€)", f"{tot_now['ingresos']:.2f}")
c2.metric("Ocupación actual (%)", f"{tot_now['ocupacion_pct']:.2f}%")
c3.metric("ADR actual (€)", f"{tot_now['adr']:.2f}")
c4.metric("Ingresos LY final (€)", f"{tot_ly_final['ingresos']:.2f}")

st.subheader("Detalle por alojamiento")
st.dataframe(by_prop_now, use_container_width=True)

# Explicación ejecutiva
st.subheader("🧠 Explicación ejecutiva (narrada)")
blocks = pro_exec_summary(tot_now, tot_ly_cut, tot_ly_final, pace={})
st.markdown(blocks["headline"])
with st.expander("Ver análisis detallado"):
    st.markdown(blocks["detail"])