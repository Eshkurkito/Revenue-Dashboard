import pandas as pd
import streamlit as st
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, save_group_csv, load_groups, GROUPS_PATH

def render_resumen_comparativo(raw):
    if raw is None:
        st.stop()
    raw.columns = [col.strip() for col in raw.columns]

    # Sidebar: parámetros
    with st.sidebar:
        st.header("Parámetros")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cutoff_rc")
        fecha_fin_mes = (pd.Timestamp.today() + pd.offsets.MonthEnd(0)).date()
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            fecha_fin_mes,
            "rc_period"
        )

        # Gestión de grupos
        st.header("Gestión de grupos")
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_rc = groups[selected_group]
            if st.button(f"Eliminar grupo '{selected_group}'"):
                df = pd.read_csv(GROUPS_PATH)
                df = df[df["Grupo"] != selected_group]
                df.to_csv(GROUPS_PATH, index=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            if "Alojamiento" not in raw.columns:
                st.warning("No se encontró la columna 'Alojamiento'. Sube un archivo válido o revisa el nombre de la columna.")
                st.stop()
            props_rc = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_rc",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_rc:
            save_group_csv(group_name, props_rc)
            st.success(f"Grupo '{group_name}' guardado.")

        compare_rc = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=True, key="cmp_rc"
        )

    # Cálculo base
    by_prop_rc, total_rc = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_rc),
        period_start=pd.to_datetime(start_rc),
        period_end=pd.to_datetime(end_rc),
        filter_props=props_rc if props_rc else None,
    )

    st.subheader("KPIs resumen comparativo")
    help_block("Resumen comparativo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_rc['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_rc['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_rc['ocupacion_pct']:.2f}%")

    # Comparativa con año anterior
    if compare_rc:
        # Cálculo para año anterior
        ly_start = pd.to_datetime(start_rc) - pd.DateOffset(years=1)
        ly_end = pd.to_datetime(end_rc) - pd.DateOffset(years=1)
        ly_cutoff = pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1)
        ly_df, ly_total = compute_kpis(
            df_all=raw,
            cutoff=ly_cutoff,
            period_start=ly_start,
            period_end=ly_end,
            filter_props=props_rc if props_rc else None,
        )
        st.subheader("KPIs año anterior")
        c4, c5, c6 = st.columns(3)
        c4.metric("Noches ocupadas LY", f"{ly_total['noches_ocupadas']:,}".replace(",", "."))
        c5.metric("Noches disponibles LY", f"{ly_total['noches_disponibles']:,}".replace(",", "."))
        c6.metric("Ocupación LY", f"{ly_total['ocupacion_pct']:.2f}%")

    # Detalle por alojamiento
    st.subheader("Detalle por alojamiento")
    st.dataframe(by_prop_rc, use_container_width=True)
    csv_detalle = by_prop_rc.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Descargar detalle por alojamiento (CSV)",
        data=csv_detalle,
        file_name="detalle_comparativo.csv",
        mime="text/csv"
    )