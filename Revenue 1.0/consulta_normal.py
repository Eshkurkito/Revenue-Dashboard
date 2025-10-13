import pandas as pd
import streamlit as st
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, compute_portal_share, save_group_csv, load_groups, GROUPS_CSV

def render_consulta_normal(raw):
    if raw is None:
        st.stop()
    # Normaliza los nombres de columna
    raw.columns = [col.strip() for col in raw.columns]

    # Mapeo flexible para 'Agente/Intermediario'
    col_portal = None
    for col in raw.columns:
        if col.lower().replace(" ", "") == "agente/intermediario".lower().replace(" ", ""):
            col_portal = col
            break

    if not col_portal:
        st.error("No se encontró la columna 'Agente/Intermediario'. Las columnas detectadas son: " + ", ".join(raw.columns))
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date.today(), key="cutoff_normal")
        c1, c2 = st.columns(2)
        # Robust extraction of end-of-month date
        fecha_fin_mes = (pd.Timestamp.today() + pd.offsets.MonthEnd(0)).date()
        start_normal, end_normal = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            fecha_fin_mes,
            "normal_period"
        )
        inv_normal = st.number_input(
            "Sobrescribir inventario (nº alojamientos)",
            min_value=0, value=0, step=1, key="inv_normal"
        )
        st.markdown("—")
        compare_normal = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal"
        )
        inv_normal_prev = st.number_input(
            "Inventario año anterior (opcional)",
            min_value=0, value=0, step=1, key="inv_normal_prev"
        )

        # Gestión de grupos
        st.header("Gestión de grupos")
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_normal = groups[selected_group]
            if st.button(f"Eliminar grupo '{selected_group}'"):
                df = pd.read_csv(GROUPS_CSV)
                df = df[df["Grupo"] != selected_group]
                df.to_csv(GROUPS_CSV=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            props_normal = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_normal",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_normal:
            save_group_csv(group_name, props_normal)
            st.success(f"Grupo '{group_name}' guardado.")

    # Calcula noches ocupadas si no existe la columna
    if "Noches ocupadas" not in raw.columns:
        if "Fecha entrada" in raw.columns and "Fecha salida" in raw.columns:
            # Calcula noches ocupadas como la diferencia de días
            raw["Noches ocupadas"] = (
                pd.to_datetime(raw["Fecha salida"]) - pd.to_datetime(raw["Fecha entrada"])
            ).dt.days
        else:
            st.error("No se encontraron las columnas 'Fecha entrada' y 'Fecha salida' para calcular noches ocupadas.")
            st.stop()

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    help_block("Consulta normal")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")

    # Cálculo de ADR usando "Alquiler con IVA (€)"
    total_ingresos = raw["Alquiler con IVA (€)"].sum()
    total_noches = raw["Noches ocupadas"].sum()  # O la columna que corresponda

    adr = total_ingresos / total_noches if total_noches > 0 else 0

    # Distribución por portal (si existe columna)
    port_df = compute_portal_share(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        filter_props=props_normal if props_normal else None,
        portal_col=col_portal  # <-- Añade este argumento si tu función lo acepta
    )
    st.subheader(f"Distribución por {col_portal} (reservas en el periodo)")
    if port_df is None:
        st.info(f"No se encontró la columna '{col_portal}'. Si tiene otro nombre, dímelo y lo mapeo.")
    elif port_df.empty:
        st.warning("No hay reservas del periodo a la fecha de corte para calcular distribución por portal.")
    else:
        port_view = port_df.copy()
        port_view["% Reservas"] = port_view["% Reservas"].round(2)
        st.dataframe(port_view, use_container_width=True)
        csv_port = port_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar distribución por portal (CSV)",
            data=csv_port,
            file_name="portales_distribucion.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar detalle (CSV)",
            data=csv,
            file_name="detalle_por_alojamiento.csv",
            mime="text/csv"
        )

    st.write("Columnas detectadas:", list(raw.columns))

    with st.sidebar:
        st.header("Selección de grupo de alojamientos")
        groups = load_groups()
        group_names = sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        props_rc = groups[selected_group] if selected_group else []

    # Filtra alojamientos según el grupo seleccionado
    if props_rc:
        raw = raw[raw["Alojamiento"].isin(props_rc)]