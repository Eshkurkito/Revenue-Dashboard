import pandas as pd
import streamlit as st
from datetime import date
from utils import period_inputs, group_selector, save_group_csv, load_groups, GROUPS_PATH

def calcular_kpis_por_alojamiento(df):
    # Calcula noches ocupadas
    df["Noches ocupadas"] = (
        pd.to_datetime(df["Fecha salida"]) - pd.to_datetime(df["Fecha entrada"])
    ).dt.days

    # Agrupa por alojamiento
    agrupado = df.groupby("Alojamiento").agg(
        noches_ocupadas=("Noches ocupadas", "sum"),
        ingresos=("Alquiler con IVA (€)", "sum"),
        reservas=("Alojamiento", "count")
    ).reset_index()

    # Calcula ADR
    agrupado["ADR"] = agrupado["ingresos"] / agrupado["noches_ocupadas"]
    agrupado["Ocupación"] = 100  # Si no tienes noches disponibles, pon 100% por defecto

    # Formato
    agrupado["ADR"] = agrupado["ADR"].apply(lambda x: f"{x:.2f} €" if pd.notnull(x) else "")
    agrupado["ingresos"] = agrupado["ingresos"].apply(lambda x: f"{x:.2f} €" if pd.notnull(x) else "")
    agrupado["Ocupación"] = agrupado["Ocupación"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

    return agrupado

def render_resumen_comparativo(raw):
    if raw is None:
        st.stop()
    raw.columns = [col.strip() for col in raw.columns]

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

    # Filtra por fechas del periodo actual
    df_actual = raw[
        (pd.to_datetime(raw["Fecha entrada"]) >= pd.to_datetime(start_rc)) &
        (pd.to_datetime(raw["Fecha entrada"]) <= pd.to_datetime(end_rc))
    ]
    if props_rc:
        df_actual = df_actual[df_actual["Alojamiento"].isin(props_rc)]

    detalle_actual = calcular_kpis_por_alojamiento(df_actual)

    # Si comparar con LY
    if compare_rc:
        ly_start = pd.to_datetime(start_rc) - pd.DateOffset(years=1)
        ly_end = pd.to_datetime(end_rc) - pd.DateOffset(years=1)
        df_ly = raw[
            (pd.to_datetime(raw["Fecha entrada"]) >= ly_start) &
            (pd.to_datetime(raw["Fecha entrada"]) <= ly_end)
        ]
        if props_rc:
            df_ly = df_ly[df_ly["Alojamiento"].isin(props_rc)]
        detalle_ly = calcular_kpis_por_alojamiento(df_ly)
        # Merge ambos detalles
        detalle = detalle_actual.merge(
            detalle_ly[["Alojamiento", "noches_ocupadas", "ADR", "ingresos"]],
            on="Alojamiento", how="left", suffixes=('', '_LY')
        )
        # Renombra para claridad
        detalle.rename(columns={
            "noches_ocupadas": "Noches ocupadas",
            "noches_ocupadas_LY": "Noches ocupadas LY",
            "ADR": "ADR",
            "ADR_LY": "ADR LY",
            "ingresos": "Ingresos",
            "ingresos_LY": "Ingresos LY"
        }, inplace=True)
    else:
        detalle = detalle_actual.copy()
        detalle["Noches ocupadas LY"] = None
        detalle["ADR LY"] = None
        detalle["Ingresos LY"] = None

    st.subheader("Detalle por alojamiento")
    st.dataframe(detalle, use_container_width=True)