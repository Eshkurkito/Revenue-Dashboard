import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, save_group_csv, load_groups
import numpy as np

def render_pickup(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_1 = st.date_input("Fecha de corte", value=date.today(), key="pickup_cutoff")
        cutoff_2 = st.date_input("Fecha de corte 2", value=date.today(), key="cutoff_2")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            date.today(),
            "pickup"
        )

        # Gestión de grupos
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_pickup = groups[selected_group]
            # Botón para eliminar grupo
            if st.button(f"Eliminar grupo '{selected_group}'"):
                df_groups = pd.read_csv("grupos_guardados.csv")
                df_groups = df_groups[df_groups["Grupo"] != selected_group]
                df_groups.to_csv("grupos_guardados.csv", index=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            props_pickup = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_pickup",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_pickup:
            save_group_csv(group_name, props_pickup)
            st.success(f"Grupo '{group_name}' guardado.")

        inv_pickup = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_pickup"
        )

    # KPIs en corte 1
    by_prop_1, total_1 = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_1),
        period_start=pd.to_datetime(period_start),
        period_end=pd.to_datetime(period_end),
        inventory_override=int(inv_pickup) if inv_pickup > 0 else None,
        filter_props=props_pickup if props_pickup else None,
    )

    # KPIs en corte 2
    by_prop_2, total_2 = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_2),
        period_start=pd.to_datetime(period_start),
        period_end=pd.to_datetime(period_end),
        inventory_override=int(inv_pickup) if inv_pickup > 0 else None,
        filter_props=props_pickup if props_pickup else None,
    )

    st.subheader("Pickup entre dos cortes")
    help_block("Pickup entre dos cortes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Noches pickup", f"{total_2['noches_ocupadas'] - total_1['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Ingresos pickup (€)", f"{total_2['ingresos'] - total_1['ingresos']:.2f}")
    c3.metric("ADR pickup (€)", f"{(total_2['ingresos'] - total_1['ingresos']) / max(1, total_2['noches_ocupadas'] - total_1['noches_ocupadas']):.2f}")

    st.divider()
    st.subheader("Detalle pickup por alojamiento")
    df_pickup = by_prop_2.copy()
    df_pickup["Noches pickup"] = df_pickup["Noches ocupadas"] - by_prop_1.set_index("Alojamiento")["Noches ocupadas"].reindex(df_pickup["Alojamiento"]).fillna(0).values
    df_pickup["Ingresos pickup"] = df_pickup["Ingresos"] - by_prop_1.set_index("Alojamiento")["Ingresos"].reindex(df_pickup["Alojamiento"]).fillna(0).values
    df_pickup["ADR pickup"] = df_pickup["Ingresos pickup"] / df_pickup["Noches pickup"].replace(0, np.nan)
    st.dataframe(df_pickup, use_container_width=True)
    csv = df_pickup.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Descargar detalle pickup (CSV)",
        data=csv,
        file_name="detalle_pickup.csv",
        mime="text/csv"
    )