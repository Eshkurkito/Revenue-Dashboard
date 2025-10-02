import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_evolucion_corte(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "evo"
        )
        inv_evo = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_evo"
        )
        days_back = st.slider("Días hacia atrás", min_value=7, max_value=120, value=30, step=1, key="evo_days")
        metric_choice = st.radio("Métrica", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="evo_metric")

        # Gestión de grupos
        from utils import save_group_csv, load_groups, group_selector
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_evo = groups[selected_group]
            # Botón para eliminar grupo
            if st.button(f"Eliminar grupo '{selected_group}'"):
                import pandas as pd
                from utils import GROUPS_PATH
                df_groups = pd.read_csv(GROUPS_PATH)
                df_groups = df_groups[df_groups["Grupo"] != selected_group]
                df_groups.to_csv(GROUPS_PATH, index=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            props_evo = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_evo",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_evo:
            save_group_csv(group_name, props_evo)
            st.success(f"Grupo '{group_name}' guardado.")

    # Evolución por fecha de corte
    rows_evo = []
    cut_start = (pd.to_datetime(date.today()) - pd.Timedelta(days=int(days_back))).normalize()
    for c in pd.date_range(cut_start, pd.to_datetime(date.today()).normalize(), freq="D"):
        _, tot_tmp = compute_kpis(
            df_all=raw,
            cutoff=c,
            period_start=pd.to_datetime(period_start),
            period_end=pd.to_datetime(period_end),
            inventory_override=int(inv_evo) if inv_evo > 0 else None,
            filter_props=props_evo if props_evo else None,
        )
        rows_evo.append({"Corte": c, **tot_tmp})
    df_evo = pd.DataFrame(rows_evo)

    st.subheader("Evolución por fecha de corte")
    help_block("Evolución por fecha de corte")

    if not df_evo.empty:
        key_map = {"Ocupación %":"ocupacion_pct","ADR (€)":"adr","RevPAR (€)":"revpar"}
        k = key_map[metric_choice]
        plot = pd.DataFrame({metric_choice: df_evo[k].values}, index=df_evo["Corte"])
        st.line_chart(plot, height=260)
        st.dataframe(df_evo[["Corte","ocupacion_pct","adr","revpar","ingresos"]]
                     .rename(columns={"ocupacion_pct":"Ocupación %","adr":"ADR (€)","revpar":"RevPAR (€)","ingresos":"Ingresos (€)"}),
                     use_container_width=True)
        st.download_button(
            "📥 Descargar evolución (CSV)",
            data=df_evo.to_csv(index=False).encode("utf-8-sig"),
            file_name="evolucion_por_corte.csv",
            mime="text/csv",
        )
    else:
        st.info("Sin datos para el rango seleccionado.")