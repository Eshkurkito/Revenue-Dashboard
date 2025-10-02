import streamlit as st
import pandas as pd
from datetime import date
from utils import period_inputs, help_block

def render_prediccion_pace(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today() + pd.offsets.MonthEnd(0)).date(),
            "predpace"
        )

        # Gestión de grupos
        from utils import save_group_csv, load_groups, group_selector
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_predpace = groups[selected_group]
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
            props_predpace = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_predpace",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_predpace:
            save_group_csv(group_name, props_predpace)
            st.success(f"Grupo '{group_name}' guardado.")

        inv_predpace = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_predpace"
        )
        d_max = st.slider("Días antes de la estancia (curva D)", min_value=7, max_value=120, value=30, step=1, key="predpace_dmax")

    # Aquí deberías tener una función prediccion_pace_series en utils.py
    from utils import prediccion_pace_series

    predpace_df = prediccion_pace_series(
        df=raw,
        period_start=pd.to_datetime(period_start),
        period_end=pd.to_datetime(period_end),
        d_max=d_max,
        props=props_predpace if props_predpace else None,
        inv_override=int(inv_predpace) if inv_predpace > 0 else None,
    )

    st.subheader("Predicción Pace")
    help_block("Predicción Pace")
    st.line_chart(predpace_df.set_index("D")[["noches_pred", "ingresos_pred"]], height=260)
    st.dataframe(predpace_df, use_container_width=True)
    csv = predpace_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Descargar predicción Pace (CSV)",
        data=csv,
        file_name="prediccion_pace.csv",
        mime="text/csv"
    )