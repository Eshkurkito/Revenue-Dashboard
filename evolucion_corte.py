import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, load_groups

def render_evolucion_corte(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            date.today(),
            "evo"
        )
        inv_evo = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_evo"
        )
        days_back = st.slider("Días hacia atrás", min_value=7, max_value=120, value=30, step=1, key="evo_days")
        metric_choice = st.radio("Métrica", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="evo_metric")

        # Gestión de grupos solo lectura
        groups = load_groups()
        group_names = ["Todos"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        if selected_group == "Todos":
            props_evo = sorted([str(x) for x in raw["Alojamiento"].dropna().unique()])
        else:
            props_evo = groups[selected_group] if selected_group else []

        # Filtro adicional por alojamiento dentro del grupo seleccionado
        st.header("Filtrar alojamientos (opcional)")
        props_evo = st.multiselect(
            "Alojamientos a mostrar",
            options=props_evo,
            default=props_evo,
            key="evo_selector"
        )

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

        # Formato para mostrar
        df_show = df_evo[["Corte","ocupacion_pct","adr","revpar","ingresos"]].rename(
            columns={"ocupacion_pct":"Ocupación %","adr":"ADR (€)","revpar":"RevPAR (€)","ingresos":"Ingresos (€)"}
        )
        st.dataframe(
            df_show.style.format({
                "Ocupación %": "{:.2f}%",
                "ADR (€)": "{:.2f} €",
                "RevPAR (€)": "{:.2f} €",
                "Ingresos (€)": "{:.2f} €"
            }),
            use_container_width=True
        )

        # Descarga CSV
        st.download_button(
            "📥 Descargar evolución (CSV)",
            data=df_evo.to_csv(index=False).encode("utf-8-sig"),
            file_name="evolucion_por_corte.csv",
            mime="text/csv",
        )

        # Descarga Excel con formato
        import io
        buffer = io.BytesIO()
        df_excel = df_show.copy()
        # Divide ocupación por 100 para formato porcentaje en Excel
        if "Ocupación %" in df_excel.columns:
            df_excel["Ocupación %"] = df_excel["Ocupación %"] / 100
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_excel.to_excel(writer, index=False, sheet_name="Evolución")
            wb = writer.book
            ws = writer.sheets["Evolución"]
            for j, col in enumerate(df_excel.columns):
                ws.set_column(j, j, 18)
            fmt_pct = wb.add_format({"num_format": "0.00%", "align": "center"})
            fmt_eur = wb.add_format({"num_format": "€ #,##0.00", "align": "center"})
            for idx, col in enumerate(df_excel.columns):
                if "Ocupación" in col:
                    ws.set_column(idx, idx, 18, fmt_pct)
                elif "ADR" in col or "RevPAR" in col or "Ingresos" in col:
                    ws.set_column(idx, idx, 18, fmt_eur)
            # Nombre de alojamientos o grupo arriba a la izquierda
            nombre_alojamientos = ", ".join(props_evo) if props_evo else f"Grupo: {selected_group}"
            ws.write(0, 0, nombre_alojamientos, wb.add_format({"bold": True, "font_color": "#003366"}))
        st.download_button(
            "📥 Descargar evolución (Excel)",
            data=buffer.getvalue(),
            file_name="evolucion_por_corte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Sin datos para el rango seleccionado.")
