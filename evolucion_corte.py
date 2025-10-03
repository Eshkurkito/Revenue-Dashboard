import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils import compute_kpis, period_inputs, group_selector, help_block, load_groups

def render_evolucion_corte(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        cut_start = st.date_input("Inicio de corte", value=date.today() - timedelta(days=30), key="evo_cut_start")
        cut_end   = st.date_input("Fin de corte",   value=date.today(), key="evo_cut_end")

        st.header("Periodo objetivo")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            date.today(),
            "evo_target"
        )

        # Selector de grupo y filtro por alojamiento
        groups = load_groups()
        group_names = ["Todos"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        if selected_group == "Todos":
            props_evo = sorted([str(x) for x in raw["Alojamiento"].dropna().unique()])
        else:
            props_evo = groups[selected_group] if selected_group else []

        st.header("Filtrar alojamientos (opcional)")
        props_evo = st.multiselect(
            "Alojamientos a mostrar",
            options=props_evo,
            default=props_evo,
            key="evo_selector"
        )

        inv_evo = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_evo"
        )
        metric_choice = st.radio("Métrica", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="evo_metric")

    # Serie de fechas de corte
    cut_start_ts = pd.to_datetime(cut_start)
    cut_end_ts   = pd.to_datetime(cut_end)
    if cut_start_ts > cut_end_ts:
        st.error("El inicio del rango de corte no puede ser posterior al fin.")
        st.stop()

    rows_evo = []
    for c in pd.date_range(cut_start_ts, cut_end_ts, freq="D"):
        _, tot_tmp = compute_kpis(
            df_all=raw,
            cutoff=c,
            period_start=pd.to_datetime(period_start),
            period_end=pd.to_datetime(period_end),
            inventory_override=int(inv_evo) if inv_evo > 0 else None,
            filter_props=props_evo if props_evo else None,
        )
        rows_evo.append({"Corte": c.normalize(), **tot_tmp})
    df_evo = pd.DataFrame(rows_evo)

    st.subheader("📈 Evolución por fecha de corte")
    help_block("Evolución por fecha de corte")

    if not df_evo.empty:
        key_map = {"Ocupación %":"ocupacion_pct","ADR (€)":"adr","RevPAR (€)":"revpar"}
        k = key_map[metric_choice]
        plot = pd.DataFrame({metric_choice: df_evo[k].values}, index=df_evo["Corte"])
        st.line_chart(plot, height=280)

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

        # Descarga Excel con formato y nombre arriba a la izquierda
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
            fmt_int = wb.add_format({"num_format": "0", "align": "center"})
            for idx, col in enumerate(df_excel.columns):
                if "Ocupación" in col:
                    ws.set_column(idx, idx, 18, fmt_pct)
                elif "ADR" in col or "RevPAR" in col or "Ingresos" in col:
                    ws.set_column(idx, idx, 18, fmt_eur)
                elif "Noches" in col:
                    ws.set_column(idx, idx, 18, fmt_int)
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
