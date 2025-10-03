import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_evolucion_corte(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date.today() - timedelta(days=30), key="evo_cut_start")
        evo_cut_end   = st.date_input("Fin de corte",   value=date.today(), key="evo_cut_end")

        st.header("Periodo objetivo")
        evo_target_start, evo_target_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "evo_target"
        )

        props_e = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_evo",
            default=[]
        )
        inv_e      = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")

        kpi_options = ["Ocupación %", "ADR (€)", "RevPAR (€)"]
        selected_kpis = st.multiselect("KPIs a mostrar", kpi_options, default=["Ocupación %"], key="kpi_multi")

        compare_e = st.checkbox("Mostrar LY (alineado por día)", value=False, key="cmp_evo")

        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📈 Evolución de KPIs vs fecha de corte")
    help_block("Evolución por corte")

    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts   = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
            st.stop()

        # ---------- Serie ACTUAL ----------
        rows_now = []
        for c in pd.date_range(cut_start_ts, cut_end_ts, freq="D"):
            _, tot = compute_kpis(
                df_all=raw,
                cutoff=c,
                period_start=pd.to_datetime(evo_target_start),
                period_end=pd.to_datetime(evo_target_end),
                inventory_override=int(inv_e) if inv_e > 0 else None,
                filter_props=props_e if props_e else None,
            )
            rows_now.append({
                "Corte": c.normalize(),
                "ocupacion_pct": float(tot["ocupacion_pct"]),
                "adr": float(tot["adr"]),
                "revpar": float(tot["revpar"]),
                "ingresos": float(tot["ingresos"]),
            })
        df_now = pd.DataFrame(rows_now)
        if df_now.empty:
            st.info("No hay datos para el rango seleccionado.")
            st.stop()

        # ---------- Serie LY (opcional) ----------
        df_prev = pd.DataFrame()
        if compare_e:
            rows_prev = []
            cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
            cut_end_prev   = cut_end_ts   - pd.DateOffset(years=1)
            target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
            target_end_prev   = pd.to_datetime(evo_target_end)   - pd.DateOffset(years=1)
            for c in pd.date_range(cut_start_prev, cut_end_prev, freq="D"):
                _, tot2 = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=target_start_prev,
                    period_end=target_end_prev,
                    inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_prev.append({
                    "Corte": (pd.to_datetime(c).normalize() + pd.DateOffset(years=1)),  # alineado al año actual
                    "ocupacion_pct": float(tot2["ocupacion_pct"]),
                    "adr": float(tot2["adr"]),
                    "revpar": float(tot2["revpar"]),
                    "ingresos": float(tot2["ingresos"]),
                })
            df_prev = pd.DataFrame(rows_prev)

        # ---------- Preparación tabla ----------
        table_df = df_now.copy()
        if compare_e and not df_prev.empty:
            table_df = table_df.merge(df_prev, on="Corte", how="left", suffixes=("", " (LY)"))

        # Renombrar columnas para mostrar
        rename_map = {
            "ocupacion_pct": "Ocupación %",
            "adr": "ADR (€)",
            "revpar": "RevPAR (€)",
            "ingresos": "Ingresos (€)",
            "ocupacion_pct (LY)": "Ocupación % (LY)",
            "adr (LY)": "ADR (€) (LY)",
            "revpar (LY)": "RevPAR (€) (LY)",
            "ingresos (LY)": "Ingresos (€) (LY)",
        }
        table_df = table_df.rename(columns=rename_map)

        # Mostrar solo KPIs seleccionados
        cols_to_show = ["Corte"] + selected_kpis
        if compare_e:
            cols_to_show += [f"{kpi} (LY)" for kpi in selected_kpis]
        if "Ingresos (€)" in table_df.columns and "Ingresos (€)" not in cols_to_show:
            cols_to_show.append("Ingresos (€)")
        table_df = table_df[[c for c in cols_to_show if c in table_df.columns]]

        # Formato y colores condicionales
        GREEN = "background-color: #d4edda; color: #155724; font-weight: 600;"
        RED   = "background-color: #f8d7da; color: #721c24; font-weight: 600;"
        def style_row(r: pd.Series):
            s = pd.Series("", index=table_df.columns, dtype="object")
            for kpi in selected_kpis:
                kpi_ly = f"{kpi} (LY)"
                if kpi in r and kpi_ly in r and pd.notna(r[kpi]) and pd.notna(r[kpi_ly]):
                    try:
                        if float(r[kpi]) > float(r[kpi_ly]): s[kpi] = GREEN
                        elif float(r[kpi]) < float(r[kpi_ly]): s[kpi] = RED
                    except Exception:
                        pass
            return s

        styler = (
            table_df.style
            .apply(style_row, axis=1)
            .format({c: "{:.2f}%" if "Ocupación" in c else "{:.2f} €" for c in table_df.columns if c != "Corte"})
        )
        st.dataframe(styler, use_container_width=True)

        # Exportar a Excel con formato y colores
        import io
        buffer = io.BytesIO()
        df_excel = table_df.copy()
        # Divide ocupación por 100 para formato porcentaje en Excel
        for col in df_excel.columns:
            if "Ocupación" in col:
                df_excel[col] = df_excel[col] / 100
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_excel.to_excel(writer, index=False, sheet_name="Evolución")
            wb = writer.book
            ws = writer.sheets["Evolución"]
            for j, col in enumerate(df_excel.columns):
                ws.set_column(j, j, 18)
            fmt_pct = wb.add_format({"num_format": "0.00%", "align": "center"})
            fmt_eur = wb.add_format({"num_format": "€ #,##0.00", "align": "center"})
            fmt_int = wb.add_format({"num_format": "0", "align": "center"})
            fmt_green = wb.add_format({"bg_color": "#d4edda", "font_color": "#155724", "bold": True})
            fmt_red   = wb.add_format({"bg_color": "#f8d7da", "font_color": "#721c24", "bold": True})
            for idx, col in enumerate(df_excel.columns):
                if "Ocupación" in col:
                    ws.set_column(idx, idx, 18, fmt_pct)
                elif "ADR" in col or "RevPAR" in col or "Ingresos" in col:
                    ws.set_column(idx, idx, 18, fmt_eur)
                elif "Noches" in col:
                    ws.set_column(idx, idx, 18, fmt_int)
            # Colores condicionales
            n = len(df_excel)
            if n > 0 and compare_e:
                from xlsxwriter.utility import xl_rowcol_to_cell
                first_row = 1
                last_row  = first_row + n - 1
                for kpi in selected_kpis:
                    if f"{kpi} (LY)" in df_excel.columns:
                        i_a  = df_excel.columns.get_loc(kpi)
                        i_ly = df_excel.columns.get_loc(f"{kpi} (LY)")
                        a_cell  = xl_rowcol_to_cell(first_row, i_a,  row_abs=False, col_abs=True)
                        ly_cell = xl_rowcol_to_cell(first_row, i_ly, row_abs=False, col_abs=True)
                        ws.conditional_format(first_row, i_a, last_row, i_a, {
                            "type": "formula", "criteria": f"={a_cell}>{ly_cell}", "format": fmt_green
                        })
                        ws.conditional_format(first_row, i_a, last_row, i_a, {
                            "type": "formula", "criteria": f"={a_cell}<{ly_cell}", "format": fmt_red
                        })
            # Nombre de alojamientos o grupo arriba a la izquierda
            nombre_alojamientos = ", ".join(props_e) if props_e else "Todos"
            ws.write(0, 0, nombre_alojamientos, wb.add_format({"bold": True, "font_color": "#003366"}))
        st.download_button(
            "📥 Descargar evolución (Excel)",
            data=buffer.getvalue(),
            file_name="evolucion_por_corte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Descarga CSV
        st.download_button(
            "📥 Descargar evolución (CSV)",
            data=table_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="evolucion_por_corte.csv",
            mime="text/csv",
        )
    else:
        st.caption("Configura los parámetros y pulsa **Calcular evolución**.")
