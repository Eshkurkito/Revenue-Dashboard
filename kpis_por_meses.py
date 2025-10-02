import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_kpis_por_meses(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_m = st.date_input("Fecha de corte", value=date.today(), key="cutoff_m")

        # Gestión de grupos
        from utils import save_group_csv, load_groups, group_selector
        groups = load_groups()
        group_names = list(groups.keys())
        selected_group = st.selectbox("Grupo guardado", group_names) if group_names else None

        if selected_group:
            props_m = groups[selected_group]
        else:
            props_m = group_selector(
                "Filtrar alojamientos (opcional)",
                list(raw["Alojamiento"].unique()),
                key_prefix="props_m",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_m:
            save_group_csv(group_name, props_m)
            st.success(f"Grupo '{group_name}' guardado.")

        st.markdown("—")
        compare_m = st.checkbox(
            "Comparar con año anterior (mismo mes)", value=True, key="cmp_m"
        )
        inv_m = st.number_input(
            "Inventario actual (opcional)",
            min_value=0, value=0, step=1, key="inv_m"
        )
        inv_m_prev = st.number_input(
            "Inventario año anterior (opcional)",
            min_value=0, value=0, step=1, key="inv_m_prev"
        )
        # Selección de meses
        months = pd.date_range(
            start=date(date.today().year, 1, 1),
            end=date(date.today().year, 12, 31),
            freq="MS"
        ).to_period("M")
        selected_months_m = st.multiselect(
            "Selecciona meses",
            options=[str(m) for m in months],
            default=[str(pd.Timestamp.today().to_period("M"))],
            key="sel_months_m"
        )

    # KPIs por mes
    rows_actual, rows_prev, rows_prev_final = [], [], []
    for ym in selected_months_m:
        p = pd.Period(ym, freq="M")
        start_m = p.to_timestamp(how="start")
        end_m = p.to_timestamp(how="end")
        _bp, _tot = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_m),
            period_start=start_m,
            period_end=end_m,
            inventory_override=int(inv_m) if inv_m > 0 else None,
            filter_props=props_m if props_m else None,
        )
        rows_actual.append({"Mes": ym, **_tot})

        if compare_m:
            p_prev = p - 12
            start_prev = p_prev.to_timestamp(how="start")
            end_prev = p_prev.to_timestamp(how="end")
            cutoff_prev = pd.to_datetime(cutoff_m) - pd.DateOffset(years=1)
            _bp2, _tot_prev = compute_kpis(
                df_all=raw,
                cutoff=cutoff_prev,
                period_start=start_prev,
                period_end=end_prev,
                inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_prev.append({"Mes": ym, **_tot_prev})

            # Ingresos finales LY (corte = fin de mes LY)
            _bp3, _tot_prev_final = compute_kpis(
                df_all=raw,
                cutoff=end_prev,
                period_start=start_prev,
                period_end=end_prev,
                inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_prev_final.append({"Mes": ym, **_tot_prev_final})

    df_actual = pd.DataFrame(rows_actual).sort_values("Mes") if rows_actual else pd.DataFrame()
    df_prev = pd.DataFrame(rows_prev).sort_values("Mes") if rows_prev else pd.DataFrame()
    df_prev_final = pd.DataFrame(rows_prev_final).sort_values("Mes") if rows_prev_final else pd.DataFrame()

    st.subheader("KPIs por meses")
    help_block("KPIs por meses")

    if df_actual.empty:
        st.warning("Sin datos para los meses seleccionados.")
    else:
        st.dataframe(df_actual, use_container_width=True)
        csvm = df_actual.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar KPIs por mes (CSV)", data=csvm, file_name="kpis_por_mes.csv", mime="text/csv")

        # Tabla compacta y Excel con colores vs LY
        try:
            if compare_m and not df_prev.empty:
                act = df_actual[['Mes', 'adr', 'ocupacion_pct', 'ingresos']].rename(columns={
                    'adr': 'ADR (€)', 'ocupacion_pct': 'Ocupación %', 'ingresos': 'Ingresos (€)'
                })
                prev = df_prev[['Mes', 'adr', 'ocupacion_pct']].rename(columns={
                    'adr': 'ADR LY (€)', 'ocupacion_pct': 'Ocupación LY %'
                })
                prev_final = df_prev_final[['Mes', 'ingresos']].rename(columns={
                    'ingresos': 'Ingresos finales LY (€)'
                })
                export_df = act.merge(prev, on='Mes', how='left').merge(prev_final, on='Mes', how='left')
                export_df = export_df[['Mes', 'ADR (€)', 'ADR LY (€)', 'Ocupación %', 'Ocupación LY %', 'Ingresos (€)', 'Ingresos finales LY (€)']]
            else:
                export_df = df_actual[['Mes', 'adr', 'ocupacion_pct', 'ingresos']].rename(columns={
                    'adr': 'ADR (€)', 'ocupacion_pct': 'Ocupación %', 'ingresos': 'Ingresos (€)'
                })

            st.subheader("Tabla comparativa (compacta)")
            st.dataframe(export_df, use_container_width=True)

            # Exportar Excel con formatos condicionales
            import io
            buffer_xlsx = io.BytesIO()
            with pd.ExcelWriter(buffer_xlsx, engine="xlsxwriter") as writer:
                sheet_name = "KPIs por meses"
                export_df.to_excel(writer, index=False, sheet_name=sheet_name)
                wb = writer.book
                ws = writer.sheets[sheet_name]

                fmt_num2 = wb.add_format({"num_format": "0.00"})
                fmt_eur  = wb.add_format({"num_format": "€ #,##0.00"})

                for j, col in enumerate(export_df.columns):
                    w = int(min(30, max(12, export_df[col].astype(str).str.len().max() if not export_df.empty else 12)))
                    if col in ("ADR (€)", "ADR LY (€)", "Ingresos (€)", "Ingresos finales LY (€)"):
                        ws.set_column(j, j, w, fmt_eur)
                    elif col in ("Ocupación %", "Ocupación LY %"):
                        ws.set_column(j, j, w, fmt_num2)
                    else:
                        ws.set_column(j, j, w)

                from xlsxwriter.utility import xl_rowcol_to_cell
                fmt_green = wb.add_format({"bg_color": "#d4edda", "font_color": "#155724", "bold": True})
                fmt_red   = wb.add_format({"bg_color": "#f8d7da", "font_color": "#721c24", "bold": True})
                n = len(export_df)
                if n > 0 and compare_m:
                    first_row = 1
                    last_row = first_row + n - 1

                    def add_cmp(actual_col: str, ly_col: str):
                        if actual_col in export_df.columns and ly_col in export_df.columns:
                            i_a  = export_df.columns.get_loc(actual_col)
                            i_ly = export_df.columns.get_loc(ly_col)
                            a_cell  = xl_rowcol_to_cell(first_row, i_a,  row_abs=False, col_abs=True)
                            ly_cell = xl_rowcol_to_cell(first_row, i_ly, row_abs=False, col_abs=True)
                            ws.conditional_format(first_row, i_a, last_row, i_a, {
                                "type": "formula", "criteria": f"={a_cell}>{ly_cell}", "format": fmt_green
                            })
                            ws.conditional_format(first_row, i_a, last_row, i_a, {
                                "type": "formula", "criteria": f"={a_cell}<{ly_cell}", "format": fmt_red
                            })

                    add_cmp("ADR (€)", "ADR LY (€)")
                    add_cmp("Ocupación %", "Ocupación LY %")
                    add_cmp("Ingresos (€)", "Ingresos finales LY (€)")

            st.download_button(
                "📥 Descargar Excel (.xlsx) – KPIs por meses",
                data=buffer_xlsx.getvalue(),
                file_name="kpis_por_mes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning(f"No se pudo generar el Excel: {e}")