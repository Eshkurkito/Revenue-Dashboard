import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_evolucion_corte(raw):
    if raw is None:
        st.stop()
    
    if not isinstance(raw, pd.DataFrame):
        st.error("No se han cargado datos o el formato no es correcto.")
        st.stop()
    if "Alojamiento" not in raw.columns:
        st.error("El archivo no contiene la columna 'Alojamiento'.")
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

        # Grupo de alojamientos
        st.header("Grupo de alojamientos")
        groups = load_groups()
        group_names = ["Todos"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        if selected_group == "Todos":
            props_group = sorted(list(raw["Alojamiento"].dropna().unique()))
        else:
            props_group = groups[selected_group] if selected_group in groups else []

        st.header("Filtrar alojamientos (opcional)")
        props_e = st.multiselect(
            "Alojamientos a mostrar",
            options=props_group,
            default=props_group,
            key="props_evo"
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

        # ---------- Preparación long-form para graficar ----------
        kpi_map = {
            "Ocupación %": ("ocupacion_pct", "occ"),
            "ADR (€)":     ("adr", "eur"),
            "RevPAR (€)":  ("revpar", "eur"),
        }
        sel_items = [(k, *kpi_map[k]) for k in selected_kpis]

        def to_long(df, label_suffix="Actual"):
            out = []
            for lbl, col, kind in sel_items:
                if col in df.columns:
                    tmp = df[["Corte", col]].copy()
                    tmp["metric_label"] = lbl if label_suffix == "Actual" else f"{lbl} (LY)"
                    tmp["value"] = tmp[col].astype(float)
                    tmp["kind"] = kind
                    tmp["series"] = label_suffix
                    out.append(tmp[["Corte", "metric_label", "value", "kind", "series"]])
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

        long_now  = to_long(df_now, "Actual")
        long_prev = to_long(df_prev, "LY") if compare_e and not df_prev.empty else pd.DataFrame()
        long_all  = pd.concat([long_now, long_prev], ignore_index=True) if not long_prev.empty else long_now

        # ==========================
        #     G R Á F I C A S
        # ==========================
        import altair as alt

        nearest = alt.selection_point(fields=["Corte"], nearest=True, on="mousemove", empty="none")

        def build_layer(data, kind, axis_orient="left", color_map=None, dash_ly=True):
            if data.empty:
                return None
            dfk = data[data["kind"] == kind]
            if dfk.empty:
                return None
            _colors = color_map or {
                "Ocupación %": "#1f77b4",
                "ADR (€)": "#ff7f0e",
                "RevPAR (€)": "#2ca02c",
                "Ocupación % (LY)": "#1f77b4",
                "ADR (€) (LY)": "#ff7f0e",
                "RevPAR (€) (LY)": "#2ca02c",
            }
            line = (
                alt.Chart(dfk)
                .mark_line(strokeWidth=2, interpolate="monotone", point=alt.OverlayMarkDef(size=30, filled=True))
                .encode(
                    x=alt.X("Corte:T", title="Fecha de corte"),
                    y=alt.Y(
                        "value:Q",
                        axis=alt.Axis(orient=axis_orient, title=list(dfk["metric_label"].unique())[0])
                    ),
                    color=alt.Color("metric_label:N", scale=alt.Scale(domain=list(_colors.keys()),
                                                                      range=[_colors[k] for k in _colors]),
                                    legend=None),
                    detail="metric_label:N",
                    tooltip=[alt.Tooltip("Corte:T", title="Día"),
                             alt.Tooltip("metric_label:N", title="KPI"),
                             alt.Tooltip("value:Q", title="Valor", format=".2f")],
                )
            )
            pts_hover = (
                alt.Chart(dfk)
                .mark_point(size=90, filled=True)
                .encode(
                    x="Corte:T",
                    y="value:Q",
                    color=alt.Color("metric_label:N", scale=alt.Scale(domain=list(_colors.keys()),
                                                                      range=[_colors[k] for k in _colors]),
                                    legend=None),
                    detail="metric_label:N",
                )
                .transform_filter(nearest)
            )
            if " (LY)" in " ".join(dfk["metric_label"].unique()):
                line = line.encode(strokeDash=alt.condition(
                    "indexof(datum.metric_label, '(LY)') >= 0",
                    alt.value([5, 3]), alt.value([0, 0])
                ), opacity=alt.condition(
                    "indexof(datum.metric_label, '(LY)') >= 0",
                    alt.value(0.35), alt.value(1.0)
                ))
            return alt.layer(line, pts_hover)

        selectors = (
            alt.Chart(long_all)
            .mark_rule(opacity=0)
            .encode(x="Corte:T")
            .add_params(nearest)
        )
        vline = (
            alt.Chart(long_all)
            .mark_rule(color="#666", strokeWidth=1)
            .encode(x="Corte:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        )

        occ_selected   = any(kind == "occ" for _, _, kind in sel_items)
        euros_selected = any(kind == "eur" for _, _, kind in sel_items)

        left_layer  = build_layer(long_all, "occ", axis_orient="left")
        right_orient = "right" if (occ_selected and euros_selected) else "left"
        right_layer = build_layer(long_all, "eur", axis_orient=right_orient)

        layers = [selectors]
        if left_layer is not None:
            layers.append(left_layer)
        if right_layer is not None:
            layers.append(right_layer)
        layers.append(vline)

        chart = alt.layer(*layers).resolve_scale(
            y="independent" if (occ_selected and euros_selected) else "shared"
        ).properties(height=380)

        zoomx = alt.selection_interval(bind="scales", encodings=["x"])
        st.altair_chart(chart.add_params(zoomx), use_container_width=True)

        # ---------- Preparación tabla ----------
        table_df = df_now.copy()
        if compare_e and not df_prev.empty:
            table_df = table_df.merge(df_prev, on="Corte", how="left", suffixes=("", " (LY)"))

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
            nombre_alojamientos = ", ".join(props_e) if props_e else "Todos"
            ws.write(0, 0, nombre_alojamientos, wb.add_format({"bold": True, "font_color": "#003366"}))
        st.download_button(
            "📥 Descargar evolución (Excel)",
            data=buffer.getvalue(),
            file_name="evolucion_por_corte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            "📥 Descargar evolución (CSV)",
            data=table_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="evolucion_por_corte.csv",
            mime="text/csv",
        )
    else:
        st.caption("Configura los parámetros y pulsa **Calcular evolución**.")
