import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, load_groups

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm = {c: str(c).strip().lower() for c in df.columns}
    def find(*keys):
        for col, n in norm.items():
            for k in keys:
                if n == k or k in n:
                    return col
        return None
    mapping = {}
    a = find("alojamiento","propiedad","listing","unidad","apartamento","room","unit")
    fi = find("fecha entrada","check in","entrada","arrival")
    fo = find("fecha salida","check out","salida","departure")
    fa = find("fecha alta","creado","booking","reserva","created")
    rev = find("alquiler con iva","ingresos","revenue","importe","total")
    if a:   mapping[a] = "Alojamiento"
    if fi:  mapping[fi] = "Fecha entrada"
    if fo:  mapping[fo] = "Fecha salida"
    if fa:  mapping[fa] = "Fecha alta"
    if rev: mapping[rev] = "Alquiler con IVA (€)"
    return df.rename(columns=mapping) if mapping else df

def render_kpis_por_meses(raw):
    with st.sidebar:
        st.header("Selección de grupo de alojamientos")
        groups = load_groups()
        # Añade opción "Todos" al principio
        group_names = ["Todos"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        if selected_group == "Todos":
            props_rc = sorted([str(x) for x in raw["Alojamiento"].dropna().unique()])
        else:
            props_rc = groups[selected_group] if selected_group else []

        # Filtro adicional por alojamiento dentro del grupo seleccionado
        st.header("Filtrar alojamientos (opcional)")
        props_rc = st.multiselect(
            "Alojamientos a mostrar",
            options=props_rc,
            default=props_rc,
            key="kpis_mes_selector"
        )

        st.header("Periodo")
        year = st.number_input("Año", min_value=2000, max_value=date.today().year, value=date.today().year)
        start_month = st.number_input("Mes inicial", min_value=1, max_value=12, value=1)
        end_month = st.number_input("Mes final", min_value=1, max_value=12, value=12)

    # Genera rango de meses
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = pd.Timestamp(year=year, month=end_month, day=1)
    months = pd.date_range(start=start, end=end, freq="MS").to_period("M")

    st.subheader(f"KPI por meses ({year})")

    resultados = []
    resultados_ly = []
    for mes in months:
        # Actual
        periodo_inicio = mes.to_timestamp()
        periodo_fin = (mes + 1).to_timestamp() - pd.Timedelta(days=1)
        by_prop, totales = compute_kpis(
            df_all=raw,
            cutoff=periodo_fin,
            period_start=periodo_inicio,
            period_end=periodo_fin,
            filter_props=props_rc
        )
        resultados.append({
            "Mes": mes.strftime("%Y-%m"),
            "Noches ocupadas": totales["noches_ocupadas"],
            "Noches disponibles": totales["noches_disponibles"],
            "Ocupación %": totales["ocupacion_pct"],
            "Ingresos": totales["ingresos"],
            "ADR": totales["adr"],
            "RevPAR": totales["revpar"]
        })

        # Año anterior (LY)
        periodo_inicio_ly = periodo_inicio - pd.DateOffset(years=1)
        periodo_fin_ly = periodo_fin - pd.DateOffset(years=1)
        by_prop_ly, totales_ly = compute_kpis(
            df_all=raw,
            cutoff=periodo_fin_ly,
            period_start=periodo_inicio_ly,
            period_end=periodo_fin_ly,
            filter_props=props_rc
        )
        resultados_ly.append({
            "Noches ocupadas LY": totales_ly["noches_ocupadas"],
            "Noches disponibles LY": totales_ly["noches_disponibles"],
            "Ocupación LY %": totales_ly["ocupacion_pct"],
            "Ingresos LY": totales_ly["ingresos"],
            "ADR LY": totales_ly["adr"],
            "RevPAR LY": totales_ly["revpar"]
        })

    # Une ambos resultados
    df_result = pd.DataFrame(resultados)
    df_result_ly = pd.DataFrame(resultados_ly)
    df_final = pd.concat([df_result, df_result_ly], axis=1)

    # Formato y colores tipo resumen comparativo
    GREEN = "background-color: #d4edda; color: #155724; font-weight: 600;"
    RED   = "background-color: #f8d7da; color: #721c24; font-weight: 600;"

    def style_row(r: pd.Series):
        s = pd.Series("", index=df_final.columns, dtype="object")
        # Marca verde si mejora respecto al año anterior, rojo si empeora
        for col, ly_col in [
            ("Ocupación %", "Ocupación LY %"),
            ("Ingresos", "Ingresos LY"),
            ("ADR", "ADR LY"),
            ("RevPAR", "RevPAR LY")
        ]:
            curr = r.get(col)
            prev = r.get(ly_col)
            if pd.notna(curr) and pd.notna(prev):
                try:
                    if float(curr) > float(prev): s[col] = GREEN
                    elif float(curr) < float(prev): s[col] = RED
                except Exception:
                    pass
        return s

    styler = (
        df_final.style
        .apply(style_row, axis=1)
        .format({
            "Noches ocupadas": "{:.0f}",
            "Noches disponibles": "{:.0f}",
            "Ocupación %": "{:.2f}%",
            "Ingresos": "{:.2f} €",
            "ADR": "{:.2f} €",
            "RevPAR": "{:.2f} €",
            "Noches ocupadas LY": "{:.0f}",
            "Noches disponibles LY": "{:.0f}",
            "Ocupación LY %": "{:.2f}%",
            "Ingresos LY": "{:.2f} €",
            "ADR LY": "{:.2f} €",
            "RevPAR LY": "{:.2f} €"
        })
    )

    st.dataframe(styler, use_container_width=True)

    # Descarga CSV
    st.download_button(
        "📥 Descargar CSV",
        data=df_final.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"kpis_mensuales_{year}.csv",
        mime="text/csv"
    )

    # Descarga Excel con formato y colores
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_final_excel = df_final.copy()
        # Divide ocupación por 100 para formato porcentaje en Excel
        for col in df_final_excel.columns:
            if "Ocupación" in col:
                df_final_excel[col] = df_final_excel[col] / 100

        df_final_excel.to_excel(writer, index=False, sheet_name="KPIs")
        wb = writer.book
        ws = writer.sheets["KPIs"]

        # Ajusta ancho de columnas
        for j, col in enumerate(df_final_excel.columns):
            ws.set_column(j, j, 18)

        # Formatos
        fmt_pct = wb.add_format({"num_format": "0.00%", "align": "center"})
        fmt_eur = wb.add_format({"num_format": "€ #,##0.00", "align": "center"})
        fmt_int = wb.add_format({"num_format": "0", "align": "center"})
        fmt_green = wb.add_format({"bg_color": "#d4edda", "font_color": "#155724", "bold": True})
        fmt_red   = wb.add_format({"bg_color": "#f8d7da", "font_color": "#721c24", "bold": True})

        # Aplica formato por columna
        for idx, col in enumerate(df_final_excel.columns):
            if "Ocupación" in col:
                ws.set_column(idx, idx, 18, fmt_pct)
            elif "Ingresos" in col or "ADR" in col or "RevPAR" in col:
                ws.set_column(idx, idx, 18, fmt_eur)
            elif "Noches" in col:
                ws.set_column(idx, idx, 18, fmt_int)

        # Colores comparativos
        pairs = [
            ("Ocupación %", "Ocupación LY %"),
            ("Ingresos", "Ingresos LY"),
            ("ADR", "ADR LY"),
            ("RevPAR", "RevPAR LY")
        ]
        n = len(df_final_excel)
        if n > 0:
            first_row = 1
            last_row  = first_row + n - 1
            from xlsxwriter.utility import xl_rowcol_to_cell
            for a_col, ly_col in pairs:
                a_idx  = df_final_excel.columns.get_loc(a_col)
                ly_idx = df_final_excel.columns.get_loc(ly_col)
                a_cell  = xl_rowcol_to_cell(first_row, a_idx,  row_abs=False, col_abs=True)
                ly_cell = xl_rowcol_to_cell(first_row, ly_idx, row_abs=False, col_abs=True)
                ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                    "type": "formula", "criteria": f"={a_cell}>{ly_cell}", "format": fmt_green
                })
                ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                    "type": "formula", "criteria": f"={a_cell}<{ly_cell}", "format": fmt_red
                })

        # Nombre de alojamientos a mostrar o grupo seleccionado arriba a la izquierda
        nombre_alojamientos = ", ".join(props_rc) if props_rc else f"Grupo: {selected_group}"
        ws.write(0, 0, nombre_alojamientos, wb.add_format({"bold": True, "font_color": "#003366"}))

    st.download_button(
        "📥 Descargar Excel",
        data=buffer.getvalue(),
        file_name=f"kpis_mensuales_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("No hay datos cargados. Sube un archivo en la barra lateral.")
        return
    df = _standardize_columns(raw.copy())
    needed = {"Fecha entrada","Fecha salida","Alquiler con IVA (€)"}
    if not needed.issubset(df.columns):
        st.warning("Faltan columnas necesarias (Fecha entrada, Fecha salida, Alquiler con IVA (€)).")
        return

    # Parámetros
    c1, c2 = st.columns(2)
    start = c1.date_input("Inicio", value=date.today().replace(day=1), key="kpm_start")
    end   = c2.date_input("Fin", value=date.today(), key="kpm_end")
    props = []
    if "Alojamiento" in df.columns:
        props = st.multiselect("Alojamientos (opcional)", sorted(df["Alojamiento"].astype(str).dropna().unique()),
                               default=[], key="kpm_props")

    # Filtro
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"]).copy()
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"])
    df["Fecha salida"]  = pd.to_datetime(df["Fecha salida"])
    if props:
        df = df[df["Alojamiento"].astype(str).isin(props)]

    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1)  # incluir noche del último día

    # Noches prorrateadas por mes y ADR por noche
    si = df["Fecha entrada"].dt.normalize().values.astype("datetime64[D]")
    eo = df["Fecha salida"].dt.normalize().values.astype("datetime64[D]")
    los_total = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    rate = (df["Alquiler con IVA (€)"] / los_total).astype(float).values

    # Genera rango mensual
    months = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")
    rows = []
    for m in months:
        ms = pd.Timestamp(m.start_time).to_datetime64()
        me = (pd.Timestamp(m.end_time) + pd.Timedelta(days=1)).to_datetime64()
        seg_s = np.maximum(si, ms)
        seg_e = np.minimum(eo, me)
        nights = (seg_e - seg_s).astype("timedelta64[D]").astype(int)
        mask = nights > 0
        if not np.any(mask):
            rows.append({"Mes": str(m), "Noches": 0, "Ingresos": 0.0, "ADR": 0.0, "RevPAR": 0.0})
            continue
        noches = nights[mask].sum()
        ingresos = float(np.sum(rate[mask] * nights[mask]))
        adr = ingresos / noches if noches > 0 else 0.0
        rows.append({"Mes": str(m), "Noches": int(noches), "Ingresos": ingresos, "ADR": adr, "RevPAR": adr})  # sin inventario
    res = pd.DataFrame(rows)

    st.dataframe(res.round(2), use_container_width=True)

    # Antes de graficar
    res["Mes"] = res["Mes"].astype(str)

    chart = (
        alt.Chart(res)
        .transform_fold(["Noches", "Ingresos", "ADR"], as_=["KPI", "Valor"])
        .mark_line(point=True)
        .encode(
            x=alt.X("Mes:N", title="Mes"),
            y=alt.Y("Valor:Q", title="Valor"),
            color=alt.Color("KPI:N", title="KPI"),
            tooltip=["Mes:N", "KPI:N", alt.Tooltip("Valor:Q", format=".2f")],
        )
        .properties(height=280)  # ← quita use_container_width de aquí
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)  # ← aquí sí

    st.download_button(
        "📥 Descargar (CSV)",
        data=res.to_csv(index=False).encode("utf-8-sig"),
        file_name="kpis_por_meses.csv",
        mime="text/csv",
    )