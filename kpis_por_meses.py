import pandas as pd
import streamlit as st
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, load_groups

def render_kpis_por_meses(raw):
    with st.sidebar:
        st.header("Selección de grupo de alojamientos")
        groups = load_groups()
        group_names = sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        props_rc = groups[selected_group] if selected_group else []

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