import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_resumen_comparativo(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cutoff_rc")
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "rc"
        )
        props_rc = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_rc",
            default=[]
        )
        compare_rc = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=True, key="cmp_rc"
        )

    # Actual
    now_df = _by_prop_with_occ(raw, cutoff_rc, start_rc, end_rc, props_rc).rename(columns={
        "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
    })

    # LY (mismo periodo y cutoff -1 año)
    ly_df = _by_prop_with_occ(
        raw,
        pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_rc
    ).rename(columns={
        "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
    })

    # LY final (resultado): mismo periodo LY, pero corte = fin del periodo LY
    ly_final_df = _by_prop_with_occ(
        raw,
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),  # corte = fin del periodo LY
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_rc
    )
    # De este solo necesitamos los ingresos finales
    ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    # Merge total
    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer") \
                    .merge(ly_final_df, on="Alojamiento", how="left")

    # Estilos comparativos
    GREEN = "background-color: #d4edda; color: #155724; font-weight: 600;"
    RED   = "background-color: #f8d7da; color: #721c24; font-weight: 600;"
    def _style_row(r: pd.Series):
        s = pd.Series("", index=resumen.columns, dtype="object")
        def mark(a, b):
            va, vb = r.get(a), r.get(b)
            if pd.notna(va) and pd.notna(vb):
                try:
                    if float(va) > float(vb): s[a] = GREEN
                    elif float(va) < float(vb): s[a] = RED
                except Exception:
                    pass
        mark("ADR actual", "ADR LY")
        mark("Ocupación actual %", "Ocupación LY %")
        mark("Ingresos actuales (€)", "Ingresos LY (€)")
        return s
    styler = (
        resumen.style
        .apply(_style_row, axis=1)
        .format({
            "ADR actual": "{:.2f}", "ADR LY": "{:.2f}",
            "Ocupación actual %": "{:.2f}", "Ocupación LY %": "{:.2f}",
            "Ingresos actuales (€)": "{:.2f}", "Ingresos LY (€)": "{:.2f}",
            "Ingresos finales LY (€)": "{:.2f}",
        })
    )

    st.subheader("Resumen comparativo")
    help_block("Resumen Comparativo")
    st.dataframe(styler, use_container_width=True)
    # Exportar a Excel
    import io
    output = io.BytesIO()
    resumen.to_excel(output, index=False, sheet_name="Comparativo")
    output.seek(0)
    st.download_button(
        "📥 Descargar detalle (Excel)",
        data=output,
        file_name="detalle_comparativo_por_alojamiento.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def _by_prop_with_occ(raw, cutoff_dt, start_dt, end_dt, props_sel=None):
    days_period = (pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days + 1
    by_prop, _ = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_dt),
        period_start=pd.to_datetime(start_dt),
        period_end=pd.to_datetime(end_dt),
        inventory_override=None,
        filter_props=props_sel if props_sel else None,
    )
    if by_prop.empty:
        return pd.DataFrame(columns=["Alojamiento","ADR","Ocupación %","Ingresos"])
    out = by_prop.copy()
    # Si no existe la columna, la creamos con 0
    if "Noches ocupadas" not in out.columns:
        out["Noches ocupadas"] = 0
    out["Ocupación %"] = (out["Noches ocupadas"] / days_period * 100.0).astype(float)
    return out[["Alojamiento","ADR","Ocupación %","Ingresos"]]