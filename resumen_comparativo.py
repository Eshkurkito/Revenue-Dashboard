import pandas as pd
import streamlit as st
from datetime import date
from utils import period_inputs, group_selector

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: int = None,
    filter_props: list = None,
):
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    inv_detected = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    inv_eff = int(inventory_override) if (inventory_override is not None and int(inventory_override) > 0) else int(inv_detected)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv_eff * days if days > 0 else 0

    if df_cut.empty:
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": noches_disponibles,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = pd.Timedelta(days=1)
    start_ns = pd.to_datetime(period_start)
    end_excl_ns = pd.to_datetime(period_end) + one_day

    arr_e = pd.to_datetime(df_cut["Fecha entrada"])
    arr_s = pd.to_datetime(df_cut["Fecha salida"])

    total_nights = (arr_s - arr_e).dt.days.clip(lower=0)
    ov_start = arr_e.clip(lower=start_ns)
    ov_end = arr_s.clip(upper=end_excl_ns)
    ov_days = (ov_end - ov_start).dt.days.clip(lower=0)

    price = df_cut["Alquiler con IVA (€)"].astype(float)
    share = ov_days / total_nights.replace(0, 1)
    income = price * share

    props = df_cut["Alojamiento"].astype(str)
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = by_prop["Ingresos"] / by_prop["Noches ocupadas"].replace(0, 1)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }
    return by_prop, tot

def render_resumen_comparativo(raw):
    if raw is None:
        st.warning("No hay datos cargados.")
        st.stop()

    with st.sidebar:
        st.header("Parámetros – Resumen comparativo")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cut_resumen_comp")
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "resumen_comp"
        )
        props_rc = group_selector(
            "Alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="resumen_comp",
            default=[]
        )

    st.subheader("📊 Resumen comparativo por alojamiento")

    days_period = (pd.to_datetime(end_rc) - pd.to_datetime(start_rc)).days + 1
    if days_period <= 0:
        st.error("El periodo no es válido (fin anterior o igual al inicio). Ajusta fechas.")
        st.stop()

    props_sel = props_rc if props_rc else None

    def _by_prop_with_occ(cutoff_dt, start_dt, end_dt, props_sel=None):
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
        out["Ocupación %"] = (out["Noches ocupadas"] / days_period * 100.0).astype(float)
        return out[["Alojamiento","ADR","Ocupación %","Ingresos"]]

    # Actual
    now_df = _by_prop_with_occ(cutoff_rc, start_rc, end_rc, props_sel).rename(columns={
        "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
    })

    # LY (mismo periodo y cutoff -1 año)
    ly_df = _by_prop_with_occ(
        pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    ).rename(columns={
        "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
    })

    # LY final (resultado): mismo periodo LY, pero corte = fin del periodo LY
    ly_final_df = _by_prop_with_occ(
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    )
    ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    # Merge total
    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer") \
                    .merge(ly_final_df, on="Alojamiento", how="left")

    if resumen.empty:
        st.info(
            "No hay reservas que intersecten el periodo **a la fecha de corte** seleccionada.\n"
            "- Prueba a ampliar el periodo o mover la fecha de corte.\n"
            "- Recuerda que se incluyen reservas con **Fecha alta ≤ corte** y estancia dentro del periodo."
        )
        st.stop()

    resumen = resumen.reindex(columns=[
        "Alojamiento",
        "ADR actual","ADR LY",
        "Ocupación actual %","Ocupación LY %",
        "Ingresos actuales (€)","Ingresos LY (€)",
        "Ingresos finales LY (€)"
    ])

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
    st.dataframe(styler, use_container_width=True)

    # Descargas
    csv_bytes = resumen.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV", data=csv_bytes,
                       file_name="resumen_comparativo.csv", mime="text/csv")

    import io
    buffer = io.BytesIO()
    try:
        from xlsxwriter.utility import xl_rowcol_to_cell
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
            wb = writer.book
            ws = writer.sheets["Resumen"]
            for j, col in enumerate(resumen.columns):
                width = int(min(38, max(12, resumen[col].astype(str).str.len().max() if not resumen.empty else 12)))
                ws.set_column(j, j, width)
            fmt_green = wb.add_format({"bg_color": "#d4edda", "font_color": "#155724", "bold": True})
            fmt_red   = wb.add_format({"bg_color": "#f8d7da", "font_color": "#721c24", "bold": True})
            pairs = [
                ("ADR actual", "ADR LY"),
                ("Ocupación actual %", "Ocupación LY %"),
                ("Ingresos actuales (€)", "Ingresos LY (€)"),
            ]
            n = len(resumen)
            if n > 0:
                first_row = 1
                last_row  = first_row + n - 1
                for a_col, ly_col in pairs:
                    a_idx  = resumen.columns.get_loc(a_col)
                    ly_idx = resumen.columns.get_loc(ly_col)
                    a_cell  = xl_rowcol_to_cell(first_row, a_idx,  row_abs=False, col_abs=True)
                    ly_cell = xl_rowcol_to_cell(first_row, ly_idx, row_abs=False, col_abs=True)
                    ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                        "type": "formula", "criteria": f"={a_cell}>{ly_cell}", "format": fmt_green
                    })
                    ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                        "type": "formula", "criteria": f"={a_cell}<{ly_cell}", "format": fmt_red
                    })
    except Exception:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
    st.download_button(
        "📥 Descargar Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="resumen_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )