import streamlit as st
import pandas as pd
import numpy as np
import calendar
import io
from datetime import date
from pathlib import Path

# --- Fallbacks si no existe utils.py ---
try:
    from utils import period_inputs, group_selector, save_group_csv, load_groups, GROUPS_CSV
except Exception:
    GROUPS_CSV = Path(__file__).resolve().parent / "grupos_guardados.csv"

    def period_inputs(label_start, label_end, default_start, default_end, key_prefix="resumen_comp"):
        c1, c2 = st.columns(2)
        start = c1.date_input(label_start, value=default_start, key=f"{key_prefix}_start")
        end   = c2.date_input(label_end,   value=default_end,   key=f"{key_prefix}_end")
        return start, end

    def group_selector(label, options, key_prefix="resumen_comp", default=None):
        return st.multiselect(label, options=options, default=(default or []), key=f"{key_prefix}_props")

    def _read_csv_any(path: Path):
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return None

    def load_groups() -> dict[str, list[str]]:
        if not GROUPS_CSV.exists():
            return {}
        df = _read_csv_any(GROUPS_CSV)
        if df is None or df.empty:
            return {}
        cols = {c.lower(): c for c in df.columns}
        if not {"grupo", "alojamiento"}.issubset(set(cols.keys())):
            return {}
        gcol, pcol = cols["grupo"], cols["alojamiento"]
        d = (
            df.dropna(subset=[gcol, pcol])
              .astype({gcol: str, pcol: str})
              .groupby(gcol)[pcol].apply(list).to_dict()
        )
        return {g: list(dict.fromkeys(v)) for g, v in d.items()}

    def save_group_csv(name: str, props: list[str]):
        rows = [{"Grupo": name, "Alojamiento": p} for p in props]
        if GROUPS_CSV.exists():
            old = _read_csv_any(GROUPS_CSV)
            dfw = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if old is not None else pd.DataFrame(rows)
        else:
            dfw = pd.DataFrame(rows)
        dfw.to_csv(GROUPS_CSV, index=False, encoding="utf-8-sig")


def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: int = None,
    filter_props: list = None,
):
    df_cut = df_all[pd.to_datetime(df_all["Fecha alta"]) <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    inv_detected = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    inv_eff = int(inventory_override) if (inventory_override is not None and int(inventory_override) > 0) else int(inv_detected)
    days = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days + 1
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

    price = pd.to_numeric(df_cut["Alquiler con IVA (€)"], errors="coerce").fillna(0)
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
    if raw is None or raw.empty:
        st.warning("No hay datos cargados.")
        st.stop()

    today = date.today()
    default_start = today.replace(day=1)
    last_day = calendar.monthrange(today.year, today.month)[1]
    default_end = date(today.year, today.month, last_day)

    with st.sidebar:
        st.header("Parámetros – Resumen comparativo")
        cutoff_rc = st.date_input("Fecha de corte", value=today, key="cut_resumen_comp")
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            default_start, default_end,
            "resumen_comp"
        )

        view_mode = st.radio("Modo de vista", ["Por periodo (actual)", "Por meses (con resumen general)"], index=0)

        st.header("Gestión de grupos")
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_rc = groups[selected_group]
            if st.button(f"Eliminar grupo '{selected_group}'"):
                try:
                    df = pd.read_csv(GROUPS_CSV, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(GROUPS_CSV)
                df = df[df["Grupo"] != selected_group]
                df.to_csv(GROUPS_CSV, index=False, encoding="utf-8-sig")
                st.success(f"Grupo '{selected_group}' eliminado.")
                try: st.rerun()
                except Exception: pass
        else:
            if "Alojamiento" not in raw.columns:
                st.warning("No se encontró la columna 'Alojamiento'.")
                st.stop()
            props_rc = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_rc",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_rc:
            save_group_csv(group_name, props_rc)
            st.success(f"Grupo '{group_name}' guardado.")

    st.subheader("📊 Resumen comparativo por alojamiento")

    days_period = (pd.to_datetime(end_rc) - pd.to_datetime(start_rc)).days + 1
    if days_period <= 0:
        st.error("El periodo no es válido (fin anterior o igual al inicio). Ajusta fechas.")
        st.stop()

    props_sel = props_rc if props_rc else None

    def _by_prop_with_occ(cutoff_dt, start_dt, end_dt, props_sel=None):
        days_local = (pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days + 1
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
        out["Ocupación %"] = (out["Noches ocupadas"] / days_local * 100.0).astype(float)
        return out[["Alojamiento","ADR","Ocupación %","Ingresos"]]

    def _make_resumen(start_dt, end_dt, cutoff_dt, props_sel):
        now_df = _by_prop_with_occ(cutoff_dt, start_dt, end_dt, props_sel).rename(columns={
            "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
        })

        ly_df = _by_prop_with_occ(
            pd.to_datetime(cutoff_dt) - pd.DateOffset(years=1),
            pd.to_datetime(start_dt) - pd.DateOffset(years=1),
            pd.to_datetime(end_dt)   - pd.DateOffset(years=1),
            props_sel
        ).rename(columns={
            "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
        })

        ly_final_df = _by_prop_with_occ(
            pd.to_datetime(end_dt)   - pd.DateOffset(years=1),
            pd.to_datetime(start_dt) - pd.DateOffset(years=1),
            pd.to_datetime(end_dt)   - pd.DateOffset(years=1),
            props_sel
        )
        ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

        ly2_df = _by_prop_with_occ(
            pd.to_datetime(cutoff_dt) - pd.DateOffset(years=2),
            pd.to_datetime(start_dt)  - pd.DateOffset(years=2),
            pd.to_datetime(end_dt)    - pd.DateOffset(years=2),
            props_sel
        ).rename(columns={
            "ADR":"ADR LY-2", "Ocupación %":"Ocupación LY-2 %", "Ingresos":"Ingresos LY-2 (€)"
        })

        ly2_final_df = _by_prop_with_occ(
            pd.to_datetime(end_dt)    - pd.DateOffset(years=2),
            pd.to_datetime(start_dt)  - pd.DateOffset(years=2),
            pd.to_datetime(end_dt)    - pd.DateOffset(years=2),
            props_sel
        )
        ly2_final_df = ly2_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY-2 (€)"})

        resumen = now_df.merge(ly_df, on="Alojamiento", how="outer") \
                        .merge(ly2_df[["Alojamiento","Ingresos LY-2 (€)"]], on="Alojamiento", how="left") \
                        .merge(ly_final_df, on="Alojamiento", how="left") \
                        .merge(ly2_final_df, on="Alojamiento", how="left")

        if resumen.empty:
            return pd.DataFrame(columns=[
                "Alojamiento",
                "ADR actual","ADR LY",
                "Ocupación actual %","Ocupación LY %",
                "Ingresos actuales (€)","Ingresos LY (€)","Ingresos LY-2 (€)",
                "Ingresos finales LY (€)","Ingresos finales LY-2 (€)"
            ])

        resumen = resumen.reindex(columns=[
            "Alojamiento",
            "ADR actual","ADR LY",
            "Ocupación actual %","Ocupación LY %",
            "Ingresos actuales (€)","Ingresos LY (€)","Ingresos LY-2 (€)",
            "Ingresos finales LY (€)","Ingresos finales LY-2 (€)"
        ])
        return resumen

    def _month_ranges(start, end):
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        starts = pd.date_range(start=s, end=e, freq='MS')
        out = []
        for st in starts:
            y, m = st.year, st.month
            last = calendar.monthrange(y, m)[1]
            st_date = pd.Timestamp(date(y, m, 1))
            ed_date = pd.Timestamp(date(y, m, last))
            if ed_date < s or st_date > e:
                continue
            out.append((max(st_date, s), min(ed_date, e)))
        return out

    GREEN = "background-color: #d4edda; color: #155724; font-weight: 600;"
    RED   = "background-color: #f8d7da; color: #721c24; font-weight: 600;"
    def _style_row_factory(resumen_df):
        def _style_row(r: pd.Series):
            s = pd.Series("", index=resumen_df.columns, dtype="object")
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
        return _style_row

    def _export_excel_general_and_months(resumen_general, months_list, resumen_by_months):
        buffer = io.BytesIO()
        try:
            from xlsxwriter.utility import xl_rowcol_to_cell
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                resumen_general.to_excel(writer, index=False, sheet_name="Resumen")
                wb = writer.book

                def _write_sheet(df, name):
                    ws = writer.sheets[name]

                    fmt_green = wb.add_format({"bg_color": "#d4edda", "font_color": "#155724", "bold": True})
                    fmt_red   = wb.add_format({"bg_color": "#f8d7da", "font_color": "#721c24", "bold": True})
                    fmt_currency = wb.add_format({"num_format": "#,##0.00 €"})
                    fmt_percent  = wb.add_format({"num_format": '0.00 " %"'})

                    # Ajustar ancho y aplicar formato por columna
                    for j, col in enumerate(df.columns):
                        width = int(min(38, max(12, df[col].astype(str).str.len().max() if not df.empty else 12)))
                        if col in ["ADR actual", "ADR LY",
                                   "Ingresos actuales (€)", "Ingresos LY (€)", "Ingresos LY-2 (€)",
                                   "Ingresos finales LY (€)", "Ingresos finales LY-2 (€)"]:
                            ws.set_column(j, j, width, fmt_currency)
                        elif col in ["Ocupación actual %", "Ocupación LY %"]:
                            ws.set_column(j, j, width, fmt_percent)
                        else:
                            ws.set_column(j, j, width)

                    # Condicionales (mantener resaltado comparativo)
                    pairs = [
                        ("ADR actual", "ADR LY"),
                        ("Ocupación actual %", "Ocupación LY %"),
                        ("Ingresos actuales (€)", "Ingresos LY (€)"),
                    ]
                    n = len(df)
                    if n > 0:
                        first_row = 1
                        last_row  = first_row + n - 1
                        for a_col, ly_col in pairs:
                            if a_col in df.columns and ly_col in df.columns:
                                a_idx  = df.columns.get_loc(a_col)
                                ly_idx = df.columns.get_loc(ly_col)
                                a_cell  = xl_rowcol_to_cell(first_row, a_idx,  row_abs=False, col_abs=True)
                                ly_cell = xl_rowcol_to_cell(first_row, ly_idx, row_abs=False, col_abs=True)
                                ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                                    "type": "formula", "criteria": f"={a_cell}>{ly_cell}", "format": fmt_green
                                })
                                ws.conditional_format(first_row, a_idx, last_row, a_idx, {
                                    "type": "formula", "criteria": f"={a_cell}<{ly_cell}", "format": fmt_red
                                })

                writer.sheets["Resumen"] = writer.sheets.get("Resumen")
                _write_sheet(resumen_general, "Resumen")

                for key, dfm in resumen_by_months.items():
                    name = key[:31]
                    dfm.to_excel(writer, index=False, sheet_name=name)
                    writer.sheets[name] = writer.sheets.get(name)
                    _write_sheet(dfm, name)
        except Exception:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                resumen_general.to_excel(writer, index=False, sheet_name="Resumen")
                for key, dfm in resumen_by_months.items():
                    name = key[:31]
                    dfm.to_excel(writer, index=False, sheet_name=name)
        return buffer.getvalue()

    # Generación según modo
    if view_mode == "Por periodo (actual)":
        resumen = _make_resumen(start_rc, end_rc, cutoff_rc, props_sel)

        if resumen.empty:
            st.info(
                "No hay reservas que intersecten el periodo **a la fecha de corte** seleccionada.\n"
                "- Prueba a ampliar el periodo o mover la fecha de corte.\n"
                "- Recuerda que se incluyen reservas con **Fecha alta ≤ corte** y estancia dentro del periodo."
            )
            st.stop()

        styler = (
            resumen.style
            .apply(_style_row_factory(resumen), axis=1)
            .format({
                "ADR actual": "{:.2f} €", "ADR LY": "{:.2f} €",
                "Ocupación actual %": "{:.2f}%", "Ocupación LY %": "{:.2f}%",
                "Ingresos actuales (€)": "{:.2f} €", "Ingresos LY (€)": "{:.2f} €",
                "Ingresos LY-2 (€)": "{:.2f} €",
                "Ingresos finales LY (€)": "{:.2f} €",
                "Ingresos finales LY-2 (€)": "{:.2f} €",
            })
        )
        st.dataframe(styler, use_container_width=True)

        csv_bytes = resumen.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar CSV", data=csv_bytes,
                           file_name="resumen_comparativo.csv", mime="text/csv")

        xlsx_bytes = _export_excel_general_and_months(resumen, {}, {})
        st.download_button(
            "📥 Descargar Excel (.xlsx)",
            data=xlsx_bytes,
            file_name="resumen_comparativo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        month_ranges = _month_ranges(start_rc, end_rc)
        if not month_ranges:
            st.info("No hay meses en el periodo seleccionado.")
            st.stop()

        resumen_general = _make_resumen(start_rc, end_rc, cutoff_rc, props_sel)
        resumen_by_months = {}
        for sdt, edt in month_ranges:
            label = sdt.strftime("%B %Y")
            resumen_by_months[label] = _make_resumen(sdt, edt, cutoff_rc, props_sel)

        tabs = st.tabs(["Resumen general"] + list(resumen_by_months.keys()))
        with tabs[0]:
            if resumen_general.empty:
                st.info("No hay datos en el resumen general para el periodo seleccionado.")
            else:
                sty = (
                    resumen_general.style
                    .apply(_style_row_factory(resumen_general), axis=1)
                    .format({
                        "ADR actual": "{:.2f} €", "ADR LY": "{:.2f} €",
                        "Ocupación actual %": "{:.2f}%", "Ocupación LY %": "{:.2f}%",
                        "Ingresos actuales (€)": "{:.2f} €", "Ingresos LY (€)": "{:.2f} €",
                        "Ingresos LY-2 (€)": "{:.2f} €",
                        "Ingresos finales LY (€)": "{:.2f} €",
                        "Ingresos finales LY-2 (€)": "{:.2f} €",
                    })
                )
                st.dataframe(sty, use_container_width=True)

        for i, (label, dfm) in enumerate(resumen_by_months.items(), start=1):
            with tabs[i]:
                if dfm.empty:
                    st.info(f"No hay datos para {label}.")
                else:
                    sty_m = (
                        dfm.style
                        .apply(_style_row_factory(dfm), axis=1)
                        .format({
                            "ADR actual": "{:.2f} €", "ADR LY": "{:.2f} €",
                            "Ocupación actual %": "{:.2f}%", "Ocupación LY %": "{:.2f}%",
                            "Ingresos actuales (€)": "{:.2f} €", "Ingresos LY (€)": "{:.2f} €",
                            "Ingresos LY-2 (€)": "{:.2f} €",
                            "Ingresos finales LY (€)": "{:.2f} €",
                            "Ingresos finales LY-2 (€)": "{:.2f} €",
                        })
                    )
                    st.dataframe(sty_m, use_container_width=True)

        csv_bytes = resumen_general.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar CSV (resumen general)", data=csv_bytes,
                           file_name="resumen_comparativo_resumen.csv", mime="text/csv")

        xlsx_bytes = _export_excel_general_and_months(resumen_general, month_ranges, resumen_by_months)
        st.download_button(
            "📥 Descargar Excel (.xlsx) (Resumen + por meses)",
            data=xlsx_bytes,
            file_name="resumen_comparativo_por_meses.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )