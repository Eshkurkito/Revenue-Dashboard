import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta

TITLE = "Informe de propietario"

# Intentar reutilizar compute_kpis si existe
try:
    from utils import compute_kpis
except Exception:
    compute_kpis = None

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Renombres suaves por si vienen otras etiquetas
    norm = {c: str(c).strip().lower() for c in df.columns}
    m = {}
    def pick(*alts):
        for k, v in norm.items():
            for a in alts:
                if v == a or a in v:
                    return k
        return None
    a = pick("alojamiento","propiedad","unidad","listing","apartamento")
    fi = pick("fecha entrada","check in","entrada")
    fo = pick("fecha salida","check out","salida")
    rev = pick("alquiler con iva", "ingresos", "revenue", "importe", "total")
    inv = pick("inventario", "noches inventario", "inventory")
    if a:   m[a] = "Alojamiento"
    if fi:  m[fi] = "Fecha entrada"
    if fo:  m[fo] = "Fecha salida"
    if rev: m[rev] = "Alquiler con IVA (€)"
    if inv: m[inv] = "Inventario (noches)"
    return df.rename(columns=m) if m else df

def _ensure_dt(s):
    s = pd.to_datetime(s, errors="coerce", dayfirst=True)
    # fallback excel serial
    if s.isna().mean() > 0.8:
        s2 = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="d", origin="1899-12-30", errors="coerce")
        if s2.notna().sum() > s.notna().sum():
            s = s2
    return s.dt.normalize()

def _simple_kpis(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, inventory_nights: float | None):
    # Requiere: Fecha entrada, Fecha salida, Alquiler con IVA (€)
    dfx = df.dropna(subset=["Fecha entrada","Fecha salida"]).copy()
    dfx["Fecha entrada"] = _ensure_dt(dfx["Fecha entrada"])
    dfx["Fecha salida"]  = _ensure_dt(dfx["Fecha salida"])
    dfx["Alquiler con IVA (€)"] = pd.to_numeric(dfx["Alquiler con IVA (€)"], errors="coerce").fillna(0.0)

    # Longitud reserva
    los_total = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    rate = dfx["Alquiler con IVA (€)"].values / los_total.values

    # Intersección de noches con el periodo [start, end] (incluye la noche de end)
    si = np.maximum(dfx["Fecha entrada"].values.astype("datetime64[D]"), start.to_datetime64())
    eo = np.minimum(dfx["Fecha salida"].values.astype("datetime64[D]"), (end + pd.Timedelta(days=1)).to_datetime64())
    nights = (eo - si).astype("timedelta64[D]").astype(int)
    mask = nights > 0

    noches_vendidas = int(nights[mask].sum())
    ingresos = float(np.sum(rate[mask] * nights[mask]))
    adr = ingresos / noches_vendidas if noches_vendidas > 0 else 0.0

    if inventory_nights is None:
        # Si viene inventario en el DF, úsalo (suma única)
        inv = None
        if "Inventario (noches)" in dfx.columns:
            inv = pd.to_numeric(dfx["Inventario (noches)"], errors="coerce").dropna()
            inv = float(inv.iloc[0]) if not inv.empty else None
        inventory_nights = inv

    if inventory_nights is None or inventory_nights <= 0:
        ocup = None
        revpar = None
    else:
        ocup = noches_vendidas / float(inventory_nights)
        revpar = ingresos / float(inventory_nights)

    return {
        "noches": noches_vendidas,
        "ingresos": ingresos,
        "adr": adr,
        "ocupacion_pct": (ocup or 0.0) * 100.0 if ocup is not None else None,
        "revpar": revpar,
    }

@st.cache_data(ttl=1800, show_spinner=False)
def _compute_periods(df: pd.DataFrame, start: date, end: date, props: list[str], inv_override: float | None):
    dfx = df.copy()
    if props and "Alojamiento" in dfx.columns:
        dfx = dfx[dfx["Alojamiento"].astype(str).isin(props)].copy()

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    if compute_kpis:
        # Usa compute_kpis del proyecto si existe
        _, tot = compute_kpis(
            df_all=_std_cols(dfx),
            cutoff=e,  # no relevante para estático
            period_start=s,
            period_end=e,
            inventory_override=int(inv_override) if inv_override and inv_override > 0 else None,
            filter_props=None  # ya filtrado
        )
        act = {
            "noches": int(tot.get("noches", 0)),
            "ingresos": float(tot.get("ingresos", 0.0)),
            "adr": float(tot.get("adr", 0.0)),
            "ocupacion_pct": float(tot.get("ocupacion_pct", 0.0)) if tot.get("ocupacion_pct") is not None else None,
            "revpar": float(tot.get("revpar", 0.0)) if tot.get("revpar") is not None else None,
        }
    else:
        act = _simple_kpis(_std_cols(dfx), s, e, inv_override)

    # LY (mismo rango hace 1 año)
    s_ly = s - pd.DateOffset(years=1)
    e_ly = e - pd.DateOffset(years=1)
    ly = _simple_kpis(_std_cols(dfx), s_ly, e_ly, inv_override)

    return act, ly

def _delta(a, b):
    if a is None or b is None:
        return None
    return a - b

def _pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0

def _monthly_series(df: pd.DataFrame, start: date, end: date, props: list[str]):
    dfx = _std_cols(df.copy())
    if props and "Alojamiento" in dfx.columns:
        dfx = dfx[dfx["Alojamiento"].astype(str).isin(props)].copy()
    dfx = dfx.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    dfx["Fecha entrada"] = _ensure_dt(dfx["Fecha entrada"])
    dfx["Fecha salida"]  = _ensure_dt(dfx["Fecha salida"])
    dfx["Alquiler con IVA (€)"] = pd.to_numeric(dfx["Alquiler con IVA (€)"], errors="coerce").fillna(0.0)

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    months = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")

    los_total = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    rate = (dfx["Alquiler con IVA (€)"] / los_total).values
    si = np.maximum(dfx["Fecha entrada"].values.astype("datetime64[D]"), s.to_datetime64())
    eo = np.minimum(dfx["Fecha salida"].values.astype("datetime64[D]"), (e + pd.Timedelta(days=1)).to_datetime64())

    rows = []
    for m in months:
        ms = pd.Timestamp(m.start_time).to_datetime64()
        me = (pd.Timestamp(m.end_time) + pd.Timedelta(days=1)).to_datetime64()
        seg_s = np.maximum(si, ms)
        seg_e = np.minimum(eo, me)
        nights = (seg_e - seg_s).astype("timedelta64[D]").astype(int)
        mask = nights > 0
        if np.any(mask):
            ingresos = float(np.sum(rate[mask] * nights[mask]))
            rows.append({"Mes": str(m), "Ingresos": ingresos})
        else:
            rows.append({"Mes": str(m), "Ingresos": 0.0})
    act = pd.DataFrame(rows)

    # LY desplazado 1 año
    act_idx = pd.PeriodIndex(act["Mes"], freq="M")
    ly_idx = act_idx - 12
    s_ly, e_ly = (pd.to_datetime(start) - pd.DateOffset(years=1)), (pd.to_datetime(end) - pd.DateOffset(years=1))
    ly_rows = []
    for m in ly_idx:
        ms = pd.Timestamp(m.start_time).date()
        me = pd.Timestamp(m.end_time).date()
        ly_rows.append({"Mes": str(m + 12), "Ingresos": float(
            _monthly_series_sum(dfx, ms, me)
        )})
    ly = pd.DataFrame(ly_rows)
    # Asegura alineación
    ser = act.merge(ly, on="Mes", how="left", suffixes=("", "_LY"))
    return ser.fillna(0.0)

def _monthly_series_sum(dfx, ms, me):
    s = pd.to_datetime(ms)
    e = pd.to_datetime(me)
    los_total = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    rate = (dfx["Alquiler con IVA (€)"] / los_total).values
    si = np.maximum(dfx["Fecha entrada"].values.astype("datetime64[D]"), s.to_datetime64())
    eo = np.minimum(dfx["Fecha salida"].values.astype("datetime64[D]"), (e + pd.Timedelta(days=1)).to_datetime64())
    nights = (eo - si).astype("timedelta64[D]").astype(int)
    mask = nights > 0
    return np.sum(rate[mask] * nights[mask])

def render_informe_propietario(raw: pd.DataFrame | None = None):
    st.header(TITLE)

    if raw is None:
        raw = st.session_state.get("df_active") or st.session_state.get("raw")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("Sube un archivo en la barra lateral para continuar.")
        return
    df = _std_cols(raw)

    # Sidebar: parámetros
    with st.sidebar:
        st.subheader("Parámetros · Informe")
        props = []
        prop_name = ""
        if "Alojamiento" in df.columns:
            all_props = sorted(df["Alojamiento"].astype(str).dropna().unique().tolist())
            props = st.multiselect("Alojamientos", all_props, default=[], key="inf_props")
            if len(props) == 1:
                prop_name = props[0]
        apto = st.text_input("Nombre de apartamento (portada)", value=prop_name, key="inf_apto")
        owner = st.text_input("Nombre del propietario", value="", key="inf_owner")

        today = date.today()
        start = st.date_input("Inicio periodo", value=today.replace(day=1), key="inf_start")
        end   = st.date_input("Fin periodo", value=today, key="inf_end")
        inv_override = st.number_input("Inventario (noches) opcional", min_value=0, step=1, value=0, help="Si lo dejas en 0, se calcula sin ocupación/RevPAR.")
        comments = st.text_area("Comentarios del Revenue Manager", height=120, key="inf_comments", placeholder="Puntos clave, acciones y próximos pasos…")
        run = st.button("Generar informe", type="primary", use_container_width=True, key="inf_run")

    if not run:
        st.info("Elige parámetros y pulsa Generar informe.")
        return
    if start > end:
        st.error("La fecha de inicio no puede ser posterior al fin.")
        return

    inv_val = int(inv_override) if inv_override and inv_override > 0 else None

    # KPIs
    act, ly = _compute_periods(df, start, end, props, inv_val)

    # Portada
    st.markdown(f"### {apto or '—'}")
    st.markdown(f"Propietario: {owner or '—'}")
    st.markdown(f"Periodo: {pd.to_datetime(start).date()} – {pd.to_datetime(end).date()}")

    # Tarjetas KPI
    def fmt_money(x): return f"€{x:,.0f}".replace(",", ".")
    def fmt_pct(x):   return f"{x:.1f}%" if x is not None else "—"
    def fmt_num(x):   return f"{x:,.0f}".replace(",", ".")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingresos", fmt_money(act["ingresos"]), f"{_pct(act['ingresos'], ly['ingresos']):.1f}% vs LY" if ly["ingresos"] else None)
    c2.metric("ADR", fmt_money(act["adr"]), f"{_pct(act['adr'], ly['adr']):.1f}% vs LY" if ly["adr"] else None)
    c3.metric("Noches vendidas", fmt_num(act["noches"]), f"{_delta(act['noches'], ly['noches']):+,.0f}".replace(",", ".") if ly["noches"] is not None else None)
    c4.metric("Ocupación", fmt_pct(act["ocupacion_pct"]), f"{_delta(act['ocupacion_pct'], ly['ocupacion_pct']):+.1f} pp" if act["ocupacion_pct"] is not None and ly["ocupacion_pct"] is not None else None)

    # Serie mensual Ingresos (Act vs LY)
    ser = _monthly_series(df, start, end, props)
    ser_long = ser.melt(id_vars=["Mes"], value_vars=["Ingresos","Ingresos_LY"], var_name="Serie", value_name="Valor")
    ser_long["Serie"] = ser_long["Serie"].replace({"Ingresos":"Act", "Ingresos_LY":"LY"})
    chart = (
        alt.Chart(ser_long)
        .mark_bar()
        .encode(
            x=alt.X("Mes:N", title="Mes"),
            y=alt.Y("Valor:Q", title="Ingresos"),
            color=alt.Color("Serie:N", title="Serie"),
            tooltip=[alt.Tooltip("Mes:N"), alt.Tooltip("Serie:N"), alt.Tooltip("Valor:Q", format=",.0f")]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    # Comentarios
    st.subheader("Comentarios del Revenue Manager")
    st.write(comments or "—")

    # Exportar a PDF (dos vías)
    st.divider()
    st.markdown("#### Exportar")
    # Opción 1: imprimir/guardar en PDF desde el navegador (más compatible)
    import streamlit.components.v1 as components
    printable = f"""
    <style>@media print {{ .stApp header, .stApp footer {{ display:none; }} }}</style>
    <button onclick="window.print()" style="padding:10px 16px;border-radius:8px;border:1px solid #ccc;background:#2e485f;color:#fff;cursor:pointer;">
      Imprimir / Guardar PDF
    </button>
    """
    components.html(printable, height=60)
    # Opción 2: pdfkit si está disponible en el servidor
    try:
        import pdfkit  # requiere wkhtmltopdf instalado en el sistema
        html = f"""
        <h2>{TITLE}</h2>
        <p><b>Apartamento:</b> {apto or '—'} &nbsp; | &nbsp; <b>Propietario:</b> {owner or '—'}</p>
        <p><b>Periodo:</b> {pd.to_datetime(start).date()} – {pd.to_datetime(end).date()}</p>
        <ul>
          <li>Ingresos: {fmt_money(act['ingresos'])} (LY: {fmt_money(ly['ingresos'])})</li>
          <li>ADR: {fmt_money(act['adr'])} (LY: {fmt_money(ly['adr'])})</li>
          <li>Noches vendidas: {fmt_num(act['noches'])} (LY: {fmt_num(ly['noches'])})</li>
          <li>Ocupación: {fmt_pct(act['ocupacion_pct'])} (LY: {fmt_pct(ly['ocupacion_pct'])})</li>
        </ul>
        <h3>Comentarios</h3>
        <p>{(comments or '—').replace('\n','<br>')}</p>
        """
        pdf_bytes = pdfkit.from_string(html, False)
        st.download_button("Descargar PDF (experimental)", data=pdf_bytes, file_name="informe_propietario.pdf", mime="application/pdf")
    except Exception:
        st.caption("Para descarga directa en PDF se necesita wkhtmltopdf en el servidor. Se habilitó el botón de Imprimir/Guardar PDF del navegador.")