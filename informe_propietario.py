import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta

TITLE = "Informe de propietario"
BUILD = "v1.5"

# ----------------- Utilidades -----------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    norm = {c: str(c).strip().lower() for c in df.columns}
    def find(*keys):
        for col, n in norm.items():
            for k in keys:
                if n == k or k in n:
                    return col
        return None
    m = {}
    a   = find("alojamiento","propiedad","listing","unidad","apartamento","room","unit")
    fi  = find("fecha entrada","check in","entrada","arrival")
    fo  = find("fecha salida","check out","salida","departure")
    rev = find("alquiler con iva","ingresos","revenue","importe","total","neto","importe total")
    adr = find("adr","avg daily","average daily rate")
    if a:   m[a] = "Alojamiento"
    if fi:  m[fi] = "Fecha entrada"
    if fo:  m[fo] = "Fecha salida"
    if rev: m[rev] = "Alquiler con IVA (€)"
    if adr: m[adr] = "ADR (opcional)"
    return df.rename(columns=m) if m else df

def _to_dt(s: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if s1.isna().mean() > 0.8:
        s2 = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="d", origin="1899-12-30", errors="coerce")
        if s2.notna().sum() > s1.notna().sum():
            s1 = s2
    return s1.dt.normalize()

def _period_label(s: date, e: date) -> str:
    return f"{pd.to_datetime(s).date()} – {pd.to_datetime(e).date()}"

# ----------------- Cálculos base -----------------
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = _std_cols(df.copy())
    need = ["Fecha entrada","Fecha salida"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {', '.join(missing)}")
    d["Fecha entrada"] = _to_dt(d["Fecha entrada"])
    d["Fecha salida"]  = _to_dt(d["Fecha salida"])
    if "Alquiler con IVA (€)" in d.columns:
        d["Alquiler con IVA (€)"] = pd.to_numeric(d["Alquiler con IVA (€)"], errors="coerce").fillna(0.0)
    else:
        d["Alquiler con IVA (€)"] = 0.0
    return d

def _overlap_nights_rows(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Expande reservas a noches individuales dentro del periodo [start, end] (incluye la noche de 'end').
    Esto corrige 'Noches vendidas = 0' y nos da series diarias robustas para ingresos/ADR.
    """
    if d.empty:
        return pd.DataFrame(columns=["Fecha","Noches","Ingresos"])
    # Filtra reservas que intersectan
    s = start.normalize()
    e = end.normalize()
    # Intersección de rangos
    d = d.dropna(subset=["Fecha entrada","Fecha salida"]).copy()
    d["los"] = (d["Fecha salida"] - d["Fecha entrada"]).dt.days.clip(lower=1)
    d["rate"] = np.where(d["los"] > 0, d["Alquiler con IVA (€)"] / d["los"], 0.0)

    # Calcula segmento intersectado con [s, e+1)
    seg_s = np.maximum(d["Fecha entrada"].values.astype("datetime64[D]"), s.to_datetime64())
    seg_e = np.minimum(d["Fecha salida"].values.astype("datetime64[D]"), (e + pd.Timedelta(days=1)).to_datetime64())
    n = (seg_e - seg_s).astype("timedelta64[D]").astype(int)
    valid = n > 0
    if not np.any(valid):
        return pd.DataFrame(columns=["Fecha","Noches","Ingresos"])

    # Expansión por reserva
    idx = np.where(valid)[0]
    rows = []
    for i in idx:
        start_i = pd.Timestamp(seg_s[i])
        nights_i = int(n[i])
        # rango nightly: [start_i, start_i + nights_i)
        days = pd.date_range(start_i, periods=nights_i, freq="D")
        rows.append(pd.DataFrame({
            "Fecha": days,
            "Noches": 1,
            "Ingresos": d["rate"].iloc[i]
        }))
    daily = pd.concat(rows, ignore_index=True)
    daily = daily.groupby("Fecha", as_index=False).agg({"Noches":"sum","Ingresos":"sum"})
    return daily

@st.cache_data(ttl=1800, show_spinner=False)
def _compute_series(df: pd.DataFrame, start: date, end: date, props: list[str] | None):
    d = _preprocess(df)
    if props and "Alojamiento" in d.columns:
        d = d[d["Alojamiento"].astype(str).isin(props)].copy()
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    act = _overlap_nights_rows(d, s, e)
    # Reindex rango completo para rellenar huecos
    rng = pd.date_range(s, e, freq="D")
    act = act.set_index("Fecha").reindex(rng, fill_value=0).rename_axis("Fecha").reset_index()
    act["ADR"] = np.where(act["Noches"] > 0, act["Ingresos"] / act["Noches"], 0.0)

    s_ly = s - pd.DateOffset(years=1)
    e_ly = e - pd.DateOffset(years=1)
    ly = _overlap_nights_rows(d, s_ly, e_ly)
    rng_ly = pd.date_range(s_ly, e_ly, freq="D")
    ly = ly.set_index("Fecha").reindex(rng_ly, fill_value=0).rename_axis("Fecha").reset_index()
    ly["ADR"] = np.where(ly["Noches"] > 0, ly["Ingresos"] / ly["Noches"], 0.0)
    return act, ly

def _agg_kpis(daily: pd.DataFrame) -> dict:
    nights = int(daily["Noches"].sum())
    ingresos = float(daily["Ingresos"].sum())
    adr = ingresos / nights if nights > 0 else 0.0
    return {"noches": nights, "ingresos": ingresos, "adr": adr}

# ----------------- Render -----------------
def render_informe_propietario(raw: pd.DataFrame | None = None):
    st.header(TITLE)
    st.caption(f"Build {BUILD}")

    if raw is None:
        raw = st.session_state.get("df_active") or st.session_state.get("raw")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        st.info("Sube un archivo en la barra lateral.")
        return

    df = _std_cols(raw)

    # Sidebar
    with st.sidebar:
        st.subheader("Parámetros · Informe")
        props = []
        if "Alojamiento" in df.columns:
            all_props = sorted(df["Alojamiento"].astype(str).dropna().unique().tolist())
            props = st.multiselect("Alojamientos", all_props, default=[], key="inf_props")

        apto = st.text_input("Nombre del apartamento (portada)", value=(props[0] if len(props)==1 else ""), key="inf_apto")
        owner = st.text_input("Nombre del propietario", value="", key="inf_owner")

        today = date.today()
        start = st.date_input("Inicio del periodo", value=today.replace(day=1), key="inf_start")
        end   = st.date_input("Fin del periodo", value=today, key="inf_end")
        gran = st.radio("Granularidad ADR", ["Día","Semana"], horizontal=True, key="inf_gran")
        run = st.button("Generar informe", type="primary", use_container_width=True, key="inf_run")

    if not run:
        st.info("Elige parámetros y pulsa Generar informe.")
        return
    if start > end:
        st.error("La fecha de inicio no puede ser posterior a la de fin.")
        return

    # Series ACT/LY
    act, ly = _compute_series(df, start, end, props)

    # KPIs
    k_act = _agg_kpis(act)
    k_ly  = _agg_kpis(ly)

    # Portada
    st.markdown(f"### {apto or '—'}")
    st.markdown(f"Propietario: {owner or '—'}")
    st.markdown(f"Periodo ACT: {_period_label(start, end)}  |  Periodo LY: {_period_label(pd.to_datetime(start)-pd.DateOffset(years=1), pd.to_datetime(end)-pd.DateOffset(years=1))}")

    # Tarjetas KPI con LY explícito
    def money(x): return f"€{x:,.0f}".replace(",", ".")
    def num(x):   return f"{x:,.0f}".replace(",", ".")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ingresos (ACT)", money(k_act["ingresos"]))
        st.caption(f"LY: {money(k_ly['ingresos'])}")
    with c2:
        st.metric("ADR (ACT)", money(k_act["adr"]))
        st.caption(f"LY: {money(k_ly['adr'])}")
    with c3:
        st.metric("Noches vendidas (ACT)", num(k_act["noches"]))
        st.caption(f"LY: {num(k_ly['noches'])}")

    # Gráfico de ingresos diarios (ACT vs LY)
    act_i = act.assign(Serie="Act")
    ly_i  = ly.assign(Serie="LY")
    # Alinear LY a las fechas de ACT para comparar visualmente (sumar 1 año)
    ly_i_plot = ly_i.copy()
    ly_i_plot["Fecha"] = ly_i_plot["Fecha"] + pd.DateOffset(years=1)
    plot_ing = pd.concat([act_i, ly_i_plot], ignore_index=True)

    chart_ing = (
        alt.Chart(plot_ing)
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("Fecha:T", title="Fecha"),
            y=alt.Y("Ingresos:Q", title="Ingresos por día"),
            color=alt.Color("Serie:N", scale=alt.Scale(range=["#2e485f","#c77b2b"])),
            tooltip=[
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Fecha:T"),
                alt.Tooltip("Ingresos:Q", format=",.0f")
            ],
        )
        .properties(height=260)
    ) + (
        alt.Chart(plot_ing)
        .mark_line()
        .encode(x="Fecha:T", y="Ingresos:Q", color="Serie:N")
    )
    st.subheader("Ingresos por día (Act vs LY alineado)")
    st.altair_chart(chart_ing, use_container_width=True)

    # ADR por día o por semana
    if gran == "Día":
        adr_act = act[["Fecha","ADR"]].assign(Serie="Act")
        adr_ly  = ly[["Fecha","ADR"]].assign(Serie="LY")
        adr_ly["Fecha"] = adr_ly["Fecha"] + pd.DateOffset(years=1)
    else:
        # Semana natural, lunes-domingo
        w_act = act.assign(Semana=act["Fecha"].dt.to_period("W-MON").dt.start_time)
        adr_act = (w_act.groupby("Semana", as_index=False).agg({"Ingresos":"sum","Noches":"sum"})
                        .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"], 0.0))
                        .rename(columns={"Semana":"Fecha"}))[["Fecha","ADR"]].assign(Serie="Act")

        w_ly = ly.assign(Semana=ly["Fecha"].dt.to_period("W-MON").dt.start_time)
        adr_ly = (w_ly.groupby("Semana", as_index=False).agg({"Ingresos":"sum","Noches":"sum"})
                        .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"], 0.0))
                        .rename(columns={"Semana":"Fecha"}))[["Fecha","ADR"]].assign(Serie="LY")
        adr_ly["Fecha"] = adr_ly["Fecha"] + pd.DateOffset(years=1)

    adr_plot = pd.concat([adr_act, adr_ly], ignore_index=True)

    st.subheader(f"ADR por {gran.lower()} (Act vs LY alineado)")
    chart_adr = (
        alt.Chart(adr_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("Fecha:T", title="Fecha"),
            y=alt.Y("ADR:Q", title="ADR"),
            color=alt.Color("Serie:N", scale=alt.Scale(range=["#2e485f","#c77b2b"])),
            tooltip=[
                alt.Tooltip("Serie:N"),
                alt.Tooltip("Fecha:T"),
                alt.Tooltip("ADR:Q", format=",.0f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )
    st.altair_chart(chart_adr, use_container_width=True)

    # Tabla resumen ACT vs LY
    st.subheader("Resumen ACT vs LY")
    summary = pd.DataFrame([
        {"Serie":"Act","Periodo":_period_label(start,end),
         "Ingresos":k_act["ingresos"],"ADR":k_act["adr"],"Noches":k_act["noches"]},
        {"Serie":"LY","Periodo":_period_label(pd.to_datetime(start)-pd.DateOffset(years=1),
                                              pd.to_datetime(end)-pd.DateOffset(years=1)),
         "Ingresos":k_ly["ingresos"],"ADR":k_ly["adr"],"Noches":k_ly["noches"]},
    ])
    st.dataframe(summary.style.format({"Ingresos":"{:,.0f}","ADR":"{:,.0f}","Noches":"{:,.0f}"}), use_container_width=True)

    # Comentarios
    st.subheader("Comentarios del Revenue Manager")
    comments = st.text_area("Comentarios", key="inf_comments", height=140, placeholder="Puntos clave, acciones, próximos pasos…")
    st.write(comments or "—")

    # Exportar a PDF
    st.divider()
    st.markdown("#### Exportar")
    import streamlit.components.v1 as components
    components.html(
        """
        <style>@media print { .stApp header, .stApp footer { display:none; } }</style>
        <button onclick="window.print()" style="padding:10px 16px;border-radius:8px;border:1px solid #ccc;background:#2e485f;color:#fff;cursor:pointer;">
          Imprimir / Guardar PDF
        </button>
        """,
        height=60
    )
    try:
        import pdfkit, os, shutil
        wk = shutil.which("wkhtmltopdf") or r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        cfg = pdfkit.configuration(wkhtmltopdf=wk) if wk and os.path.exists(wk) else None
        pdf_bytes = pdfkit.from_string(html, False, configuration=cfg) if cfg else None
        if pdf_bytes:
            st.download_button("Descargar PDF (experimental)", data=pdf_bytes,
                               file_name="informe_propietario.pdf", mime="application/pdf")
        else:
            st.caption("wkhtmltopdf no encontrado. Usa el botón Imprimir/Guardar PDF.")
    except Exception:
        st.caption("Si instalas wkhtmltopdf en el servidor, se habilitará la descarga PDF directa. De momento usa Imprimir/Guardar PDF del navegador.")