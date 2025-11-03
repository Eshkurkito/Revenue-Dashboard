import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
import io, base64, shutil, os, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TITLE = "Informe de propietario"
BUILD = "v1.6-diag"

# --- diagnóstico wkhtmltopdf ---
def _wkhtmltopdf_candidates():
    return [
        os.environ.get("WKHTMLTOPDF_PATH"),
        getattr(st.secrets, "wkhtmltopdf_path", None) if hasattr(st, "secrets") else None,
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
        shutil.which("wkhtmltopdf"),
        "/usr/bin/wkhtmltopdf",
        "/usr/local/bin/wkhtmltopdf",
    ]

def _wkhtmltopdf_detect():
    for p in _wkhtmltopdf_candidates():
        if p and os.path.exists(p):
            return p
    return None

def _wkhtmltopdf_version(path: str | None):
    if not path:
        return None
    try:
        out = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def _pdfkit_config():
    path = _wkhtmltopdf_detect()
    if not path:
        return None
    try:
        import pdfkit
        return pdfkit.configuration(wkhtmltopdf=path)
    except Exception:
        return None

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

    # --- Estado persistente ---
    if "inf_ready" not in st.session_state:
        st.session_state.inf_ready = False
    if "inf_cfg" not in st.session_state:
        st.session_state.inf_cfg = {}

    # --- Sidebar: formulario de parámetros (persistente) ---
    with st.sidebar:
        st.subheader("Parámetros · Informe")
        with st.form("inf_form", clear_on_submit=False):
            props = []
            if "Alojamiento" in df.columns:
                all_props = sorted(df["Alojamiento"].astype(str).dropna().unique().tolist())
                # Valores por defecto desde el último informe si existen
                prev = st.session_state.inf_cfg.get("props", [])
                props = st.multiselect("Alojamientos", all_props, default=prev, key="inf_props_form")

            apto_default = st.session_state.inf_cfg.get("apto", (props[0] if len(props)==1 else ""))
            owner_default = st.session_state.inf_cfg.get("owner", "")
            apto = st.text_input("Nombre del apartamento (portada)", value=apto_default, key="inf_apto_form")
            owner = st.text_input("Nombre del propietario", value=owner_default, key="inf_owner_form")

            today = date.today()
            start_default = st.session_state.inf_cfg.get("start", today.replace(day=1))
            end_default   = st.session_state.inf_cfg.get("end", today)
            start = st.date_input("Inicio del periodo", value=start_default, key="inf_start_form")
            end   = st.date_input("Fin del periodo", value=end_default, key="inf_end_form")

            gran = st.radio("Granularidad ADR", ["Día","Semana"],
                            horizontal=True,
                            index=(0 if st.session_state.inf_cfg.get("gran","Día")=="Día" else 1),
                            key="inf_gran_form")

            submitted = st.form_submit_button("Generar informe", use_container_width=True)

        # Botón utilitario para refrescar caché
        if st.button("Forzar recarga", use_container_width=True, key="inf_force_reload"):
            st.cache_data.clear(); st.rerun()

    # Si se envía el formulario, guarda y fija estado
    if submitted:
        if start > end:
            st.error("La fecha de inicio no puede ser posterior a la de fin.")
            return
        st.session_state.inf_cfg = {
            "props": st.session_state.get("inf_props_form", []),
            "apto": st.session_state.get("inf_apto_form", ""),
            "owner": st.session_state.get("inf_owner_form", ""),
            "start": pd.to_datetime(start).date(),
            "end": pd.to_datetime(end).date(),
            "gran": st.session_state.get("inf_gran_form", "Día"),
        }
        st.session_state.inf_ready = True
        st.rerun()

    # Si aún no se ha generado, muestra guía y sal
    if not st.session_state.inf_ready:
        st.info("Elige parámetros y pulsa Generar informe.")
        return

    # --- Usa parámetros fijados para renderizar el informe ---
    cfg = st.session_state.inf_cfg
    props = cfg["props"]; apto = cfg["apto"]; owner = cfg["owner"]
    start = cfg["start"]; end = cfg["end"]; gran = cfg["gran"]

    # Botón para volver a editar
    st.button("Editar parámetros", key="inf_edit_params",
              on_click=lambda: st.session_state.update(inf_ready=False))

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

    # ================= Exportar a PDF (plantilla) =================
    st.divider()
    st.markdown("#### Exportar a PDF (plantilla HTML + wkhtmltopdf)")

    # Diagnóstico rápido visible
    cfg = _pdfkit_config()
    wk_path = _wkhtmltopdf_detect()
    wk_ver = _wkhtmltopdf_version(wk_path)
    tpl_path = Path(__file__).resolve().parent / "templates" / "informe_propietario.html"
    logo_guess = Path(__file__).resolve().parent / "assets" / "florit.flats-logo.png"
    st.caption(f"Build: {BUILD} · wkhtmltopdf: {'OK' if cfg else 'NO'} · Ruta: {wk_path or '—'} · Versión: {wk_ver or '—'}")
    st.caption(f"Plantilla: {'OK' if tpl_path.exists() else 'NO'} ({tpl_path}) · Logo: {'OK' if logo_guess.exists() else 'NO'} ({logo_guess})")

    # Botón generar PDF
    if st.button("Generar PDF", type="primary", use_container_width=True, key="inf_pdf"):
        if cfg is None:
            st.error("wkhtmltopdf no encontrado. Define WKHTMLTOPDF_PATH o añade su ruta al PATH.")
        elif not tpl_path.exists():
            st.error(f"No existe la plantilla: {tpl_path}")
        else:
            try:
                chart_ing_b64 = _plot_ingresos_png(act, ly)
                chart_adr_b64 = _plot_adr_png(act, ly, gran)
                ctx = {
                    "apto": apto, "owner": owner,
                    "period_act": _period_label(start, end),
                    "period_ly": _period_label(pd.to_datetime(start)-pd.DateOffset(years=1), pd.to_datetime(end)-pd.DateOffset(years=1)),
                    "act": {"ingresos": _fmt_money(k_act['ingresos']), "adr": _fmt_money(k_act['adr']), "noches": f"{k_act['noches']:,}".replace(",",".")},
                    "ly":  {"ingresos": _fmt_money(k_ly['ingresos']),  "adr": _fmt_money(k_ly['adr']),  "noches": f"{k_ly['noches']:,}".replace(",",".")},
                    "chart_ingresos": chart_ing_b64,
                    "chart_adr": chart_adr_b64,
                    "gran_label": gran.lower(),
                    "comments": st.session_state.get("inf_comments") or "",
                    "logo_b64": _logo_b64(),
                }
                # render
                from jinja2 import Environment, FileSystemLoader
                tpl_dir = Path(__file__).resolve().parent / "templates"
                env = Environment(loader=FileSystemLoader(str(tpl_dir)), autoescape=True)
                html = env.get_template("informe_propietario.html").render(**ctx)

                import pdfkit
                options = {
                    "enable-local-file-access": None,
                    "page-size": "A4",
                    "margin-top": "10mm",
                    "margin-bottom": "10mm",
                    "margin-left": "10mm",
                    "margin-right": "10mm",
                    "encoding": "UTF-8",
                    "quiet": "",
                }
                pdf_bytes = pdfkit.from_string(html, False, configuration=cfg, options=options)
                st.download_button("Descargar PDF", data=pdf_bytes, file_name=f"Informe_{apto or 'propietario'}.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error("No se pudo generar el PDF.")
                with st.expander("Detalle"):
                    st.exception(e)

    # Info para despliegue en la nube
    if os.environ.get("STREAMLIT_RUNTIME") == "cloud" or "mount/src" in str(Path.cwd()):
        st.info("Si estás viendo la app en Streamlit Cloud, wkhtmltopdf puede no estar disponible. En la nube usa el botón Imprimir/Guardar PDF o despliega localmente con wkhtmltopdf instalado.")

def _fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")

def _logo_b64() -> str | None:
    # Busca el logo en ./assets (florit.flats-logo.png prioritario)
    from pathlib import Path
    base = Path(__file__).resolve().parent / "assets"
    for name in ["florit.flats-logo.png","florit.flats_logo.png","logo.png","logo.jpg","logo.jpeg","logo.svg"]:
        p = base / name
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None
    return None

def _png_from_plt(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _plot_ingresos_png(act: pd.DataFrame, ly: pd.DataFrame) -> str:
    ly_plot = ly.copy()
    ly_plot["Fecha"] = ly_plot["Fecha"] + pd.DateOffset(years=1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(act["Fecha"], act["Ingresos"], color="#2e485f", label="Act")
    ax.plot(ly_plot["Fecha"], ly_plot["Ingresos"], color="#c77b2b", label="LY")
    ax.fill_between(act["Fecha"], 0, act["Ingresos"], color="#2e485f", alpha=0.15)
    ax.fill_between(ly_plot["Fecha"], 0, ly_plot["Ingresos"], color="#c77b2b", alpha=0.10)
    ax.set_ylabel("€")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _png_from_plt(fig)

def _plot_adr_png(act: pd.DataFrame, ly: pd.DataFrame, gran: str) -> str:
    if gran == "Semana":
        w_act = act.assign(Sem=act["Fecha"].dt.to_period("W-MON").dt.start_time)
        actp = (w_act.groupby("Sem", as_index=False).agg({"Ingresos":"sum","Noches":"sum"})
                .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"],0.0))
                .rename(columns={"Sem":"Fecha"}))
        w_ly = ly.assign(Sem=ly["Fecha"].dt.to_period("W-MON").dt.start_time)
        lyp = (w_ly.groupby("Sem", as_index=False).agg({"Ingresos":"sum","Noches":"sum"})
               .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"],0.0))
               .rename(columns={"Sem":"Fecha"}))
    else:
        actp, lyp = act.copy(), ly.copy()

    lyp = lyp.copy()
    lyp["Fecha"] = lyp["Fecha"] + pd.DateOffset(years=1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(actp["Fecha"], actp["ADR"], marker="o", ms=3, color="#2e485f", label="Act")
    ax.plot(lyp["Fecha"], lyp["ADR"], marker="o", ms=3, color="#c77b2b", label="LY")
    ax.set_ylabel("€")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _png_from_plt(fig)