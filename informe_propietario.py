import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
import io, base64, shutil, os, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # backend headless para Cloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TITLE = "Informe de propietario"
BUILD = "v1.8-pro"   # versión

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
    # Cloud: /usr/bin/wkhtmltopdf; Local: Windows rutas típicas o PATH
    candidates = [
        os.environ.get("WKHTMLTOPDF_PATH"),
        (st.secrets.get("wkhtmltopdf_path") if hasattr(st, "secrets") else None),
        "/usr/bin/wkhtmltopdf", "/usr/local/bin/wkhtmltopdf",            # Linux (Cloud)
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
        shutil.which("wkhtmltopdf"),
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        return None
    import pdfkit
    return pdfkit.configuration(wkhtmltopdf=path)

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
    ch  = find("portal","canal","channel","ota","fuente","agencia","source")         # ← NUEVO
    fa  = find("fecha alta","alta","creación","creado","booking date","booked","created")  # ← NUEVO
    if a:   m[a] = "Alojamiento"
    if fi:  m[fi] = "Fecha entrada"
    if fo:  m[fo] = "Fecha salida"
    if rev: m[rev] = "Alquiler con IVA (€)"
    if adr: m[adr] = "ADR (opcional)"
    if ch:  m[ch] = "Portal"                                                        # ← NUEVO
    if fa:  m[fa] = "Fecha alta"                                                    # ← NUEVO
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
    if "Fecha alta" in d.columns:                                                  # ← NUEVO
        d["Fecha alta"] = _to_dt(d["Fecha alta"])
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

# ==== NUEVO: ritmo de ocupación (ocupación acumulada del periodo) ====
def _pace_dataframe(act: pd.DataFrame, ly: pd.DataFrame, start: date, end: date, inv_units: int) -> pd.DataFrame:
    """
    Devuelve un DataFrame con la ocupación acumulada (%) por fecha para ACT y LY
    alineando LY al año de ACT. inv_units es el nº de alojamientos considerados.
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    total_days = (e - s).days + 1
    total_cap = max(1, int(inv_units)) * total_days  # noches disponibles en el periodo

    p_act = act[["Fecha", "Noches"]].copy()
    p_act["CumNoches"] = p_act["Noches"].cumsum()
    p_act["PacePct"] = (p_act["CumNoches"] / total_cap) * 100.0
    p_act["Serie"] = "Act"

    p_ly = ly[["Fecha", "Noches"]].copy()
    p_ly["CumNoches"] = p_ly["Noches"].cumsum()
    p_ly["PacePct"] = (p_ly["CumNoches"] / total_cap) * 100.0
    p_ly["Fecha"] = p_ly["Fecha"] + pd.DateOffset(years=1)  # alinear con ACT
    p_ly["Serie"] = "LY"

    return pd.concat(
        [p_act[["Fecha", "PacePct", "Serie"]], p_ly[["Fecha", "PacePct", "Serie"]]],
        ignore_index=True
    )

# ====== NUEVO: helpers faltantes ======
def _fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")

# NUEVO: reservas por portal en el periodo (por fecha de alta si existe; si no, por check-in)
def _bookings_by_portal(df_raw: pd.DataFrame, start: date, end: date, props: list[str] | None) -> pd.DataFrame:
    d = _preprocess(df_raw)
    if props and "Alojamiento" in d.columns:
        d = d[d["Alojamiento"].astype(str).isin(props)].copy()
    # Fecha de referencia
    if "Fecha alta" in d.columns:
        ref = d["Fecha alta"]
    else:
        ref = d["Fecha entrada"]
    mask = (ref >= pd.to_datetime(start)) & (ref <= pd.to_datetime(end))
    d = d.loc[mask].copy()
    if "Portal" not in d.columns:
        d["Portal"] = "Sin portal"
    g = (d.groupby("Portal", dropna=False)
           .agg(Reservas=("Portal","size"), Ingresos=("Alquiler con IVA (€)","sum"))
           .reset_index()
           .sort_values(["Reservas","Ingresos"], ascending=[False, False]))
    return g

# NUEVO: gráfico PNG para PDF (reservas por portal)
def _plot_portales_png(portal_df: pd.DataFrame) -> str:
    if portal_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Sin datos de reservas en el periodo", ha="center", va="center")
        ax.axis("off")
        return _png_from_plt(fig)
    df = portal_df.copy()
    df = df.sort_values("Reservas", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(df["Portal"], df["Reservas"], color="#2e485f")
    for i, v in enumerate(df["Reservas"]):
        ax.text(v + max(df["Reservas"])*0.01, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("Reservas")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _png_from_plt(fig)

# ===== Helpers requeridos =====
def _png_from_plt(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _plot_adr_png(act: pd.DataFrame, ly: pd.DataFrame, gran: str) -> str:
    # ADR por día o semana (para PDF)
    if gran == "Semana":
        w_act = act.assign(Sem=act["Fecha"].dt.to_period("W-MON").dt.start_time)
        actp = (w_act.groupby("Sem", as_index=False)
                .agg({"Ingresos":"sum","Noches":"sum"})
                .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"], 0.0))
                .rename(columns={"Sem":"Fecha"}))
        w_ly = ly.assign(Sem=ly["Fecha"].dt.to_period("W-MON").dt.start_time)
        lyp = (w_ly.groupby("Sem", as_index=False)
               .agg({"Ingresos":"sum","Noches":"sum"})
               .assign(ADR=lambda x: np.where(x["Noches"]>0, x["Ingresos"]/x["Noches"], 0.0))
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

def _logo_b64() -> str | None:
    base = Path(__file__).resolve().parent / "assets"
    candidates = [
        "florit.flats-logo.png",
        "florit.flats_logo.png",
        "logo.png", "logo.jpg", "logo.jpeg", "logo.svg",
    ]
    for name in candidates:
        p = base / name
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None
    if base.exists():
        for p in base.iterdir():
            if p.is_file() and "logo" in p.name.lower():
                try:
                    with open(p, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    return None
    return None

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
            submitted = st.form_submit_button("Generar informe", use_container_width=True)  # ← quitamos objetivo ocupación
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

    # Capacidad diaria (nº de unidades seleccionadas). Si no hay selección, 1.
    inv_units = max(1, len(props) if props else (df["Alojamiento"].nunique() if "Alojamiento" in df.columns else 1))

    # Botón para volver a editar
    st.button("Editar parámetros", key="inf_edit_params",
              on_click=lambda: st.session_state.update(inf_ready=False))

    # Series ACT/LY
    act, ly = _compute_series(df, start, end, props)

    # KPIs
    k_act = _agg_kpis(act)
    k_ly  = _agg_kpis(ly)

    # Deltas YoY (para las métricas)
    def _pct_delta(a: float, b: float) -> float | None:
        try:
            return None if b == 0 else (a - b) / b * 100.0
        except Exception:
            return None
    d_ing = _pct_delta(k_act["ingresos"], k_ly["ingresos"])
    d_adr = _pct_delta(k_act["adr"],      k_ly["adr"])
    d_nch = _pct_delta(k_act["noches"],   k_ly["noches"])

    # Portada
    st.markdown(f"### {apto or '—'}")
    st.markdown(f"Propietario: {owner or '—'}")
    st.markdown(f"Periodo ACT: {_period_label(start, end)}  |  Periodo LY: {_period_label(pd.to_datetime(start)-pd.DateOffset(years=1), pd.to_datetime(end)-pd.DateOffset(years=1))}")

    # Tarjetas KPI con LY explícito
    def money(x): return f"€{x:,.0f}".replace(",", ".")
    def num(x):   return f"{x:,.0f}".replace(",", ".")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ingresos (ACT)", money(k_act["ingresos"]),
                  delta=(f"{d_ing:+.1f}%" if d_ing is not None else "—"))
        st.caption(f"LY: {money(k_ly['ingresos'])}")
    with c2:
        st.metric("ADR (ACT)", money(k_act["adr"]),
                  delta=(f"{d_adr:+.1f}%" if d_adr is not None else "—"))
        st.caption(f"LY: {money(k_ly['adr'])}")
    with c3:
        st.metric("Noches vendidas (ACT)", num(k_act["noches"]),
                  delta=(f"{d_nch:+.1f}%" if d_nch is not None else "—"))
        st.caption(f"LY: {num(k_ly['noches'])}")

    # NUEVO: Reservas por portal (periodo)
    st.subheader("Reservas por portal en el periodo")
    portal_df = _bookings_by_portal(df, start, end, props)
    if portal_df.empty:
        st.info("No hay reservas en ese periodo.")
    else:
        chart_portal = (
            alt.Chart(portal_df)
              .mark_bar(color="#2e485f")
              .encode(
                  y=alt.Y("Portal:N", sort="-x", title="Portal"),
                  x=alt.X("Reservas:Q", title="Reservas"),
                  tooltip=[
                      alt.Tooltip("Portal:N"),
                      alt.Tooltip("Reservas:Q"),
                      alt.Tooltip("Ingresos:Q", format=",.0f", title="Ingresos (€)")
                  ]
              ).properties(height=max(160, 22*len(portal_df)))
        )
        st.altair_chart(chart_portal, use_container_width=True)
        st.dataframe(
            portal_df.assign(Ingresos_fmt=portal_df["Ingresos"].map(_fmt_money))
                     .rename(columns={"Ingresos_fmt":"Ingresos (€)"}).drop(columns=["Ingresos"]),
            use_container_width=True
        )

    # ADR por día o por semana (se mantiene)
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
    # Corrige el nombre del logo del diagnóstico
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
                chart_portales_b64 = _plot_portales_png(portal_df)        # ← NUEVO
                chart_adr_b64      = _plot_adr_png(act, ly, gran)
                ctx = {
                    "apto": apto, "owner": owner,
                    "period_act": _period_label(start, end),
                    "period_ly": _period_label(pd.to_datetime(start)-pd.DateOffset(years=1), pd.to_datetime(end)-pd.DateOffset(years=1)),
                    "act": {"ingresos": _fmt_money(k_act['ingresos']), "adr": _fmt_money(k_act['adr']), "noches": f"{k_act['noches']:,}".replace(",",".")},
                    "ly":  {"ingresos": _fmt_money(k_ly['ingresos']),  "adr": _fmt_money(k_ly['adr']),  "noches": f"{k_ly['noches']:,}".replace(",",".")},
                    "chart_portales": chart_portales_b64,               # ← NUEVO
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
                # pdfkit
                import pdfkit
                options = {
                    "enable-local-file-access": None,
                    "page-size": "A4",
                    "margin-top": "12mm",
                    "margin-bottom": "12mm",
                    "margin-left": "12mm",
                    "margin-right": "12mm",
                    "encoding": "UTF-8",
                    "quiet": "",
                    "footer-right": "Generado: " + date.today().isoformat() + "   |   Página [page]/[toPage]",  # ← pie profesional
                    "footer-font-size": "8",
                    "footer-spacing": "3",
                    "header-left": "Florit Flats · Informe de propietario",    # ← cabecera simple
                    "header-font-size": "9",
                    "header-spacing": "3",
                }
                pdf_bytes = pdfkit.from_string(html, False, configuration=cfg, options=options)
                st.download_button("Descargar PDF", data=pdf_bytes, file_name=f"Informe_{apto or 'propietario'}.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error("No se pudo generar el PDF.")
                with st.expander("Detalle"):
                    st.exception(e)

# QUITAR: función _plot_pace_png (ya no se usa)