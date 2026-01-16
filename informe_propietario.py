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
import re
import calendar

TITLE = "Informe de propietario"
BUILD = "v1.9-portales"

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
    ch  = find("portal","canal","channel","ota","fuente","agencia","source","agente","intermediario")
    # ← ampliamos sin confundir con “Nº reserva”
    fa  = (find("fecha alta","fecha de alta","fecha reserva","fecha de reserva",
                "fecha confirmación","fecha confirmacion",
                "fecha creación","fecha creacion",
                "booking date","book date","fecha booking",
                "creada","creado","booked","created"))
    if a:   m[a] = "Alojamiento"
    if fi:  m[fi] = "Fecha entrada"
    if fo:  m[fo] = "Fecha salida"
    if rev: m[rev] = "Alquiler con IVA (€)"
    if adr: m[adr] = "ADR (opcional)"
    if ch:  m[ch] = "Portal"
    if fa:  m[fa] = "Fecha alta"
    out = df.rename(columns=m).copy() if m else df.copy()
    # Usar la columna E como Portal SOLO si su cabecera es “Agente/Intermediario”
    if "Portal" not in out.columns and out.shape[1] >= 5:
        col_e = out.columns[4]
        if str(col_e).strip().lower() in {"agente/intermediario","agente - intermediario","agente","intermediario"}:
            out = out.rename(columns={col_e: "Portal"})
    return out

# Normalización del nombre del portal (Booking, Airbnb, Vrbo, Expedia Group, Directo, etc.)
def _canon_portal(x) -> str:
    if x is None:
        return "Directa"
    s = str(x).strip()
    if s == "":
        return "Directa"
    sl = re.sub(r"[\s_\-\.]+", " ", s.lower())
    sl = (sl.replace("á","a").replace("é","e").replace("í","i")
             .replace("ó","o").replace("ú","u").replace("ü","u").replace("ñ","n"))

    # Directa
    if any(k in sl for k in ["direct", "directo", "directa", "propia", "web", "sitio", "pagina"]):
        return "Directa"
    # Principales OTAs
    if "book" in sl or "bkg" in sl:
        return "Booking.com"
    if "air" in sl and "bnb" in sl:
        return "Airbnb"
    if "vrbo" in sl or "homeaway" in sl:
        return "Vrbo"
    if any(k in sl for k in ["expedia","hotels com","hoteis com","orbitz","travelocity","ebookers","mrjet","wotif","cheaptickets"]):
        return "Expedia Group"
    # Otros canales frecuentes
    if any(k in sl for k in ["telefono","phone","call"]):
        return "Teléfono"
    if "walk" in sl:
        return "Walk-in"
    if any(k in sl for k in ["agency","agencia","touroper","touroperator","wholesaler","mayorista","agente","intermediario"]):
        return "Agencia"
    if "owner" in sl or "propiet" in sl:
        return "Propietario"
    if any(k in sl for k in ["manual","otros","other"]):
        return "Otros"
    return s  # deja el nombre legible original

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
    # Evitar ZeroDivisionError: usar arrays NumPy para que la división no se evalúe cuando el denominador es 0
    act_noches = act["Noches"].to_numpy(dtype=float)
    act_ing = act["Ingresos"].to_numpy(dtype=float)
    act["ADR"] = np.where(act_noches > 0, act_ing / act_noches, 0.0)

    s_ly = s - pd.DateOffset(years=1)
    e_ly = e - pd.DateOffset(years=1)
    ly = _overlap_nights_rows(d, s_ly, e_ly)
    rng_ly = pd.date_range(s_ly, e_ly, freq="D")
    ly = ly.set_index("Fecha").reindex(rng_ly, fill_value=0).rename_axis("Fecha").reset_index()
    ly_noches = ly["Noches"].to_numpy(dtype=float)
    ly_ing = ly["Ingresos"].to_numpy(dtype=float)
    ly["ADR"] = np.where(ly_noches > 0, ly_ing / ly_noches, 0.0)
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
def _fmt_money(x: float, decimals: int = 2) -> str:
    if x is None:
        return "—"
    try:
        s = f"{x:,.{decimals}f}"
        # convertir a formato español: miles con '.' y decimales con ','
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"€{s}"
    except Exception:
        return f"€{x:.{decimals}f}"

def _fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None:
        return "—"
    try:
        s = f"{x:.{decimals}f}".replace(".", ",")
        return f"{s} %"
    except Exception:
        return f"{x:.{decimals}f} %"

# NUEVO: reservas por portal en el periodo (por fecha de alta si existe; si no, por check-in)
def _bookings_by_portal(df_raw: pd.DataFrame, start: date, end: date, props: list[str] | None) -> pd.DataFrame:
    d = _preprocess(df_raw)
    if props and "Alojamiento" in d.columns:
        d = d[d["Alojamiento"].astype(str).isin(props)].copy()
    ref = d["Fecha alta"] if "Fecha alta" in d.columns else d["Fecha entrada"]
    mask = (ref >= pd.to_datetime(start)) & (ref <= pd.to_datetime(end))
    d = d.loc[mask].copy()

    # Portal: si falta la columna o hay valores NaN/empty, asignar "FloritFlats"
    if "Portal" not in d.columns:
        d["Portal"] = "FloritFlats"
    else:
        # sustituye NA o cadenas vacías por "FloritFlats" antes de normalizar
        d["Portal"] = d["Portal"].where(d["Portal"].notna() & d["Portal"].astype(str).str.strip().ne(""), "FloritFlats")

    d["Portal"] = d["Portal"].astype(str).map(_canon_portal)
    # asegurar que no queden vacíos
    d.loc[d["Portal"].eq("") | d["Portal"].isna(), "Portal"] = "FloritFlats"

    g = (d.groupby("Portal", dropna=False)
           .agg(Reservas=("Portal","size"),
                Ingresos=("Alquiler con IVA (€)","sum"))
           .reset_index()
           .sort_values(["Reservas","Ingresos"], ascending=[False, False]))
    g["Reservas"] = g["Reservas"].astype(int)
    g["Ingresos"] = g["Ingresos"].astype(float)
    return g

# NUEVO: gráfico PNG para PDF (reservas por portal)
def _plot_portales_png(portal_df: pd.DataFrame) -> str:
    if portal_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Sin datos de reservas en el periodo", ha="center", va="center")
        ax.axis("off")
        return _png_from_plt(fig)
    df = portal_df.copy().sort_values("Reservas", ascending=True)
    n = len(df)
    # grosor de barra más fino; aún más fino si solo hay 1-2 portales
    bar_h = 0.12 if n <= 2 else (0.22 if n <= 6 else 0.45)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(df["Portal"], df["Reservas"], color="#2e485f", height=bar_h, align="center")
    for i, v in enumerate(df["Reservas"]):
        ax.text(v + max(df["Reservas"])*0.01, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("Reservas")
    ax.grid(axis="x", alpha=0.2)
    ax.margins(y=0.25)  # añade aire vertical para que no se vea “gorda”
    fig.tight_layout()
    return _png_from_plt(fig)

def _png_from_plt(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")  # ← más resolución
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _plot_adr_png(act: pd.DataFrame, ly: pd.DataFrame, gran: str) -> str:
    # ADR por día o semana (para PDF)
    if gran == "Semana":
        w_act = act.assign(Sem=act["Fecha"].dt.to_period("W-MON").dt.start_time)
        actp = w_act.groupby("Sem", as_index=False).agg({"Ingresos":"sum","Noches":"sum"}).rename(columns={"Sem":"Fecha"})
        # safe ADR computation (avoid division by zero)
        a_noches = actp["Noches"].to_numpy(dtype=float)
        a_ing = actp["Ingresos"].to_numpy(dtype=float)
        a_den = np.where(a_noches == 0, 1.0, a_noches)
        actp["ADR"] = np.where(a_noches > 0, a_ing / a_den, 0.0)

        w_ly = ly.assign(Sem=ly["Fecha"].dt.to_period("W-MON").dt.start_time)
        lyp = w_ly.groupby("Sem", as_index=False).agg({"Ingresos":"sum","Noches":"sum"}).rename(columns={"Sem":"Fecha"})
        l_noches = lyp["Noches"].to_numpy(dtype=float)
        l_ing = lyp["Ingresos"].to_numpy(dtype=float)
        l_den = np.where(l_noches == 0, 1.0, l_noches)
        lyp["ADR"] = np.where(l_noches > 0, l_ing / l_den, 0.0)
    else:
        actp, lyp = act.copy(), ly.copy()
    lyp = lyp.copy()
    lyp["Fecha"] = lyp["Fecha"] + pd.DateOffset(years=1)

    fig, ax = plt.subplots(figsize=(10, 4.0))   # ← más grande
    ax.plot(actp["Fecha"], actp["ADR"], marker="o", ms=3, color="#2e485f", label="Act")
    ax.plot(lyp["Fecha"], lyp["ADR"], marker="o", ms=3, color="#c77b2b", label="LY")
    ax.set_ylabel("€")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _png_from_plt(fig)

# === NUEVO: estadísticas de estancia y antelación ===
def _booking_stats(df_raw: pd.DataFrame, start: date, end: date, props: list[str] | None) -> dict:
    d = _preprocess(df_raw)
    if props and "Alojamiento" in d.columns:
        d = d[d["Alojamiento"].astype(str).isin(props)].copy()
    # Tomamos reservas con check-in dentro del periodo
    m = (d["Fecha entrada"] >= pd.to_datetime(start)) & (d["Fecha entrada"] <= pd.to_datetime(end))
    d = d.loc[m].copy()
    if d.empty:
        return {"los_avg": None, "lead_avg": None}

    los = (d["Fecha salida"] - d["Fecha entrada"]).dt.days.clip(lower=1)
    los_avg = float(los.mean()) if len(los) else None

    lead_avg = None
    if "Fecha alta" in d.columns:
        lead = (d["Fecha entrada"].dt.normalize() - d["Fecha alta"].dt.normalize()).dt.days
        lead = lead[lead >= 0]  # descarta negativos/NaN
        if len(lead):
            lead_avg = float(lead.mean())

    return {"los_avg": los_avg, "lead_avg": lead_avg}

# ----------------- Render -----------------
def render_informe_propietario(raw: pd.DataFrame | None = None):
    st.header(TITLE)
    st.caption(f"Build {BUILD} · fuente: {Path(__file__).name}")

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

    # Estadísticas LOS y Antelación
    stats = _booking_stats(df, start, end, props)

    # Deltas YoY (para las métricas)
    def _pct_delta(a: float, b: float) -> float | None:
        try:
            return None if b == 0 else (a - b) / b * 100.0
        except Exception:
            return None
    d_ing = _pct_delta(k_act["ingresos"], k_ly["ingresos"])
    d_adr = _pct_delta(k_act["adr"],      k_ly["adr"])
    d_nch = _pct_delta(k_act["noches"],   k_ly["noches"])

    # --- NUEVAS ESTADÍSTICAS SIMPLES ---
    total_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    revpar_act = k_act["ingresos"] / (max(1, inv_units) * total_days) if total_days > 0 else 0.0
    revpar_ly  = k_ly["ingresos"]  / (max(1, inv_units) * total_days) if total_days > 0 else 0.0

    # Top portals con % share (para UI y PDF)
    portal_df = _bookings_by_portal(df, start, end, props)
    total_portal_ing = float(portal_df["Ingresos"].sum()) if not portal_df.empty else 0.0
    if not portal_df.empty and total_portal_ing > 0:
        portal_df = portal_df.assign(Share=lambda x: x["Ingresos"] / total_portal_ing * 100.0)
    else:
        portal_df = portal_df.assign(Share=0.0)
    top_portals = portal_df.sort_values("Ingresos", ascending=False).head(5).to_dict(orient="records")

    # Ocupación por día de la semana (media sobre los días del periodo)
    rng = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D")
    days_by_wd = {i: int((rng.weekday == i).sum()) for i in range(7)}
    weekday_names = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    wgroup = act.copy()
    wgroup["wd"] = wgroup["Fecha"].dt.weekday
    nights_by_wd = wgroup.groupby("wd")["Noches"].sum().to_dict()
    weekday_rows = []
    for i, name in enumerate(weekday_names):
        days_count = days_by_wd.get(i, 0)
        nights = float(nights_by_wd.get(i, 0.0))
        pct = (nights / (max(1, inv_units) * days_count) * 100.0) if days_count > 0 else 0.0
        weekday_rows.append({"day": name, "pct": _fmt_pct(pct, 2)})

    # LOS median y buckets; Lead-time buckets
    d_stats = _preprocess(df)
    if props and "Alojamiento" in d_stats.columns:
        d_stats = d_stats[d_stats["Alojamiento"].astype(str).isin(props)].copy()
    # filtrar por fecha entrada en el periodo
    mask_in = (d_stats["Fecha entrada"] >= pd.to_datetime(start)) & (d_stats["Fecha entrada"] <= pd.to_datetime(end))
    d_stats = d_stats.loc[mask_in].copy()
    los_median = None
    los_buckets = {"1":0.0,"2":0.0,"3-6":0.0,"7+":0.0}
    lead_buckets = {"<7d":0.0,"7-30d":0.0,">30d":0.0}
    if not d_stats.empty:
        los = (d_stats["Fecha salida"] - d_stats["Fecha entrada"]).dt.days.clip(lower=1)
        los_median = float(los.median()) if len(los) else None
        total_res = len(los)
        los_buckets = {
            "1": int((los == 1).sum()) / total_res * 100.0,
            "2": int((los == 2).sum()) / total_res * 100.0,
            "3-6": int(((los >= 3) & (los <= 6)).sum()) / total_res * 100.0,
            "7+": int((los >= 7).sum()) / total_res * 100.0,
        }
        # lead time
        if "Fecha alta" in d_stats.columns:
            lead = (d_stats["Fecha entrada"].dt.normalize() - d_stats["Fecha alta"].dt.normalize()).dt.days
            lead = lead.dropna()
            if len(lead):
                total_lead = len(lead)
                lead_buckets = {
                    "<7d": int((lead < 7).sum()) / total_lead * 100.0,
                    "7-30d": int(((lead >= 7) & (lead <= 30)).sum()) / total_lead * 100.0,
                    ">30d": int((lead > 30).sum()) / total_lead * 100.0,
                }

    # Top months (por ingresos) — para mostrar en la portada/HTML también
    monthly_rows_raw = _monthly_summary(df, pd.to_datetime(start), pd.to_datetime(end), props, inv_units)
    # intentar cargar forecast y añadir columna 'forecast' prorrateada por periodo
    fdf = _load_forecast_db()
    forecast_by_month, forecast_total = _forecast_for_period(fdf, pd.to_datetime(start), pd.to_datetime(end), props)
    for r in monthly_rows_raw:
        r["forecast"] = float(forecast_by_month.get(r.get("mes"), 0.0))
    top_months = sorted(monthly_rows_raw, key=lambda r: r.get("ing_act", 0.0), reverse=True)[:3]
    # progreso global ACT vs forecast
    forecast_total = float(forecast_total or sum(r.get("forecast", 0.0) for r in monthly_rows_raw))
    prog_pct = int(min(100, round((k_act["ingresos"] / forecast_total * 100) if forecast_total > 0 else 0)))
    st.subheader("Progreso ingresos vs previsión")
    st.write(f"ACT: { _fmt_money(k_act['ingresos'],2) }  ·  Previsión periodo: { _fmt_money(forecast_total,2) }  ·  {prog_pct}%")
    st.progress(prog_pct)

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

    # Fila extra de métricas (estancia media y antelación)
    c4, c5 = st.columns(2)
    with c4:
        st.metric("Estancia media (LOS)", f"{stats['los_avg']:.1f} días" if stats["los_avg"] is not None else "—")
    with c5:
        st.metric("Antelación media", f"{stats['lead_avg']:.0f} días" if stats["lead_avg"] is not None else "—")
    if "Fecha alta" not in df.columns:
        st.caption("Antelación media: no se encontró una columna de fecha de reserva (usa “Fecha alta”, “Fecha reserva”, “Fecha confirmación”…).")

    # (Tabla previa de "Reservas por portal" en la UI eliminada;
    #  se mantiene el bloque de portales en el PDF y el cálculo arriba)

    # ADR (sube la altura)
    st.subheader(f"ADR por {gran.lower()} (Act vs LY alineado)")

    # === construir adr_plot (ACT vs LY) ===
    if gran == "Semana":
        actp = (act.assign(Sem=act["Fecha"].dt.to_period("W-MON").dt.start_time)
                  .groupby("Sem", as_index=False)
                  .agg({"Ingresos":"sum","Noches":"sum"}))
        # safe division: evitar ZeroDivisionError
        _a_noches = actp["Noches"].to_numpy(dtype=float)
        _a_ing = actp["Ingresos"].to_numpy(dtype=float)
        _a_den = np.where(_a_noches == 0, 1.0, _a_noches)
        actp["ADR"] = np.where(_a_noches > 0, _a_ing / _a_den, 0.0)
        actp = actp.rename(columns={"Sem":"Fecha"})
 
        lyp = (ly.assign(Sem=ly["Fecha"].dt.to_period("W-MON").dt.start_time)
                 .groupby("Sem", as_index=False)
                 .agg({"Ingresos":"sum","Noches":"sum"}))
        _l_noches = lyp["Noches"].to_numpy(dtype=float)
        _l_ing = lyp["Ingresos"].to_numpy(dtype=float)
        _l_den = np.where(_l_noches == 0, 1.0, _l_noches)
        lyp["ADR"] = np.where(_l_noches > 0, _l_ing / _l_den, 0.0)
        lyp = lyp.rename(columns={"Sem":"Fecha"})
    else:
        actp, lyp = act.copy(), ly.copy()
    lyp = lyp.copy()
    lyp["Fecha"] = lyp["Fecha"] + pd.DateOffset(years=1)

    adr_plot = pd.concat(
        [
            actp[["Fecha","ADR"]].assign(Serie="Act"),
            lyp[["Fecha","ADR"]].assign(Serie="LY"),
        ],
        ignore_index=True
    )

    chart_adr = (
        alt.Chart(adr_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("Fecha:T", title="Fecha"),
            y=alt.Y("ADR:Q", title="ADR"),
            color=alt.Color("Serie:N", scale=alt.Scale(range=["#2e485f","#c77b2b"])),
            tooltip=[alt.Tooltip("Serie:N"), alt.Tooltip("Fecha:T"), alt.Tooltip("ADR:Q", format=",.0f")],
        )
        .properties(height=340)
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

    # ================= Exportar a PDF =================
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
                # chart_portales_b64 = _plot_portales_png(portal_df)   # ← eliminado
                chart_adr_b64      = _plot_adr_png(act, ly, gran)
                # calcular resumen mensual para la segunda página
                monthly_rows_raw = monthly_rows_raw  # (se calculó arriba)
                # formatear para plantilla: € con 2 decimales y % con 2 decimales
                monthly_rows = []
                for r in monthly_rows_raw:
                    monthly_rows.append({
                        "mes": r.get("mes"),
                        "ing_act": _fmt_money(r.get("ing_act", 0.0), 2),
                        "ing_ly":  _fmt_money(r.get("ing_ly", 0.0), 2),
                        "forecast": _fmt_money(r.get("forecast", 0.0), 2),
                        "adr_act": _fmt_money(r.get("adr_act", 2), 2),
                        "adr_ly":  _fmt_money(r.get("adr_ly", 2), 2),
                        "ocup_act": _fmt_pct(r.get("ocup_act", 0.0), 2),
                        "ocup_ly":  _fmt_pct(r.get("ocup_ly", 0.0), 2),
                        "nights_act": f"{r.get('nights_act',0):,}".replace(",", "."),
                        "nights_ly":  f"{r.get('nights_ly',0):,}".replace(",", "."),
                    })
                # formatear top_portals y top_months, weekday, buckets para plantilla
                tp_formatted = []
                for p in top_portals:
                    tp_formatted.append({
                        "Portal": p.get("Portal"),
                        "Ingresos": _fmt_money(p.get("Ingresos", 0.0), 2),
                        "Share": f"{p.get('Share',0.0):.2f} %"
                    })
                wd_formatted = [{"day": w["day"], "pct": w["pct"]} for w in weekday_rows]
                los_b_fmt = {k: f"{v:.1f} %" for k,v in los_buckets.items()}
                lead_b_fmt = {k: f"{v:.1f} %" for k,v in lead_buckets.items()}
                top_months_fmt = [{"mes": m["mes"], "ing": _fmt_money(m.get("ing_act",0.0),2)} for m in top_months]

                # Unificar tabla de portales (Reservas + Ingresos + Share) para el PDF
                portal_rows_pdf = []
                try:
                    if not portal_df.empty:
                        total_res = int(portal_df["Reservas"].sum()) if not pd.isna(portal_df["Reservas"].sum()) else 0
                        for _, pr in portal_df.iterrows():
                            reservas = int(pr["Reservas"]) if not pd.isna(pr["Reservas"]) else 0
                            ingresos_val = float(pr["Ingresos"]) if not pd.isna(pr["Ingresos"]) else 0.0
                            pct_res = (reservas / total_res * 100.0) if total_res > 0 else 0.0
                            portal_rows_pdf.append({
                                "Portal": pr["Portal"],
                                "Reservas": reservas,
                                "PctReservas": f"{pct_res:.2f} %",
                                "Ingresos": _fmt_money(ingresos_val, 2),
                                # "% share" eliminado de la tabla PDF
                            })
                except Exception:
                    portal_rows_pdf = []

                # Si el usuario seleccionó varios pisos, generar un resumen mensual por piso
                monthly_by_property = []
                try:
                    if props and isinstance(props, (list, tuple)) and len(props) > 1:
                        for p in props:
                            mr_raw = _monthly_summary(df, pd.to_datetime(start), pd.to_datetime(end), [p], 1)
                            # obtener forecast por mes para este piso (si hay DB)
                            prop_forecast_map, _ = _forecast_for_period(fdf, pd.to_datetime(start), pd.to_datetime(end), [p])
                            mr_fmt = []
                            for r in mr_raw:
                                mr_fmt.append({
                                    "mes": r.get("mes"),
                                    "ing_act": _fmt_money(r.get("ing_act", 0.0), 2),
                                    "ing_ly":  _fmt_money(r.get("ing_ly", 0.0), 2),
                                    "forecast": _fmt_money(float(prop_forecast_map.get(r.get("mes"), 0.0)), 2),
                                    "adr_act": _fmt_money(r.get("adr_act", 2), 2),
                                    "adr_ly":  _fmt_money(r.get("adr_ly", 2), 2),
                                    "ocup_act": _fmt_pct(r.get("ocup_act", 0.0), 2),
                                    "ocup_ly":  _fmt_pct(r.get("ocup_ly", 0.0), 2),
                                    "nights_act": f"{r.get('nights_act',0):,}".replace(",","."),
                                    "nights_ly":  f"{r.get('nights_ly',0):,}".replace(",","."),
                                })
                            monthly_by_property.append({"apto": p, "monthly_rows": mr_fmt})
                except Exception:
                    monthly_by_property = []

                ctx = {
                    "apto": apto, "owner": owner,
                    "period_act": _period_label(start, end),
                    "period_ly": _period_label(pd.to_datetime(start)-pd.DateOffset(years=1),
                                               pd.to_datetime(end)-pd.DateOffset(years=1)),
                    "act": {"ingresos": _fmt_money(k_act['ingresos']), "adr": _fmt_money(k_act['adr']), "noches": f"{k_act['noches']:,}".replace(",",".")},
                    "ly":  {"ingresos": _fmt_money(k_ly['ingresos']),  "adr": _fmt_money(k_ly['adr']),  "noches": f"{k_ly['noches']:,}".replace(",",".")},
                    "portal_rows": portal_rows_pdf,
                    "chart_adr": chart_adr_b64,
                    "gran_label": gran.lower(),
                    "comments": st.session_state.get("inf_comments") or "",
                    "logo_b64": _load_logo_b64(),
                    "los_avg": f"{stats['los_avg']:.1f}" if stats["los_avg"] is not None else "—",
                    "lead_avg": f"{stats['lead_avg']:.0f}" if stats["lead_avg"] is not None else "—",
                    "monthly_rows": monthly_rows,   # ← existente
                    "monthly_by_property": monthly_by_property,
                    # NUEVOS campos:
                    "revpar": {"act": _fmt_money(revpar_act,2), "ly": _fmt_money(revpar_ly,2)},
                    "top_portals": tp_formatted,
                    "weekday_rows": wd_formatted,
                    "los_median": (f"{los_median:.1f}" if los_median is not None else "—"),
                    "los_buckets": los_b_fmt,
                    "lead_buckets": lead_b_fmt,
                    "top_months": top_months_fmt,
                    # Forecast / progreso
                    "forecast_total": _fmt_money(forecast_total, 2),
                    "forecast_progress_pct": f"{prog_pct}%",
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
                    "disable-smart-shrinking": None,  # ← evita que encoja
                    "zoom": "1.0",
                    "footer-right": "Generado: " + date.today().isoformat() + "   |   Página [page]/[toPage]",
                    "footer-font-size": "8",
                    "footer-spacing": "3",
                    "header-left": "Florit Flats · Informe de propietario",
                    "header-font-size": "9",
                    "header-spacing": "3",
                }
                pdf_bytes = pdfkit.from_string(html, False, configuration=cfg, options=options)
                st.download_button("Descargar PDF", data=pdf_bytes, file_name=f"Informe_{apto or 'propietario'}.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error("No se pudo generar el PDF.")
                with st.expander("Detalle"):
                    st.exception(e)

# ===== Helpers de fechas y logo =====
def _to_dt(s: pd.Series) -> pd.Series:
    # Strings/fechas
    out = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # Números: Excel y UNIX (s/ms/us/ns)
    nums = pd.to_numeric(s, errors="coerce")
    is_num = nums.notna()
    if not is_num.any():
        return out

    excel_mask = is_num & nums.between(20000, 80000)  # días Excel ~1955–2120
    sec_mask   = is_num & nums.between(1_000_000_000, 9_999_999_999)
    ms_mask    = is_num & nums.between(1_000_000_000_000, 9_999_999_999_999)
    us_mask    = is_num & nums.between(1_000_000_000_000_000, 9_999_999_999_999_999)
    ns_mask    = is_num & nums.between(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)

    base = pd.Timestamp("1899-12-30")
    if excel_mask.any():
        out.loc[excel_mask] = base + pd.to_timedelta(nums.loc[excel_mask], unit="D")
    if sec_mask.any():
        out.loc[sec_mask] = pd.to_datetime(nums.loc[sec_mask], unit="s", origin="unix", errors="coerce")
    if ms_mask.any():
        out.loc[ms_mask] = pd.to_datetime(nums.loc[ms_mask], unit="ms", origin="unix", errors="coerce")
    if us_mask.any():
        out.loc[us_mask] = pd.to_datetime(nums.loc[us_mask], unit="us", origin="unix", errors="coerce")
    if ns_mask.any():
        out.loc[ns_mask] = pd.to_datetime(nums.loc[ns_mask], unit="ns", origin="unix", errors="coerce")
    return out

def _period_label(s: date | pd.Timestamp, e: date | pd.Timestamp) -> str:
    sd = pd.to_datetime(s).date()
    ed = pd.to_datetime(e).date()
    return f"{sd.strftime('%d/%m/%Y')} – {ed.strftime('%d/%m/%Y')}"
    
# Añadido: devuelve el logo como base64 (o None si no existe)
def _load_logo_b64() -> str | None:
    """Devuelve el logo en base64 si existe."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "assets" / "florit-flats-logo.png",
        here.parent / "assets" / "florit-flats-logo.png",
        here / "assets" / "logo.png",
    ]
    for p in candidates:
        try:
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            pass
    return None

# --- NEW: rangos por mes y resumen mensual ---
def _month_ranges(start: date | pd.Timestamp, end: date | pd.Timestamp):
    s = pd.to_datetime(start).date().replace(day=1)
    e = pd.to_datetime(end).date()
    out = []
    cur = s
    while cur <= e:
        y, m = cur.year, cur.month
        last = calendar.monthrange(y, m)[1]
        ms = pd.Timestamp(date(y, m, 1))
        me = pd.Timestamp(date(y, m, last))
        out.append((ms, me))
        # next month
        if m == 12:
            cur = date(y + 1, 1, 1)
        else:
            cur = date(y, m + 1, 1)
    return out

def _monthly_summary(df_raw: pd.DataFrame, start: date, end: date, props: list[str] | None, inv_units: int) -> list[dict]:
    """
    Devuelve lista de dicts por mes con métricas ACT vs LY:
    {'mes': 'Enero 2026', 'ing_act':float, 'ing_ly':float, 'adr_act':float, 'adr_ly':float, 'ocup_act':float, 'ocup_ly':float}
    """
    d = _preprocess(df_raw)
    if props and "Alojamiento" in d.columns:
        d = d[d["Alojamiento"].astype(str).isin(props)].copy()
    months = _month_ranges(start, end)
    rows = []
    for ms, me in months:
        # ACT
        daily_act = _overlap_nights_rows(d, ms, me)
        nights_act = int(daily_act["Noches"].sum()) if not daily_act.empty else 0
        ing_act = float(daily_act["Ingresos"].sum()) if not daily_act.empty else 0.0
        adr_act = (ing_act / nights_act) if nights_act > 0 else 0.0
        days_month = (pd.to_datetime(me).date() - pd.to_datetime(ms).date()).days + 1
        ocup_act = (nights_act / (max(1, int(inv_units)) * days_month) * 100.0) if days_month > 0 else 0.0

        # LY (mismo mes año anterior)
        ms_ly = ms - pd.DateOffset(years=1)
        me_ly = me - pd.DateOffset(years=1)
        daily_ly = _overlap_nights_rows(d, ms_ly, me_ly)
        nights_ly = int(daily_ly["Noches"].sum()) if not daily_ly.empty else 0
        ing_ly = float(daily_ly["Ingresos"].sum()) if not daily_ly.empty else 0.0
        adr_ly = (ing_ly / nights_ly) if nights_ly > 0 else 0.0
        ocup_ly = (nights_ly / (max(1, int(inv_units)) * days_month) * 100.0) if days_month > 0 else 0.0

        rows.append({
            "mes": ms.strftime("%B %Y"),
            "ing_act": ing_act, "ing_ly": ing_ly,
            "adr_act": adr_act, "adr_ly": adr_ly,
            "ocup_act": ocup_act, "ocup_ly": ocup_ly,
            "days": days_month,
            "nights_act": nights_act, "nights_ly": nights_ly,
        })
    return rows

# NUEVO: carga y cálculo de forecast desde data/forecast_db.csv
def _parse_euro(s: str) -> float:
    try:
        if pd.isna(s):
            return 0.0
        t = str(s).replace("�", "").strip()
        # eliminar espacios y normalizar
        t = t.replace(" ", "")
        # quitar miles '.' y convertir decimal ',' -> '.'
        # si hay tanto '.' como ',' asumimos '.'=miles y ','=decimales
        if t.count(",") == 1 and t.count(".") > 0:
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", ".").replace(" ", "")
        # dejar sólo dígitos, punto y signo
        t = re.sub(r"[^\d\.\-]", "", t)
        return float(t) if t != "" else 0.0
    except Exception:
        return 0.0

def _load_forecast_db() -> pd.DataFrame | None:
    base = Path(__file__).resolve().parent
    f = base / "data" / "forecast_db.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f, sep=";", dtype=str, encoding="utf-8", engine="python")
    except Exception:
        try:
            df = pd.read_csv(f, sep=";", dtype=str, encoding="latin-1", engine="python")
        except Exception:
            return None
    # limpiar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]
    # normalizar apartment name column
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Apartamento"})
    # parse numeric month columns
    for c in df.columns[1:]:
        df[c] = df[c].apply(_parse_euro)
    # normalizar apartment key
    df["__apt_key"] = df["Apartamento"].astype(str).str.strip().str.lower()
    return df

def _forecast_for_period(forecast_df: pd.DataFrame | None, start: date, end: date, props: list[str] | None) -> tuple[dict, float]:
    """
    Devuelve (dict mes->forecast, total_forecast) donde 'mes' coincide con monthly_summary 'mes' (p.e. 'Enero 2026').
    Prorratea meses parciales según días cubiertos.
    """
    if forecast_df is None or forecast_df.empty:
        return {}, 0.0
    # columnas meses en el csv esperadas en español (encabezados)
    meses_es = [c.strip() for c in forecast_df.columns[1:-1]] if "__apt_key" in forecast_df.columns else [c.strip() for c in forecast_df.columns[1:]]
    # map apt keys if props given
    selected_mask = None
    if props and len(props) > 0:
        keys = [str(p).strip().lower() for p in props]
        selected_mask = forecast_df["__apt_key"].isin(keys)
    else:
        selected_mask = pd.Series(True, index=forecast_df.index)

    months = _month_ranges(start, end)
    out = {}
    total = 0.0
    for ms, me in months:
        # nombre del mes en formato igual al que produce _monthly_summary: "Enero 2026"
        mes_label = ms.strftime("%B %Y")
        # buscar columna correspondiente en forecast_df (coincidir por nombre mes en español)
        # intentar varias capitalizaciones/espacios
        col_name = None
        target = ms.month  # 1..12
        # buscar columna cuyo nombre contiene número de mes en español
        for c in forecast_df.columns[1:]:
            cc = c.strip().lower()
            # comparar por nombre de mes en español
            meses_map = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
            if meses_map[target-1] in cc:
                col_name = c
                break
        # si no encontrado por nombre exacto, intentar por posición (asumir columnas están en orden enero..diciembre)
        if col_name is None:
            cols_months = [c for c in forecast_df.columns[1:] ]
            if len(cols_months) >= target:
                col_name = cols_months[target-1]
        # si aún no hay columna, poner 0
        if col_name is None:
            f_val = 0.0
        else:
            # sumar filas seleccionadas y prorratear según días del mes que están en el periodo
            days_in_month = (pd.to_datetime(me).date() - pd.to_datetime(ms).date()).days + 1
            # overlap with period (in case start/end cut month)
            ov_s = max(pd.to_datetime(ms).date(), pd.to_datetime(start).date())
            ov_e = min(pd.to_datetime(me).date(), pd.to_datetime(end).date())
            covered_days = (ov_e - ov_s).days + 1
            covered_days = max(0, covered_days)
            month_sum = float(forecast_df.loc[selected_mask, col_name].sum(skipna=True))
            # prorrateo
            f_val = month_sum * (covered_days / days_in_month) if days_in_month > 0 else 0.0
        out[mes_label] = f_val
        total += f_val
    return out, total