import pandas as pd
import streamlit as st
from datetime import date
from utils import period_inputs
from landing_ui import render_landing, LOGO_PATH

# --- Define la función aquí ---
def cargar_dataframe():
    # Devuelve un DataFrame vacío por defecto
    return pd.DataFrame()

def _get_raw():
    return st.session_state.get("df_active") or st.session_state.get("raw")

def _safe_call(view_fn):
    raw = _get_raw()
    try:
        return view_fn(raw)
    except TypeError:
        return view_fn()

from consulta_normal import render_consulta_normal
from resumen_comparativo import render_resumen_comparativo
from kpis_por_meses import render_kpis_por_meses
from evolucion_cutoff import render_evolucion_corte
from pickup import render_pickup
from pace import render_pace
from prediccion_pace import render_prediccion_pace
from cuadro_mando_pro import render_cuadro_mando_pro
from alerts_module import render_alerts_module
from what_if import render_what_if
from evolucion_cutoff import render_evolucion_cutoff

# Configuración de página
st.set_page_config(page_title="Revenue Dashboard", page_icon="📊", layout="wide")

# Estilos globales (tema claro + color corporativo)
st.markdown("""
<style>
:root{
  --brand:#2e485f;        /* Florit Flats */
  --brand-600:#264052;    /* hover/bordes */
  --brand-50:#f3f6f9;     /* fondos suaves */
}
.stApp { background:#ffffff; }

/* Tarjetas y paneles */
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] { background: var(--brand-50); }

/* Botones */
.stButton > button, .btn-primary {
  background: var(--brand);
  color: #fff;
  border: 1px solid var(--brand-600);
  border-radius: 10px;
}
.stButton > button:hover, .btn-primary:hover {
  background: var(--brand-600);
  border-color: var(--brand-600);
}

/* Métricas */
[data-testid="stMetricDelta"] { color: var(--brand) !important; }

/* Tablas: cabecera sutil */
[data-testid="stTable"], .stDataFrame thead tr th {
  background: var(--brand-50) !important;
}

/* Enlaces/selecciones activas */
a, .st-af { color: var(--brand); }
</style>
""", unsafe_allow_html=True)

# Añade un helper de rerun compatible
def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

with st.sidebar:
    # Logo en la sidebar
    try:
        st.image(LOGO_PATH, use_column_width=True)
    except Exception:
        st.write("Florit Flats")
    if st.button("⬅️ Volver al inicio"):
        st.session_state.view = "landing"
        _rerun()

if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view == "landing":
    render_landing()

elif st.session_state.view == "consulta":
    from consulta_normal import render_consulta_normal
    _safe_call(render_consulta_normal)

elif st.session_state.view == "pro":
    from cuadro_mando_pro import render_cuadro_mando_pro
    _safe_call(render_cuadro_mando_pro)

elif st.session_state.view == "whatif":
    from what_if import render_what_if
    _safe_call(render_what_if)

elif st.session_state.view == "evolucion":
    from evolucion_cutoff import render_evolucion_cutoff
    _safe_call(render_evolucion_cutoff)

# -------- Sidebar: carga de datos --------
st.sidebar.header("Carga de datos")

# Subida de archivo
uploaded_file = st.sidebar.file_uploader("Carga tu archivo Excel/CSV", type=["xlsx", "csv"])

# Botón para confirmar el archivo
if uploaded_file is not None:
    if st.sidebar.button("Usar este archivo"):
        if uploaded_file.name.endswith(".xlsx"):
            st.session_state.raw = pd.read_excel(uploaded_file)
        else:
            st.session_state.raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente.")
else:
    if "raw" not in st.session_state:
        st.session_state.raw = cargar_dataframe()  # Reemplaza por tu función de carga
raw = st.session_state.raw

# Menú de modos
st.sidebar.header("Menú principal")
mode = st.sidebar.selectbox(
    "Selecciona modo",
    [
        "Consulta normal",
        "Resumen Comparativo",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pickup (entre dos cortes)",
        "Pace (curva D)",
        "Predicción (Pace)",
        "Cuadro de mando (PRO)",
        "Panel de alertas",
        "What‑if (escenarios)"
    ]
)

# Routing: llama al modo seleccionado
if raw is None:
    st.warning("⚠️ No hay datos cargados. Sube tu archivo en la barra lateral.")
else:
    if mode == "Consulta normal":
        render_consulta_normal(raw)
    elif mode == "Resumen Comparativo":
        render_resumen_comparativo(raw)
    elif mode == "KPIs por meses":
        render_kpis_por_meses(raw)
    elif mode == "Evolución por fecha de corte":
        render_evolucion_corte(raw)
    elif mode == "Pickup (entre dos cortes)":
        render_pickup(raw)
    elif mode == "Pace (curva D)":
        render_pace(raw)
    elif mode == "Predicción (Pace)":
        render_prediccion_pace(raw)
    elif mode == "Cuadro de mando (PRO)":
        render_cuadro_mando_pro(raw)
    elif mode == "Panel de alertas":
        render_alerts_module(raw)
    elif mode == "What‑if (escenarios)":
        render_what_if(raw)

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import date

BRAND = "#2e485f"

def _ensure_parsed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    dfx = df.copy()
    for c in ["Fecha alta","Fecha entrada","Fecha salida"]:
        if c in dfx.columns:
            dfx[c] = pd.to_datetime(dfx[c], errors="coerce")
    # nombre de importe habitual
    if "Alquiler con IVA (€)" not in dfx.columns:
        # intenta encontrar alguna columna de ingresos razonable
        for cand in ["Ingresos","Revenue","Importe","Total","Precio total"]:
            if cand in dfx.columns:
                dfx["Alquiler con IVA (€)"] = pd.to_numeric(dfx[cand], errors="coerce")
                break
    dfx["Alquiler con IVA (€)"] = pd.to_numeric(dfx.get("Alquiler con IVA (€)"), errors="coerce").fillna(0.0)
    dfx = dfx.dropna(subset=["Fecha entrada","Fecha salida"])
    dfx = dfx[dfx["Fecha salida"] > dfx["Fecha entrada"]].copy()
    dfx["los"] = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    dfx["adr_reserva"] = np.where(dfx["los"] > 0, dfx["Alquiler con IVA (€)"] / dfx["los"], 0.0)
    # normaliza Alojamiento a str si existe
    if "Alojamiento" in dfx.columns:
        dfx["Alojamiento"] = dfx["Alojamiento"].astype(str)
    return dfx

def _overlap_nights_and_revenue(dfx: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Cálculo vectorizado de noches e ingresos dentro del periodo."""
    p_start = pd.to_datetime(p_start).normalize()
    p_end_i = pd.to_datetime(p_end).normalize() + pd.Timedelta(days=1)  # fin exclusivo
    in_start = dfx["Fecha entrada"].clip(lower=p_start)
    in_end = dfx["Fecha salida"].clip(upper=p_end_i)
    n_in = (in_end - in_start).dt.days.clip(lower=0)
    rev_in = n_in.to_numpy(dtype=float) * dfx["adr_reserva"].to_numpy(dtype=float)
    return n_in.to_numpy(dtype=float), rev_in

def render_evolucion_cutoff(raw: pd.DataFrame | None = None):
    st.header("📈 Evolución por fecha de corte")

    if raw is None:
        raw = st.session_state.get("df_active") or st.session_state.get("raw")
    df = _ensure_parsed(raw if isinstance(raw, pd.DataFrame) else pd.DataFrame())
    if df.empty:
        st.info("No hay datos cargados. Vuelve a la portada y sube un CSV/Excel.")
        return

    with st.sidebar:
        st.subheader("Parámetros")
        today = date.today()
        colp1, colp2 = st.columns(2)
        p_start = colp1.date_input("Inicio del periodo (estancias)", value=today.replace(day=1))
        p_end   = colp2.date_input("Fin del periodo (estancias)", value=today)

        colc1, colc2 = st.columns(2)
        c_start = colc1.date_input("Corte inicial", value=today.replace(day=1))
        c_end   = colc2.date_input("Corte final", value=today)

        # Filtro de alojamientos opcional
        props = sorted(df["Alojamiento"].dropna().unique()) if "Alojamiento" in df.columns else []
        selected_props = st.multiselect("Alojamientos (opcional)", options=props, default=[])

        st.caption("Consejo: elige de 5 a 30 fechas de corte para un gráfico fluido.")

    dfx = df.copy()
    if selected_props:
        dfx = dfx[dfx["Alojamiento"].isin(selected_props)]

    # Rangos válidos
    if pd.to_datetime(c_end) < pd.to_datetime(c_start):
        st.error("El corte final debe ser posterior o igual al corte inicial.")
        return
    if pd.to_datetime(p_end) < pd.to_datetime(p_start):
        st.error("El fin del periodo debe ser posterior o igual al inicio.")
        return

    cutoffs = pd.date_range(pd.to_datetime(c_start), pd.to_datetime(c_end), freq="D")
    if len(cutoffs) == 0:
        st.warning("Rango de cortes vacío.")
        return

    # Pre-filtrado por solape con periodo (independiente del corte)
    p_start_dt = pd.to_datetime(p_start).normalize()
    p_end_dt = pd.to_datetime(p_end).normalize()
    stays_overlap = (dfx["Fecha salida"] > p_start_dt) & (dfx["Fecha entrada"] <= p_end_dt + pd.Timedelta(days=1))
    dfx = dfx[stays_overlap].copy()
    if dfx.empty:
        st.warning("No hay estancias que solapen el periodo seleccionado.")
        return

    rows = []
    # Cálculo por corte (filtra reservas con Fecha alta <= corte)
    for cut in cutoffs:
        dfc = dfx[(dfx["Fecha alta"].notna()) & (dfx["Fecha alta"].dt.normalize() <= cut.normalize())]
        if dfc.empty:
            rows.append({"corte": cut.date(), "noches": 0.0, "ingresos": 0.0, "adr": 0.0})
            continue
        n_in, rev_in = _overlap_nights_and_revenue(dfc, p_start_dt, p_end_dt)
        nights = float(np.nansum(n_in))
        revenue = float(np.nansum(rev_in))
        adr = (revenue / nights) if nights > 0 else 0.0
        rows.append({"corte": cut.date(), "noches": nights, "ingresos": revenue, "adr": adr})

    evo = pd.DataFrame(rows)
    if evo.empty:
        st.warning("Sin resultados para el rango de cortes.")
        return

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Ingresos OTB (último corte) €", f"{evo['ingresos'].iloc[-1]:,.2f}")
    c2.metric("Noches OTB (último corte)", f"{evo['noches'].iloc[-1]:,.0f}".replace(",", "."))
    c3.metric("ADR OTB (último corte) €", f"{evo['adr'].iloc[-1]:,.2f}")

    # Gráficos
    base = alt.Chart(evo).encode(x=alt.X("corte:T", title="Fecha de corte"))
    line_rev = base.mark_line(color=BRAND, strokeWidth=3).encode(y=alt.Y("ingresos:Q", title="Ingresos (€)"))
    line_n = base.mark_line(color="#9e9e9e", strokeDash=[4,4]).encode(y=alt.Y("noches:Q", title="Noches", axis=alt.Axis(titleColor="#6b7280")))
    st.altair_chart((line_rev + line_n).properties(height=360), use_container_width=True)

    st.subheader("Detalle")
    st.dataframe(evo, use_container_width=True)
    st.download_button(
        "📥 Descargar evolución (CSV)",
        data=evo.to_csv(index=False).encode("utf-8-sig"),
        file_name="evolucion_por_corte.csv",
        mime="text/csv",
    )
