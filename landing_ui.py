import os, json, requests
from pathlib import Path
import streamlit as st
from streamlit_lottie import st_lottie

# Activa/Desactiva animaciones (Lottie)
USE_ANIMATIONS = False  # ← pon True si quieres animaciones

# Import opcional (no rompe si no está instalado)
try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None
    USE_ANIMATIONS = False

# Rutas (si usas animaciones)
LOTTIE = {
    "consulta": "assets/lottie/consulta.json",
    "pro": "assets/lottie/pro.json",
    "whatif": "assets/lottie/whatif.json",
}

def _load_lottie(src: str):
    if not src:
        return None
    try:
        if os.path.exists(src):  # archivo local
            with open(src, "r", encoding="utf-8") as f:
                return json.load(f)
        if src.lower().startswith(("http://", "https://")):  # URL
            r = requests.get(src, timeout=7)
            if r.ok:
                return r.json()
    except Exception:
        pass
    return None

def _safe_lottie(data, height: int, key: str, label: str):
    # Evita que streamlit_lottie lance excepción si data es None
    try:
        if isinstance(data, dict):
            st_lottie(data, height=height, key=key)
        else:
            st.markdown(
                f'<div style="height:{height}px;display:flex;align-items:center;justify-content:center;opacity:.6;">🖼️ {label}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            f'<div style="height:{height}px;display:flex;align-items:center;justify-content:center;opacity:.6;">🖼️ {label}</div>',
            unsafe_allow_html=True,
        )

def _inject_css():
    st.markdown(
        """
        <style>
        :root{ --brand:#2e485f; --brand-600:#264052; --brand-50:#f3f6f9; }
        .hero { padding:18px 26px; border-radius:14px; background:linear-gradient(135deg,#fff 0%,#f6f8fb 100%); color:#1f2937; border:1px solid rgba(0,0,0,0.04); box-shadow:0 8px 24px rgba(0,0,0,0.06); }
        .tile { padding:16px; border-radius:14px; background:#fff; border:1px solid rgba(0,0,0,0.06); transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease; display:flex; flex-direction:column; height:100%; }
        .tile:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.12); border-color:var(--brand); }
        .btn-primary { background:var(--brand); color:#fff; padding:8px 14px; border-radius:10px; border:1px solid var(--brand-600); }
        .btn-primary:hover { background:var(--brand-600); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def render_landing():
    _inject_css()
    # Logo en cabecera eliminado para no duplicar con el de la barra lateral
    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0">📊 Revenue Dashboard</h2>
          <p style="margin:.25rem 0 0; opacity:.9">Elige un módulo para empezar. Puedes volver aquí en cualquier momento.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    # Tarjetas del menú
    tiles = [
        ("Consulta normal", "consulta", "KPIs, ocupación, ADR e ingresos por periodo."),
        ("Cuadro de mando PRO", "pro", "Análisis avanzado, pace y narrativa ejecutiva."),
        ("What-if", "whatif", "Simula precio y pickup por grupos/propiedades."),
        ("Evolución (cut-off)", "evolucion", "Comparativa por fecha de corte."),
        ("Resumen comparativo", "resumen", "Comparativa por alojamiento: ADR, ocupación e ingresos."),
        ("KPIs por meses", "kpis_por_meses", "Resumen mensual: Noches, ADR, RevPAR e Ingresos."),
        ("Reservas por día", "reservas_por_dia", "Reservas recibidas por fecha de alta con comparación LY y LY-2."),
        ("Informe de propietario", "informe_propietario", "KPIs, comparación LY, gráfica y comentarios; exportable a PDF."),
    ]

    clicks = {}
    for row in _chunks(tiles, 4):  # ← máx. 4 por fila
        cols = st.columns(4, gap="large")
        for i in range(4):
            with cols[i]:
                if i >= len(row):
                    st.empty()
                    continue
                title, key, desc = row[i]
                with st.container():
                    st.markdown(f"<div class='tile'><div style='min-height:120px'><b>{title}</b><br><span style='color:#556'>{desc}</span></div></div>", unsafe_allow_html=True)
                    if st.button("Entrar", key=f"go_{key}", use_container_width=True):
                        clicks[key] = True
        st.write("")  # separación entre filas

    # Navegación
    if clicks.get("consulta"):
        st.session_state.view = "consulta"; st.rerun()
    elif clicks.get("pro"):
        st.session_state.view = "pro"; st.rerun()
    elif clicks.get("whatif"):
        st.session_state.view = "whatif"; st.rerun()
    elif clicks.get("evolucion"):
        st.session_state.view = "evolucion"; st.rerun()
    elif clicks.get("resumen"):
        st.session_state.view = "resumen"; st.rerun()
    elif clicks.get("kpis_por_meses"):
        st.session_state.view = "kpis_por_meses"; st.rerun()
    elif clicks.get("reservas_por_dia"):
        st.session_state.view = "reservas_por_dia"; st.rerun()
    elif clicks.get("informe_propietario"):
        st.session_state.view = "informe_propietario"; st.rerun()