import os, json, requests
import streamlit as st
from streamlit_lottie import st_lottie

LOGO_PATH = "assets/images/florit-flats-logo.png"  # ← tu logo

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
        .hero {
            padding: 18px 26px;
            border-radius: 14px;
            background: linear-gradient(135deg, #ffffff 0%, #f6f8fb 100%);
            color: #1f2937;
            border: 1px solid rgba(0,0,0,0.04);
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        .tile {
            padding: 16px;
            border-radius: 14px;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.06);
            transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
        }
        .tile:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            border-color: var(--brand);
        }
        .btn-primary {
            background: var(--brand);
            color: #fff;
            padding: 8px 14px;
            border-radius: 10px;
            border: 1px solid var(--brand-600);
        }
        .btn-primary:hover { background: var(--brand-600); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _tile_media(title: str, key: str):
    if USE_ANIMATIONS and st_lottie:
        anim = _load_lottie(LOTTIE.get(key))
        if isinstance(anim, dict):
            st_lottie(anim, height=140, key=f"lot_{key}")
            return
    # Placeholder estático (sin animación)
    st.markdown(
        f'''
        <div style="height:140px;display:flex;align-items:center;justify-content:center;
                    border-radius:12px;background:#f3f6f9;color:#2e485f;border:1px solid #e5e7eb;">
            <span style="font-weight:600;">{title}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )

# Helper de rerun compatible
def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def render_landing():
    _inject_css()

    # Logo arriba del landing
    try:
        st.image(LOGO_PATH, width=160)
    except Exception:
        st.caption("Logo no disponible (revisa la ruta LOGO_PATH).")

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
    cols = st.columns(3, gap="large")
    tiles = [
        ("Consulta normal", "consulta", "KPIs, ocupación, ADR e ingresos por periodo."),
        ("Cuadro de mando PRO", "pro", "Análisis avanzado, pace y narrativa ejecutiva."),
        ("What‑if", "whatif", "Simula precio y pickup por grupos/propiedades."),
        ("Evolución por fecha de corte", "evolucion", "Cómo evolucionan KPIs al mover la fecha de corte."),
    ]
    clicks = {}

    for col, (title, key, desc) in zip(cols, tiles):
        with col:
            st.markdown('<div class="tile">', unsafe_allow_html=True)
            _tile_media(title, key)  # ← reemplaza st_lottie(...)
            st.markdown(f"**{title}**")
            st.caption(desc)
            clicks[key] = st.button("Entrar", key=f"btn_{key}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if clicks.get("consulta"):
        st.session_state.view = "consulta"
        _rerun()
    if clicks.get("pro"):
        st.session_state.view = "pro"
        _rerun()
    if clicks.get("whatif"):
        st.session_state.view = "whatif"
        _rerun()
    if clicks.get("evolucion"):
        st.session_state.view = "evolucion"
        _rerun()