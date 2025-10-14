import requests
import streamlit as st
from streamlit_lottie import st_lottie

LOGO_PATH = "assets/images/florit-flats-logo.png"  # ← tu logo

LOTTIE = {
    "consulta": "https://assets7.lottiefiles.com/packages/lf20_5ngs2ksb.json",
    "pro": "https://assets7.lottiefiles.com/packages/lf20_x62chJ.json",
    "whatif": "https://assets9.lottiefiles.com/packages/lf20_9Zoynq.json",
}

def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

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
    ]
    clicks = {}

    for col, (title, key, desc) in zip(cols, tiles):
        with col:
            st.markdown('<div class="tile">', unsafe_allow_html=True)
            anim = _load_lottie(LOTTIE[key])
            st_lottie(anim, height=140, key=f"lot_{key}")
            st.markdown(f"**{title}**")
            st.caption(desc)
            clicks[key] = st.button("Entrar", key=f"btn_{key}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if clicks.get("consulta"):
        st.session_state.view = "consulta"
        st.experimental_rerun()
    if clicks.get("pro"):
        st.session_state.view = "pro"
        st.experimental_rerun()
    if clicks.get("whatif"):
        st.session_state.view = "whatif"
        st.experimental_rerun()