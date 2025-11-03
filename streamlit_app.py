import streamlit as st
st.set_page_config(page_title="Revenue Dashboard", layout="wide", initial_sidebar_state="expanded")

from pathlib import Path
import pandas as pd
from auth import require_login, logout_button
from landing_ui import render_landing, get_logo_path

# --- helpers ---
def _get_raw():
    # intenta devolver el DF cargado en la sesión
    return st.session_state.get("df_active") or st.session_state.get("raw")

def _user_id() -> str | None:
    u = st.session_state.get("auth_user") or {}
    return u.get("username")

def _user_data_dir() -> Path:
    return Path(__file__).resolve().parent / "_user_data"

def _user_paths() -> tuple[Path, Path] | tuple[None, None]:
    uid = _user_id()
    if not uid:
        return None, None
    d = _user_data_dir(); d.mkdir(parents=True, exist_ok=True)
    return d / f"{uid}.parquet", d / f"{uid}.csv"

def _persist_user_dataset(df: pd.DataFrame):
    p_parq, p_csv = _user_paths()
    if not p_parq:
        return
    try:
        df.to_parquet(p_parq, index=False)
    except Exception:
        df.to_csv(p_csv, index=False, encoding="utf-8")

def _restore_user_dataset() -> bool:
    p_parq, p_csv = _user_paths()
    if not p_parq:
        return False
    try:
        if p_parq.exists():
            st.session_state.raw = pd.read_parquet(p_parq)
            return True
        if p_csv.exists():
            st.session_state.raw = pd.read_csv(p_csv)
            return True
    except Exception:
        pass
    return False

def _delete_user_dataset():
    p_parq, p_csv = _user_paths()
    for p in (p_parq, p_csv):
        try:
            if p and p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            pass

def _rerun():
    try:
        st.rerun()
    except Exception:
        pass

def clear_data(reset_view: bool = True):
    for k in ["raw", "df_active", "_last_file_sig", "uploader_has_file"]:
        st.session_state.pop(k, None)
    _delete_user_dataset()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    if reset_view:
        st.session_state.view = "landing"
        _rerun()

def _safe_call(view_fn):
    raw = _get_raw()
    if raw is None or (hasattr(raw, "empty") and getattr(raw, "empty", False)):
        st.info("Sube un archivo en la barra lateral y luego pulsa el botón Generar en el módulo.")
        return
    try:
        return view_fn(raw)            # ← pasa el DF al módulo
    except TypeError:
        return view_fn()
    except Exception as e:
        st.error("Ocurrió un error en este módulo.")
        with st.expander("Detalles del error"):
            st.exception(e)
        return

# --- login ---
if not require_login():
    st.stop()

# Restaura automáticamente el último archivo del usuario
if _get_raw() is None and _restore_user_dataset():
    st.toast("Datos restaurados de la sesión anterior.", icon="💾")

# --- SIDEBAR ---
with st.sidebar:
    logo = get_logo_path()
    if logo:
        st.image(logo, width=140)

    if st.button("⬅️ Volver al inicio", key="btn_home", use_container_width=True):
        st.session_state.view = "landing"; _rerun()
    logout_button()  # borra también el dataset persistido

    st.header("Carga de datos")
    st.checkbox("Borrar datos al quitar el archivo", value=True, key="auto_clear_on_remove")

    uploaded_file = st.file_uploader("Carga tu archivo Excel/CSV", type=["xlsx", "csv"], key="uploader")

    prev_has = st.session_state.get("uploader_has_file", False)
    curr_has = uploaded_file is not None
    st.session_state.uploader_has_file = curr_has

    if curr_has:
        try:
            size = getattr(uploaded_file, "size", None) or uploaded_file.getbuffer().nbytes
        except Exception:
            size = None
        sig = (uploaded_file.name, size)

        if st.session_state.get("_last_file_sig") != sig:
            try:
                uploaded_file.seek(0)
                if uploaded_file.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file, engine="openpyxl")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding="latin-1")
                st.session_state.raw = df
                _persist_user_dataset(df)  # ← guarda en disco por usuario
                st.session_state._last_file_sig = sig
                st.toast(f"Archivo cargado: {df.shape[0]:,} filas · {df.shape[1]} columnas", icon="✅")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    if prev_has and not curr_has and st.session_state.get("auto_clear_on_remove", True) and _get_raw() is not None:
        clear_data(reset_view=False)
        st.success("Datos borrados al quitar el archivo.")

    if _get_raw() is not None:
        if st.button("🧹 Quitar datos cargados", key="btn_clear_data", use_container_width=True):
            clear_data()

# --- Router ---
VALID_VIEWS = {
    "landing","consulta","pro","whatif","evolucion","resumen",
    "kpis_por_meses","reservas_por_dia","informe_propietario"   # ← añadido
}
if st.session_state.get("view") not in VALID_VIEWS:
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
elif st.session_state.view == "resumen":
    from resumen_comparativo import render_resumen_comparativo
    _safe_call(render_resumen_comparativo)
elif st.session_state.view == "kpis_por_meses":
    from kpis_por_meses import render_kpis_por_meses
    _safe_call(render_kpis_por_meses)
elif st.session_state.view == "reservas_por_dia":
    from reservas_por_dia import render_reservas_por_dia
    _safe_call(render_reservas_por_dia)
elif st.session_state.view == "informe_propietario":  # ← nuevo
    from informe_propietario import render_informe_propietario
    _safe_call(render_informe_propietario)

def render_landing():
    st.title("Bienvenido a la herramienta de análisis de datos")
    st.write("Esta herramienta te permite analizar y visualizar tus datos de manera interactiva.")

    # --- Sección de carga de datos ---
    st.header("Carga de datos")
    st.write("Para comenzar, carga tu archivo de datos en formato Excel o CSV.")

    # Botón para restablecer la vista
    if st.button("↻ Restablecer vista", key="btn_reset_view", help="Restablece la vista actual"):
        clear_data()

    # Muestra un mensaje si no hay datos cargados
    if _get_raw() is None:
        st.info("No hay datos cargados. Por favor, sube un archivo en la barra lateral.")
    else:
        df = _get_raw()
        st.success(f"Datos cargados: {df.shape[0]:,} filas · {df.shape[1]} columnas")

    # --- Ejemplo de datos ---
    st.subheader("Ejemplo de datos")
    raw = _get_raw()
    if raw is not None and not raw.empty:
        st.write("Aquí tienes una vista previa de tus datos:")
        st.dataframe(raw.head(10))
    else:
        st.write("No hay datos disponibles para mostrar.")

    # --- Navegación a módulos ---
    st.header("Análisis y visualización")
    clicks = {}
    cols = st.columns(3)
    with cols[0]:
        clicks["consulta"] = st.button("Consulta normal", help="Consulta y analiza tus datos.")
    with cols[1]:
        clicks["pro"] = st.button("Cuadro de mando PRO", help="Análisis avanzado con gráficos.")
    with cols[2]:
        clicks["whatif"] = st.button("Análisis What-If", help="Simula escenarios hipotéticos.")

    # Navegación:
    if clicks.get("consulta"):
        st.session_state.view = "consulta"; _rerun()
    elif clicks.get("pro"):
        st.session_state.view = "pro"; _rerun()
    elif clicks.get("whatif"):
        st.session_state.view = "whatif"; _rerun()
    elif clicks.get("informe_propietario"):             # ← nuevo
        st.session_state.view = "informe_propietario"; _rerun()

    # --- Información adicional ---
    st.subheader("Información adicional")
    st.write("Utiliza los módulos disponibles para realizar un análisis detallado de tus datos.")
    st.write("Para más información, consulta la documentación de la herramienta.")
