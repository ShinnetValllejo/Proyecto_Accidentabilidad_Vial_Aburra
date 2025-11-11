import streamlit as st
from pathlib import Path

# ======================================================
# CONFIGURACIÓN DE RUTAS BASE
# ======================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
PAGES_DIR = APP_DIR / "Pages"
STYLE_DIR = PAGES_DIR / "Style"

# Importar vistas
from APP_FRONT.Pages.Portada import mostrar_portada
from APP_FRONT.Pages.IndicadoresGenerales import mostrar_indicadores

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
st.set_page_config(
    page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
    layout="wide",
    page_icon="🚧"
)

# Ocultar la navegación lateral de Streamlit
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# FUNCIÓN DE CARGA DE CSS
# ======================================================
def load_css(file_name: str):
    css_path = STYLE_DIR / file_name
    if css_path.is_file():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"No se encontró el archivo CSS: {css_path}")

# ======================================================
# CONTROL DE PÁGINAS
# ======================================================
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"

if st.session_state["pagina"] == "inicio":
    load_css("style.css")
    mostrar_portada()

elif st.session_state["pagina"] == "indicadores":
    load_css("indicadores.css")
    mostrar_indicadores()

else:
    st.write("Vista en desarrollo.")
