import streamlit as st
from pathlib import Path

# Importar vistas
from Pages.Portada import mostrar_portada
from Pages.IndicadoresGenerales import mostrar_indicadores

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
st.set_page_config(
    page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
    layout="wide",
    page_icon="🚧"
)

# --- INICIO DE LA MODIFICACIÓN ---
# Este CSS se aplica a TODAS las vistas y oculta la navegación automática
# que Streamlit crea en el panel lateral.
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
# --- FIN DE LA MODIFICACIÓN ---

# ======================================================
# FUNCIONES DE CARGA CSS
# ======================================================
BASE_DIR = Path(__file__).parent.resolve()

def load_css(file_name: str):
    css_path = BASE_DIR / "Pages" / "Style" / file_name
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