# ======================================================
# DASHBOARD ACCIDENTALIDAD VIAL — APP PRINCIPAL
# ======================================================

import streamlit as st
from pathlib import Path

# ======================================================
# CONFIGURACIÓN DE RUTAS BASE
# ======================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
PAGES_DIR = APP_DIR / "Pages"
STYLE_DIR = PAGES_DIR / "Style"

# ======================================================
# IMPORTACIÓN DE VISTAS
# ======================================================
from APP_FRONT.Pages.Portada import mostrar_portada
from APP_FRONT.Pages.IndicadoresGenerales import mostrar_indicadores
from APP_FRONT.Pages.GravedadAccidente import mostrar_gravedad
from APP_FRONT.Pages.Clasificacion import mostrar_clasificacion
from APP_FRONT.Pages.Prediccion import mostrar_prediccion

# ======================================================
# CONFIGURACIÓN INICIAL DE LA APLICACIÓN
# ======================================================
st.set_page_config(
    page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
    page_icon="🚧",
    layout="wide"
)

# ======================================================
# FUNCIÓN DE CARGA DE CSS GLOBAL / ESPECÍFICO
# ======================================================
def load_css(file_name: str):
    """Carga un archivo CSS desde la carpeta Style."""
    css_path = STYLE_DIR / file_name
    if css_path.is_file():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"No se encontró el archivo CSS: {file_name}")

# ======================================================
# CONTROLADOR PRINCIPAL DE PÁGINAS
# ======================================================
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"

pagina_actual = st.session_state["pagina"]

# ======================================================
# ENRUTAMIENTO DE VISTAS
# ======================================================
if pagina_actual == "inicio":
    load_css("style.css")
    mostrar_portada()

elif pagina_actual == "indicadores":
    load_css("indicadores.css")
    mostrar_indicadores()

elif pagina_actual == "gravedad":
    load_css("gravedad.css")
    mostrar_gravedad()

elif pagina_actual == "clasificacion":
    load_css("clasificacion.css")
    mostrar_clasificacion()

elif pagina_actual == "predictivo":
    load_css("prediccion.css")
    mostrar_prediccion()

else:
    st.error("❌ Página no encontrada. Verifica la variable 'st.session_state[\"pagina\"]'.")

# ======================================================
# FIN DE LA APLICACIÓN
# ======================================================
