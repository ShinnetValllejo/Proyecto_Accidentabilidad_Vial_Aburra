from pathlib import Path
import streamlit as st
import base64

# ==========================================================
# CONFIGURACIÓN DE RUTA BASE DEL PROYECTO
# ==========================================================
def find_project_root(marker: str = '.project_root') -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(
        f"No se pudo encontrar la raíz del proyecto. "
        f"Asegúrate de que existe un archivo llamado '{marker}' en la carpeta raíz."
    )

PROJECT_ROOT = find_project_root()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
IMG_PATH = APP_DIR / "Static" / "Map_portada.jpeg"

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ==========================================================
# FUNCIÓN PRINCIPAL DE LA PORTADA
# ==========================================================
def mostrar_portada():
    # Ocultar barra lateral
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Fondo con imagen
    if IMG_PATH.exists():
        bg_base64 = get_base64_of_image(IMG_PATH)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bg_base64}");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"No se encontró la imagen de portada: {IMG_PATH}")

    # Contenido principal
    st.markdown(
        """
        <div class="content-container">
            <div class="title">
                Análisis de datos y modelado predictivo<br>
                sobre la accidentabilidad vial<br>
                en el Valle de Aburrá<br>
                (2015–2019)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Botones de navegación
    st.markdown("<div class='button-area'>", unsafe_allow_html=True)
    col_vacio, col1, col2, col3, col4 = st.columns([0.3, 1, 1, 1, 1])
    with col1:
        if st.button("🛣️ Indicadores Generales", use_container_width=True):
            st.session_state["pagina"] = "indicadores"
            st.rerun()
    with col2:
        if st.button("🚑 Gravedad Accidente", use_container_width=True):
            st.session_state["pagina"] = "modelo"
            st.rerun()
    with col3:
        if st.button("🔬 Modelo de Clasificación", use_container_width=True):
            st.session_state["pagina"] = "curvas"
            st.rerun()
    with col4:
        if st.button("🔮 Modelo Predictivo", use_container_width=True):
            st.session_state["pagina"] = "resumen"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
