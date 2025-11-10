from pathlib import Path 
import streamlit as st
import base64

# Ruta de imagen desde raíz del proyecto
BASE_DIR = Path(__file__).parents[1]
IMG_PATH = BASE_DIR / "Static" / "Map_portada.jpeg"

def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def mostrar_portada():
    # --- INICIO DE LA MODIFICACIÓN ---
    # Este CSS se aplica SOLO a la portada y oculta el panel lateral COMPLETO
    # (incluido el botón para abrirlo).
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
    # --- FIN DE LA MODIFICACIÓN ---

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