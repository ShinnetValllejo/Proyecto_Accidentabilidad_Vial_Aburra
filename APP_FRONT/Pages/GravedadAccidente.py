# ==========================================================
# MÓDULO: GRAVEDAD ACCIDENTE
# ==========================================================

import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from sqlalchemy import create_engine
from APP_FRONT.nav import generar_barra_navegacion

# ==========================================================
# FUNCIONES DE UTILIDAD
# ==========================================================

def find_project_root(marker: str = '.project_root') -> Path:
    """Busca la raíz del proyecto a partir de un marcador."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"No se encontró el marcador '{marker}' en el árbol de carpetas.")

def get_base64_of_image(img_path: Path) -> str:
    """Convierte una imagen a base64 para su uso como fondo."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_image_data_uri(img_path: Path) -> str:
    """Convierte una imagen a data URI con el MIME type correcto."""
    with open(img_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode()
    ext = img_path.suffix.lower()
    mime_map = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png'}
    mime = mime_map.get(ext, 'jpeg')
    return f"data:image/{mime};base64,{base64_data}"

def load_css(file_name: str):
    """Carga y aplica un archivo CSS externo."""
    css_path = STYLE_DIR / file_name
    if css_path.is_file():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"No se encontró el archivo CSS: {file_name}")

# ==========================================================
# CONFIGURACIÓN DE RUTAS
# ==========================================================
PROJECT_ROOT = find_project_root()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
STYLE_DIR = APP_DIR / "Pages" / "Style"
IMG_PATH = APP_DIR / "Static" / "FondoVistas.png"
DB_PATH = PROJECT_ROOT / "DATASETS" / "Destino" / "Proyecto_Accidentalidad_Vial_Antioquia.db"
GRAFICAS_DIR = PROJECT_ROOT / "ETL_MODULES" / "Transform" / "Graficas_Salida"

# ==========================================================
# CARGA DE DATOS
# ==========================================================
@st.cache_data
def load_data():
    """Carga los datos de accidentalidad desde la base de datos."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM Accidentalidad_Vial_Antioquia", engine)
    df.columns = [col.strip().upper() for col in df.columns]
    return df

df = load_data()

# ==========================================================
# FUNCIÓN PRINCIPAL DE LA VISTA
# ==========================================================
def mostrar_gravedad():
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
        st.warning(f"No se encontró la imagen de fondo: {IMG_PATH}")

    # Estilos CSS
    load_css("gravedad.css")

    # Espaciado superior mínimo
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ==========================================================
    # BARRA DE NAVEGACIÓN
    # ==========================================================
    generar_barra_navegacion()

    # ==========================================================
    # TÍTULOS
    # ==========================================================
    st.markdown(
        """
        <div class="title-block">
            <h1 class="dashboard-title">🚑 Gravedad Accidente</h1>
            <h2 class="dashboard-subtitle">
                Accidentabilidad vial en el Valle de Aburrá (2015–2019)
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================
    # VISUALIZACIÓN DE GRÁFICAS
    # ==========================================================
    
    # CSS para forzar la misma altura en las gráficas
    st.markdown("""
    <style>
    .uniform-height {
        height: 500px !important;
        object-fit: contain !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ✅ PRIMER PAR: Accidentes por Comuna y Jornada
    st.markdown("### Distribución de Accidentes por Comuna y Jornada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        accidentes_comuna_path = GRAFICAS_DIR / "Accidentes_Comuna.jpg"
        if accidentes_comuna_path.exists():
            st.markdown(
                f'<img src="{get_image_data_uri(accidentes_comuna_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
        else:
            st.warning("No se encontró: Accidentes_Comuna.jpg")
    
    with col2:
        accidentes_jornada_path = GRAFICAS_DIR / "Accidentes_Jornada.jpg"
        if accidentes_jornada_path.exists():
            st.markdown(
                f'<img src="{get_image_data_uri(accidentes_jornada_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
        else:
            st.warning("No se encontró: Accidentes_Jornada.jpg")
    
    # Espacio entre secciones
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    
    # ✅ SEGUNDO PAR: Gravedad por Comuna y Hora/Día
    st.markdown("### Análisis de Gravedad y Distribución Temporal")
    
    col3, col4 = st.columns(2)
    
    with col3:
        gravedad_comuna_path = GRAFICAS_DIR / "GRAVEDAD_COMUNA.jpg"
        if gravedad_comuna_path.exists():
            st.markdown(
                f'<img src="{get_image_data_uri(gravedad_comuna_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
        else:
            st.warning("No se encontró: GRAVEDAD_COMUNA.jpg")
    
    with col4:
        hora_diasemana_path = GRAFICAS_DIR / "HORA_DÍASEMANA.jpg"
        if hora_diasemana_path.exists():
            st.markdown(
                f'<img src="{get_image_data_uri(hora_diasemana_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
        else:
            st.warning("No se encontró: HORA_DÍASEMANA.jpg")


