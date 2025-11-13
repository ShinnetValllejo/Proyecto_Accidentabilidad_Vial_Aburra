
# ==========================================================
# MÓDULO: CLASIFICACION
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
def mostrar_clasificacion():
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
    load_css("clasificacion.css")

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
            <h1 class="dashboard-title">🔬 Modelo de Clasificación</h1>
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
    
    # Matrices de confusión lado a lado
    st.markdown("### Matrices de Confusión")
    col1, col2 = st.columns(2)
    
    with col1:
        matriz_arbol_path = GRAFICAS_DIR / "Matriz_Confusion_Arbol_Decision.jpg"
        if matriz_arbol_path.exists():
            st.image(str(matriz_arbol_path), use_container_width=True, caption="Árbol de Decisión")
        else:
            st.warning("Matriz de Confusión - Árbol de Decisión no encontrada")
    
    with col2:
        matriz_rf_path = GRAFICAS_DIR / "Matriz_Confusion_RF.jpg"
        if matriz_rf_path.exists():
            st.image(str(matriz_rf_path), use_container_width=True, caption="Random Forest")
        else:
            st.warning("Matriz de Confusión - Random Forest no encontrada")
    
    # Comparación de modelos y curva ROC lado a lado con altura fija
    st.markdown("### Métricas de Comparación")
    col3, col4 = st.columns(2)
    
    # CSS para forzar la misma altura en ambas gráficas
    st.markdown("""
    <style>
    .uniform-height {
        height: 500px !important;
        object-fit: contain !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col3:
        comparacion_path = GRAFICAS_DIR / "Comparacion_Modelos.jpg"
        if comparacion_path.exists():
            # Usar markdown con CSS personalizado para controlar el tamaño
            st.markdown(
                f'<img src="data:image/jpg;base64,{get_base64_of_image(comparacion_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
            st.markdown("<p style='text-align: center; margin-top: 10px;'>Comparación de Modelos</p>", unsafe_allow_html=True)
        else:
            st.warning("Comparación de Modelos no encontrada")
    
    with col4:
        roc_path = GRAFICAS_DIR / "Curva_ROC_Comparativa.jpg"
        if roc_path.exists():
            # Usar markdown con CSS personalizado para controlar el tamaño
            st.markdown(
                f'<img src="data:image/jpg;base64,{get_base64_of_image(roc_path)}" class="uniform-height" style="width:100%">',
                unsafe_allow_html=True
            )
            st.markdown("<p style='text-align: center; margin-top: 10px;'>Curva ROC Comparativa</p>", unsafe_allow_html=True)
        else:
            st.warning("Curva ROC Comparativa no encontrada")