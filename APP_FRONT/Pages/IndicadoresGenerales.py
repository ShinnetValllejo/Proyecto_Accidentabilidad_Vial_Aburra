# ==========================================================
# MÓDULO: INDICADORES GENERALES
# ==========================================================

import streamlit as st
import pandas as pd
import base64
from PIL import Image
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
def mostrar_indicadores():
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
    load_css("indicadores.css")

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
            <h1 class="dashboard-title">🛣️ Indicadores Generales</h1>
            <h2 class="dashboard-subtitle">
                Accidentabilidad vial en el Valle de Aburrá (2015–2019)
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================
    # CÁLCULO DE TASAS (PORCENTAJES)
    # ==========================================================
    # Total de accidentes
    total_accidentes = len(df)
    
    # Contar por gravedad
    totales = df.groupby("GRAVEDAD_ACCIDENTE").size().to_dict()
    
    # Calcular tasas (porcentajes)
    muertos_count = totales.get("MUERTOS", 0)
    heridos_count = totales.get("HERIDOS", 0)
    danos_count = totales.get("DAÑOS", 0)
    
    tasa_muertos = (muertos_count / total_accidentes * 100) if total_accidentes > 0 else 0
    tasa_heridos = (heridos_count / total_accidentes * 100) if total_accidentes > 0 else 0
    tasa_danos = (danos_count / total_accidentes * 100) if total_accidentes > 0 else 0

    # ==========================================================
    # TARJETAS DE MÉTRICAS
    # ==========================================================
    col1, col2, col3, col4 = st.columns(4, gap="small")

    tarjetas = [
        ("#FF6B6B", "Tasa Muertos", f"{tasa_muertos:.2f}%", f"({muertos_count:,} casos)"),
        ("#FFD93D", "Tasa Heridos", f"{tasa_heridos:.2f}%", f"({heridos_count:,} casos)"),
        ("#6BCB77", "Tasa Daños", f"{tasa_danos:.2f}%", f"({danos_count:,} casos)"),
        ("#4D96FF", "Total Accidentes", f"{total_accidentes:,}", "")
    ]

    for col, (color, title, value, subtitle) in zip([col1, col2, col3, col4], tarjetas):
        col.markdown(
            f"""
            <div class='metric-card' style='background-color:{color};'>
                <div class='metric-title'>{title}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-subtitle'>{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Espaciador
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ==========================================================
    # VISUALIZACIÓN DE GRÁFICAS
    # ==========================================================
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='graph-title'>Evolución Temporal por Gravedad</h3>", unsafe_allow_html=True)
    _, col_centro, _ = st.columns([0.5, 2, 0.5])
    
    with col_centro:
        img_clase_path = GRAFICAS_DIR / "Evolucion_Temporal_Gravedad.jpg"
        if img_clase_path.exists():
            st.image(str(img_clase_path), use_container_width=True)
        else:
            st.warning(f"Gráfica no encontrada: {img_clase_path.name}")
    
    # Matrices de confusión lado a lado
    st.markdown("### Accidentes por Gravedad y Clase")
    col1, col2 = st.columns(2)
    
    with col1:
        accidentes_path = GRAFICAS_DIR / "Accidentes_Gravedad.jpg"
        if accidentes_path.exists():
            st.image(str(accidentes_path), use_container_width=True, caption="Accidentes por Gravedad del Siniestro")
        else:
            st.warning("Accidentes_Gravedad no encontrada")
    
    with col2:
        clase_path = GRAFICAS_DIR / "Accidentes_Clase.jpg"
        if clase_path.exists():
            st.image(str(clase_path), use_container_width=True, caption="Mayor Número de Accidentes por Clase")
        else:
            st.warning("Accidentes_Clase no encontrada")

    # Espaciador
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)