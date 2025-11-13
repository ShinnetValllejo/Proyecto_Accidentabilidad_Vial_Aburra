# ==========================================================
# MÓDULO: PREDICCION
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

def format_percentage(value):
    """Formatea valores numéricos como porcentaje."""
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    elif isinstance(value, str) and '%' in value:
        return value
    else:
        return str(value)

def style_risk_level(val):
    """Aplica estilos CSS a los niveles de riesgo."""
    if val == "ALTO":
        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
    elif val == "MEDIO":
        return 'background-color: #ffa500; color: white; font-weight: bold;'
    elif val == "BAJO":
        return 'background-color: #4CAF50; color: white; font-weight: bold;'
    return ''

def style_confidence_level(val):
    """Aplica estilos CSS a los niveles de confianza."""
    if val == "ALTA":
        return 'background-color: #4CAF50; color: white; font-weight: bold;'
    elif val == "MEDIA":
        return 'background-color: #ffa500; color: white; font-weight: bold;'
    elif val == "BAJA":
        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
    return ''

def style_prediction(val):
    """Aplica estilos CSS a las predicciones."""
    if val == "CON HERIDOS":
        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
    elif val == "SOLO DAÑOS":
        return 'background-color: #4CAF50; color: white; font-weight: bold;'
    return ''

# ==========================================================
# CONFIGURACIÓN DE RUTAS
# ==========================================================
PROJECT_ROOT = find_project_root()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
STYLE_DIR = APP_DIR / "Pages" / "Style"
IMG_PATH = APP_DIR / "Static" / "FondoVistas.png"
DB_PATH = PROJECT_ROOT / "DATASETS" / "Destino" / "Proyecto_Accidentalidad_Vial_Antioquia.db"
PREDICCIONES_CSV_PATH = PROJECT_ROOT / "ETL_MODULES" / "Transform" / "Modelo_Predict" / "Predicciones_Random_Forest.csv"

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

@st.cache_data
def load_predictions():
    """Carga las predicciones del modelo Random Forest."""
    if PREDICCIONES_CSV_PATH.exists():
        try:
            df = pd.read_csv(PREDICCIONES_CSV_PATH, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(PREDICCIONES_CSV_PATH, encoding='latin-1')
        
        df.columns = [col.strip().upper() for col in df.columns]
        return df
    else:
        st.error(f"Archivo de predicciones no encontrado: {PREDICCIONES_CSV_PATH}")
        return pd.DataFrame()
    
df = load_data()
df_predicciones = load_predictions()

# ==========================================================
# FUNCIÓN PRINCIPAL DE LA VISTA
# ==========================================================
def mostrar_prediccion():
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

    load_css("prediccion.css")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    generar_barra_navegacion()

    st.markdown(
        """
        <div class="title-block">
            <h1 class="dashboard-title">🔮 Modelo Predictivo</h1>
            <h2 class="dashboard-subtitle">
                Accidentabilidad vial en el Valle de Aburrá (2015–2019)
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if not df_predicciones.empty:
        st.markdown("### 📊 Resumen de Predicciones")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predicciones", len(df_predicciones))
        with col2:
            st.metric("Alto Riesgo", len(df_predicciones[df_predicciones['RIESGO'] == 'ALTO']))
        with col3:
            st.metric("Con Heridos", len(df_predicciones[df_predicciones['PREDICCION'] == 'CON HERIDOS']))
        with col4:
            st.metric("Alta Confianza", len(df_predicciones[df_predicciones['CONFIANZA'] == 'ALTA']))
        
        st.markdown("### 🔍 Filtros de Predicciones")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            municipios = ['TODOS'] + sorted(df_predicciones['MUNICIPIO'].dropna().unique().tolist())
            municipio_sel = st.selectbox("Municipio", municipios)
        with col_f2:
            comunas = ['TODOS'] + sorted(df_predicciones['COMUNA'].dropna().unique().tolist())
            comuna_sel = st.selectbox("Comuna", comunas)
        with col_f3:
            riesgos = ['TODOS'] + sorted(df_predicciones['RIESGO'].dropna().unique().tolist())
            riesgo_sel = st.selectbox("Nivel de Riesgo", riesgos)
        with col_f4:
            confianzas = ['TODOS'] + sorted(df_predicciones['CONFIANZA'].dropna().unique().tolist())
            confianza_sel = st.selectbox("Nivel de Confianza", confianzas)
        
        col_f5, col_f6, col_f7 = st.columns(3)
        with col_f5:
            jornadas = ['TODOS'] + sorted(df_predicciones['JORNADA'].dropna().unique().tolist())
            jornada_sel = st.selectbox("Jornada", jornadas)
        with col_f6:
            clases = ['TODOS'] + sorted(df_predicciones['CLASE'].dropna().unique().tolist())
            clase_sel = st.selectbox("Clase de Accidente", clases)
        with col_f7:
            predicciones = ['TODOS'] + sorted(df_predicciones['PREDICCION'].dropna().unique().tolist())
            prediccion_sel = st.selectbox("Tipo de Predicción", predicciones)

        df_filtrado = df_predicciones.copy()
        if municipio_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['MUNICIPIO'] == municipio_sel]
        if comuna_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['COMUNA'] == comuna_sel]
        if riesgo_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['RIESGO'] == riesgo_sel]
        if confianza_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['CONFIANZA'] == confianza_sel]
        if jornada_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['JORNADA'] == jornada_sel]
        if clase_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['CLASE'] == clase_sel]
        if prediccion_sel != 'TODOS':
            df_filtrado = df_filtrado[df_filtrado['PREDICCION'] == prediccion_sel]

        st.markdown(f"### 📋 Detalle de Predicciones ({len(df_filtrado)} registros)")
        
        columnas_disponibles = [
            'MUNICIPIO', 'COMUNA', 'CLASE', 'JORNADA', 'PREDICCION', 
            'PROBABILIDAD_HERIDOS', 'CONFIANZA', 'RIESGO', 'NUM_HORA',
            'FIN_DE_SEMANA', 'HORA_PICO', 'HORA_NOCTURNA'
        ]
        
        columnas_seleccionadas = st.multiselect(
            "Seleccionar columnas a mostrar:",
            options=columnas_disponibles,
            default=['MUNICIPIO', 'COMUNA', 'JORNADA', 'PREDICCION', 'PROBABILIDAD_HERIDOS', 'CONFIANZA', 'RIESGO']
        )
        
        if columnas_seleccionadas:
            df_display = df_filtrado[columnas_seleccionadas].copy()
            
            styled_df = df_display.style\
                .map(style_risk_level, subset=['RIESGO'] if 'RIESGO' in df_display.columns else [])\
                .map(style_confidence_level, subset=['CONFIANZA'] if 'CONFIANZA' in df_display.columns else [])\
                .map(style_prediction, subset=['PREDICCION'] if 'PREDICCION' in df_display.columns else [])
            
            # ==========================================================
            # INICIO DEL CAMBIO: CÁLCULO DE ALTURA DINÁMICA
            # ==========================================================
            
            # Calcula la altura para un máximo de 6 filas de datos + la fila de encabezado.
            # Cada fila tiene una altura aproximada de 35px.
            # (min(len(df_display), 6) + 1) calcula el número de filas a mostrar.
            table_height = (min(len(df_display), 6) + 1) * 35 + 3

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=table_height  # Se usa la nueva altura dinámica en lugar de 400
            )
            # ==========================================================
            # FIN DEL CAMBIO
            # ==========================================================
            
            st.markdown("### 📥 Exportar Datos")
            csv = df_filtrado.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="Descargar predicciones filtradas (CSV)",
                data=csv,
                file_name="predicciones_filtradas.csv",
                mime="text/csv"
            )
        else:
            st.warning("Por favor selecciona al menos una columna para mostrar.")
        
    else:
        st.warning("No se encontraron datos de predicciones para mostrar.")