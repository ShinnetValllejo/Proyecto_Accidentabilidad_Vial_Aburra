# ==========================================================
# Indicadores Generales - Configuración y Carga de Datos
# ==========================================================
import streamlit as st
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

# ==========================================================
# FUNCIÓN PARA ENCONTRAR LA RAÍZ DEL PROYECTO
# ==========================================================
def find_project_root(marker: str = '.project_root') -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"No se encontró '{marker}' en el árbol de carpetas.")

# ==========================================================
# CONFIGURACIÓN DE RUTAS
# ==========================================================
PROJECT_ROOT = find_project_root()
APP_DIR = PROJECT_ROOT / "APP_FRONT"
STYLE_DIR = APP_DIR / "Pages" / "Style"
DB_PATH = PROJECT_ROOT / "DATASETS" / "Destino" / "Proyecto_Accidentalidad_Vial_Antioquia.db"

# ==========================================================
# CARGA DE DATOS
# ==========================================================
@st.cache_data
def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM Accidentalidad_Vial_Antioquia", engine)
    df.columns = [col.strip().upper() for col in df.columns]
    return df

df = load_data()

# ==========================================================
# CARGA DE ESTILOS
# ==========================================================
def load_css(file_name: str):
    css_path = STYLE_DIR / file_name
    if css_path.is_file():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"No se encontró el archivo CSS: {css_path}")

# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================
def mostrar_indicadores():
    load_css("indicadores.css")

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # Barra de navegación
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    col_home, col_indic, col_gravedad, col_clasif, col_pred = st.columns(5, gap="small")

    with col_home:
        if st.button("🏠 Home"):
            st.session_state["pagina"] = "inicio"
            st.rerun()
    with col_indic:
        st.button("🛣️ Indicadores Generales", disabled=True)
    with col_gravedad:
        if st.button("🚑 Gravedad Accidente"):
            st.session_state["pagina"] = "gravedad"
            st.rerun()
    with col_clasif:
        if st.button("🔬 Modelo de Clasificación"):
            st.session_state["pagina"] = "clasificacion"
            st.rerun()
    with col_pred:
        if st.button("🔮 Modelo Predictivo"):
            st.session_state["pagina"] = "predictivo"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='nav-separator'></div>", unsafe_allow_html=True)
    st.markdown("<h1 class='dashboard-title'>🛣️ Indicadores Generales — Accidentabilidad Vial</h1>", unsafe_allow_html=True)

    # Filtros
    with st.sidebar:
        st.header("Filtros")
        anios = sorted(df['AÑO'].dropna().unique())
        anio_sel = st.selectbox("AÑO", ["Todos"] + list(anios))

        municipios = sorted(df['MUNICIPIO'].dropna().unique())
        if anio_sel != "Todos":
            municipios = sorted(df[df['AÑO'] == anio_sel]['MUNICIPIO'].dropna().unique())
        municipio_sel = st.selectbox("MUNICIPIO", ["Todos"] + municipios)

        comunas = sorted(df['COMUNA'].dropna().unique())
        if municipio_sel != "Todos":
            comunas = sorted(df[df['MUNICIPIO'] == municipio_sel]['COMUNA'].dropna().unique())
        comuna_sel = st.selectbox("COMUNA", ["Todas"] + comunas)

        barrios = sorted(df['BARRIO'].dropna().unique())
        if comuna_sel != "Todas":
            barrios = sorted(df[df['COMUNA'] == comuna_sel]['BARRIO'].dropna().unique())
        barrio_sel = st.selectbox("BARRIO", ["Todos"] + barrios)

        clases = sorted(df['CLASE'].dropna().unique())
        if barrio_sel != "Todos":
            clases = sorted(df[df['BARRIO'] == barrio_sel]['CLASE'].dropna().unique())
        clase_sel = st.selectbox("CLASE", ["Todas"] + clases)

    # Filtrado
    df_filtrado = df.copy()
    if anio_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['AÑO'] == anio_sel]
    if municipio_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['MUNICIPIO'] == municipio_sel]
    if comuna_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado['COMUNA'] == comuna_sel]
    if barrio_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['BARRIO'] == barrio_sel]
    if clase_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado['CLASE'] == clase_sel]

    st.write(f"Registros filtrados: {len(df_filtrado)}")

    # Tarjetas resumen
    col1, col2, col3, col4 = st.columns(4, gap="small")
    totales = df_filtrado.groupby("GRAVEDAD_ACCIDENTE").size().to_dict()
    tasa_muertos = totales.get("MUERTOS", 0)
    tasa_heridos = totales.get("HERIDOS", 0)
    tasa_danos = totales.get("DAÑOS", 0)
    total_accidentes = len(df_filtrado)

    for col, color, title, value in zip(
        [col1, col2, col3, col4],
        ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"],
        ["Tasa Muertos", "Tasa Heridos", "Tasa Daños", "Total Accidentes"],
        [tasa_muertos, tasa_heridos, tasa_danos, total_accidentes]
    ):
        col.markdown(
            f"""
            <div class='metric-card' style='background-color:{color};'>
                <div class='metric-title'>{title}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True
        )

    # Gráfica
    if not df_filtrado.empty:
        st.markdown("<h2>📊 Evolución de Accidentes por Gravedad</h2>", unsafe_allow_html=True)
        pivot_df = df_filtrado.groupby(['AÑO', 'GRAVEDAD_ACCIDENTE']).size().reset_index(name='TOTAL')
        pivot_df = pivot_df.pivot(index='AÑO', columns='GRAVEDAD_ACCIDENTE', values='TOTAL').fillna(0)
        st.line_chart(pivot_df)
    else:
        st.warning("No hay registros para los filtros seleccionados.")
