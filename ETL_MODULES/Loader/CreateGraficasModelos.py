
# =============================================================================
# ANÁLISIS PREDICTIVO DE ACCIDENTALIDAD VIAL - VERSIÓN OPTIMIZADA
# =============================================================================

# =============================================================================
# BLOQUE 1: CONFIGURACIÓN GLOBAL
# =============================================================================

import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report, 
    precision_recall_curve, auc
)
from sklearn import tree

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

def find_project_root(marker: str = '.project_root') -> Path:
    """
    Busca la raíz del proyecto subiendo directorios hasta encontrar un archivo/marcador específico.
    """
    current_dir = Path(__file__).resolve().parent    
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    
    raise FileNotFoundError(
        f"No se pudo encontrar la raíz del proyecto. "
        f"Asegúrate de que existe un archivo llamado '{marker}' en la carpeta raíz."
    )

BASE_DIR = find_project_root()
DB_PATH = BASE_DIR / "DATASETS" / "Destino"/ "Proyecto_Accidentalidad_Vial_Antioquia.db"
OUT_DIR = BASE_DIR / "ETL_MODULES" / "Transform" / "Graficas_Salida"
MODEL_DIR = BASE_DIR / "ETL_MODULES" / "Transform" / "Modelo_Predict"

# =============================================================================
# FUNCIONES UTILITARIAS 
# =============================================================================

# PALETAS DE COLORES  PARA MEJOR CONTRASTE

PALETTE_VIBRANT = [
    "#137598","#026937", "#f9a12c", "#8dc63f", "#3ebdac","#70205b","#ef434d",
    "#35944b", "#43b649","#069a7e", "#0e7774", "069a7e",
]

PALETTE_GRADIENT = ["#FF9E80", "#FF6D42", "#FF3D00", "#DD2C00"]
PALETTE_BRIGHT = ["#f9a12c", "#8dc63f", "#3ebdac","#70205b","#ef434d"]
PALETTE_MODEL_1 = "#137598"
PALETTE_MODEL_2 = "#8dc63f"

def load_table(db_path: Path, table: str) -> pd.DataFrame:
    """Carga datos desde la base de datos SQLite"""
    if not db_path.exists():
        raise FileNotFoundError(f"No existe la base de datos: {db_path}")
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception as e:
        raise Exception(f"Error al cargar datos: {e}")

def save_fig(fig, path: Path):
    """Guarda figura en formato JPG con alta calidad"""
    try:
        fig.savefig(path, format="jpg", bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none', pad_inches=0.5)
        plt.close(fig)
    except Exception as e:
        print(f"Error guardando figura {path}: {e}")

def num_fmt(v):
    """Formatea números con separadores de miles"""
    return f"{int(v):,}"

def paleta_presentacion(n: int):
    """Genera paleta de colores"""
    if n <= 0:
        return []
    if n <= len(PALETTE_VIBRANT):
        return PALETTE_VIBRANT[:n]
    else:
        return sns.color_palette("husl", n_colors=n)

def format_torta(series: pd.Series, title: str, path: Path):
    """Genera gráfica de torta"""
    try:
        series = series.copy()
        mask = (~series.index.isna()) & (series.index.notna()) & (series.notna())
        series = series[mask]
        
        series = series[~series.index.str.contains('DESCONOCIDO|SIN INFORMACIÓN|NAN', case=False, na=False)]
        
        if series.empty or len(series) == 0:
            print(f"No hay datos válidos para: {title}")
            return
            
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.weight'] = 'bold'
        
        colors = paleta_presentacion(len(series))
        fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
        
        total = series.sum()
        
        def autopct_func(pct):
            absolute = int(round(pct * total / 100.0))
            return f'{absolute:,}'
        
        explode = [0.03] * len(series)
        if 'MUERTOS' in series.index:
            idx_muertos = list(series.index).index('MUERTOS')
            explode[idx_muertos] = 0.15
        
        wedges, texts, autotexts = ax.pie(
            series.values, 
            labels=None,
            autopct=autopct_func,
            startangle=10,
            colors=colors,
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
            explode=explode,
            shadow=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
            autotext.set_bbox(dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7, edgecolor='none'))
            
        legend_labels = [f'{label} ({value:,})' for label, value in zip(series.index, series.values)]
        ax.legend(wedges, legend_labels, title="Categorías", 
                 loc="center left", bbox_to_anchor=(1.1, 0, 0.5, 1),
                 fontsize=9, title_fontsize=10)
        
        ax.set_title(title, fontsize=16, fontweight="bold", pad=25, color='#2C3E50')
        
        plt.tight_layout()
        save_fig(fig, path)
        
    except Exception as e:
        print(f"Error en gráfica de torta '{title}': {e}")

def format_barra(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path):
    """Genera gráfica de barras horizontal con mejor legibilidad"""
    try:
        series = series.copy()
        mask = (~series.index.isna()) & (series.index.notna()) & (series.notna())
        series = series[mask]
        
        series = series[~series.index.str.contains('DESCONOCIDO|SIN INFORMACIÓN|NAN', case=False, na=False)]
        
        if series.empty or len(series) == 0:
            print(f"No hay datos válidos para: {title}")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            ax.text(0.5, 0.5, "No hay datos válidos", 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', color='#2C3E50')
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
            save_fig(fig, path)
            return
        
        series = series.sort_values(ascending=True)
        values = series.values.astype(float)
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        ax.set_facecolor('#F8F9F9')
        colors = paleta_presentacion(len(series))
        
        bars = ax.barh(range(len(series)), values, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=1.2, height=0.7)
        
        max_val = max(values)
        for i, (val, bar) in enumerate(zip(values, bars)):
            width = bar.get_width()
            
            if width >= max_val * 0.1:
                ax.text(width * 0.95, i, f"{num_fmt(val)}",
                       ha='right', va='center', fontweight='bold', 
                       fontsize=9, color='white',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7, edgecolor='none'))
            else:
                ax.text(width + max_val * 0.01, i, f"{num_fmt(val)}",
                       ha='left', va='center', fontweight='bold', 
                       fontsize=9, color='#2C3E50')
        
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#BDC3C7')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        plt.tight_layout()
        save_fig(fig, path)
        
    except Exception as e:
        print(f"Error en gráfica de barras '{title}': {e}")

def mapa_calor_accidentes(df: pd.DataFrame, path: Path):
    """Genera mapa de calor de concentración de accidentes por ubicación"""
    try:
        df_clean = df.copy()
        
        # Filtrar y limpiar datos de ubicación
        df_clean = df_clean[~df_clean["MUNICIPIO"].str.contains('DESCONOCIDO|SIN INFORMACIÓN|NAN', case=False, na=False)]
        df_clean = df_clean[~df_clean["COMUNA"].str.contains('DESCONOCIDO|SIN INFORMACIÓN|NAN', case=False, na=False)]
        
        if df_clean.empty:
            print("No hay datos válidos para el mapa de calor")
            return
        
        # Crear matriz de frecuencia por municipio y comuna
        location_counts = df_clean.groupby(['MUNICIPIO', 'COMUNA']).size().reset_index(name='ACCIDENTES')
        
        # Seleccionar top 15 combinaciones para mejor visualización
        top_locations = location_counts.nlargest(15, 'ACCIDENTES')
        
        # Preparar datos para heatmap
        pivot_data = top_locations.pivot(index='MUNICIPIO', columns='COMUNA', values='ACCIDENTES').fillna(0)
        
        # Configuración de visualización
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.weight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        ax.set_facecolor('#F8F9F9')
        
        # Paleta de colores secuencial para heatmap
        heatmap_cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        # Generar heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".0f",
            cmap=heatmap_cmap,
            cbar_kws={
                'label': 'Cantidad de Accidentes', 
                'shrink': 0.8,
                'format': '%.0f'
            },
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            annot_kws={
                'fontsize': 9,
                'fontweight': 'bold',
                'color': 'black'
            }
        )
        
        ax.set_title(
            'MAPA DE CALOR - CONCENTRACIÓN DE ACCIDENTES POR MUNICIPIO Y COMUNA',
            fontsize=16, 
            fontweight='bold', 
            pad=20, 
            color='#2C3E50'
        )
        ax.set_xlabel('COMUNA', fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        ax.set_ylabel('MUNICIPIO', fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        
        # Mejorar legibilidad de etiquetas
        plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')
        plt.yticks(rotation=0, fontsize=9, fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout()
        save_fig(fig, path)
        
        print("Mapa de calor de accidentes generado correctamente")
        
    except Exception as e:
        print(f"Error generando mapa de calor: {e}")

# Agregar llamada a la función en analisis_rapido
def analisis_rapido(df: pd.DataFrame):
    """Análisis exploratorio rápido con gráficas"""
    print("INICIANDO ANÁLISIS EXPLORATORIO ")
    
    required_columns = ["GRAVEDAD_ACCIDENTE", "JORNADA", "CLASE", "COMUNA", "MUNICIPIO"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Columnas faltantes en el dataset: {missing_columns}")
    
    try:
        df_clean = df.copy()
        
        for col in required_columns:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .replace(['None', 'nan', 'NAN', 'NONE', '', 'Sin Información', 'SIN INFORMACIÓN'], 'DESCONOCIDO')
                    .str.strip()
                    .str.upper()
                )
        
        print("Generando gráfica de gravedad mejorada...")
        gravedad_series = df_clean["GRAVEDAD_ACCIDENTE"].value_counts()
        format_torta(
            gravedad_series,
            "DISTRIBUCIÓN POR GRAVEDAD DE ACCIDENTES",
            OUT_DIR / "Accidentes_Gravedad.jpg"
        )
        
        print("Generando gráfica de jornadas mejorada...")
        jornada_series = df_clean["JORNADA"].value_counts()
        format_barra(
            jornada_series,
            "CANTIDAD DE ACCIDENTES POR JORNADA",
            "Número de Accidentes", 
            "Franja Horaria",
            OUT_DIR / "Accidentes_Jornada.jpg"
        )
        
        print("Generando gráfica de clases mejorada...")
        df_clase = df_clean[~df_clean["CLASE"].str.contains('SIN INFORMACIÓN|DESCONOCIDO', case=False, na=False)]
        clase_series = df_clase["CLASE"].value_counts().head(10)
        format_barra(
            clase_series,
            "TOP 10 - TIPOS DE ACCIDENTES MÁS FRECUENTES",
            "Número de Accidentes", 
            "Tipo de Accidente",
            OUT_DIR / "Accidentes_Clase.jpg"
        )
        
        print("Generando gráfica de comunas mejorada...")
        df_comuna = df_clean[~df_clean["COMUNA"].str.contains('SIN INFORMACIÓN|DESCONOCIDO', case=False, na=False)]
        comuna_series = df_comuna["COMUNA"].value_counts().head(10)
        format_barra(
            comuna_series,
            "TOP 10 - COMUNAS CON MÁS ACCIDENTES",
            "Número de Accidentes", 
            "Comuna",
            OUT_DIR / "Accidentes_Comuna.jpg"
        )
        
        print("Generando mapa de calor de accidentes...")
        mapa_calor_accidentes(
            df_clean,
            OUT_DIR / "Mapa_Calor_Accidentes.jpg"
        )
        
        print(f"Gráficas generadas en: {OUT_DIR}")
        
    except Exception as e:
        print(f"Error en análisis exploratorio: {e}")
        raise

# =============================================================================
# ANÁLISIS EXPLORATORIO 
# =============================================================================

def analisis_rapido(df: pd.DataFrame):
    """Análisis exploratorio rápido con gráficas """
    print("INICIANDO ANÁLISIS EXPLORATORIO ")
    
    required_columns = ["GRAVEDAD_ACCIDENTE", "JORNADA", "CLASE", "COMUNA"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Columnas faltantes en el dataset: {missing_columns}")
    
    try:
        df_clean = df.copy()
        
        for col in required_columns:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .replace(['None', 'nan', 'NAN', 'NONE', '', 'Sin Información', 'SIN INFORMACIÓN'], 'DESCONOCIDO')
                    .str.strip()
                    .str.upper()
                )
        
        print("Generando gráfica de gravedad mejorada...")
        gravedad_series = df_clean["GRAVEDAD_ACCIDENTE"].value_counts()
        format_torta(
            gravedad_series,
            "DISTRIBUCIÓN POR GRAVEDAD DE ACCIDENTES",
            OUT_DIR / "Accidentes_Gravedad.jpg"
        )
        
        print("Generando gráfica de jornadas mejorada...")
        jornada_series = df_clean["JORNADA"].value_counts()
        format_barra(
            jornada_series,
            "CANTIDAD DE ACCIDENTES POR JORNADA",
            "Número de Accidentes", 
            "Franja Horaria",
            OUT_DIR / "Accidentes_Jornada.jpg"
        )
        
        print("Generando gráfica de clases mejorada...")
        df_clase = df_clean[~df_clean["CLASE"].str.contains('SIN INFORMACIÓN|DESCONOCIDO', case=False, na=False)]
        clase_series = df_clase["CLASE"].value_counts().head(10)
        format_barra(
            clase_series,
            "TOP 10 - TIPOS DE ACCIDENTES MÁS FRECUENTES",
            "Número de Accidentes", 
            "Tipo de Accidente",
            OUT_DIR / "Accidentes_Clase.jpg"
        )
        
        print("Generando gráfica de comunas mejorada...")
        df_comuna = df_clean[~df_clean["COMUNA"].str.contains('SIN INFORMACIÓN|DESCONOCIDO', case=False, na=False)]
        comuna_series = df_comuna["COMUNA"].value_counts().head(10)
        format_barra(
            comuna_series,
            "TOP 10 - COMUNAS CON MÁS ACCIDENTES",
            "Número de Accidentes", 
            "Comuna",
            OUT_DIR / "Accidentes_Comuna.jpg"
        )
        
        print(f"Gráficas  generadas en: {OUT_DIR}")
        
    except Exception as e:
        print(f"Error en análisis exploratorio : {e}")
        raise

# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================

def preparar_datos(data: pd.DataFrame):
    """Preparación robusta de datos para modelado"""
    print("PREPARANDO DATOS - VERSIÓN MEJORADA")
    
    try:
        df = data.copy()
        
        df["GRAVEDAD_ACCIDENTE"] = (
            df["GRAVEDAD_ACCIDENTE"]
            .astype(str)
            .replace(['None', 'nan', 'NAN', 'NONE', ''], 'DESCONOCIDO')
            .str.upper()
            .str.strip()
        )
        
        df["HERIDOS_MUERTOS"] = (
            df["GRAVEDAD_ACCIDENTE"]
            .str.contains('HERIDOS|MUERTOS', case=False, na=False)
            .astype(np.uint8)
        )
        
        df["FIN_DE_SEMANA"] = df["NUM_DIA_SEMANA"].isin([6, 7]).astype(np.uint8)
        df["HORA_PICO"] = ((df["NUM_HORA"] >= 6) & (df["NUM_HORA"] <= 9) | 
                          (df["NUM_HORA"] >= 17) & (df["NUM_HORA"] <= 19)).astype(np.uint8)
        df["HORA_NOCTURNA"] = ((df["NUM_HORA"] >= 20) | (df["NUM_HORA"] <= 5)).astype(np.uint8)
        df["MES_ALTA"] = df["NUM_MES"].isin([1, 7, 12]).astype(np.uint8)
        
        categorical_features = ['CLASE', 'MUNICIPIO', 'COMUNA', 'JORNADA']
        for col in categorical_features:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace(['None', 'nan', 'NAN', 'NONE', '', 'Sin Información', 'SIN INFORMACIÓN'], 'DESCONOCIDO')
                    .str.upper()
                    .str.strip()
                )
        
        numeric_features = [
            'NUM_MES', 'NUM_DIA_SEMANA', 'NUM_HORA', 
            'FIN_DE_SEMANA', 'HORA_PICO', 'HORA_NOCTURNA', 'MES_ALTA'
        ]
        
        missing_features = [f for f in numeric_features + categorical_features if f not in df.columns]
        if missing_features:
            raise KeyError(f"Features faltantes: {missing_features}")
        
        X = df[numeric_features + categorical_features]
        y = df['HERIDOS_MUERTOS']
        
        class_dist = y.value_counts()
        print(f"Distribución de clases: {class_dist.to_dict()}")
        print(f"Porcentaje clase positiva: {y.mean():.2%}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y,
            shuffle=True
        )
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='DESCONOCIDO')),
                ('encoder', OneHotEncoder(
                    handle_unknown='ignore', 
                    sparse_output=False, 
                    min_frequency=0.01,
                    max_categories=30
                ))
            ]), categorical_features)
        ])
        
        print(f"Datos preparados - X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train, X_test, y_train, y_test, preprocessor
        
    except Exception as e:
        print(f"Error en preparación de datos: {e}")
        raise

# =============================================================================
# MATRIZ DE CONFUSIÓN
# =============================================================================

def matriz_confusion_vibrante(y_test, y_pred, model_name="Modelo"):
    """Genera matriz de confusión con mejor legibilidad"""
    try:
        cm = confusion_matrix(y_test, y_pred)
        cm_percentage = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.weight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.set_facecolor("#000000")
        
        custom_cmap = sns.light_palette(PALETTE_MODEL_1, as_cmap=True, reverse=False)
        
        sns.heatmap(
            cm, 
            annot=False, 
            fmt='d', 
            cmap=custom_cmap,
            cbar_kws={'label': '', 'shrink': 0.75},
            ax=ax,
            square=True,
            linewidths=1.5,
            linecolor='white'
        )
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.3, f'{cm[i, j]:,}', 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='black'
                )
                ax.text(
                    j + 0.5, i + 0.7, f'({cm_percentage[i, j]:.1f}%)', 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='black'
                )
        
        class_names = ['SOLO DAÑOS', 'CON HERIDOS']
        ax.set_xticklabels(class_names, fontsize=11, fontweight='bold', color='#2C3E50')
        ax.set_yticklabels(class_names, fontsize=11, fontweight='bold', color='#2C3E50', rotation=0)
        ax.set_xlabel('PREDICCIÓN DEL MODELO', fontsize=12, fontweight='bold', labelpad=15, color='#2C3E50')
        ax.set_ylabel('', fontsize=12, fontweight='bold', labelpad=15, color='#2C3E50')
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        title = (f'MATRIZ DE CONFUSIÓN - {model_name.upper()}\n')
                # f'Exactitud: {accuracy:.2%} | Precisión: {precision:.2%} | '
                # f'Sensibilidad: {recall:.2%} | F1-Score: {f1:.4f}')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        
        ax.axhline(y=1, color='white', linewidth=2)
        ax.axvline(x=1, color='white', linewidth=2)
        
        for spine in ax.spines.values():
            spine.set_color('#2C3E50')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        return fig, ax, cm
        
    except Exception as e:
        print(f"Error en matriz de confusión: {e}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Error en Matriz de Confusión - {model_name}")
        return fig, ax, np.array([[0, 0], [0, 0]])

# =============================================================================
# GRÁFICA ROC MEJORADA
# =============================================================================

def grafica_roc(y_test, y_proba_dt, y_proba_rf, resultados_dt, resultados_rf):
    """Genera gráfica ROC comparativa con mejor legibilidad"""
    try:
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.weight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('#F8F9F9')
        
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
        roc_auc_dt = resultados_dt.get('roc_auc', auc(fpr_dt, tpr_dt))
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        roc_auc_rf = resultados_rf.get('roc_auc', auc(fpr_rf, tpr_rf))
        
        ax.plot(fpr_dt, tpr_dt, color=PALETTE_MODEL_1, lw=2.5, 
                label=f'Árbol de Decisión (AUC = {roc_auc_dt:.4f})')
        ax.plot(fpr_rf, tpr_rf, color=PALETTE_MODEL_2, lw=2.5,
                label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
        
        ax.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', alpha=0.8,
                label='Clasificador Aleatorio (AUC = 0.5000)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', 
                     fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', 
                     fontsize=12, fontweight='bold', color='#2C3E50', labelpad=12)
        ax.set_title('CURVAS ROC COMPARATIVAS - ÁRBOL DE DECISIÓN vs RANDOM FOREST', 
                    fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        
        ax.grid(True, linestyle='--', alpha=0.7, color='#BDC3C7')
        ax.legend(loc="lower right", frameon=True, fancybox=True, 
                 shadow=True, fontsize=10)
        
        ax.fill_between(fpr_dt, tpr_dt, alpha=0.1, color=PALETTE_MODEL_1)
        ax.fill_between(fpr_rf, tpr_rf, alpha=0.1, color=PALETTE_MODEL_2)
        
        youden_dt = tpr_dt - fpr_dt
        idx_dt = np.argmax(youden_dt)
        ax.plot(fpr_dt[idx_dt], tpr_dt[idx_dt], 'o', color=PALETTE_MODEL_1, 
                markersize=8, label=f'Mejor punto Árbol ({(fpr_dt[idx_dt]):.3f}, {(tpr_dt[idx_dt]):.3f})')
        
        youden_rf = tpr_rf - fpr_rf
        idx_rf = np.argmax(youden_rf)
        ax.plot(fpr_rf[idx_rf], tpr_rf[idx_rf], 'o', color=PALETTE_MODEL_2, 
                markersize=8, label=f'Mejor punto RF ({(fpr_rf[idx_rf]):.3f}, {(tpr_rf[idx_rf]):.3f})')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Curva_ROC_Comparativa.jpg")
        
        print("Gráfica ROC comparativa generada correctamente")
        return fig
        
    except Exception as e:
        print(f"Error generando gráfica ROC: {e}")
        return None

# =============================================================================
# MODELO ÁRBOL DE DECISIÓN 
# =============================================================================

def entrenar_arbol_decision(X_train, X_test, y_train, y_test, preprocessor):
    """Entrenamiento  de Árbol de Decisión con parámetros reducidos"""
    print("ENTRENANDO ÁRBOL DE DECISIÓN ")
    
    try:
        dt_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        # PARÁMETROS REDUCIDOS PARA EVITAR DEMORAS
        param_grid = {
            'classifier__max_depth': [10, 15, 20],
            'classifier__min_samples_split': [20, 50],
            'classifier__min_samples_leaf': [10, 20],
            'classifier__max_features': ['sqrt']
        }
        
        grid_search = GridSearchCV(
            dt_pipeline,
            param_grid,
            cv=3,  # REDUCIDO DE 5 A 3
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Buscando mejores hiperparámetros...")
        grid_search.fit(X_train, y_train)
        
        best_dt = grid_search.best_estimator_
        y_pred = best_dt.predict(X_test)
        y_proba = best_dt.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("MÉTRICAS ÁRBOL DE DECISIÓN:")
        print(f"   • Mejores parámetros: {grid_search.best_params_}")
        print(f"   • Exactitud: {acc:.4f}")
        print(f"   • Precisión: {prec:.4f}")
        print(f"   • Sensibilidad: {rec:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        print(f"   • AUC-ROC: {roc_auc:.4f}")
        
        fig, ax, cm = matriz_confusion_vibrante(y_test, y_pred, "Árbol de Decisión")
        save_fig(fig, OUT_DIR / "Matriz_Confusion_Arbol_Decision.jpg")
        
        return best_dt, {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1, 
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'y_proba': y_proba
        }
        
    except Exception as e:
        print(f"Error entrenando Árbol de Decisión: {e}")
        raise

# =============================================================================
# MODELO RANDOM FOREST 
# =============================================================================

def entrenar_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    """Entrenamiento  de Random Forest con parámetros reducidos"""
    print("ENTRENANDO RANDOM FOREST ")
    
    try:
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        # PARÁMETROS REDUCIDOS PARA EVITAR DEMORAS
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [15, 20],
            'classifier__min_samples_split': [20, 50],
            'classifier__min_samples_leaf': [10, 20],
            'classifier__max_features': ['sqrt']
        }
        
        grid_search = GridSearchCV(
            rf_pipeline,
            param_grid,
            cv=3,  # REDUCIDO DE 5 A 3
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Buscando mejores hiperparámetros para Random Forest...")
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("MÉTRICAS RANDOM FOREST:")
        print(f"   • Mejores parámetros: {grid_search.best_params_}")
        print(f"   • Exactitud: {acc:.4f}")
        print(f"   • Precisión: {prec:.4f}")
        print(f"   • Sensibilidad: {rec:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        print(f"   • AUC-ROC: {roc_auc:.4f}")
        
        fig, ax, cm = matriz_confusion_vibrante(y_test, y_pred, "Random Forest ")
        save_fig(fig, OUT_DIR / "Matriz_Confusion_RF.jpg")
        
        return best_rf, {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1, 
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'y_proba': y_proba
        }
        
    except Exception as e:
        print(f"Error entrenando Random Forest: {e}")
        raise

# =============================================================================
# COMPARACIÓN DE MODELOS MEJORADA
# =============================================================================

def comparar_modelos(resultados_dt, resultados_rf, modelo_dt, modelo_rf, y_test):
    """Comparación visual mejorada entre modelos"""
    print("COMPARANDO MODELOS CON VISUALIZACIÓN MEJORADA")
    
    try:
        comparacion = pd.DataFrame({
            'Métrica': ['Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score', 'AUC-ROC'],
            'Árbol de Decisión': [
                resultados_dt.get('accuracy', 0),
                resultados_dt.get('precision', 0), 
                resultados_dt.get('recall', 0),
                resultados_dt.get('f1', 0),
                resultados_dt.get('roc_auc', 0)
            ],
            'Random Forest': [
                resultados_rf.get('accuracy', 0),
                resultados_rf.get('precision', 0),
                resultados_rf.get('recall', 0),
                resultados_rf.get('f1', 0),
                resultados_rf.get('roc_auc', 0)
            ]
        })
        
        comparacion['Mejor Modelo'] = comparacion[['Árbol de Decisión', 'Random Forest']].idxmax(axis=1)
        
        print("COMPARACIÓN POR MÉTRICAS:")
        print(comparacion.to_string(index=False))
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#F8F9F9')
        
        x = np.arange(len(comparacion))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparacion['Árbol de Decisión'], width, 
                      label='Árbol de Decisión', color=PALETTE_MODEL_1, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, comparacion['Random Forest'], width, 
                      label='Random Forest', color=PALETTE_MODEL_2, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Métricas de Evaluación', fontsize=12, fontweight='bold', 
                     color='#2C3E50', labelpad=12)
        ax.set_ylabel('Puntuación', fontsize=12, fontweight='bold', 
                     color='#2C3E50', labelpad=12)
        ax.set_title('COMPARACIÓN DE MODELOS: ÁRBOL DE DECISIÓN vs RANDOM FOREST', 
                    fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xticks(x)
        ax.set_xticklabels(comparacion['Métrica'], fontsize=10, fontweight='bold', 
                          color='#2C3E50')
        
        ax.legend(frameon=True, fancybox=True, shadow=True, 
                 fontsize=10, loc='upper right')
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#BDC3C7')
        ax.set_axisbelow(True)
        
        for i, (v_dt, v_rf) in enumerate(zip(comparacion['Árbol de Decisión'], 
                                            comparacion['Random Forest'])):
            ax.text(i - width/2, v_dt + 0.01, f'{v_dt:.3f}', 
                   ha='center', va='bottom', fontweight='bold', 
                   fontsize=9, color=PALETTE_MODEL_1)
            ax.text(i + width/2, v_rf + 0.01, f'{v_rf:.3f}', 
                   ha='center', va='bottom', fontweight='bold', 
                   fontsize=9, color=PALETTE_MODEL_2)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        ax.axhline(y=0.5, color='#E74C3C', linestyle=':', alpha=0.6, linewidth=1.5)
        
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Comparacion_Modelos.jpg")
        
        grafica_roc(
            y_test,
            resultados_dt.get('y_proba'),
            resultados_rf.get('y_proba'),
            resultados_dt,
            resultados_rf
        )
        
        score_dt = (resultados_dt.get('f1', 0) + resultados_dt.get('roc_auc', 0)) / 2
        score_rf = (resultados_rf.get('f1', 0) + resultados_rf.get('roc_auc', 0)) / 2
        
        if score_rf > score_dt:
            mejor_modelo = modelo_rf
            mejor_resultados = resultados_rf
            mejor_nombre = "Random Forest"
            print(f"MEJOR MODELO: RANDOM FOREST (Score: {score_rf:.4f})")
        else:
            mejor_modelo = modelo_dt
            mejor_resultados = resultados_dt
            mejor_nombre = "Árbol de Decisión"
            print(f"MEJOR MODELO: ÁRBOL DE DECISIÓN (Score: {score_dt:.4f})")
        
        comparacion.to_csv(MODEL_DIR / "Comparacion_Modelos_Detallada.csv", 
                          index=False, encoding='utf-8')
        
        return mejor_modelo, mejor_resultados, mejor_nombre
        
    except Exception as e:
        print(f"Error comparando modelos: {e}")
        return modelo_rf, resultados_rf, "Random Forest"

# =============================================================================
# ANÁLISIS DE IMPORTANCIA DE VARIABLES 
# =============================================================================

def analizar_importancia_vibrante(modelo: Pipeline, preprocessor: ColumnTransformer, nombre_modelo: str):
    """Análisis de importancia de variables con mejor legibilidad"""
    print(f"ANALIZANDO IMPORTANCIA - {nombre_modelo}")
    
    try:
        classifier = modelo.named_steps['classifier']
        importancias = classifier.feature_importances_
        
        def obtener_nombres_features(preprocessor):
            try:
                numeric_features = preprocessor.transformers[0][2]
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                categorical_features = preprocessor.transformers[1][2]
                cat_features_names = cat_encoder.get_feature_names_out(categorical_features)
                return list(numeric_features) + list(cat_features_names)
            except Exception as e:
                print(f"Error obteniendo nombres de features: {e}")
                return [f'Feature_{i}' for i in range(len(importancias))]
        
        feature_names = obtener_nombres_features(preprocessor)
        
        if len(feature_names) != len(importancias):
            print(f"Discrepancia en número de features: {len(feature_names)} vs {len(importancias)}")
            feature_names = [f'Feature_{i}' for i in range(len(importancias))]
        
        importancia_df = pd.DataFrame({
            'Variable': feature_names,
            'Importancia': importancias
        }).sort_values('Importancia', ascending=False)
        
        top_vars = importancia_df.head(10)
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titleweight'] = 'bold'
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('#F8F9F9')
        
        colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top_vars)))
        
        bars = ax.barh(range(len(top_vars)), top_vars['Importancia'], 
                      color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, height=0.7)
        
        ax.set_yticks(range(len(top_vars)))
        ax.set_yticklabels(top_vars['Variable'], fontsize=9, fontweight='bold')
        ax.set_xlabel('Importancia Relativa', fontsize=11, fontweight='bold', 
                     color='#2C3E50', labelpad=12)
        ax.set_title(f'TOP 10 VARIABLES MÁS IMPORTANTES - {nombre_modelo.upper()}', 
                    fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#BDC3C7')
        ax.set_axisbelow(True)
        
        for i, (_, row) in enumerate(top_vars.iterrows()):
            ax.text(row['Importancia'] + 0.001, i, f'{row["Importancia"]:.4f}', 
                   va='center', fontsize=8, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        plt.tight_layout()
        save_fig(fig, MODEL_DIR / f"Importancia_Variables_{nombre_modelo.replace(' ', '_')}.jpg")
        
        importancia_df.to_csv(
            MODEL_DIR / f"Importancia_Variables_{nombre_modelo.replace(' ', '_')}.csv", 
            index=False, encoding='utf-8'
        )
        
        print(f"Importancia mejorada guardada para {nombre_modelo}")
        print("TOP 5 VARIABLES MÁS IMPORTANTES:")
        for i, row in top_vars.head().iterrows():
            print(f"   {i+1}. {row['Variable']}: {row['Importancia']:.4f}")
        
        return importancia_df
        
    except Exception as e:
        print(f"Error analizando importancia: {e}")
        return pd.DataFrame()

# =============================================================================
# FUNCIONES AUXILIARES FINALES 
# =============================================================================

def hacer_predicciones(modelo, nombre_modelo):
    """Predicciones robustas con nuevo datos"""
    print(f"REALIZANDO PREDICCIONES - {nombre_modelo}")
    
    try:
        nuevos_casos = pd.DataFrame({
            'NUM_MES': [1, 7, 12, 3, 6],
            'NUM_DIA_SEMANA': [6, 2, 4, 1, 7],
            'NUM_HORA': [20, 14, 8, 18, 7],
            'FIN_DE_SEMANA': [1, 0, 0, 0, 1],
            'HORA_PICO': [0, 0, 1, 0, 1],
            'HORA_NOCTURNA': [1, 0, 0, 0, 0],
            'MES_ALTA': [1, 1, 1, 0, 0],
            'CLASE': ['CHOQUE', 'ATROPELLO', 'CHOQUE', 'VOLCAMIENTO', 'CAIDA OCUPANTE'],
            'MUNICIPIO': ['MEDELLÍN', 'MEDELLÍN', 'BELLO', 'ITAGÜÍ', 'MEDELLÍN'],
            'COMUNA': ['LAURELES ESTADIO', 'LA CANDELARIA', 'CENTRO', 'SANTA ANA', 'EL POBLADO'],
            'JORNADA': ['NOCHE', 'TARDE', 'MAÑANA', 'TARDE', 'MAÑANA']
        })
        
        preds = modelo.predict(nuevos_casos)
        proba = modelo.predict_proba(nuevos_casos)[:, 1]
        
        resultados = nuevos_casos.copy()
        resultados['PREDICCION'] = np.where(preds == 1, 'CON HERIDOS', 'SOLO DAÑOS')
        resultados['PROBABILIDAD_HERIDOS'] = [f"{p:.1%}" for p in proba]
        resultados['PROBABILIDAD_NUM'] = proba
        resultados['CONFIANZA'] = np.where(proba > 0.7, 'ALTA', 
                                         np.where(proba > 0.5, 'MEDIA', 'BAJA'))
        resultados['RIESGO'] = np.select(
            [proba > 0.7, proba > 0.4], 
            ['ALTO', 'MEDIO'], 
            default='BAJO'
        )
        
        save_path = MODEL_DIR / f"Predicciones_{nombre_modelo.replace(' ', '_')}.csv"
        resultados.to_csv(save_path, index=False, encoding='utf-8')
        
        print(f"{len(resultados)} predicciones guardadas en {save_path}")
        print("MUESTRA DE PREDICCIONES:")
        print(resultados[['CLASE', 'MUNICIPIO', 'PREDICCION', 'PROBABILIDAD_HERIDOS', 'RIESGO']].head())
        
        return resultados
        
    except Exception as e:
        print(f"Error en predicciones: {e}")
        return pd.DataFrame()

def guardar_modelo(modelo, nombre_modelo, resultados):
    """Guardado robusto de modelos"""
    try:
        path_modelo = MODEL_DIR / f"Modelo_{nombre_modelo.replace(' ', '_')}_SVA.joblib"
        joblib.dump(modelo, path_modelo)
        
        metadatos = {
            'nombre_modelo': nombre_modelo,
            'fecha_entrenamiento': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metricas': resultados,
            'version': '6.0'
        }
        
        path_metadatos = MODEL_DIR / f"Metadatos_{nombre_modelo.replace(' ', '_')}_SVA.joblib"
        joblib.dump(metadatos, path_metadatos)
        
        print(f"Modelo {nombre_modelo} guardado en: {path_modelo}")
        return path_modelo
        
    except Exception as e:
        print(f"Error guardando modelo: {e}")
        return None

def generar_resumen_ejecutivo(df: pd.DataFrame, resultados_dt: dict, resultados_rf: dict, mejor_modelo: str):
    """Genera resumen ejecutivo robusto"""
    resumen_path = MODEL_DIR / "Resumen_Ejecutivo_Modelos_Comparados.txt"
    
    try:
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write("RESUMEN EJECUTIVO - COMPARACIÓN DE MODELOS PREDICTIVOS \n")
            f.write("="*70 + "\n\n")
            
            total_accidentes = len(df)
            accidentes_graves = df.get('HERIDOS_MUERTOS', pd.Series([0] * len(df))).sum()
            tasa_graves = accidentes_graves / total_accidentes if total_accidentes > 0 else 0
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write(f"   • Total de accidentes analizados: {total_accidentes:,}\n")
            f.write(f"   • Accidentes con heridos/muertos: {accidentes_graves:,}\n")
            f.write(f"   • Tasa de accidentes graves: {tasa_graves:.2%}\n\n")
            
            f.write("COMPARACIÓN DE MODELOS :\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Métrica':<15} {'Árbol':<8} {'Random Forest':<14} {'Mejor':<8}\n")
            f.write("-" * 50 + "\n")
            
            metricas = [
                ('Exactitud', 'accuracy'),
                ('Precisión', 'precision'), 
                ('Sensibilidad', 'recall'),
                ('F1-Score', 'f1'),
                ('AUC-ROC', 'roc_auc')
            ]
            
            for nombre, clave in metricas:
                dt_val = resultados_dt.get(clave, 0)
                rf_val = resultados_rf.get(clave, 0)
                mejor = "Árbol" if dt_val > rf_val else "RF"
                f.write(f"{nombre:<15} {dt_val:.4f}    {rf_val:.4f}         {mejor:<8}\n")
            
            f.write("-" * 50 + "\n\n")
            
            f.write("MODELO RECOMENDADO:\n")
            f.write(f"   • {mejor_modelo}\n\n")
            
            f.write("RECOMENDACIONES:\n")
            f.write("   1. Implementar sistema de alerta temprana\n")
            f.write("   2. Focalizar recursos en horarios de alto riesgo\n")
            f.write("   3. Monitorear performance del modelo regularmente\n")
            f.write("   4. Actualizar modelo con nuevos datos periódicamente\n\n")
            
            f.write("="*70 + "\n")
            f.write(f"Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Resumen ejecutivo guardado en: {resumen_path}")
        
    except Exception as e:
        print(f"Error generando resumen: {e}")

# =============================================================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# =============================================================================

def main():
    """Función principal optimizada"""
    try:
        print("INICIANDO ANÁLISIS ")
        print("=" * 70)
        
        print("CARGANDO DATOS...")
        df = load_table(DB_PATH, "Accidentalidad_Vial_Antioquia")
        print(f"Datos cargados: {len(df):,} registros, {len(df.columns)} columnas")
        
        analisis_rapido(df)
        
        X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)
        
        modelo_dt, resultados_dt = entrenar_arbol_decision(X_train, X_test, y_train, y_test, preprocessor)
        modelo_rf, resultados_rf = entrenar_random_forest(X_train, X_test, y_train, y_test, preprocessor)
        
        mejor_modelo, mejores_resultados, mejor_nombre = comparar_modelos(
            resultados_dt, resultados_rf, modelo_dt, modelo_rf, y_test
        )
        
        analizar_importancia_vibrante(modelo_dt, preprocessor, "Arbol_Decision")
        analizar_importancia_vibrante(modelo_rf, preprocessor, "Random_Forest")
        
        hacer_predicciones(mejor_modelo, mejor_nombre)
        
        guardar_modelo(modelo_dt, "Arbol_Decision", resultados_dt)
        guardar_modelo(modelo_rf, "Random_Forest", resultados_rf)
        guardar_modelo(mejor_modelo, f"MEJOR_MODELO_{mejor_nombre}", mejores_resultados)
        
        generar_resumen_ejecutivo(df, resultados_dt, resultados_rf, mejor_nombre)
        
        print("\nANÁLISIS  COMPLETADO")
        print("=" * 70)
        print("RESULTADOS FINALES:")
        print(f"   • Mejor modelo: {mejor_nombre}")
        print(f"   • F1-Score: {mejores_resultados.get('f1', 0):.4f}")
        print(f"   • AUC-ROC: {mejores_resultados.get('roc_auc', 0):.4f}")
        print(f"   • Gráficas : {OUT_DIR}")
        print(f"   • Modelos: {MODEL_DIR}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)

if __name__ == "__main__":
    main()