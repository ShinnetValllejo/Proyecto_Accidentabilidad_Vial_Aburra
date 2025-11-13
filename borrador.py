import pandas as pd
from sqlalchemy import create_engine
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
APP_DIR = PROJECT_ROOT
DB_PATH = APP_DIR / "DATASETS" / "Destino" / "Proyecto_Accidentalidad_Vial_Antioquia.db"

# Verificar existencia de la base de datos
if not DB_PATH.exists():
    raise FileNotFoundError(f"No se encontró la base de datos en: {DB_PATH}")

# Crear conexión
engine = create_engine(f"sqlite:///{DB_PATH}")

# Consultar estructura de la tabla
df_info = pd.read_sql("PRAGMA table_info(Accidentalidad_Vial_Antioquia);", con=engine)

print(df_info)

