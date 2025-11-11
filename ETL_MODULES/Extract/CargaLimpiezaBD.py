# =============================================================================
# Carga y limpieza de datos - Proyecto Accidentalidad Vial Antioquia
# =============================================================================

import re
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# ---------- CONFIGURACIÓN ----------
# CSV remoto
CSV_PATH = "https://raw.githubusercontent.com/ShinnetValllejo/Proyecto_Accidentabilidad_Vial_Aburra/main/DATASETS/Fuentes/AMVA_Accidentalidad_20191022_2.csv"

# Función para encontrar la raíz del proyecto
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

# Directorios y DB
BASE_DIR = find_project_root()
DB_DIR = BASE_DIR / "DATASETS" / "Destino"
DB_PATH = DB_DIR / "Proyecto_Accidentalidad_Vial_Antioquia.db"

# Configuración de tabla y CSV
TABLE_NAME = "Accidentalidad_Vial_Antioquia"
SEPARATOR = ";"
ENCODING = "latin-1"

# ---------- FUNCIONES DE LIMPIEZA ----------
def clean_fecha(fecha):
    if pd.isna(fecha):
        return None
    s = str(fecha).strip()
    match = re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s)
    if not match:
        return None
    for fmt in ["%d/%m/%Y", "%m/%d/%Y"]:
        try:
            return pd.to_datetime(match.group(0), format=fmt).strftime("%d/%m/%Y")
        except:
            continue
    return None

def clean_hora(hora):
    if pd.isna(hora):
        return None
    s = re.sub(r"\s+", " ", str(hora).strip().replace("\u00A0", " "))
    m = re.search(r"(\d{1,2}:\d{2}(:\d{2})?)", s)
    s = m.group(1) if m else s
    s = s.replace("p m", "PM").replace("pm", "PM").replace("a m", "AM").replace("am", "AM")
    return s.strip()

def try_parse_time(val):
    for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
        t = pd.to_datetime(val, format=fmt, errors="coerce")
        if pd.notna(t):
            return t
    return None

def clasificar_jornada(hora_str):
    if pd.isna(hora_str):
        return None
    try:
        h = int(hora_str.split(":")[0])
        if 0 <= h < 6: return "MADRUGADA"
        if 6 <= h < 12: return "MAÑANA"
        if 12 <= h < 18: return "TARDE"
        if 18 <= h < 24: return "NOCHE"
    except:
        return None

# ---------- CARGA ----------
df = pd.read_csv(CSV_PATH, sep=SEPARATOR, encoding=ENCODING, low_memory=False)

# ---------- LIMPIEZA Y NORMALIZACIÓN ----------
df.columns = df.columns.str.strip()

rename_map = {
    "GRAVEDAÑOSSADAÑOSS": "GRAVEDAD_ACCIDENTE",
    "DÍA DE LA SEMANA": "NOM_DIA_SEMANA",
    "DIA DE LA SEMANA": "NOM_DIA_SEMANA"
}
df.rename(columns=rename_map, inplace=True)

df["FECHA"] = df["FECHA"].astype(str).map(clean_fecha)
df["HORA"] = df["HORA"].astype(str).map(clean_hora)

df["HORA_dt"] = df["HORA"].apply(try_parse_time)
df["NUM_HORA"] = df["HORA_dt"].apply(
    lambda t: (t.hour + t.minute / 60.0 + t.second / 3600.0) if pd.notna(t) else None
)
df['hora_redondeada'] = df['HORA_dt'].dt.floor('h').dt.time
df["HORA"] = df["HORA_dt"].dt.strftime("%H:%M:%S")
df.drop(columns=["HORA_dt"], inplace=True)

df["JORNADA"] = df["HORA"].map(clasificar_jornada)
df["FECHA_dt"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y", errors="coerce")
df["NUM_DIA_SEMANA"] = df["FECHA_dt"].dt.weekday + 1
df["NUM_MES"] = df["FECHA_dt"].dt.month
df["NOM_MES"] = df["FECHA_dt"].dt.month_name(locale='es_ES')
df["AÑO"] = df["FECHA_dt"].dt.year
df.drop(columns=["FECHA_dt"], inplace=True)
df["AÑO_MES"] = df["AÑO"].astype(str) + "/" + df["NUM_MES"].astype(str).str.zfill(2)

for col in df.columns:
    if col in ("NUM_HORA", "NUM_DIA_SEMANA", "NUM_MES"):
        continue
    if df[col].dtype == "object":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

df["FECHA"] = df["FECHA"].replace("NAN", None)
df["HORA"] = df["HORA"].replace("NAN", None)
df["NUM_HORA"] = pd.to_numeric(df["NUM_HORA"], errors="coerce")

# ---------- GUARDADO ----------
engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)

# ---------- VALIDACIÓN DE NULOS POR CAMPO ----------
print("\n=== VALIDACIÓN DE NULOS POR CAMPO ===")
nulos_por_columna = df.isna().sum()
for col, nulos in nulos_por_columna.items():
    total = len(df)
    pct = (nulos / total) * 100
    print(f"{col:<25} -> {nulos:>6} nulos ({pct:5.2f}%)")
print("======================================\n")
