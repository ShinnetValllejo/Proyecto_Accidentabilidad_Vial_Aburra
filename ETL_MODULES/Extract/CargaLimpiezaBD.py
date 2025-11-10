# =============================================================================
# Carga y limpieza de datos - Proyecto Accidentalidad Vial Antioquia
# =============================================================================

import re
import pandas as pd
from sqlalchemy import create_engine

# ---------- CONFIGURACIÓN ----------
CSV_PATH = "C:/Users/DanielaVallejo/Desktop/Proyecto_Accidentalidad_Vial_Antioquia/Stage/AMVA_Accidentalidad_20191022_2.csv"
DB_PATH = "Proyecto_Accidentalidad_Vial_Antioquia.db"
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

# Limpiar fecha y hora (strings)
df["FECHA"] = df["FECHA"].astype(str).map(clean_fecha)
df["HORA"] = df["HORA"].astype(str).map(clean_hora)

# Normalizar hora -> obtener objeto datetime en columna temporal HORA_dt
df["HORA_dt"] = df["HORA"].apply(try_parse_time)

# Crear NUM_HORA: hora en formato numérico continuo (horas decimales: H + M/60 + S/3600)
# Ejemplo: 08:30:00 -> 8.5
df["NUM_HORA"] = df["HORA_dt"].apply(
    lambda t: (t.hour + t.minute / 60.0 + t.second / 3600.0) if pd.notna(t) else None
)

# Mantener HORA como string normalizado HH:MM:SS (si parse falla quedará NaN)
df["HORA"] = df["HORA_dt"].dt.strftime("%H:%M:%S")
# eliminar columna temporal HORA_dt
df.drop(columns=["HORA_dt"], inplace=True)

# Clasificar jornada (sigue usando HORA string limpio)
df["JORNADA"] = df["HORA"].map(clasificar_jornada)

# Día, número de semana, número de mes y nombre del mes
df["FECHA_dt"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y", errors="coerce")
df["NUM_DIA_SEMANA"] = df["FECHA_dt"].dt.weekday + 1
df["NUM_MES"] = df["FECHA_dt"].dt.month
df["NOM_MES"] = df["FECHA_dt"].dt.month_name(locale='es_ES')  # Nombre del mes en español
df["ANIO"] = df["FECHA_dt"].dt.year
df.drop(columns=["FECHA_dt"], inplace=True)
# Crear campo ANIO/MES en formato YYYY/MM
df["ANIO_MES"] = df["ANIO"].astype(str) + "/" + df["NUM_MES"].astype(str).str.zfill(2)

# Normalización de todos los textos (mantengo HORA y NUM_HORA fuera; HORA ya es string)
for col in df.columns:
    # no normalizar columnas numéricas que acabamos de crear
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

# ---------- VALIDACIÓN ----------
df["FECHA"] = df["FECHA"].replace("NAN", None)
df["HORA"] = df["HORA"].replace("NAN", None)

# Asegurar tipo numérico para NUM_HORA (float, permite nulos)
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
