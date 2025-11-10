import os

db_path = "Proyecto_Accidentalidad_Vial_Antioquia.db"

if os.path.exists(db_path):
    os.remove(db_path)
    print("✅ Base de datos eliminada correctamente.")
else:
    print("⚠️ No se encontró el archivo de base de datos.")