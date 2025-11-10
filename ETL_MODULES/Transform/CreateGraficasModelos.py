# =============================================================================
# BLOQUE 1: CONFIGURACIÓN GLOBAL
# =============================================================================

import warnings, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report )

# WARNINGS
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# CONEXIÓN BD
DB_PATH = Path(r"C:/Users/DanielaVallejo/Desktop/Proyecto_Accidentalidad_Vial_Antioquia/Proyecto_Accidentalidad_Vial_Antioquia.db")
OUT_DIR = Path(r"C:/Users/DanielaVallejo/Desktop/Proyecto_Accidentalidad_Vial_Antioquia/Graficas_Salida")
MODEL_DIR = Path(r"C:/Users/DanielaVallejo/Desktop/Proyecto_Accidentalidad_Vial_Antioquia/Modelo_Predict")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PALETTE_GREEN = ["#267344", "#37A85B", "#A9E4B4"]

# UTILIDADES
def load_table(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"No existe la base de datos: {db_path}")
    with create_engine(f"sqlite:///{db_path}").connect() as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)

def save_fig(fig, path: Path):
    fig.savefig(path, format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)

def num_fmt(v): return f"{int(v):,}"

def txt_color(rgb):
    return "black" if (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) > 0.65 else "white"

def paleta_antioquia(n: int):
    if n <= 0: return []
    return PALETTE_GREEN[:n] if n <= len(PALETTE_GREEN) else sns.blend_palette(PALETTE_GREEN, n_colors=n)

def format_torta(series: pd.Series, title: str, path: Path):
    colors = paleta_antioquia(len(series)) or paleta_antioquia(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(series.values, labels=series.index, autopct="%1.1f%%", startangle=60,
           colors=colors[::-1], textprops={"fontsize": 12})
    ax.set_title(title, fontsize=18, fontweight="bold")
    save_fig(fig, path)

def format_barra(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    series = series.dropna()
    if series.empty:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        save_fig(fig, path)
        return
    values = series.values.astype(float)
    palette = paleta_antioquia(len(series))
    total, max_w = values.sum() or 1, values.max() or 1
    sns.barplot(x=values, y=series.index, palette=palette, ax=ax)
    for p, val in zip(ax.patches, values):
        w, y = p.get_width(), p.get_y() + p.get_height()/2
        pct, rel = (val/total)*100, w/max_w
        color = txt_color(p.get_facecolor()[:3]) if rel >= 0.12 else "black"
        ax.text(w*0.5 if rel >= 0.12 else w+(max_w*0.01), y,
                f"{num_fmt(val)} ({pct:.1f}%)",
                ha="center" if rel >= 0.12 else "left", va="center",
                fontweight="bold", fontsize=11, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    save_fig(fig, path)

# =============================================================================
# BLOQUE 2: ANÁLISIS EXPLORATORIO RÁPIDO
# =============================================================================

def analisis_rapido(df: pd.DataFrame):
    print("\n📄 ANÁLISIS EXPLORATORIO RÁPIDO")
    required = ["GRAVEDAD_ACCIDENTE", "JORNADA", "CLASE", "COMUNA"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Faltan columnas: {missing}")

    #Accidentes por Gravedad
    format_torta(df["GRAVEDAD_ACCIDENTE"].value_counts(),
                 "Distribución por Gravedad de Accidentes",
                 OUT_DIR / "Accidentes_Gravedad_SVA.jpg")
    #Accidentes por Jornada
    format_barra(df["JORNADA"].value_counts(),
                 "Cantidad de Accidentes por Jornada",
                 "Número de Accidentes", "Franja Horaria",
                 OUT_DIR / "Accidentes_Jornada_SVA.jpg")
    #Accidentes por Clase
    df_clase = df[df["CLASE"].str.upper() != "SIN INFORMACIÓN"]
    format_barra(df_clase["CLASE"].value_counts().head(10),
                 "Cantidad de Accidentes por Clase",
                 "Número de Accidentes", "Tipo de Accidente",
                 OUT_DIR / "Accidentes_Clase_SVA.jpg")
    #Accidentes por Comuna
    df_comuna = df[df["COMUNA"].str.upper() != "SIN INFORMACIÓN"]
    format_barra(df_comuna["COMUNA"].value_counts().head(10),
                 "Top 10 - Accidentes por Comuna",
                 "Número de Accidentes", "Comuna",
                 OUT_DIR / "Accidentes_Comuna_SVA.jpg")

    print(f"✔️  Gráficas generadas en: {OUT_DIR}")
    print("="*60)

# =============================================================================
# BLOQUE 3: PREPARACIÓN DE DATOS
# =============================================================================

def preparar_datos(data: pd.DataFrame):
    print("📄 PREPARANDO DATOS PARA RANDOM FOREST")
    df = data.copy()

    df["HERIDOS_MUERTOS"] = df["GRAVEDAD_ACCIDENTE"].str.upper().isin(["HERIDOS", "MUERTOS"]).astype(np.uint8)
    df["FIN_DE_SEMANA"] = df["NUM_DIA_SEMANA"].isin([6,7]).astype(np.uint8)

    numeric_features = ['NUM_MES', 'NUM_DIA_SEMANA', 'NUM_HORA', 'FIN_DE_SEMANA']
    categorical_features = ['CLASE', 'MUNICIPIO', 'COMUNA', 'JORNADA']

    X = df[numeric_features + categorical_features]
    y = df['HERIDOS_MUERTOS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y )

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
    ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    print(f"✔️  Forma después del preprocesamiento: {X_train_preprocessed.shape}")
    return X_train, X_test, y_train, y_test, preprocessor

# =============================================================================
# BLOQUE 4: ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def entrenar_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    print("\n🤖 ENTRENAMIENTO RANDOM FOREST")
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]

    acc, prec, rec, f1 = (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    )
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Exactitud: {acc:.4f} | Precisión: {prec:.4f} | Sensibilidad: {rec:.4f} | F1: {f1:.4f} | AUC-ROC: {roc_auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.blend_palette(PALETTE_GREEN[::-1], as_cmap=True),
                xticklabels=['Solo Daños', 'Con Heridos'], yticklabels=['Solo Daños', 'Con Heridos'], ax=ax)
    ax.set_title('Matriz de Confusión - Clasificación de Accidentes', fontsize=16)
    save_fig(fig, OUT_DIR / "Matriz_Confusion_SVA.jpg")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color=PALETTE_GREEN[1], lw=2, label=f'ROC (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    ax.set_xlim([-0.00, 1.01]); ax.set_ylim([-0.00, 1.01])
    ax.set_title('Curva ROC - Clasificación de Severidad de Accidentes')
    ax.set_xlabel('Tasa Falsos Positivos'); ax.set_ylabel('Tasa Verdaderos Positivos')
    ax.legend(loc="lower right")
    save_fig(fig, OUT_DIR / "Curva_ROC_SVA.jpg")

    print(classification_report(y_test, y_pred, target_names=['Solo Daños', 'Con Heridos']))
    return model, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}
    print("="*60)

# =============================================================================
# BLOQUE 5: PREDICCIÓN Y GUARDE DEL MODELO
# =============================================================================

def hacer_predicciones(modelo):
    nuevos = pd.DataFrame({
        'NUM_MES': [1,7,12,3],
        'NUM_DIA_SEMANA': [6,2,4,1],
        'NUM_HORA': [20,14,8,18],
        'FIN_DE_SEMANA': [1,0,0,0],
        'CLASE': ['CHOQUE','ATROPELLO','CHOQUE','VOLCAMIENTO'],
        'MUNICIPIO': ['MEDELLÍN']*4,
        'COMUNA': ['LAURELES ESTADIO','LA CANDELARIA','CASTILLA','ROBLEDO'],
        'JORNADA': ['NOCHE','TARDE','MAÑANA','TARDE']
    })
    preds = modelo.predict(nuevos)
    proba = modelo.predict_proba(nuevos)[:, 1]
    resultados = nuevos.copy()
    resultados['PREDICCION'] = np.where(preds==1,'CON HERIDOS','SOLO DAÑOS')
    resultados['PROBABILIDAD_HERIDOS'] = [f"{p:.1%}" for p in proba]
    resultados['RIESGO'] = np.select([proba>0.7, proba>0.5], ['ALTO','MEDIO'], default='BAJO')
    save_path = MODEL_DIR / "Predicciones_Nuevos_Accidentes.csv"
    resultados.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✔️ Predicciones guardadas en {save_path}")
    return resultados

def guardar_modelo(modelo):
    path = MODEL_DIR / "Modelo_RandomForest_SVA.joblib"
    joblib.dump(modelo, path)
    print(f"✔️ Modelo guardado en: {path}")

# =============================================================================
# BLOQUE 6: IMPORTANCIA DE VARIABLES
# =============================================================================

def analizar_importancia_variables(modelo: Pipeline, preprocessor: ColumnTransformer):
    rf = modelo.named_steps['classifier']
    num = ['NUM_MES','NUM_DIA_SEMANA','NUM_HORA','FIN_DE_SEMANA']
    cat = ['CLASE','MUNICIPIO','COMUNA','JORNADA']
    ohe = preprocessor.named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(cat)
    names = num + list(cat_names)
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({'Variable': names, 'Importancia': imp}).sort_values('Importancia', ascending=False)
    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(x='Importancia', y='Variable', data=imp_df.head(10), palette=paleta_antioquia(10), ax=ax)
    ax.set_title('Top 10 Variables Más Importantes - Random Forest', fontsize=16)
    save_fig(fig, MODEL_DIR / "Importancia_Variables_RF.jpg")
    imp_df.to_csv(MODEL_DIR / "Importancia_Variables_RF.csv", index=False, encoding='utf-8-sig')
    print("✔️ Importancia de variables guardada.")
    return imp_df
    print("="*60)

# =============================================================================
# BLOQUE 7: RESUMEN EJECUTIVO
# =============================================================================

def generar_resumen_final(df: pd.DataFrame, resultados: dict):

    resumen_path = MODEL_DIR / "Resumen_Ejecutivo_Modelo.txt"
    with open(resumen_path, "w", encoding="utf-8") as f:
        print("\n" + "="*60)
        print("📄 RESUMEN EJECUTIVO DEL PROYECTO")
        print("="*60)
        f.write("📄 RESUMEN EJECUTIVO DEL PROYECTO\n")
        f.write("="*60 + "\n")

        # ESTADÍSTICAS GENERALES
        total_accidentes = len(df)
        accidentes_con_heridos = len(df[df['GRAVEDAD_ACCIDENTE'].str.upper().isin(['HERIDOS', 'MUERTOS'])])
        tasa_heridos = accidentes_con_heridos / total_accidentes

        texto_stats = (
            f"📈 ESTADÍSTICAS GENERALES:\n"
            f"   • Total de accidentes analizados: {total_accidentes:,}\n"
            f"   • Accidentes con heridos/muertos: {accidentes_con_heridos:,}\n"
            f"   • Tasa de accidentes con heridos: {tasa_heridos:.2%}\n\n" )
        print(texto_stats)
        f.write(texto_stats)

        # RESULTADOS DEL MODELO
        texto_modelo = (
            f"🤖 MODELO RANDOM FOREST:\n"
            f"   • Exactitud: {resultados['accuracy']:.2%}\n"
            f"   • Precisión: {resultados['precision']:.2%}\n"
            f"   • Sensibilidad: {resultados['recall']:.2%}\n"
            f"   • F1-Score: {resultados['f1']:.4f}\n"
            f"   • AUC-ROC: {resultados['roc_auc']:.4f}\n\n" )
        print(texto_modelo)
        f.write(texto_modelo)

        # HALLAZGOS PRINCIPALES
        print("🔍 HALLAZGOS PRINCIPALES:")
        f.write("🔍 HALLAZGOS PRINCIPALES:\n")
        try:
            franja_peligrosa = (
                df.groupby('JORNADA')['GRAVEDAD_ACCIDENTE']
                .apply(lambda x: x.str.upper().isin(['HERIDOS', 'MUERTOS']).mean())
                .idxmax() )
            tipo_peligroso = (
                df.groupby('CLASE')['GRAVEDAD_ACCIDENTE']
                .apply(lambda x: x.str.upper().isin(['HERIDOS', 'MUERTOS']).mean())
                .idxmax() )
            comuna_peligrosa = (
                df.groupby('COMUNA')['GRAVEDAD_ACCIDENTE']
                .apply(lambda x: x.str.upper().isin(['HERIDOS', 'MUERTOS']).mean())
                .idxmax() )
            hallazgos = (
                f"   • Franja horaria más peligrosa: {franja_peligrosa}\n"
                f"   • Tipo de accidente más peligroso: {tipo_peligroso}\n"
                f"   • Comuna con mayor tasa de heridos: {comuna_peligrosa}\n\n" )
            print(hallazgos)
            f.write(hallazgos)

            # RECOMENDACIONES
            recomendaciones = (
                "💡 RECOMENDACIONES:\n"
                f"   1. Reforzar vigilancia en: {franja_peligrosa}\n"
                f"   2. Implementar campañas preventivas para: {tipo_peligroso}\n"
                f"   3. Focalizar recursos de control en: {comuna_peligrosa}\n"
                "   4. Utilizar el modelo predictivo para priorizar zonas de riesgo.\n" )
            print(recomendaciones)
            f.write(recomendaciones)

        except Exception as e:
            msg = f"❌ No se pudieron generar hallazgos detallados: {e}\n"
            print(msg)
            f.write(msg)
        print(f"\n✔️ Resumen ejecutivo guardado en: {resumen_path}")
        f.write(f"\n✔️ Resumen ejecutivo guardado en: {resumen_path}\n")

# =============================================================================
# BLOQUE FINAL DE EJECUCIÓN CONTROLADA
# =============================================================================

if __name__ == "__main__":
    try:
        df = load_table(DB_PATH, "Accidentalidad_Vial_Antioquia")
        analisis_rapido(df)
        X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)
        modelo_rf, resultados = entrenar_random_forest(X_train, X_test, y_train, y_test, preprocessor)
        hacer_predicciones(modelo_rf)
        guardar_modelo(modelo_rf)
        analizar_importancia_variables(modelo_rf, preprocessor)
        generar_resumen_final(df, resultados)
        print("\n✔️ Flujo completo: ejecutado correctamente.")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
