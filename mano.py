"""Entrena y guarda un clasificador Random Forest a partir de datos_mano.csv."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "datos_mano.csv"
RUTA_MODELO = CARPETA / "clasificador_manos.pkl"
NUM_CARACTERISTICAS = 42
CLASES_VALIDAS = {"0", "1", "2", "3"}

if not RUTA_CSV.is_file():
    sys.exit("Error: no se encontró datos_mano.csv. Ejecuta primero aprendizaje_corregido.py.")

try:
    datos = pd.read_csv(RUTA_CSV, header=None)
except pd.errors.EmptyDataError:
    sys.exit("Error: datos_mano.csv está vacío. Recoge muestras antes de entrenar.")

if datos.shape[1] != NUM_CARACTERISTICAS + 1:
    sys.exit(
        f"Error: se esperaban {NUM_CARACTERISTICAS + 1} columnas (42 coordenadas y una etiqueta), "
        f"pero el CSV tiene {datos.shape[1]}. Revisa datos_mano.csv."
    )

# Fuerza las coordenadas a valores numéricos y elimina filas corruptas.
X = datos.iloc[:, :NUM_CARACTERISTICAS].apply(pd.to_numeric, errors="coerce")
y = datos.iloc[:, NUM_CARACTERISTICAS].astype(str).str.strip()
validas = X.notna().all(axis=1) & y.isin(CLASES_VALIDAS)
X, y = X.loc[validas], y.loc[validas]

if len(X) < 5:
    sys.exit(f"Error: solo hay {len(X)} muestras válidas. Recoge más datos antes de entrenar.")
if y.nunique() < 2:
    sys.exit("Error: se necesitan muestras de al menos dos clases distintas para entrenar.")

# Estratificar si cada clase tiene suficientes registros para el conjunto de prueba.
conteos = y.value_counts()
usar_estratificacion = conteos.min() >= 2 and len(X) >= 10
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if usar_estratificacion else None,
)

print(f"Entrenando con {len(X_train)} muestras; evaluando con {len(X_test)}...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo.fit(X_train, y_train)
precision = modelo.score(X_test, y_test)
print(f"Entrenamiento completado. Exactitud en prueba: {precision * 100:.2f}%")

joblib.dump(modelo, RUTA_MODELO)
print(f"Modelo guardado en: {RUTA_MODELO}")
print("Clases aprendidas:", ", ".join(sorted(modelo.classes_)))
