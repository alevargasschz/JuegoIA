"""Recolecta coordenadas de la mano para entrenar el clasificador.

Ejecutar desde la carpeta que contiene este archivo y hand_landmarker.task.
Pulsa 0=izquierda, 1=derecha, 2=arriba, 3=abajo para guardar una muestra.
Pulsa q para salir.
"""
import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "hand_landmarker.task"
RUTA_CSV = CARPETA / "datos_mano.csv"

if not RUTA_MODELO.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo de MediaPipe: {RUTA_MODELO}\n"
        "Descarga hand_landmarker.task y guárdalo junto a este script."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(RUTA_MODELO)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Comprueba permisos o cambia el índice de cámara.")

# El CSV contiene 42 características (x,y de 21 puntos) y una etiqueta.
try:
    with HandLandmarker.create_from_options(opciones) as detector, RUTA_CSV.open(
        "a", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.writer(archivo)
        print("INSTRUCCIONES: 0=IZQUIERDA | 1=DERECHA | 2=ARRIBA | 3=ABAJO")
        print("Coloca la mano frente a la cámara y pulsa la tecla de la clase deseada.")
        print("Pulsa q para salir.")
        timestamp_anterior = 0

        while True:
            correcto, frame = cap.read()
            if not correcto:
                print("No se pudo leer un fotograma de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Usa un timestamp creciente para cumplir el modo VIDEO.
            timestamp_ms = max(int(time.monotonic() * 1000), timestamp_anterior + 1)
            timestamp_anterior = timestamp_ms
            resultado = detector.detect_for_video(imagen, timestamp_ms)

            puntos = None
            if resultado.hand_landmarks:
                puntos = []
                for punto in resultado.hand_landmarks[0]:
                    puntos.extend([punto.x, punto.y])
                cv2.putText(frame, "Mano detectada", (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 180, 0), 2)
            else:
                cv2.putText(frame, "No se detecta mano", (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            cv2.putText(frame, "0 Izq | 1 Der | 2 Arriba | 3 Abajo | q Salir",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imshow("Recolector de datos", frame)

            # Solo hay una llamada a waitKey por fotograma, para no perder teclas.
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla in (ord("0"), ord("1"), ord("2"), ord("3")):
                if puntos is None:
                    print("No se guardó la muestra: primero coloca la mano dentro del encuadre.")
                else:
                    etiqueta = chr(tecla)
                    escritor.writerow(puntos + [etiqueta])
                    archivo.flush()
                    print(f"Muestra guardada para la clase {etiqueta} ({len(puntos)} características).")
finally:
    cap.release()
    cv2.destroyAllWindows()
