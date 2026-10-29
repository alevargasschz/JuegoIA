"""Juego Pygame: mueve un rectángulo según la dirección predicha de la mano."""
from pathlib import Path
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pygame

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO_IA = CARPETA / "clasificador_manos.pkl"
RUTA_MODELO_MANO = CARPETA / "hand_landmarker.task"

if not RUTA_MODELO_IA.is_file():
    sys.exit("Error: falta clasificador_manos.pkl. Ejecuta primero mano_corregido.py.")
if not RUTA_MODELO_MANO.is_file():
    sys.exit("Error: falta hand_landmarker.task. Guárdalo junto a este script.")

try:
    modelo = joblib.load(RUTA_MODELO_IA)
except (OSError, ValueError, EOFError) as error:
    sys.exit(f"No se pudo cargar el clasificador: {error}")

if not hasattr(modelo, "n_features_in_") or modelo.n_features_in_ != 42:
    sys.exit("Error: el clasificador no espera 42 características. Vuelve a entrenarlo con mano_corregido.py.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(RUTA_MODELO_MANO)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

ANCHO, ALTO = 600, 400
RECT_ANCHO, RECT_ALTO = 100, 60
VELOCIDAD = 5
FONDO = (144, 238, 144)
AZUL = (0, 0, 255)
NEGRO = (0, 0, 0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: no se pudo abrir la cámara. Comprueba los permisos o cambia el índice de cámara.")

pygame.init()
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Mover rectángulo con IA de mano")
reloj = pygame.time.Clock()
fuente = pygame.font.SysFont("Arial", 24)
rect = pygame.Rect(250, 150, RECT_ANCHO, RECT_ALTO)
gesto_detectado = "Esperando mano..."
timestamp_anterior = 0

try:
    with HandLandmarker.create_from_options(opciones) as detector:
        ejecutando = True
        while ejecutando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    ejecutando = False

            if not ejecutando:
                break

            correcto, frame = cap.read()
            if correcto:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = max(int(time.monotonic() * 1000), timestamp_anterior + 1)
                timestamp_anterior = timestamp_ms
                resultado = detector.detect_for_video(imagen, timestamp_ms)

                if resultado.hand_landmarks:
                    puntos = []
                    for punto in resultado.hand_landmarks[0]:
                        puntos.extend([punto.x, punto.y])

                    entrada = np.asarray(puntos, dtype=np.float32).reshape(1, 42)
                    prediccion = str(modelo.predict(entrada)[0])
                    if prediccion == "0":
                        rect.x -= VELOCIDAD
                        gesto_detectado = "IZQUIERDA"
                    elif prediccion == "1":
                        rect.x += VELOCIDAD
                        gesto_detectado = "DERECHA"
                    elif prediccion == "2":
                        rect.y -= VELOCIDAD
                        gesto_detectado = "ARRIBA"
                    elif prediccion == "3":
                        rect.y += VELOCIDAD
                        gesto_detectado = "ABAJO"
                    else:
                        gesto_detectado = f"Clase desconocida: {prediccion}"
                else:
                    gesto_detectado = "Ninguna mano en pantalla"
            else:
                gesto_detectado = "No se pudo leer la cámara"

            # Mantener el rectángulo dentro de la ventana.
            rect.clamp_ip(pygame.Rect(0, 0, ANCHO, ALTO))
            ventana.fill(FONDO)
            pygame.draw.rect(ventana, AZUL, rect)
            texto = fuente.render(f"Comando IA: {gesto_detectado}", True, NEGRO)
            ventana.blit(texto, (10, 10))
            pygame.display.flip()
            reloj.tick(30)
finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
