"""
Demo en tiempo real - Detección de vehículos con webcam
Ejecutar: python demo_realtime.py
Presiona 'q' para salir
"""

import cv2
from ultralytics import YOLO

# Cargar modelo pre-entrenado
model = YOLO("yolov8n.pt")

# Clases de vehículos que nos interesan
VEHICULOS = {
    2: ("Auto", (0, 255, 0)),        # Verde
    3: ("Moto", (255, 255, 0)),      # Cyan
    5: ("Bus", (0, 165, 255)),       # Naranja
    7: ("Camion", (0, 0, 255)),      # Rojo
    1: ("Bicicleta", (255, 0, 255)), # Magenta
}

# Abrir webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la webcam")
    exit()

print("=" * 40)
print("  DEMO DETECCION DE VEHICULOS")
print("  Presiona 'q' para salir")
print("=" * 40)

conteo = {name: 0 for _, (name, _) in VEHICULOS.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos
    results = model(frame, conf=0.5, verbose=False)

    # Contador por frame
    conteo_frame = {name: 0 for _, (name, _) in VEHICULOS.items()}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])

            # Solo mostrar vehículos
            if class_id not in VEHICULOS:
                continue

            nombre, color = VEHICULOS[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta
            label = f"{nombre} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            conteo_frame[nombre] += 1

    # Panel de conteo
    y = 30
    cv2.rectangle(frame, (5, 5), (200, 35 + len(VEHICULOS) * 25), (0, 0, 0), -1)
    cv2.putText(frame, "VEHICULOS DETECTADOS", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 25

    for class_id, (nombre, color) in VEHICULOS.items():
        cantidad = conteo_frame[nombre]
        cv2.putText(frame, f"{nombre}: {cantidad}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 25

    # Mostrar
    cv2.imshow("Deteccion de Vehiculos - Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Demo finalizada.")
