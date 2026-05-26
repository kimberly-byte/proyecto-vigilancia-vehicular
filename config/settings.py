"""
Configuración principal del sistema de vigilancia vehicular.
"""

# --- Modelo de detección ---
YOLO_MODEL = "yolov8s.pt"  # Small: mejor precisión en clasificación de vehículos
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.45

# --- Clases de vehículos (COCO dataset IDs) ---
VEHICLE_CLASSES = {
    1: "bicicleta",
    2: "auto",
    3: "moto",
    5: "bus",       # Se subclasifica en: bus, micro_red, colectivo
    7: "camion",
}

# --- Clasificación chilena (subclasificación de buses) ---
# Un bus detectado por YOLO se reclasifica según color y tamaño:
#   Rojo + grande  → micro_red  (bus del sistema RED de Santiago)
#   Otro + pequeño → colectivo  (taxi colectivo / minibus)
#   Otro + grande  → bus        (bus interurbano o privado)
CHILEAN_VEHICLE_NAMES = {
    "auto": "Auto",
    "moto": "Moto",
    "bus": "Bus",
    "micro_red": "Micro RED",
    "colectivo": "Colectivo",
    "camion": "Camión",
    "bicicleta": "Bicicleta",
}

# --- Tracking ---
TRACK_MAX_AGE = 30          # Frames antes de eliminar un track perdido
TRACK_MIN_HITS = 3          # Detecciones mínimas para confirmar un track
TRACK_IOU_THRESHOLD = 0.3   # IOU mínimo para asociar detección con track

# --- Análisis de trayectoria ---
TRAJECTORY_MIN_POINTS = 10          # Puntos mínimos para analizar dirección
ANGLE_STRAIGHT_THRESHOLD = 15.0     # Grados: < 15° = sigue derecho
ANGLE_TURN_THRESHOLD = 30.0         # Grados: > 30° = giro confirmado
ANGLE_UTURN_THRESHOLD = 140.0       # Grados: > 140° = vuelta en U

# --- Video / Cámaras ---
CAMERA_SOURCES = [
    0,  # Webcam por defecto
    # "rtsp://usuario:password@192.168.1.100:554/stream",  # Cámara IP ejemplo
    # "/ruta/al/video.mp4",  # Video local
]
TARGET_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Visualización ---
SHOW_BOUNDING_BOXES = True
SHOW_TRAJECTORIES = True
SHOW_VEHICLE_CLASS = True
SHOW_DIRECTION = True
BOX_COLORS = {
    "auto": (0, 255, 0),        # Verde
    "moto": (255, 255, 0),      # Cyan
    "bus": (0, 165, 255),       # Naranja
    "micro_red": (0, 0, 255),   # Rojo (micros RED)
    "colectivo": (255, 0, 255), # Magenta (colectivos)
    "camion": (255, 0, 0),      # Azul
    "bicicleta": (0, 255, 255), # Amarillo
}

# --- Análisis de tráfico ---
ANALYSIS_INTERVAL_MINUTES = [1, 5, 15, 60]  # Intervalos de agrupación en minutos
PEAK_PERIOD_TOP_N = 3                        # Cuántos periodos punta mostrar

# --- Reportes ---
REPORT_OUTPUT_DIR = "data/output/reportes"
REPORT_CSV_FILE = "data/output/reportes/detalle_vehiculos.csv"
REPORT_JSON_FILE = "data/output/reportes/resumen_estadistico.json"
REPORT_TXT_FILE = "data/output/reportes/reporte_terminal.txt"

# --- Metadata municipalidad ---
MUNICIPALIDAD_NOMBRE = "Municipalidad"
UBICACION_CAMARA = "Intersección principal"

# --- Salida ---
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = "data/output/resultado.mp4"
LOG_FILE = "data/output/registro_vehiculos.csv"
