"""Test rápido: captura 30 frames de webcam, detecta vehículos, genera reporte."""
import sys
import time
import cv2
sys.path.insert(0, ".")

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer
from src.analysis.traffic_analyzer import TrafficAnalyzer
from src.reports.report_generator import ReportGenerator
from config.settings import TARGET_FPS, REPORT_OUTPUT_DIR
from datetime import datetime

print("Cargando modelo YOLOv8...")
detector = VehicleDetector()
tracker = VehicleTracker()
traj_analyzer = TrajectoryAnalyzer()

print("Abriendo webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No se pudo abrir la webcam")
    sys.exit(1)

log_data = []
start_time = datetime.now()
MAX_FRAMES = 60  # ~2 segundos

print(f"Capturando {MAX_FRAMES} frames...")
for frame_count in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    active_tracks = tracker.update(detections)
    direction_results = traj_analyzer.analyze_batch(active_tracks)

    for track in active_tracks:
        if track.id in direction_results:
            result = direction_results[track.id]
            if result["direction"] != "DESCONOCIDO":
                log_data.append({
                    "frame": frame_count,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera": 0,
                    "track_id": track.id,
                    "vehicle_type": track.class_name,
                    "direction": result["direction"],
                    "angle": result["angle"],
                    "confidence": result["confidence"],
                })

    if frame_count % 10 == 0:
        print(f"  Frame {frame_count}/{MAX_FRAMES} - Detecciones: {len(detections)} - Tracks: {len(active_tracks)}")

cap.release()

print(f"\nFrames procesados: {MAX_FRAMES}")
print(f"Registros en log: {len(log_data)}")

if log_data:
    analyzer = TrafficAnalyzer(log_data=log_data, fps=TARGET_FPS, total_frames=MAX_FRAMES, start_time=start_time)
    results = analyzer.analyze()
    reporter = ReportGenerator(results, log_data)
    print(reporter.generate_terminal_report())
    paths = reporter.generate_all(REPORT_OUTPUT_DIR)
    print("Archivos generados correctamente.")
else:
    print("\nNo se detectaron vehículos (es normal si la webcam no apunta a tráfico).")
    print("Tip: apunta la cámara a tu celular mostrando un video de YouTube de tráfico.")

print("\nTest completado.")
