"""Test con video de tráfico de ejemplo."""
import sys
import cv2
from datetime import datetime
sys.path.insert(0, ".")

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer
from src.analysis.traffic_analyzer import TrafficAnalyzer
from src.reports.report_generator import ReportGenerator
from config.settings import TARGET_FPS, REPORT_OUTPUT_DIR

VIDEO = "data/input/traffic_sample.mp4"

print("=" * 50)
print("  TEST CON VIDEO DE TRAFICO")
print("=" * 50)

print("Cargando modelo YOLOv8...")
detector = VehicleDetector()
tracker = VehicleTracker()
traj_analyzer = TrajectoryAnalyzer()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: No se pudo abrir {VIDEO}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO}")
print(f"FPS: {fps:.0f} | Frames totales: {total_video_frames}")
print(f"Duración: {total_video_frames/fps:.1f} segundos")
print("-" * 50)

log_data = []
start_time = datetime.now()
frame_count = 0

while True:
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

    if frame_count % 30 == 0:
        print(f"  Frame {frame_count}/{total_video_frames} | Detecciones: {len(detections)} | Tracks activos: {len(active_tracks)} | Registros: {len(log_data)}")

    frame_count += 1

cap.release()

print("-" * 50)
print(f"Procesamiento completo: {frame_count} frames")
print(f"Total registros: {len(log_data)}")

if log_data:
    analyzer = TrafficAnalyzer(log_data=log_data, fps=fps, total_frames=frame_count, start_time=start_time)
    results = analyzer.analyze()
    reporter = ReportGenerator(results, log_data)
    print(reporter.generate_terminal_report())
    paths = reporter.generate_all(REPORT_OUTPUT_DIR)
    print("Archivos de reporte generados correctamente.")
else:
    print("No se detectaron vehículos en el video.")
