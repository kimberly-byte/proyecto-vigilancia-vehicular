"""Test con video - registra TODAS las detecciones incluyendo DESCONOCIDO."""
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
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO} | FPS: {fps:.0f} | Frames: {total_video_frames} | Duración: {total_video_frames/fps:.1f}s")
print("-" * 50)

log_data = []
start_time = datetime.now()
frame_count = 0
total_detections = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    total_detections += len(detections)
    active_tracks = tracker.update(detections)
    direction_results = traj_analyzer.analyze_batch(active_tracks)

    # Registrar TODOS los tracks activos (incluyendo DESCONOCIDO)
    for track in active_tracks:
        direction_info = direction_results.get(track.id, {"direction": "DESCONOCIDO", "angle": 0, "confidence": 0})
        log_data.append({
            "frame": frame_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": 0,
            "track_id": track.id,
            "vehicle_type": track.class_name,
            "direction": direction_info["direction"],
            "angle": direction_info["angle"],
            "confidence": direction_info["confidence"],
        })

    if frame_count % 30 == 0:
        det_str = ", ".join([f"{d['class_name']}({d['confidence']:.0%})" for d in detections]) if detections else "ninguna"
        print(f"  Frame {frame_count:>3}/{total_video_frames} | Det: {det_str} | Tracks: {len(active_tracks)}")

    frame_count += 1

cap.release()

print("-" * 50)
print(f"Frames: {frame_count} | Detecciones totales: {total_detections} | Registros: {len(log_data)}")

if log_data:
    analyzer = TrafficAnalyzer(log_data=log_data, fps=fps, total_frames=frame_count, start_time=start_time)
    results = analyzer.analyze()
    reporter = ReportGenerator(results, log_data)
    print(reporter.generate_terminal_report())
    paths = reporter.generate_all(REPORT_OUTPUT_DIR)
    print("Reportes generados en data/output/reportes/")
else:
    print("No se detectaron vehículos.")
