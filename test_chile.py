"""
Test con videos de tráfico - simula 3 cámaras con diferentes ángulos.
Genera reporte completo al finalizar.
"""
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

VIDEOS = {
    "Cam 1 - Autopista": "data/input/autopista.mp4",
    "Cam 2 - Interseccion": "data/input/interseccion.mp4",
    "Cam 3 - Flujo": "data/input/trafico_flujo.mp4",
}

print("=" * 60)
print("  SISTEMA DE VIGILANCIA VEHICULAR - DEMO MULTI-CAMARA")
print("=" * 60)

detector = VehicleDetector()
all_log_data = []
start_time = datetime.now()
total_frames = 0

for cam_name, video_path in VIDEOS.items():
    print(f"\n{'─' * 60}")
    print(f"  Procesando: {cam_name}")
    print(f"  Archivo: {video_path}")
    print(f"{'─' * 60}")

    tracker = VehicleTracker()
    traj_analyzer = TrajectoryAnalyzer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(video_frames, 300)  # Procesar máximo 300 frames por video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolución: {w}x{h} | FPS: {fps:.0f} | Procesando: {max_frames} frames")

    cam_detections = 0
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        cam_detections += len(detections)
        active_tracks = tracker.update(detections)
        direction_results = traj_analyzer.analyze_batch(active_tracks)

        for track in active_tracks:
            direction_info = direction_results.get(
                track.id, {"direction": "DESCONOCIDO", "angle": 0, "confidence": 0}
            )
            all_log_data.append({
                "frame": total_frames + frame_count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_name,
                "track_id": track.id,
                "vehicle_type": track.class_name,
                "direction": direction_info["direction"],
                "angle": direction_info["angle"],
                "confidence": direction_info["confidence"],
            })

        if frame_count % 50 == 0:
            types = {}
            for d in detections:
                types[d["class_name"]] = types.get(d["class_name"], 0) + 1
            det_str = ", ".join([f"{v}x {k}" for k, v in types.items()]) if types else "—"
            print(f"  Frame {frame_count:>3}/{max_frames} | {det_str} | Tracks: {len(active_tracks)}")

        frame_count += 1

    total_frames += frame_count
    cap.release()
    print(f"  ✓ {cam_name}: {cam_detections} detecciones en {frame_count} frames")

print(f"\n{'=' * 60}")
print(f"  PROCESAMIENTO COMPLETO")
print(f"  Total frames: {total_frames} | Registros: {len(all_log_data)}")
print(f"{'=' * 60}")

if all_log_data:
    analyzer = TrafficAnalyzer(
        log_data=all_log_data, fps=30, total_frames=total_frames, start_time=start_time
    )
    results = analyzer.analyze()
    reporter = ReportGenerator(results, all_log_data)
    print(reporter.generate_terminal_report())
    paths = reporter.generate_all(REPORT_OUTPUT_DIR)
    print("Reportes exportados correctamente.")
else:
    print("No se detectaron vehículos.")
