"""
Procesamiento masivo de videos de intersección.
Procesa todos los MP4 de una carpeta y genera reporte consolidado.

Uso: python procesar_todos.py /ruta/a/carpeta/videos
"""

import sys
# Forzar flush inmediato de stdout
sys.stdout.reconfigure(line_buffering=True)
import os
import csv
import json
import time
import glob
from datetime import datetime, timedelta
from collections import defaultdict

import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker, Track
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer
from config.settings import TARGET_FPS


def process_video(video_path, detector, analyzer):
    """Procesa un video y retorna los resultados."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    tracker = VehicleTracker()
    Track._next_id = 1

    frame_count = 0
    processed = 0
    log_data = []

    # Procesar 1 frame por segundo (30x mas rapido)
    skip = max(1, int(fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip != 0:
            continue
        processed += 1

        # Detectar vehiculos
        detections = detector.detect(frame)

        # Tracking
        active_tracks = tracker.update(detections)

        # Analizar direccion
        direction_results = analyzer.analyze_batch(active_tracks)

        # Registrar
        for track in active_tracks:
            direction_info = direction_results.get(
                track.id, {"direction": "DESCONOCIDO", "angle": 0, "confidence": 0}
            )
            log_data.append({
                "frame": frame_count,
                "track_id": track.id,
                "vehicle_type": track.class_name,
                "direction": direction_info["direction"],
                "angle": direction_info["angle"],
                "confidence": direction_info["confidence"],
            })

    cap.release()

    # Deduplicar: un registro por track_id (tomar el último)
    vehicles = {}
    for record in log_data:
        vehicles[record["track_id"]] = record
    unique = list(vehicles.values())

    return {
        "total_frames": frame_count,
        "duration_sec": round(duration_sec, 1),
        "fps": fps,
        "vehicles": unique,
    }


def main():
    if len(sys.argv) < 2:
        print("Uso: python procesar_todos.py /ruta/a/carpeta/videos")
        sys.exit(1)

    input_path = sys.argv[1]

    # Buscar todos los MP4 recursivamente
    video_files = sorted(glob.glob(os.path.join(input_path, "**", "*.mp4"), recursive=True))

    if not video_files:
        print(f"No se encontraron videos MP4 en: {input_path}")
        sys.exit(1)

    print("=" * 65)
    print("  PROCESAMIENTO MASIVO DE VIDEOS DE INTERSECCION")
    print(f"  Videos encontrados: {len(video_files)}")
    print("=" * 65)

    detector = VehicleDetector()
    analyzer = TrajectoryAnalyzer()

    all_results = []
    all_vehicles = []
    total_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        folder = os.path.basename(os.path.dirname(video_path))
        print(f"\n[{i}/{len(video_files)}] {folder}/{filename}")

        start = time.time()
        result = process_video(video_path, detector, analyzer)
        elapsed = time.time() - start

        if result is None:
            continue

        n_vehicles = len(result["vehicles"])
        print(f"  {n_vehicles} vehiculos | {result['duration_sec']}s | procesado en {elapsed:.1f}s")

        # Agregar info del archivo
        for v in result["vehicles"]:
            v["archivo"] = filename
            v["carpeta"] = folder

        all_vehicles.extend(result["vehicles"])
        all_results.append({
            "archivo": filename,
            "carpeta": folder,
            "vehiculos": n_vehicles,
            "duracion_seg": result["duration_sec"],
            "frames": result["total_frames"],
        })

    total_elapsed = time.time() - total_start

    # --- Generar reporte consolidado ---
    print("\n" + "=" * 65)
    print("  REPORTE CONSOLIDADO")
    print("=" * 65)

    total_vehicles = len(all_vehicles)

    # Conteo por tipo
    type_counts = defaultdict(int)
    for v in all_vehicles:
        type_counts[v["vehicle_type"]] += 1

    # Conteo por direccion
    dir_counts = defaultdict(int)
    for v in all_vehicles:
        dir_counts[v["direction"]] += 1

    # Conteo por tipo + direccion
    type_dir_counts = defaultdict(lambda: defaultdict(int))
    for v in all_vehicles:
        type_dir_counts[v["vehicle_type"]][v["direction"]] += 1

    # Conteo por carpeta
    folder_counts = defaultdict(lambda: defaultdict(int))
    for v in all_vehicles:
        folder_counts[v["carpeta"]][v["direction"]] += 1

    print(f"\n  Total de videos procesados: {len(all_results)}")
    print(f"  Total de vehiculos detectados: {total_vehicles}")
    print(f"  Tiempo total de procesamiento: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Por tipo de vehiculo
    print(f"\n  {'TIPO DE VEHICULO':<20} | {'CANTIDAD':>10} | {'%':>8}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*8}")
    for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / max(total_vehicles, 1) * 100
        print(f"  {vtype:<20} | {count:>10} | {pct:>7.1f}%")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'TOTAL':<20} | {total_vehicles:>10} |  100.0%")

    # Por direccion
    print(f"\n  {'DIRECCION':<20} | {'CANTIDAD':>10} | {'%':>8}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*8}")
    for direction in ["DERECHO", "IZQUIERDA", "DERECHA", "VUELTA_EN_U", "DESCONOCIDO"]:
        count = dir_counts.get(direction, 0)
        pct = count / max(total_vehicles, 1) * 100
        print(f"  {direction:<20} | {count:>10} | {pct:>7.1f}%")

    # Tabla cruzada: tipo x direccion
    directions = ["DERECHO", "IZQUIERDA", "DERECHA", "VUELTA_EN_U", "DESCONOCIDO"]
    print(f"\n  {'TIPO':<15}", end="")
    for d in directions:
        print(f" | {d:>12}", end="")
    print(f" | {'TOTAL':>8}")
    print(f"  {'-'*15}", end="")
    for _ in directions:
        print(f"-+-{'-'*12}", end="")
    print(f"-+-{'-'*8}")

    for vtype in sorted(type_counts.keys()):
        print(f"  {vtype:<15}", end="")
        row_total = 0
        for d in directions:
            count = type_dir_counts[vtype].get(d, 0)
            row_total += count
            print(f" | {count:>12}", end="")
        print(f" | {row_total:>8}")

    # Por carpeta/ubicacion
    print(f"\n  POR UBICACION:")
    for folder in sorted(folder_counts.keys()):
        counts = folder_counts[folder]
        total_folder = sum(counts.values())
        print(f"\n  [{folder}] - {total_folder} vehiculos")
        for d in directions:
            count = counts.get(d, 0)
            if count > 0:
                pct = count / max(total_folder, 1) * 100
                print(f"    {d:<20}: {count:>6} ({pct:.1f}%)")

    # --- Exportar CSV ---
    output_dir = "data/output/reportes_masivo"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "detalle_todos_vehiculos.csv")
    if all_vehicles:
        fieldnames = ["carpeta", "archivo", "track_id", "vehicle_type", "direction", "angle", "confidence"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in all_vehicles:
                writer.writerow({k: v.get(k, "") for k in fieldnames})
        print(f"\n  CSV exportado: {csv_path}")

    # Exportar JSON resumen
    json_path = os.path.join(output_dir, "resumen_consolidado.json")
    resumen = {
        "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(all_results),
        "total_vehiculos": total_vehicles,
        "tiempo_procesamiento_seg": round(total_elapsed, 1),
        "conteo_por_tipo": dict(type_counts),
        "conteo_por_direccion": dict(dir_counts),
        "conteo_tipo_x_direccion": {k: dict(v) for k, v in type_dir_counts.items()},
        "conteo_por_ubicacion": {k: dict(v) for k, v in folder_counts.items()},
        "detalle_videos": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"  JSON exportado: {json_path}")

    print("\n" + "=" * 65)
    print("  PROCESAMIENTO COMPLETO")
    print("=" * 65)


if __name__ == "__main__":
    main()
