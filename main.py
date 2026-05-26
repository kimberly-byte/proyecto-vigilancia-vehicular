"""
Sistema de Vigilancia Vehicular en Tiempo Real
================================================
Detecta, clasifica y rastrea vehículos, determinando su dirección de movimiento.

Uso:
    python main.py                          # Usa webcam
    python main.py --source video.mp4       # Usa archivo de video
    python main.py --source rtsp://...      # Usa cámara IP
    python main.py --source 0 1 2           # Múltiples cámaras
"""

import argparse
import sys
import time
import csv
import os
from datetime import datetime, timedelta
import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer
from src.analysis.traffic_analyzer import TrafficAnalyzer
from src.reports.report_generator import ReportGenerator
from src.utils.visualizer import Visualizer
from config.settings import (
    CAMERA_SOURCES,
    TARGET_FPS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SAVE_OUTPUT_VIDEO,
    OUTPUT_VIDEO_PATH,
    LOG_FILE,
    REPORT_OUTPUT_DIR,
)


class TrafficSurveillance:
    def __init__(self, sources=None):
        self.sources = sources or CAMERA_SOURCES
        self.detector = VehicleDetector()
        self.trackers = {}       # Un tracker por cámara
        self.analyzer = TrajectoryAnalyzer()
        self.visualizer = Visualizer()
        self.captures = {}
        self.writers = {}
        self.video_fps = {}
        self.frame_count = 0
        self.log_data = []
        self.start_time = None

    def start(self):
        """Inicia el sistema de vigilancia."""
        print("=" * 50)
        print("  SISTEMA DE VIGILANCIA VEHICULAR")
        print("=" * 50)

        # Abrir fuentes de video
        for i, source in enumerate(self.sources):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir la fuente: {source}")
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.captures[i] = cap
            self.trackers[i] = VehicleTracker()

            # Configurar writer para guardar video
            if SAVE_OUTPUT_VIDEO:
                os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                path = OUTPUT_VIDEO_PATH.replace(".mp4", f"_cam{i}.mp4")
                writer = cv2.VideoWriter(path, fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                self.writers[i] = writer

            self.video_fps[i] = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
            print(f"[OK] Cámara {i} conectada: {source} ({self.video_fps[i]:.0f} FPS)")

        if not self.captures:
            print("[ERROR] No hay fuentes de video disponibles.")
            sys.exit(1)

        print(f"\nProcesando {len(self.captures)} fuente(s)...")
        print("Presiona 'q' para salir\n")

        self._run()

    def _run(self):
        """Loop principal de procesamiento."""
        self.start_time = datetime.now()
        try:
            while True:
                start_time = time.time()

                for cam_id, cap in list(self.captures.items()):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[AVISO] Fin del video en cámara {cam_id}")
                        self.captures.pop(cam_id)
                        continue

                    # 1. Detectar vehículos
                    detections = self.detector.detect(frame)

                    # 2. Rastrear vehículos
                    active_tracks = self.trackers[cam_id].update(detections)

                    # 3. Analizar trayectorias
                    direction_results = self.analyzer.analyze_batch(active_tracks)

                    # 4. Registrar datos
                    self._log_results(cam_id, active_tracks, direction_results)

                    # 5. Visualizar
                    output_frame = self.visualizer.draw(frame, active_tracks, direction_results)

                    # Mostrar FPS
                    fps = 1.0 / max(time.time() - start_time, 0.001)
                    cv2.putText(
                        output_frame,
                        f"FPS: {fps:.1f}",
                        (output_frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Mostrar ventana
                    window_name = f"Vigilancia - Camara {cam_id}"
                    cv2.imshow(window_name, output_frame)

                    # Guardar video
                    if cam_id in self.writers:
                        self.writers[cam_id].write(output_frame)

                self.frame_count += 1

                if not self.captures:
                    print("Todas las fuentes terminaron.")
                    break

                # Controlar velocidad de reproducción (velocidad real del video)
                elapsed = time.time() - start_time
                frame_time = 1.0 / TARGET_FPS
                wait_ms = max(1, int((frame_time - elapsed) * 1000))

                # Salir con 'q'
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    print("\nDetenido por el usuario.")
                    break

        finally:
            self._cleanup()

    def _log_results(self, cam_id, tracks, direction_results):
        """Registra resultados para análisis posterior."""
        fps = self.video_fps.get(cam_id, TARGET_FPS)
        elapsed = self.frame_count / max(fps, 1)
        timestamp = self.start_time + timedelta(seconds=elapsed)

        for track in tracks:
            direction_info = direction_results.get(
                track.id, {"direction": "DESCONOCIDO", "angle": 0, "confidence": 0}
            )
            self.log_data.append({
                "frame": self.frame_count,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_id,
                "track_id": track.id,
                "vehicle_type": track.class_name,
                "direction": direction_info["direction"],
                "angle": direction_info["angle"],
                "confidence": direction_info["confidence"],
            })

    def _save_log(self):
        """Guarda el registro en CSV."""
        if not self.log_data:
            return

        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"Registro guardado en: {LOG_FILE}")

    def _cleanup(self):
        """Libera recursos y genera reportes."""
        for cap in self.captures.values():
            cap.release()
        for writer in self.writers.values():
            writer.release()
        cv2.destroyAllWindows()
        self._save_log()

        # Generar análisis y reportes
        if self.log_data:
            avg_fps = (
                sum(self.video_fps.values()) / len(self.video_fps)
                if self.video_fps else TARGET_FPS
            )
            analyzer = TrafficAnalyzer(
                log_data=self.log_data,
                fps=avg_fps,
                total_frames=self.frame_count,
                start_time=self.start_time,
            )
            results = analyzer.analyze()

            reporter = ReportGenerator(results, self.log_data)

            # Mostrar reporte en terminal
            terminal_report = reporter.generate_terminal_report()
            print(terminal_report)

            # Exportar archivos
            paths = reporter.generate_all(REPORT_OUTPUT_DIR)
            print(f"Total de frames procesados: {self.frame_count}")
        else:
            print(f"\nTotal de frames procesados: {self.frame_count}")
            print("No se detectaron vehículos.")


def main():
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia Vehicular")
    parser.add_argument(
        "--source",
        nargs="+",
        default=None,
        help="Fuente(s) de video: archivo, URL RTSP, o índice de cámara",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ruta al modelo YOLO (default: yolov8n.pt)",
    )
    args = parser.parse_args()

    # Parsear fuentes
    sources = None
    if args.source:
        sources = []
        for s in args.source:
            try:
                sources.append(int(s))  # Índice de cámara
            except ValueError:
                sources.append(s)       # Archivo o URL

    app = TrafficSurveillance(sources=sources)
    app.start()


if __name__ == "__main__":
    main()
