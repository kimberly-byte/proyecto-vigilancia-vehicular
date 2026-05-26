"""
Stream en vivo desde YouTube - Vigilancia Vehicular
Usa streamlink + PyAV para decodificar sin necesitar ffmpeg del sistema.

Ejecutar: python stream_live.py
Presiona 'q' para salir
"""

import sys
import time
import threading
import numpy as np
import cv2
import av
import streamlink

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.trajectory.trajectory_analyzer import TrajectoryAnalyzer
from src.utils.visualizer import Visualizer
from config.settings import FRAME_WIDTH, FRAME_HEIGHT


# URL del stream en vivo - Ñuñoa, Santiago (720p, nivel de calle)
YOUTUBE_URL = "https://www.youtube.com/watch?v=E9FK3QGdpqE"
STREAM_QUALITY = "720p"


def main():
    print("=" * 50)
    print("  VIGILANCIA VEHICULAR - STREAM EN VIVO")
    print("  Ñuñoa - Santiago de Chile (720p)")
    print("=" * 50)
    print(f"\nConectando a: {YOUTUBE_URL}")

    # Obtener stream con streamlink
    streams = streamlink.streams(YOUTUBE_URL)
    if not streams:
        print("[ERROR] No se encontraron streams disponibles")
        sys.exit(1)

    print(f"Calidades disponibles: {list(streams.keys())}")

    quality = STREAM_QUALITY if STREAM_QUALITY in streams else "best"
    print(f"Calidad seleccionada: {quality}")

    stream_fd = streams[quality].open()
    print("[OK] Stream conectado.")

    # Obtener la URL directa del stream para PyAV
    stream_url = streams[quality].url
    container = av.open(stream_url, options={"http_persistent": "0"})
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"

    print(f"[OK] Video: {video_stream.width}x{video_stream.height}")
    print("Procesando... Presiona 'q' para salir.\n")

    detector = VehicleDetector()
    tracker = VehicleTracker()
    analyzer = TrajectoryAnalyzer()
    visualizer = Visualizer()

    frame_count = 0
    fps_start = time.time()

    try:
        for packet in container.demux(video_stream):
            for av_frame in packet.decode():
                # Convertir frame de PyAV a numpy BGR para OpenCV
                frame = av_frame.to_ndarray(format="bgr24")

                # Redimensionar si es necesario
                h, w = frame.shape[:2]
                if w > FRAME_WIDTH:
                    scale = FRAME_WIDTH / w
                    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))

                # 1. Detectar vehiculos
                detections = detector.detect(frame)

                # 2. Tracking
                active_tracks = tracker.update(detections)

                # 3. Analizar direccion
                direction_results = analyzer.analyze_batch(active_tracks)

                # 4. Visualizar
                output_frame = visualizer.draw(frame, active_tracks, direction_results)

                # FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(
                        output_frame,
                        f"FPS: {fps:.1f} | EN VIVO - Santiago, Chile",
                        (output_frame.shape[1] - 400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Vigilancia Vehicular - Santiago EN VIVO", output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nDetenido por el usuario.")
                    raise StopIteration

    except (StopIteration, KeyboardInterrupt):
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        container.close()
        try:
            stream_fd.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"Total frames procesados: {frame_count}")


if __name__ == "__main__":
    main()
