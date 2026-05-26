"""
Módulo de visualización.
Dibuja bounding boxes, trayectorias y etiquetas sobre los frames.
"""

import cv2
import numpy as np
from config.settings import (
    BOX_COLORS,
    SHOW_BOUNDING_BOXES,
    SHOW_TRAJECTORIES,
    SHOW_VEHICLE_CLASS,
    SHOW_DIRECTION,
)


class Visualizer:
    def draw(self, frame, tracks, direction_results):
        """
        Dibuja toda la información visual sobre el frame.

        Args:
            frame: imagen BGR
            tracks: lista de tracks activos
            direction_results: dict {track_id: resultado_análisis}

        Returns:
            frame con visualización
        """
        overlay = frame.copy()

        for track in tracks:
            color = BOX_COLORS.get(track.class_name, (255, 255, 255))
            x1, y1, x2, y2 = track.bbox

            # Bounding box
            if SHOW_BOUNDING_BOXES:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Trayectoria
            if SHOW_TRAJECTORIES and len(track.trajectory) > 1:
                points = np.array(track.trajectory, dtype=np.int32)
                cv2.polylines(overlay, [points], False, color, 2)
                # Punto actual
                cv2.circle(overlay, track.trajectory[-1], 4, color, -1)

            # Etiqueta
            label_parts = [f"ID:{track.id}"]

            if SHOW_VEHICLE_CLASS:
                label_parts.append(track.class_name.upper())

            if SHOW_DIRECTION and track.id in direction_results:
                direction = direction_results[track.id]["direction"]
                if direction != "DESCONOCIDO":
                    label_parts.append(direction)

            label = " | ".join(label_parts)

            # Fondo de la etiqueta
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        # Panel de estadísticas
        self._draw_stats_panel(overlay, tracks, direction_results)

        return overlay

    def _draw_stats_panel(self, frame, tracks, direction_results):
        """Dibuja un panel con estadísticas en la esquina superior izquierda."""
        # Contar vehículos por tipo
        vehicle_counts = {}
        for track in tracks:
            vehicle_counts[track.class_name] = vehicle_counts.get(track.class_name, 0) + 1

        # Contar direcciones
        direction_counts = {"DERECHO": 0, "IZQUIERDA": 0, "DERECHA": 0, "VUELTA_EN_U": 0}
        for result in direction_results.values():
            d = result["direction"]
            if d in direction_counts:
                direction_counts[d] += 1

        # Dibujar panel semitransparente
        panel_h = 30 + len(vehicle_counts) * 22 + len(direction_counts) * 22 + 40
        panel_w = 250
        sub = frame[10:10 + panel_h, 10:10 + panel_w]
        black_rect = np.zeros(sub.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub, 0.4, black_rect, 0.6, 0)
        frame[10:10 + panel_h, 10:10 + panel_w] = res

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30

        # Total
        cv2.putText(frame, f"Vehiculos: {len(tracks)}", (20, y), font, 0.5, (255, 255, 255), 1)
        y += 25

        # Por tipo
        for vtype, count in vehicle_counts.items():
            color = BOX_COLORS.get(vtype, (255, 255, 255))
            cv2.putText(frame, f"  {vtype}: {count}", (20, y), font, 0.45, color, 1)
            y += 22

        y += 10
        cv2.putText(frame, "Direcciones:", (20, y), font, 0.45, (255, 255, 255), 1)
        y += 22

        for direction, count in direction_counts.items():
            cv2.putText(frame, f"  {direction}: {count}", (20, y), font, 0.45, (200, 200, 200), 1)
            y += 22
