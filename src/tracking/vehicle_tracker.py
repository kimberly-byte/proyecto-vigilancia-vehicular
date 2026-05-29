"""
Módulo de tracking de vehículos.
Asigna IDs únicos a cada vehículo y los sigue entre frames.
Implementación basada en IoU (Intersection over Union).
"""

import numpy as np
from collections import defaultdict
from config.settings import (
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
    TRACK_IOU_THRESHOLD,
)


def compute_iou(bbox1, bbox2):
    """Calcula Intersection over Union entre dos bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class Track:
    """Representa un vehículo siendo rastreado."""
    _next_id = 1

    def __init__(self, detection):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = detection["bbox"]
        self.class_name = detection["class_name"]
        self.class_id = detection["class_id"]
        self.confidence = detection["confidence"]
        self.age = 0                # Frames desde última detección
        self.hits = 1               # Total de detecciones asociadas
        self.trajectory = []        # Lista de centros (x, y)
        self._class_votes = defaultdict(int)  # Votación de clasificación
        self._class_votes[detection["class_name"]] += 1
        self._update_center()

    def _update_center(self):
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.trajectory.append((cx, cy))

    def update(self, detection):
        """Actualiza el track con una nueva detección."""
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.age = 0
        self.hits += 1
        # Votar por la clase: el nombre más frecuente gana
        self._class_votes[detection["class_name"]] += 1
        self.class_name = max(self._class_votes, key=self._class_votes.get)
        self._update_center()

    def mark_missed(self):
        """Marca el track como no detectado en este frame."""
        self.age += 1

    @property
    def is_confirmed(self):
        return self.hits >= TRACK_MIN_HITS

    @property
    def is_dead(self):
        return self.age > TRACK_MAX_AGE


class VehicleTracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """
        Actualiza tracks con nuevas detecciones.

        Args:
            detections: lista de detecciones del detector

        Returns:
            lista de tracks activos confirmados con sus trayectorias
        """
        if not self.tracks:
            # Primer frame: crear tracks para todas las detecciones
            for det in detections:
                self.tracks.append(Track(det))
            return self._get_active_tracks()

        # Calcular matriz de IoU entre tracks existentes y nuevas detecciones
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = compute_iou(track.bbox, det["bbox"])

        # Asociación greedy por IoU
        matched_tracks = set()
        matched_detections = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < TRACK_IOU_THRESHOLD:
                break

            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            self.tracks[t_idx].update(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0

        # Marcar tracks no asociados como perdidos
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Crear nuevos tracks para detecciones no asociadas
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks.append(Track(det))

        # Eliminar tracks muertos
        self.tracks = [t for t in self.tracks if not t.is_dead]

        return self._get_active_tracks()

    def _get_active_tracks(self):
        """Retorna tracks confirmados."""
        return [t for t in self.tracks if t.is_confirmed]

    def reset(self):
        """Reinicia todos los tracks."""
        self.tracks = []
        Track._next_id = 1
