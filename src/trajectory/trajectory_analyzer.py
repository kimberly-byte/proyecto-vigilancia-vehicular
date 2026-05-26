"""
Módulo de análisis de trayectoria.
Determina la dirección de movimiento de cada vehículo:
- DERECHO: el vehículo sigue en línea recta
- IZQUIERDA: el vehículo dobla a la izquierda
- DERECHA: el vehículo dobla a la derecha
"""

import math
import numpy as np
from config.settings import (
    TRAJECTORY_MIN_POINTS,
    ANGLE_STRAIGHT_THRESHOLD,
    ANGLE_TURN_THRESHOLD,
    ANGLE_UTURN_THRESHOLD,
)


class Direction:
    UNKNOWN = "DESCONOCIDO"
    STRAIGHT = "DERECHO"
    LEFT = "IZQUIERDA"
    RIGHT = "DERECHA"
    UTURN = "VUELTA_EN_U"


class TrajectoryAnalyzer:
    def analyze(self, trajectory):
        """
        Analiza la trayectoria de un vehículo y determina su dirección.

        Args:
            trajectory: lista de puntos (x, y) del centro del vehículo

        Returns:
            {
                "direction": str,
                "angle": float,
                "confidence": float
            }
        """
        if len(trajectory) < TRAJECTORY_MIN_POINTS:
            return {
                "direction": Direction.UNKNOWN,
                "angle": 0.0,
                "confidence": 0.0,
            }

        # Suavizar la trayectoria para reducir ruido
        smoothed = self._smooth_trajectory(trajectory)

        # Calcular el ángulo de cambio de dirección
        angle = self._calculate_turn_angle(smoothed)

        # Determinar dirección basada en el ángulo
        direction = self._classify_direction(angle)

        # Calcular confianza basada en la cantidad de puntos
        confidence = min(1.0, len(trajectory) / (TRAJECTORY_MIN_POINTS * 3))

        return {
            "direction": direction,
            "angle": round(angle, 2),
            "confidence": round(confidence, 2),
        }

    def _smooth_trajectory(self, trajectory, window=5):
        """Suaviza la trayectoria usando media móvil."""
        if len(trajectory) < window:
            return trajectory

        points = np.array(trajectory, dtype=float)
        smoothed = []
        for i in range(len(points)):
            start = max(0, i - window // 2)
            end = min(len(points), i + window // 2 + 1)
            smoothed.append(points[start:end].mean(axis=0))

        return [(int(p[0]), int(p[1])) for p in smoothed]

    def _calculate_turn_angle(self, trajectory):
        """
        Calcula el ángulo de giro comparando la dirección inicial
        con la dirección final de la trayectoria.

        Retorna ángulo en grados:
        - Positivo = giro a la derecha
        - Negativo = giro a la izquierda
        """
        n = len(trajectory)
        segment_size = max(3, n // 4)

        # Vector de dirección inicial (primeros puntos)
        start_points = trajectory[:segment_size]
        v_start = (
            start_points[-1][0] - start_points[0][0],
            start_points[-1][1] - start_points[0][1],
        )

        # Vector de dirección final (últimos puntos)
        end_points = trajectory[-segment_size:]
        v_end = (
            end_points[-1][0] - end_points[0][0],
            end_points[-1][1] - end_points[0][1],
        )

        # Calcular ángulo entre los dos vectores
        angle = self._angle_between_vectors(v_start, v_end)

        # Determinar signo (izquierda/derecha) usando producto cruzado
        cross = v_start[0] * v_end[1] - v_start[1] * v_end[0]
        if cross < 0:
            angle = -angle  # Giro a la izquierda

        return angle

    def _angle_between_vectors(self, v1, v2):
        """Calcula el ángulo en grados entre dos vectores 2D."""
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))

        return math.degrees(math.acos(cos_angle))

    def _classify_direction(self, angle):
        """Clasifica la dirección basada en el ángulo de giro."""
        abs_angle = abs(angle)

        if abs_angle >= ANGLE_UTURN_THRESHOLD:
            return Direction.UTURN
        elif abs_angle < ANGLE_STRAIGHT_THRESHOLD:
            return Direction.STRAIGHT
        elif abs_angle >= ANGLE_TURN_THRESHOLD:
            return Direction.LEFT if angle < 0 else Direction.RIGHT
        else:
            # Zona intermedia: probablemente derecho con algo de desviación
            return Direction.STRAIGHT

    def analyze_batch(self, tracks):
        """
        Analiza la trayectoria de múltiples vehículos.

        Args:
            tracks: lista de objetos Track con atributo .trajectory

        Returns:
            dict {track_id: resultado_análisis}
        """
        results = {}
        for track in tracks:
            results[track.id] = self.analyze(track.trajectory)
        return results
