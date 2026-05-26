"""
Clasificador de color para vehículos chilenos.
Distingue micros RED (rojos) de buses normales y colectivos.
"""

import cv2
import numpy as np


# Umbral mínimo de área del bbox para considerar un bus (vs colectivo)
BUS_MIN_AREA_RATIO = 0.02  # 2% del frame


class ColorClassifier:
    # Rangos HSV para rojo (el rojo cruza 0° en HSV, se necesitan 2 rangos)
    RED_LOWER_1 = np.array([0, 80, 80])
    RED_UPPER_1 = np.array([12, 255, 255])
    RED_LOWER_2 = np.array([168, 80, 80])
    RED_UPPER_2 = np.array([180, 255, 255])

    # Umbral: si más del 20% de píxeles son rojos, es micro RED
    RED_THRESHOLD = 0.15

    @staticmethod
    def classify_bus(frame, bbox):
        """
        Subclasifica un bus detectado por YOLO en categoría chilena.

        Args:
            frame: imagen BGR completa
            bbox: [x1, y1, x2, y2]

        Returns:
            "micro_red" si es rojo (bus del sistema RED)
            "colectivo" si es pequeño
            "bus" si es bus normal
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        # Asegurar que el bbox está dentro del frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "bus"

        # Verificar si es rojo (micro RED)
        if ColorClassifier._is_red(roi):
            return "micro_red"

        # Clasificar por tamaño: colectivo vs bus
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = w_frame * h_frame
        area_ratio = bbox_area / frame_area

        if area_ratio < BUS_MIN_AREA_RATIO:
            return "colectivo"

        return "bus"

    @staticmethod
    def _is_red(roi):
        """Detecta si la región tiene color rojo dominante."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, ColorClassifier.RED_LOWER_1, ColorClassifier.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, ColorClassifier.RED_LOWER_2, ColorClassifier.RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)

        red_ratio = np.count_nonzero(mask) / mask.size
        return red_ratio > ColorClassifier.RED_THRESHOLD
