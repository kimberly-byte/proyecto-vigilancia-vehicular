"""
Módulo de detección de vehículos usando YOLOv8.
Detecta y clasifica vehículos en cada frame.
"""

from ultralytics import YOLO
import numpy as np
from config.settings import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    VEHICLE_CLASSES,
)
from src.detection.color_classifier import ColorClassifier


class VehicleDetector:
    def __init__(self, model_path=None):
        model_path = model_path or YOLO_MODEL
        self.model = YOLO(model_path)
        self.vehicle_class_ids = set(VEHICLE_CLASSES.keys())

    def detect(self, frame):
        """
        Detecta vehículos en un frame.

        Args:
            frame: imagen BGR (numpy array)

        Returns:
            lista de detecciones, cada una con:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str
            }
        """
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in self.vehicle_class_ids:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                class_name = VEHICLE_CLASSES[class_id]

                # Subclasificar buses en categorías chilenas
                if class_id == 5:
                    class_name = ColorClassifier.classify_bus(frame, bbox)

                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                })

        return detections

    def detect_batch(self, frames):
        """Detecta vehículos en múltiples frames (batch processing)."""
        all_detections = []
        results = self.model(
            frames,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id not in self.vehicle_class_ids:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])

                    frame_detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": VEHICLE_CLASSES[class_id],
                    })
            all_detections.append(frame_detections)

        return all_detections
