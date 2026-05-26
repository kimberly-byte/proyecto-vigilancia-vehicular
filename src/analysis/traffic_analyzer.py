"""
Módulo de análisis de tráfico vehicular.
Procesa los datos crudos de detección/tracking y genera estadísticas.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from config.settings import (
    ANALYSIS_INTERVAL_MINUTES,
    PEAK_PERIOD_TOP_N,
    MUNICIPALIDAD_NOMBRE,
    UBICACION_CAMARA,
)


class TrafficAnalyzer:
    def __init__(self, log_data, fps=30, total_frames=0, start_time=None):
        """
        Args:
            log_data: lista de dicts con {frame, camera, track_id, vehicle_type,
                      direction, angle, confidence, timestamp}
            fps: frames por segundo del video
            total_frames: total de frames procesados
            start_time: datetime de inicio del procesamiento
        """
        self.log_data = log_data
        self.fps = fps
        self.total_frames = total_frames
        self.start_time = start_time or datetime.now()

        # Deduplicar: un registro por track_id (tomar el último, más confiable)
        self.unique_vehicles = self._deduplicate()

    def _deduplicate(self):
        """
        Un track_id aparece en múltiples frames.
        Toma el último registro de cada track (tiene más datos de trayectoria).
        """
        vehicles = {}
        for record in self.log_data:
            tid = record["track_id"]
            # Siempre sobrescribe: el último frame tiene la info más completa
            vehicles[tid] = record
        return list(vehicles.values())

    def analyze(self):
        """Ejecuta todo el análisis y retorna un dict con los resultados."""
        duration_seconds = self.total_frames / max(self.fps, 1)

        results = {
            "metadata": {
                "fecha_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duracion_video": self._format_duration(duration_seconds),
                "duracion_segundos": round(duration_seconds, 1),
                "total_frames": self.total_frames,
                "fps": self.fps,
                "municipalidad": MUNICIPALIDAD_NOMBRE,
                "ubicacion": UBICACION_CAMARA,
            },
            "resumen": self._summary_stats(duration_seconds),
            "conteo_por_tipo": self._count_by_vehicle_type(),
            "distribucion_direcciones": self._direction_distribution(),
            "conteo_por_periodo": {},
            "periodos_punta": {},
            "tipo_por_periodo": {},
        }

        # Análisis por cada intervalo de tiempo configurado
        for interval_min in ANALYSIS_INTERVAL_MINUTES:
            key = f"{interval_min}min"
            results["conteo_por_periodo"][key] = self._count_by_time_period(interval_min)
            results["periodos_punta"][key] = self._find_peak_periods(interval_min)
            results["tipo_por_periodo"][key] = self._vehicle_type_by_period(interval_min)

        return results

    def _count_by_vehicle_type(self):
        """Cuenta vehículos únicos por tipo."""
        counts = defaultdict(int)
        for v in self.unique_vehicles:
            counts[v["vehicle_type"]] += 1
        return dict(counts)

    def _count_by_time_period(self, interval_minutes):
        """
        Agrupa vehículos únicos por periodo de tiempo.
        Calcula el periodo a partir del número de frame y FPS.
        """
        period_counts = defaultdict(set)  # periodo -> set de track_ids

        for record in self.log_data:
            elapsed_seconds = record["frame"] / max(self.fps, 1)
            period_idx = int(elapsed_seconds // (interval_minutes * 60))
            period_start = period_idx * interval_minutes
            period_label = self._minutes_to_label(period_start, interval_minutes)
            period_counts[period_label].add(record["track_id"])

        # Convertir sets a conteos
        return {k: len(v) for k, v in sorted(period_counts.items())}

    def _direction_distribution(self):
        """Distribución porcentual de direcciones."""
        total = len(self.unique_vehicles)
        if total == 0:
            return {}

        counts = defaultdict(int)
        for v in self.unique_vehicles:
            counts[v["direction"]] += 1

        return {
            direction: {
                "cantidad": count,
                "porcentaje": round(count / total * 100, 1),
            }
            for direction, count in counts.items()
        }

    def _find_peak_periods(self, interval_minutes):
        """Encuentra los N periodos con más tráfico."""
        period_data = self._count_by_time_period(interval_minutes)
        sorted_periods = sorted(period_data.items(), key=lambda x: x[1], reverse=True)
        return [
            {"periodo": period, "vehiculos": count}
            for period, count in sorted_periods[:PEAK_PERIOD_TOP_N]
        ]

    def _vehicle_type_by_period(self, interval_minutes):
        """Tabla cruzada: tipo de vehículo × periodo de tiempo."""
        table = defaultdict(lambda: defaultdict(set))

        for record in self.log_data:
            elapsed_seconds = record["frame"] / max(self.fps, 1)
            period_idx = int(elapsed_seconds // (interval_minutes * 60))
            period_start = period_idx * interval_minutes
            period_label = self._minutes_to_label(period_start, interval_minutes)
            table[period_label][record["vehicle_type"]].add(record["track_id"])

        # Convertir sets a conteos
        return {
            period: {vtype: len(ids) for vtype, ids in vtypes.items()}
            for period, vtypes in sorted(table.items())
        }

    def _summary_stats(self, duration_seconds):
        """Estadísticas globales de resumen."""
        total = len(self.unique_vehicles)
        duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 1

        # Tipo más frecuente
        type_counts = self._count_by_vehicle_type()
        tipo_predominante = max(type_counts, key=type_counts.get) if type_counts else "N/A"

        # Dirección predominante
        dir_dist = self._direction_distribution()
        dir_predominante = max(
            dir_dist, key=lambda d: dir_dist[d]["cantidad"]
        ) if dir_dist else "N/A"

        return {
            "total_vehiculos": total,
            "vehiculos_por_minuto": round(total / duration_minutes, 1),
            "tipo_predominante": tipo_predominante,
            "tipo_predominante_pct": round(
                type_counts.get(tipo_predominante, 0) / max(total, 1) * 100, 1
            ),
            "direccion_predominante": dir_predominante,
        }

    # --- Utilidades ---

    def _format_duration(self, seconds):
        """Formatea segundos como HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _minutes_to_label(self, start_minutes, interval):
        """Convierte minutos a etiqueta de periodo."""
        h1, m1 = divmod(int(start_minutes), 60)
        h2, m2 = divmod(int(start_minutes + interval), 60)
        return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
