"""
Módulo de generación de reportes.
Exporta resultados del análisis en CSV, JSON y texto para terminal.
"""

import csv
import json
import os
from config.settings import (
    REPORT_CSV_FILE,
    REPORT_JSON_FILE,
    REPORT_TXT_FILE,
    CHILEAN_VEHICLE_NAMES,
)


class ReportGenerator:
    def __init__(self, analysis_results, log_data):
        """
        Args:
            analysis_results: dict retornado por TrafficAnalyzer.analyze()
            log_data: datos crudos de detección para el CSV detallado
        """
        self.results = analysis_results
        self.log_data = log_data
        self.vehicle_types = sorted(CHILEAN_VEHICLE_NAMES.keys())

    def generate_all(self, output_dir=None):
        """Genera todos los formatos de reporte."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        csv_path = REPORT_CSV_FILE
        json_path = REPORT_JSON_FILE
        txt_path = REPORT_TXT_FILE

        self.generate_csv(csv_path)
        self.generate_json(json_path)
        self.save_terminal_report(txt_path)

        return {
            "csv": os.path.abspath(csv_path),
            "json": os.path.abspath(json_path),
            "txt": os.path.abspath(txt_path),
        }

    def generate_csv(self, filepath):
        """
        Genera CSV detallado compatible con Excel chileno.
        Usa separador ';' y BOM UTF-8 para acentos.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # CSV detallado
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "frame", "timestamp", "camara", "id_vehiculo",
                "tipo_vehiculo", "direccion", "angulo", "confianza",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()

            for record in self.log_data:
                writer.writerow({
                    "frame": record["frame"],
                    "timestamp": record.get("timestamp", ""),
                    "camara": record["camera"],
                    "id_vehiculo": record["track_id"],
                    "tipo_vehiculo": record["vehicle_type"],
                    "direccion": record["direction"],
                    "angulo": record["angle"],
                    "confianza": record["confidence"],
                })

        # CSV resumen (archivo separado)
        summary_path = filepath.replace(".csv", "_resumen.csv")
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Tipo Vehículo", "Cantidad", "Porcentaje"])

            conteo = self.results["conteo_por_tipo"]
            total = sum(conteo.values())
            for vtype, count in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
                pct = round(count / max(total, 1) * 100, 1)
                writer.writerow([vtype, count, f"{pct}%"])
            writer.writerow(["TOTAL", total, "100.0%"])

    def generate_json(self, filepath):
        """Genera JSON con el análisis completo."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def generate_terminal_report(self):
        """Genera reporte formateado para imprimir en terminal."""
        meta = self.results["metadata"]
        resumen = self.results["resumen"]
        conteo = self.results["conteo_por_tipo"]
        direcciones = self.results["distribucion_direcciones"]
        total = resumen["total_vehiculos"]

        lines = []
        sep = "=" * 64

        # Encabezado
        lines.append("")
        lines.append(sep)
        lines.append("  REPORTE DE TRAFICO VEHICULAR")
        lines.append(f"  Municipalidad: {meta['municipalidad']}")
        lines.append(f"  Ubicacion: {meta['ubicacion']}")
        lines.append(f"  Fecha: {meta['fecha_analisis']}  |  Duracion: {meta['duracion_video']}")
        lines.append(sep)

        # Resumen general
        lines.append("")
        lines.append("  RESUMEN GENERAL")
        lines.append(f"  Total vehiculos detectados:    {total}")
        lines.append(f"  Vehiculos por minuto (prom):   {resumen['vehiculos_por_minuto']}")
        lines.append(f"  Tipo predominante:             {resumen['tipo_predominante'].upper()} ({resumen['tipo_predominante_pct']}%)")
        lines.append(f"  Direccion predominante:        {resumen['direccion_predominante']}")

        # Conteo por tipo
        lines.append("")
        lines.append("  CONTEO POR TIPO DE VEHICULO")
        lines.append(f"  {'Tipo':<14}| {'Cantidad':>8} | {'Porcentaje':>10}")
        lines.append(f"  {'-'*14}|{'-'*10}|{'-'*12}")

        for vtype, count in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
            pct = round(count / max(total, 1) * 100, 1)
            display_name = CHILEAN_VEHICLE_NAMES.get(vtype, vtype.capitalize())
            lines.append(f"  {display_name:<14}| {count:>8} | {pct:>9.1f}%")

        lines.append(f"  {'-'*14}|{'-'*10}|{'-'*12}")
        lines.append(f"  {'TOTAL':<14}| {total:>8} | {'100.0':>9}%")

        # Distribución de direcciones
        lines.append("")
        lines.append("  DISTRIBUCION DE DIRECCIONES")
        lines.append(f"  {'Direccion':<14}| {'Cantidad':>8} | {'Porcentaje':>10}")
        lines.append(f"  {'-'*14}|{'-'*10}|{'-'*12}")

        for direction, data in sorted(
            direcciones.items(), key=lambda x: x[1]["cantidad"], reverse=True
        ):
            lines.append(
                f"  {direction:<14}| {data['cantidad']:>8} | {data['porcentaje']:>9.1f}%"
            )

        # Periodos punta (usando intervalo de 5 minutos)
        punta_key = "5min"
        if punta_key in self.results["periodos_punta"]:
            punta = self.results["periodos_punta"][punta_key]
            if punta:
                lines.append("")
                lines.append(f"  PERIODOS DE MAYOR TRAFICO (cada 5 min)")
                for i, p in enumerate(punta, 1):
                    lines.append(
                        f"  #{i}  {p['periodo']}  ->  {p['vehiculos']} vehiculos"
                    )

        # Tabla cruzada tipo × periodo (top 10 periodos de 5min)
        tipo_periodo_key = "5min"
        if tipo_periodo_key in self.results["tipo_por_periodo"]:
            tipo_periodo = self.results["tipo_por_periodo"][tipo_periodo_key]
            if tipo_periodo:
                lines.append("")
                lines.append(f"  DETALLE POR PERIODO (cada 5 min) - Top 10")

                # Header
                header_types = self.vehicle_types
                header = f"  {'Periodo':<14}"
                for vt in header_types:
                    name = CHILEAN_VEHICLE_NAMES.get(vt, vt)[:7]
                    header += f"| {name:>7} "
                header += f"| {'Total':>6}"
                lines.append(header)
                lines.append(f"  {'-'*14}" + f"|{'-'*9}" * len(header_types) + f"|{'-'*8}")

                # Filas ordenadas por total descendente
                sorted_periods = sorted(
                    tipo_periodo.items(),
                    key=lambda x: sum(x[1].values()),
                    reverse=True,
                )[:10]

                for period, vtypes in sorted_periods:
                    row = f"  {period:<14}"
                    period_total = 0
                    for vt in header_types:
                        count = vtypes.get(vt, 0)
                        period_total += count
                        row += f"| {count:>7} "
                    row += f"| {period_total:>6}"
                    lines.append(row)

        # Footer
        lines.append("")
        lines.append(sep)
        lines.append("  Archivos exportados:")
        lines.append(f"  - CSV:  {REPORT_CSV_FILE}")
        lines.append(f"  - JSON: {REPORT_JSON_FILE}")
        lines.append(f"  - TXT:  {REPORT_TXT_FILE}")
        lines.append(sep)
        lines.append("")

        return "\n".join(lines)

    def save_terminal_report(self, filepath):
        """Guarda el reporte terminal en un archivo .txt."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        report = self.generate_terminal_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
