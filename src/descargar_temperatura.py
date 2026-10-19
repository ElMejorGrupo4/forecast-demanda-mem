"""
Descarga la temperatura horaria historica de Buenos Aires desde Open-Meteo
(API abierta, sin API key) para el mismo periodo que la serie de demanda.

Fuente: Open-Meteo Historical Weather API (reanalisis ERA5).
    https://archive-api.open-meteo.com/v1/archive
Punto: coordenadas de Aeroparque (CABA), la misma estacion que usa el dataset
regional para GBA. GBA + Buenos Aires son ~50 % de la demanda del MEM.

Salida:
    data/processed/temperatura_horaria_ba.csv   columnas: timestamp, temperatura_c

Uso:
    python src/descargar_temperatura.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
DEMANDA = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA = BASE / "data" / "processed" / "temperatura_horaria_ba.csv"

URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUD, LONGITUD = -34.56, -58.42


def main() -> None:
    demanda = pd.read_csv(DEMANDA, parse_dates=["timestamp"], usecols=["timestamp"])
    inicio, fin = demanda["timestamp"].min(), demanda["timestamp"].max()

    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": inicio.date().isoformat(),
        "end_date": fin.date().isoformat(),
        "hourly": "temperature_2m",
        # hora local de Argentina (UTC-3, sin horario de verano), igual que la serie de demanda
        "timezone": "America/Argentina/Buenos_Aires",
    }
    print(f"Descargando temperatura horaria {params['start_date']} -> {params['end_date']} ...")
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    datos = r.json()["hourly"]

    temp = pd.DataFrame({
        "timestamp": pd.to_datetime(datos["time"]),
        "temperatura_c": datos["temperature_2m"],
    })

    # Validaciones: misma grilla horaria que la demanda, sin nulos y valores fisicamente posibles.
    assert temp["timestamp"].equals(demanda["timestamp"]), "la grilla horaria no coincide con la serie de demanda"
    assert temp["temperatura_c"].notna().all(), "hay horas sin temperatura"
    assert temp["temperatura_c"].between(-10, 45).all(), "temperaturas fuera de rango para Buenos Aires"

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    temp.to_csv(SALIDA, index=False, float_format="%.1f")
    print(f"Guardado: {SALIDA.relative_to(BASE)} ({len(temp):,} horas, "
          f"{temp.temperatura_c.min():.1f} a {temp.temperatura_c.max():.1f} °C)")


if __name__ == "__main__":
    main()
