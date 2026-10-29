"""
Descarga la temperatura horaria historica de Buenos Aires desde Open-Meteo
(API abierta, sin API key) para el mismo periodo que la serie de demanda.
Es la temperatura con la que se ENTRENA el modelo.

Fuente: Open-Meteo Historical Weather API, modelo ECMWF IFS (9 km).
    https://open-meteo.com/en/docs/historical-weather-api
Se fija models=ecmwf_ifs. Sin ese parametro la API usa "best_match", que
combina IFS, ERA5 y ERA5-Land: la serie original de este proyecto se bajo asi
y resulto identica a ecmwf_ifs (verificado hora por hora, 2021-2026). Fijarlo
hace la descarga reproducible y deja explicito el producto.

Es la misma familia de modelo que los pronosticos del test
(src/descargar_temperatura_pronosticada.py). No es una medicion de estacion:
es la serie que arma Open-Meteo con las primeras horas de cada corrida de IFS.
Open-Meteo la actualiza "cada 6 horas, sin demora" (a diferencia de ERA5, que
tiene 5 dias de demora). Para el corte temporal del entrenamiento se usa un
supuesto conservador: el valor de la hora t esta disponible a las t + 12 h
(ciclo de 6 h + 6 h de publicacion). Ver notebook 02, seccion 4.

Punto: coordenadas de Aeroparque (CABA), la misma estacion que usa el dataset
regional para GBA. GBA + Buenos Aires son ~50 % de la demanda del MEM.

Salidas:
    data/processed/temperatura_horaria_ba.csv         columnas: timestamp, temperatura_c
    data/processed/temperatura_horaria_ba.meta.json   procedencia y parametros de la consulta

Uso:
    python src/descargar_temperatura.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
DEMANDA = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA = BASE / "data" / "processed" / "temperatura_horaria_ba.csv"
META = SALIDA.with_suffix(".meta.json")

URL = "https://archive-api.open-meteo.com/v1/archive"
MODELO = "ecmwf_ifs"
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
        "models": MODELO,
        # hora local de Argentina (UTC-3, sin horario de verano), igual que la serie de demanda
        "timezone": "America/Argentina/Buenos_Aires",
    }
    print(f"Descargando temperatura horaria ({MODELO}) {params['start_date']} -> {params['end_date']} ...")
    r = requests.get(URL, params=params, timeout=180)
    r.raise_for_status()
    respuesta = r.json()
    datos = respuesta["hourly"]

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
    META.write_text(json.dumps({
        "fuente": URL,
        "documentacion": "https://open-meteo.com/en/docs/historical-weather-api",
        "parametros": params,
        "punto_de_grilla_devuelto": [respuesta["latitude"], respuesta["longitude"]],
        "disponibilidad_supuesta": "valor de la hora t disponible a las t + 12 h (actualizacion cada 6 h sin demora, segun Open-Meteo)",
        "descargado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {SALIDA.relative_to(BASE)} ({len(temp):,} horas, "
          f"{temp.temperatura_c.min():.1f} a {temp.temperatura_c.max():.1f} °C) y {META.name}")


if __name__ == "__main__":
    main()
