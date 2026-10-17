"""
Descarga la temperatura horaria PRONOSTICADA con un dia de anticipacion para
Buenos Aires, en el periodo de test del modelo (2025 en adelante).

Por que: el modelo se entrena con temperatura observada, pero al pronosticar
la demanda de manana solo se conoce el pronostico del tiempo. Evaluar el test
con la temperatura que efectivamente hizo equivale a suponer un pronostico del
tiempo perfecto (fuga de informacion). Con esta serie el test se evalua con la
misma informacion que tendria el operador en la realidad.

Fuente: Open-Meteo Previous Runs API, variable temperature_2m_previous_day1:
el valor que el modelo meteorologico predijo 24 horas antes de cada hora.
    https://previous-runs-api.open-meteo.com/v1/forecast
Punto: coordenadas de Aeroparque, las mismas que la temperatura observada.

Se descarga desde el dia anterior al inicio del test para poder calcular el
promedio movil de 24 h de la primera hora del test.

Salida:
    data/processed/temperatura_pronosticada_ba.csv   columnas: timestamp, temperatura_pron_c

Uso:
    python src/descargar_temperatura_pronosticada.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
DEMANDA = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA = BASE / "data" / "processed" / "temperatura_pronosticada_ba.csv"

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LATITUD, LONGITUD = -34.56, -58.42
INICIO_TEST = pd.Timestamp("2025-01-01")
# un dia antes del test, para el promedio movil de 24 h
INICIO_DESCARGA = INICIO_TEST - pd.Timedelta(days=1)


def main() -> None:
    demanda = pd.read_csv(DEMANDA, parse_dates=["timestamp"], usecols=["timestamp"])
    fin = demanda["timestamp"].max()

    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": INICIO_DESCARGA.date().isoformat(),
        "end_date": fin.date().isoformat(),
        "hourly": "temperature_2m_previous_day1",
        # hora local de Argentina (UTC-3, sin horario de verano), igual que la serie de demanda
        "timezone": "America/Argentina/Buenos_Aires",
    }
    print(f"Descargando temperatura pronosticada {params['start_date']} -> {params['end_date']} ...")
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    datos = r.json()["hourly"]

    pron = pd.DataFrame({
        "timestamp": pd.to_datetime(datos["time"]),
        "temperatura_pron_c": datos["temperature_2m_previous_day1"],
    })

    # Validaciones: misma grilla horaria que la demanda en ese tramo, sin nulos y valores fisicamente posibles.
    esperado = demanda.loc[demanda["timestamp"] >= INICIO_DESCARGA, "timestamp"].reset_index(drop=True)
    assert pron["timestamp"].equals(esperado), "la grilla horaria no coincide con la serie de demanda"
    assert pron["temperatura_pron_c"].notna().all(), "hay horas sin pronostico"
    assert pron["temperatura_pron_c"].between(-10, 45).all(), "temperaturas fuera de rango para Buenos Aires"

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pron.to_csv(SALIDA, index=False, float_format="%.1f")
    print(f"Guardado: {SALIDA.relative_to(BASE)} ({len(pron):,} horas, "
          f"{pron.temperatura_pron_c.min():.1f} a {pron.temperatura_pron_c.max():.1f} °C)")


if __name__ == "__main__":
    main()
