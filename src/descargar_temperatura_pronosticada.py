"""
Descarga, para cada dia del periodo de test, la temperatura horaria que
pronosticaba el modelo meteorologico en el momento en que se emite el
pronostico de demanda del dia siguiente.

Por que: el modelo de demanda se entrena con temperatura observada, pero al
pronosticar solo se conoce el pronostico del tiempo. Para que el test sea
realista, todas las horas de un dia se tienen que pronosticar con la misma
informacion: la que existia en un unico momento de emision.

Supuestos (documentados en el notebook 02):
  - EMISION: el pronostico de demanda del dia D se emite el dia D-1 a las
    12:00 hora local. Es un supuesto razonable para la programacion del dia
    siguiente, no el horario oficial de CAMMESA.
  - CORRIDA: se usa la corrida de las 06 UTC del dia D-1 del modelo ECMWF IFS
    (03:00 hora local). Open-Meteo indica que los modelos globales se publican
    4 a 6 horas despues de iniciada la corrida, asi que a las 12:00 locales
    (15 UTC) ya esta disponible. Si una corrida falta en el archivo de
    Open-Meteo, se usa la anterior que exista (hasta 3 dias antes), igual que
    haria un operador si no llega la corrida nueva.

Para cada dia D se guardan 36 horas de esa corrida: desde la emision (D-1 12:00)
hasta el final de D (D 23:00). Las 12 horas de D-1 sirven para calcular el
promedio movil de 24 h de las primeras horas de D: antes de la emision se usa
temperatura observada, despues, la pronosticada.

Fuente: Open-Meteo Single Runs API, modelo ecmwf_ifs (archivado desde marzo de 2024).
    https://open-meteo.com/en/docs/single-runs-api
Punto: coordenadas de Aeroparque, las mismas que la temperatura observada.

Salida:
    data/processed/temperatura_pronosticada_ba.csv
    columnas: dia_objetivo, timestamp, temperatura_pron_c, corrida_utc, horas_desde_emision,
              antiguedad_corrida_h (horas entre la corrida usada y la habitual de las 06 UTC de D-1)

Uso:
    python src/descargar_temperatura_pronosticada.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
DEMANDA = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA = BASE / "data" / "processed" / "temperatura_pronosticada_ba.csv"
# cada corrida bajada se guarda tal cual: si la descarga se corta, se retoma sin repetir pedidos
CACHE = BASE / "data" / "raw" / "open_meteo_ecmwf_corridas"

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODELO = "ecmwf_ifs"
LATITUD, LONGITUD = -34.56, -58.42
INICIO_TEST = pd.Timestamp("2025-01-01")
HORA_EMISION = 12                      # hora local del dia anterior
# corridas candidatas, de la mas reciente a la mas vieja, como desfase respecto de D-1 00:00 UTC:
# 06 UTC de D-1 (la habitual) y, si falta, cada corrida anterior de 6 en 6 horas hasta 3 dias antes
CORRIDAS = [pd.Timedelta(hours=6 - 6 * k) for k in range(13)]
HORAS_PEDIDAS = 120                    # alcanza para cubrir el dia D aun desde una corrida de 3 dias antes


def bajar_corrida(corrida_utc: pd.Timestamp, hasta: pd.Timestamp) -> pd.Series | None:
    """Temperatura horaria (hora local) de una corrida; None si la corrida no existe en el archivo."""
    archivo = CACHE / f"{corrida_utc:%Y%m%d_%H}.json"
    if archivo.exists():
        datos = json.loads(archivo.read_text())
        if datos is None:
            return None
        serie = pd.Series(datos["temperature_2m"], index=pd.to_datetime(datos["time"]), dtype=float)
        if serie.index.max() >= hasta:     # la copia guardada alcanza; si no, se vuelve a pedir mas larga
            return serie

    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "run": corrida_utc.strftime("%Y-%m-%dT%H:%M"),
        "models": MODELO,
        "hourly": "temperature_2m",
        "forecast_hours": HORAS_PEDIDAS,
        "timezone": "America/Argentina/Buenos_Aires",
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    for intento in range(5):
        try:
            r = requests.get(URL, params=params, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(5 * (intento + 1))   # corte o demora de la conexion: reintentar
            continue
        if r.status_code == 400:            # la corrida no esta archivada
            archivo.write_text("null")
            return None
        if r.ok:
            h = r.json()["hourly"]
            archivo.write_text(json.dumps({"time": h["time"], "temperature_2m": h["temperature_2m"]}))
            return pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), dtype=float)
        time.sleep(5 * (intento + 1))       # error transitorio del servidor: reintentar
    raise RuntimeError(f"no se pudo bajar la corrida {corrida_utc} despues de 5 intentos")


def pronostico_del_dia(dia: pd.Timestamp) -> pd.DataFrame:
    """Las 36 horas (D-1 12:00 -> D 23:00) de la ultima corrida disponible al momento de emision."""
    emision = dia - pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_EMISION)
    horas = pd.date_range(emision, dia + pd.Timedelta(hours=23), freq="h")
    for desfase in CORRIDAS:
        corrida = (dia - pd.Timedelta(days=1)) + desfase      # en UTC
        serie = bajar_corrida(corrida, horas[-1])
        if serie is not None and serie.reindex(horas).notna().all():
            return pd.DataFrame({
                "dia_objetivo": dia.date(),
                "timestamp": horas,
                "temperatura_pron_c": serie.reindex(horas).values,
                "corrida_utc": corrida.strftime("%Y-%m-%d %H:%M"),
                "horas_desde_emision": ((horas - emision) / pd.Timedelta(hours=1)).astype(int),
                "antiguedad_corrida_h": int((CORRIDAS[0] - desfase) / pd.Timedelta(hours=1)),
            })
    raise RuntimeError(f"no hay ninguna corrida disponible para {dia.date()}")


def main() -> None:
    demanda = pd.read_csv(DEMANDA, parse_dates=["timestamp"], usecols=["timestamp"])
    dias = pd.date_range(INICIO_TEST, demanda["timestamp"].max().normalize(), freq="D")
    print(f"Descargando {len(dias)} corridas de {MODELO} ({dias[0].date()} -> {dias[-1].date()}) ...")

    partes = []
    for i, dia in enumerate(dias, start=1):
        partes.append(pronostico_del_dia(dia))
        if i % 50 == 0:
            print(f"  {i}/{len(dias)}")
        time.sleep(0.2)                    # no saturar la API gratuita
    pron = pd.concat(partes, ignore_index=True)

    # Validaciones: 36 horas por dia, sin nulos, valores fisicamente posibles,
    # y la corrida usada siempre anterior a la emision.
    assert (pron.groupby("dia_objetivo").size() == 36).all(), "algun dia no tiene sus 36 horas"
    assert pron["temperatura_pron_c"].notna().all(), "hay horas sin pronostico"
    assert pron["temperatura_pron_c"].between(-10, 45).all(), "temperaturas fuera de rango para Buenos Aires"
    usadas = pron.drop_duplicates("dia_objetivo")
    print("Corridas usadas (hora UTC):", usadas["corrida_utc"].str[-5:].value_counts().to_dict())
    print("Dias que necesitaron una corrida mas vieja que la habitual:",
          (usadas["antiguedad_corrida_h"] > 0).sum(), usadas.loc[usadas["antiguedad_corrida_h"] > 0, "dia_objetivo"].astype(str).tolist())

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pron.to_csv(SALIDA, index=False, float_format="%.1f")
    print(f"Guardado: {SALIDA.relative_to(BASE)} ({pron['dia_objetivo'].nunique()} dias, {len(pron):,} filas)")


if __name__ == "__main__":
    main()
