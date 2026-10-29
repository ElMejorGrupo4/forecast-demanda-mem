"""
Descarga, para cada dia del periodo de test, la temperatura horaria que se
conocia en el momento en que se emite el pronostico de demanda del dia
siguiente, usando solo corridas del modelo ECMWF IFS que, con una demora de
publicacion supuesta, ya estaban disponibles.

Por que: el modelo de demanda se entrena con la temperatura historica, pero al
pronosticar solo se conoce el pronostico del tiempo. Para que el test sea
realista, todas las horas de un dia se tienen que pronosticar con la
informacion que existia en un unico momento de emision.

Supuestos (documentados en el notebook 02):
  - EMISION: el pronostico de demanda del dia D se emite el dia D-1 a las
    12:00 hora local (15 UTC). Es un supuesto razonable para la programacion
    del dia siguiente, no el horario oficial de CAMMESA.
  - CORTE: se reserva 1 hora para entrenar y emitir, asi que solo se usa lo
    que estaba publicado a las 11:00 locales de D-1 (14 UTC).
  - PUBLICACION: Open-Meteo indica que los modelos globales se publican 4 a 6
    horas despues de iniciada la corrida. Se toma el caso conservador: una
    corrida iniciada a la hora i esta disponible a la hora i + 6 h.
    La corrida de las 06 UTC de D-1 queda disponible a las 12 UTC (09:00
    locales), antes del corte.

Para cada dia D se guardan 47 horas: de D-1 01:00 a D 23:00, hora local.
Las de D son las 24 horas a pronosticar; las 23 de D-1 se necesitan para el
promedio movil de 24 h de las primeras horas de D. Cada hora se toma de la
corrida mas reciente que (1) estuviera publicada antes del corte y (2) cubra
esa hora. En la practica: D-1 01:00 y 02:00 salen de la corrida de las 00 UTC
de D-1, y el resto, de la de las 06 UTC de D-1. Si una corrida falta en el
archivo de Open-Meteo, se usa la anterior publicada, como haria un operador.

Fuente: Open-Meteo Single Runs API, modelo ecmwf_ifs (9 km; archivado desde
marzo de 2024). Es el mismo modelo de la temperatura historica de
entrenamiento (src/descargar_temperatura.py).
    https://open-meteo.com/en/docs/single-runs-api
Punto: coordenadas de Aeroparque.

Salidas:
    data/processed/temperatura_pronosticada_ba.csv
        columnas: dia_objetivo, timestamp, temperatura_pron_c, corrida_utc,
                  disponible_supuesta_utc, horas_desde_emision
    data/processed/temperatura_pronosticada_ba.meta.json   (procedencia y parametros)

Uso:
    python src/descargar_temperatura_pronosticada.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
DEMANDA = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA = BASE / "data" / "processed" / "temperatura_pronosticada_ba.csv"
META = SALIDA.with_suffix(".meta.json")
# cada corrida bajada se guarda tal cual: si la descarga se corta, se retoma sin repetir pedidos
CACHE = BASE / "data" / "raw" / "open_meteo_ecmwf_corridas"

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODELO = "ecmwf_ifs"
LATITUD, LONGITUD = -34.56, -58.42
ZONA = "America/Argentina/Buenos_Aires"
UTC_LOCAL = pd.Timedelta(hours=-3)      # Argentina: UTC-3 todo el año, sin horario de verano
INICIO_TEST = pd.Timestamp("2025-01-01")
HORA_EMISION = pd.Timedelta(hours=12)   # D-1 12:00 local
MARGEN_EMISION = pd.Timedelta(hours=1)  # tiempo reservado para entrenar y emitir
DEMORA_PUBLICACION = pd.Timedelta(hours=6)
CICLOS = 13                             # se buscan corridas hasta 3 dias antes (de 6 en 6 horas)
HORAS_PEDIDAS = 120                     # alcanza para cubrir D aun desde una corrida de 3 dias antes


def bajar_corrida(corrida_utc: pd.Timestamp, hasta_local: pd.Timestamp) -> pd.Series | None:
    """Temperatura horaria (hora local) de una corrida; None si la corrida no existe en el archivo."""
    archivo = CACHE / f"{corrida_utc:%Y%m%d_%H}.json"
    if archivo.exists():
        datos = json.loads(archivo.read_text())
        if datos is None:
            return None
        serie = pd.Series(datos["temperature_2m"], index=pd.to_datetime(datos["time"]), dtype=float)
        if serie.index.max() >= hasta_local:   # la copia guardada alcanza; si no, se vuelve a pedir mas larga
            return serie

    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "run": corrida_utc.strftime("%Y-%m-%dT%H:%M"),
        "models": MODELO,
        "hourly": "temperature_2m",
        "forecast_hours": HORAS_PEDIDAS,
        "timezone": ZONA,
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
    """Las 47 horas (D-1 01:00 -> D 23:00, local) con la corrida mas reciente publicada antes del corte."""
    emision = dia - pd.Timedelta(days=1) + HORA_EMISION                 # local
    corte_utc = emision - MARGEN_EMISION - UTC_LOCAL                    # D-1 14 UTC
    horas = pd.date_range(dia - pd.Timedelta(hours=23), dia + pd.Timedelta(hours=23), freq="h")

    # corridas candidatas: ciclos de 6 h ya publicados al corte, de la mas reciente a la mas vieja
    ultima = (corte_utc - DEMORA_PUBLICACION).floor("6h")
    candidatas = [ultima - pd.Timedelta(hours=6 * k) for k in range(CICLOS)]

    valores = pd.Series(float("nan"), index=horas)
    corrida = pd.Series(pd.NaT, index=horas, dtype="datetime64[ns]")
    for c in candidatas:
        faltan = valores.isna()
        if not faltan.any():
            break
        serie = bajar_corrida(c, horas[-1])
        if serie is None:
            continue
        # una corrida solo aporta las horas desde su inicio (las anteriores no las pronostica)
        cubre = faltan & (horas >= c + UTC_LOCAL) & horas.isin(serie.index)
        valores[cubre] = serie.reindex(horas[cubre]).values
        corrida[cubre] = c
    if valores.isna().any():
        raise RuntimeError(f"no hay corridas publicadas que cubran todo el dia {dia.date()}")

    return pd.DataFrame({
        "dia_objetivo": dia.date(),
        "timestamp": horas,
        "temperatura_pron_c": valores.values,
        "corrida_utc": corrida.dt.strftime("%Y-%m-%d %H:%M").values,
        "disponible_supuesta_utc": (corrida + DEMORA_PUBLICACION).dt.strftime("%Y-%m-%d %H:%M").values,
        "horas_desde_emision": ((horas - emision) / pd.Timedelta(hours=1)).astype(int),
    })


def main() -> None:
    demanda = pd.read_csv(DEMANDA, parse_dates=["timestamp"], usecols=["timestamp"])
    dias = pd.date_range(INICIO_TEST, demanda["timestamp"].max().normalize(), freq="D")
    print(f"Armando {len(dias)} dias con corridas de {MODELO} ({dias[0].date()} -> {dias[-1].date()}) ...")

    partes = []
    for i, dia in enumerate(dias, start=1):
        partes.append(pronostico_del_dia(dia))
        if i % 100 == 0:
            print(f"  {i}/{len(dias)}")
    pron = pd.concat(partes, ignore_index=True)

    # Validaciones: 47 horas por dia, sin nulos, valores fisicamente posibles,
    # y todas las corridas disponibles antes del corte (D-1 11:00 local) BAJO LA DEMORA SUPUESTA:
    # la API registra la hora de inicio de cada corrida, no la de publicacion.
    assert (pron.groupby("dia_objetivo").size() == 47).all(), "algun dia no tiene sus 47 horas"
    assert pron["temperatura_pron_c"].notna().all(), "hay horas sin pronostico"
    assert pron["temperatura_pron_c"].between(-10, 45).all(), "temperaturas fuera de rango para Buenos Aires"
    corte_local = pd.to_datetime(pron["dia_objetivo"]) - pd.Timedelta(days=1) + HORA_EMISION - MARGEN_EMISION
    publicada_local = pd.to_datetime(pron["disponible_supuesta_utc"]) + UTC_LOCAL
    assert (publicada_local <= corte_local).all(), "con la demora supuesta, se uso una corrida no disponible al corte"

    por_dia = pron.groupby("dia_objetivo")["corrida_utc"].agg(lambda s: tuple(sorted(set(s))))
    print("Combinaciones de corridas usadas (hora UTC):",
          por_dia.map(lambda t: " + ".join(x[-5:] for x in t)).value_counts().to_dict())

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pron.to_csv(SALIDA, index=False, float_format="%.1f")
    META.write_text(json.dumps({
        "fuente": URL,
        "documentacion": "https://open-meteo.com/en/docs/single-runs-api",
        "modelo": MODELO,
        "latitud": LATITUD, "longitud": LONGITUD, "zona_horaria": ZONA,
        "emision": "dia D-1 a las 12:00 local", "corte": "dia D-1 a las 11:00 local",
        "demora_publicacion_supuesta_h": DEMORA_PUBLICACION / pd.Timedelta(hours=1),
        "horas_por_dia": "D-1 01:00 a D 23:00 local",
        "descargado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {SALIDA.relative_to(BASE)} ({pron['dia_objetivo'].nunique()} dias, {len(pron):,} filas) y {META.name}")


if __name__ == "__main__":
    main()
