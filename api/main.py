"""
API de pronostico de demanda electrica horaria del MEM.

Sirve el modelo de calendario y clima que arma src/entrenar_modelo.py
(models/modelo_demanda.joblib): para una fecha devuelve la demanda de las 24
horas con una banda del 80 %.

Endpoints:
    GET  /salud       estado de la API
    GET  /modelo      metadatos del modelo (datos, parametros, metricas, advertencias)
    POST /pronostico  pronostico de un dia

Uso:
    uvicorn api.main:app --reload        (desde la carpeta del proyecto)
    Documentacion interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import modelo_demanda as md   # noqa: E402  (variables y reglas fijadas en el notebook 02)

ARCHIVO_MODELOS = BASE / "models" / "modelo_demanda.joblib"
ARCHIVO_METADATOS = BASE / "models" / "modelo_demanda.json"
URL_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
LATITUD, LONGITUD = -34.56, -58.42   # Aeroparque, igual que en el entrenamiento
TEMP_MIN, TEMP_MAX = -20.0, 50.0     # valores fisicamente posibles en Buenos Aires (con margen)

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar la API."""
    if not ARCHIVO_MODELOS.exists():
        raise RuntimeError("Falta models/modelo_demanda.joblib: correr primero python src/entrenar_modelo.py")
    estado["modelos"] = joblib.load(ARCHIVO_MODELOS)
    estado["metadatos"] = json.loads(ARCHIVO_METADATOS.read_text(encoding="utf-8"))
    yield
    estado.clear()


app = FastAPI(
    title="Forecast de demanda eléctrica del MEM",
    version="1.0",
    description=(
        "Pronóstico de la demanda eléctrica horaria del Mercado Eléctrico Mayorista argentino a partir del "
        "calendario y la temperatura de Buenos Aires. Proyecto final de Data Science — Programa Energía Digital 2026."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas de pedido y respuesta
# ---------------------------------------------------------------------------
class PedidoPronostico(BaseModel):
    fecha: date = Field(..., description="Día a pronosticar (AAAA-MM-DD).", examples=["2026-11-05"])
    es_feriado: bool = Field(
        False,
        description="Marcar si la fecha es feriado o día no laborable. Sábados y domingos ya se toman como no hábiles.",
    )
    temperaturas: list[float] | None = Field(
        None, min_length=24, max_length=24,
        description="24 temperaturas horarias (°C) del día, de 0 a 23 h. Si se omite, se usa el pronóstico "
                    "de ECMWF IFS de Open-Meteo. Sirve para simular escenarios o usar la API sin internet.",
    )
    temperaturas_dia_anterior: list[float] | None = Field(
        None, min_length=24, max_length=24,
        description="Opcional, junto con `temperaturas`: las 24 del día anterior, para el promedio de 24 h "
                    "de las primeras horas del día.",
    )

    @field_validator("temperaturas", "temperaturas_dia_anterior")
    @classmethod
    def rango_fisico(cls, valores):
        if valores is not None and not all(TEMP_MIN <= v <= TEMP_MAX for v in valores):
            raise ValueError(f"las temperaturas tienen que estar entre {TEMP_MIN} y {TEMP_MAX} °C")
        return valores

    @model_validator(mode="after")
    def dia_anterior_requiere_dia(self):
        if self.temperaturas_dia_anterior is not None and self.temperaturas is None:
            raise ValueError("temperaturas_dia_anterior solo se puede mandar junto con temperaturas")
        return self


class HoraPronosticada(BaseModel):
    hora: int
    timestamp: str
    temperatura_c: float
    demanda_mwh: float = Field(description="Demanda pronosticada para el intervalo de esa hora (MWh, equivale a MW medios).")
    banda_inferior_mwh: float
    banda_superior_mwh: float


class Resumen(BaseModel):
    demanda_media_mwh: float
    demanda_maxima_mwh: float
    hora_de_la_maxima: int
    energia_del_dia_mwh: float


class RespuestaPronostico(BaseModel):
    fecha: date
    tipo_dia: Literal["habil", "no_habil"]
    fuente_temperatura: str
    horas: list[HoraPronosticada]
    resumen: Resumen
    advertencias: list[str]


# ---------------------------------------------------------------------------
# Logica
# ---------------------------------------------------------------------------
def tipo_de_dia(fecha: date, es_feriado: bool) -> str:
    """Sabados, domingos y feriados indicados por el usuario son no habiles (mismo binario que CAMMESA)."""
    return "no_habil" if es_feriado or fecha.weekday() >= 5 else "habil"


def temperaturas_open_meteo(fecha: date) -> pd.Series:
    """Temperatura horaria del dia anterior y del dia pedido (48 h), del pronostico de ECMWF IFS en Open-Meteo."""
    params = {
        "latitude": LATITUD, "longitude": LONGITUD, "hourly": "temperature_2m", "models": "ecmwf_ifs",
        "timezone": "America/Argentina/Buenos_Aires",
        "start_date": (fecha - timedelta(days=1)).isoformat(), "end_date": fecha.isoformat(),
    }
    try:
        r = requests.get(URL_OPEN_METEO, params=params, timeout=30)
    except requests.exceptions.RequestException:
        raise HTTPException(503, "No se pudo consultar Open-Meteo. Mandá las 24 temperaturas en el campo `temperaturas`.")
    if r.status_code == 400:
        motivo = r.json().get("reason", "")
        raise HTTPException(422, f"Open-Meteo no tiene pronóstico para esa fecha ({motivo}). "
                                 "Mandá las 24 temperaturas en el campo `temperaturas`.")
    if not r.ok:
        raise HTTPException(503, "Open-Meteo respondió con un error. Mandá las temperaturas en el pedido.")
    h = r.json()["hourly"]
    serie = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), dtype=float)
    if len(serie) != 48 or serie.isna().any():
        raise HTTPException(503, "El pronóstico de Open-Meteo vino incompleto. Mandá las temperaturas en el pedido.")
    return serie


def construir_variables(fecha: date, temperaturas: pd.Series, tipo: str) -> pd.DataFrame:
    """Las variables del modelo para las 24 horas del dia, en el mismo orden que en el entrenamiento."""
    inicio = pd.Timestamp(fecha)
    clima = md.variables_temperatura(temperaturas).loc[inicio:]
    assert len(clima) == 24
    X = pd.DataFrame({
        "hora": clima.index.hour,
        "dia_semana": clima.index.dayofweek,
        "mes": clima.index.month,
        "es_habil": int(tipo == "habil"),
    }, index=clima.index)
    X[md.CLIMATICAS] = clima[md.CLIMATICAS]
    return X[md.VARIABLES]


def pronosticar(X: pd.DataFrame) -> pd.DataFrame:
    """Valor central y banda calibrada del 80 %."""
    modelos, ajuste = estado["modelos"], estado["metadatos"]["banda"]["ajuste_conformal_mw"]
    central = modelos["central"].predict(X)
    inferior = modelos["q10"].predict(X) - ajuste
    superior = modelos["q90"].predict(X) + ajuste
    # los modelos de cuantiles se entrenan por separado y pueden cruzarse con el valor central:
    # en ese caso la banda se extiende para contenerlo
    return pd.DataFrame({
        "central": central,
        "inferior": np.minimum(inferior, central),
        "superior": np.maximum(superior, central),
    }, index=X.index)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/salud", summary="Estado de la API")
def salud():
    return {"estado": "ok", "modelo_cargado": "modelos" in estado}


@app.get("/modelo", summary="Metadatos del modelo")
def modelo():
    return estado["metadatos"]


@app.post("/pronostico", response_model=RespuestaPronostico, summary="Pronóstico de las 24 horas de un día")
def pronostico(pedido: PedidoPronostico):
    advertencias = [estado["metadatos"]["advertencias"][0]]
    tipo = tipo_de_dia(pedido.fecha, pedido.es_feriado)
    if tipo == "habil":
        advertencias.append("Se tomó la fecha como día hábil. Si es feriado o día no laborable, indicarlo con `es_feriado`.")

    inicio = pd.Timestamp(pedido.fecha)
    if pedido.temperaturas is None:
        temperaturas = temperaturas_open_meteo(pedido.fecha)
        fuente = "Open-Meteo, pronóstico de ECMWF IFS 9 km"
    else:
        horas_dia = pd.date_range(inicio, periods=24, freq="h")
        temperaturas = pd.Series(pedido.temperaturas, index=horas_dia)
        if pedido.temperaturas_dia_anterior is not None:
            anterior = pd.Series(pedido.temperaturas_dia_anterior, index=horas_dia - pd.Timedelta(days=1))
            temperaturas = pd.concat([anterior, temperaturas])
        else:
            advertencias.append("Sin temperaturas del día anterior, el promedio de 24 h de las primeras horas "
                                "se calcula solo con las horas disponibles del día.")
        fuente = "Temperaturas enviadas en el pedido"

    minimo, maximo = estado["metadatos"]["rango_temperatura_entrenamiento_c"]
    del_dia = temperaturas.loc[inicio:]
    if del_dia.min() < minimo or del_dia.max() > maximo:
        advertencias.append(f"Hay temperaturas fuera del rango visto en el entrenamiento ({minimo} a {maximo} °C): "
                            "el modelo no extrapola y el pronóstico es menos confiable.")

    X = construir_variables(pedido.fecha, temperaturas, tipo)
    pred = pronosticar(X)
    horas = [
        HoraPronosticada(
            hora=t.hour, timestamp=t.isoformat(), temperatura_c=round(float(del_dia[t]), 1),
            demanda_mwh=round(float(fila.central), 1),
            banda_inferior_mwh=round(float(fila.inferior), 1), banda_superior_mwh=round(float(fila.superior), 1),
        )
        for t, fila in pred.iterrows()
    ]
    resumen = Resumen(
        demanda_media_mwh=round(float(pred["central"].mean()), 1),
        demanda_maxima_mwh=round(float(pred["central"].max()), 1),
        hora_de_la_maxima=int(pred["central"].idxmax().hour),
        energia_del_dia_mwh=round(float(pred["central"].sum()), 1),
    )
    return RespuestaPronostico(fecha=pedido.fecha, tipo_dia=tipo, fuente_temperatura=fuente,
                               horas=horas, resumen=resumen, advertencias=advertencias)
