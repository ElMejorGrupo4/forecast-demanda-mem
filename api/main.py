"""
API de pronostico de demanda electrica horaria del MEM.

Sirve un snapshot del modelo de calendario y clima (models/<id>/, generado por
src/entrenar_modelo.py): para una fecha devuelve la demanda de las 24 horas con una
banda de nivel nominal 80 %, y un bloque `alcance` que identifica de donde salen los
datos y que tan comparable es la respuesta con la evaluacion publicada.

Modos:
    pronostico_actual   sin temperaturas en el pedido: se consulta el pronostico de ECMWF IFS
                        en Open-Meteo (dia anterior + dia pedido, historial termico completo).
    simulacion_manual   con las temperaturas en el pedido. Historial completo si ademas se mandan
                        las del dia anterior; incompleto solo si se pide explicitamente
                        (historial_incompleto=true): es una aproximacion que la evaluacion no valida.

Endpoints:
    GET  /salud       estado de la API
    GET  /modelo      snapshot cargado: datos, configuracion, entorno, antiguedad y resultados de experimentos
    POST /pronostico  pronostico de un dia

Configuracion (variables de entorno):
    MODELO_ID   snapshot a cargar (por defecto gb_calendario_clima_2026-08-01, el evaluado en agosto de 2026)

Uso:
    uvicorn api.main:app --reload        (desde la carpeta del proyecto)
    Documentacion interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import modelo_demanda as md   # noqa: E402  (reglas unicas compartidas con la evaluacion)

MODELO_POR_DEFECTO = "gb_calendario_clima_2026-08-01"
REGISTRO_RESULTADOS = BASE / "models" / "resultados_experimentos.json"
URL_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
LATITUD, LONGITUD = -34.56, -58.42   # Aeroparque, igual que en el entrenamiento
ZONA = "America/Argentina/Buenos_Aires"
UTC_OFFSET_SEGUNDOS = -3 * 3600      # Argentina: UTC-3 todo el año
TEMP_MIN, TEMP_MAX = -20.0, 50.0     # valores fisicamente posibles en Buenos Aires (con margen)
DIAS_ANTIGUEDAD_AVISO = 30

estado: dict = {}


def ahora() -> datetime:
    """Momento de la consulta, con zona horaria (se reemplaza en las pruebas)."""
    return datetime.now(ZoneInfo(ZONA))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cargar_snapshot(modelo_id: str) -> dict:
    """Carga un snapshot verificando su huella y la version de scikit-learn con la que se genero."""
    carpeta = BASE / "models" / modelo_id
    artefacto, archivo_meta = carpeta / "modelo.joblib", carpeta / "metadatos.json"
    if not artefacto.exists() or not archivo_meta.exists():
        raise RuntimeError(f"No existe el snapshot {modelo_id}: correr src/entrenar_modelo.py --fecha-objetivo AAAA-MM-DD")
    meta = json.loads(archivo_meta.read_text(encoding="utf-8"))
    if sha256(artefacto) != meta["artefacto"]["sha256"]:
        raise RuntimeError(f"El archivo de {modelo_id} no coincide con la huella registrada en sus metadatos")
    version = meta["entorno"]["scikit-learn"]
    if sklearn.__version__ != version:
        raise RuntimeError(f"El snapshot se generó con scikit-learn {version} y el entorno tiene {sklearn.__version__}. "
                           "Instalar el entorno fijado: pip install -r requirements.txt")
    registro = json.loads(REGISTRO_RESULTADOS.read_text(encoding="utf-8")) if REGISTRO_RESULTADOS.exists() else {}
    resultados = registro.get(modelo_id)
    if resultados and resultados.get("artefacto_sha256") != meta["artefacto"]["sha256"]:
        resultados = None   # las metricas no se atribuyen a un artefacto distinto del evaluado
    return {"modelos": joblib.load(artefacto), "metadatos": meta, "resultados": resultados}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el snapshot una sola vez, al arrancar la API."""
    estado.update(cargar_snapshot(os.environ.get("MODELO_ID", MODELO_POR_DEFECTO)))
    yield
    estado.clear()


app = FastAPI(
    title="Forecast de demanda eléctrica del MEM",
    version="2.0",
    description=(
        "Pronóstico de la demanda eléctrica horaria del Mercado Eléctrico Mayorista argentino a partir del "
        "calendario y la temperatura de Buenos Aires. Proyecto final de Data Science — Programa Energía Digital 2026. "
        "Cada respuesta incluye un bloque `alcance` que dice de dónde salen los datos y qué tan comparable es "
        "con la evaluación publicada."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------
class PedidoPronostico(BaseModel):
    fecha: date = Field(..., description="Día a pronosticar (AAAA-MM-DD).", examples=["2026-11-05"])
    es_feriado: bool = Field(
        False, description="Marcar si la fecha es feriado o día no laborable. Sábados y domingos ya se toman como no hábiles.")
    temperaturas: list[float] | None = Field(
        None, min_length=24, max_length=24,
        description="Simulación manual: 24 temperaturas horarias (°C) del día, de 0 a 23 h. "
                    "Si se omite, se consulta el pronóstico de ECMWF IFS en Open-Meteo.")
    temperaturas_dia_anterior: list[float] | None = Field(
        None, min_length=24, max_length=24,
        description="Simulación manual con historial completo (el modo evaluado): las 24 temperaturas del día "
                    "anterior. El promedio de 24 h de las primeras horas usa las de 01:00 a 23:00; la de 00:00 no se usa.")
    historial_incompleto: bool = Field(
        False, description="Permite una simulación aproximada sin las temperaturas del día anterior: el promedio de "
                           "24 h de las primeras horas se calcula solo con horas del día. Ese modo no está validado "
                           "por las métricas publicadas.")

    model_config = {"json_schema_extra": {"examples": [
        {"fecha": "2026-11-05"},
        {"fecha": "2026-08-12", "temperaturas": [9, 8.5, 8, 7.5, 7, 7, 7, 7.5, 8.5, 10, 11.5, 13, 14, 15, 15.5, 15.5, 15, 14,
                                                 12.5, 11.5, 11, 10.5, 10, 9.5],
         "temperaturas_dia_anterior": [10, 9.5, 9, 9, 8.5, 8.5, 8, 8.5, 9.5, 11, 12.5, 14, 15, 16, 16.5, 16.5, 16, 15,
                                       13.5, 12.5, 12, 11, 10, 9.5]},
    ]}}

    @field_validator("temperaturas", "temperaturas_dia_anterior")
    @classmethod
    def rango_fisico(cls, valores):
        if valores is not None and not all(math.isfinite(v) and TEMP_MIN <= v <= TEMP_MAX for v in valores):
            raise ValueError(f"las temperaturas tienen que ser números entre {TEMP_MIN} y {TEMP_MAX} °C")
        return valores

    @model_validator(mode="after")
    def combinaciones_validas(self):
        if self.temperaturas is None:
            if self.temperaturas_dia_anterior is not None:
                raise ValueError("temperaturas_dia_anterior solo se puede mandar junto con temperaturas")
            if self.historial_incompleto:
                raise ValueError("historial_incompleto solo aplica a la simulación manual (con temperaturas)")
        else:
            if self.temperaturas_dia_anterior is None and not self.historial_incompleto:
                raise ValueError("faltan temperaturas_dia_anterior: el modo evaluado necesita el historial térmico completo. "
                                 "Para una simulación aproximada sin ellas, mandar historial_incompleto=true")
            if self.temperaturas_dia_anterior is not None and self.historial_incompleto:
                raise ValueError("historial_incompleto=true no corresponde si se mandan temperaturas_dia_anterior")
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


class FuenteMeteorologica(BaseModel):
    proveedor: str
    modelo: str | None
    corrida: str | None = Field(description="Corrida del modelo meteorológico, si el proveedor la informa.")
    detalle: str


class ModeloUsado(BaseModel):
    id: str
    fecha_objetivo: str
    corte_de_informacion: str
    ultimo_dato_de_entrenamiento: str
    antiguedad_del_corte_dias: int
    artefacto_generado_utc: str
    artefacto_sha256: str


class Alcance(BaseModel):
    modo: Literal["pronostico_actual", "simulacion_manual"]
    historial_termico: Literal["completo", "incompleto"]
    consultado_en: datetime = Field(description="Momento en que se procesó el pedido, con zona horaria.")
    horizonte_horas: dict[str, float] | None = Field(
        description="Horas entre la consulta y la primera y la última hora pedidas (solo en el pronóstico actual).")
    fecha_pasada: bool
    fuente_meteorologica: FuenteMeteorologica
    banda: dict
    relacion_con_la_evaluacion: str
    modelo: ModeloUsado


class RespuestaPronostico(BaseModel):
    fecha: date
    tipo_dia: Literal["habil", "no_habil"]
    alcance: Alcance
    horas: list[HoraPronosticada]
    resumen: Resumen
    advertencias: list[str]


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
def consultar_open_meteo(fecha: date) -> tuple[pd.Series, dict]:
    """Pronostico de ECMWF IFS en Open-Meteo para el dia anterior y el pedido (48 h), validado."""
    params = {
        "latitude": LATITUD, "longitude": LONGITUD, "hourly": "temperature_2m", "models": "ecmwf_ifs", "timezone": ZONA,
        "start_date": (fecha - timedelta(days=1)).isoformat(), "end_date": fecha.isoformat(),
    }
    try:
        r = requests.get(URL_OPEN_METEO, params=params, timeout=30)
    except requests.exceptions.RequestException:
        raise HTTPException(503, "Open-Meteo no respondió. Se puede mandar las temperaturas en el pedido (simulación manual).")
    if r.status_code == 400:
        try:
            motivo = str(r.json().get("reason", ""))
        except ValueError:
            motivo = ""
        if "out of allowed range" in motivo:
            raise HTTPException(422, f"Open-Meteo no tiene pronóstico para esa fecha ({motivo}). "
                                     "Mandá las temperaturas en el pedido (simulación manual).")
        raise HTTPException(502, f"Open-Meteo rechazó la consulta: {motivo or 'sin motivo informado'}")
    if r.status_code == 429 or r.status_code >= 500:
        raise HTTPException(503, f"Open-Meteo no está disponible (HTTP {r.status_code}). Probar más tarde o mandar las temperaturas.")
    if r.status_code != 200:
        raise HTTPException(502, f"Respuesta inesperada de Open-Meteo (HTTP {r.status_code})")
    return procesar_respuesta_open_meteo(r, fecha)


def procesar_respuesta_open_meteo(r, fecha: date) -> tuple[pd.Series, dict]:
    """Valida la respuesta de Open-Meteo. Cualquier problema es una falla del proveedor (HTTP 502)."""
    def invalida(motivo):
        return HTTPException(502, f"Respuesta inválida de Open-Meteo: {motivo}")

    try:
        datos = r.json()
    except ValueError:
        raise invalida("no es JSON")
    if not isinstance(datos, dict) or not isinstance(datos.get("hourly"), dict):
        raise invalida("falta el bloque 'hourly'")
    if datos.get("timezone") != ZONA or datos.get("utc_offset_seconds") != UTC_OFFSET_SEGUNDOS:
        raise invalida(f"zona horaria {datos.get('timezone')!r} / desfase {datos.get('utc_offset_seconds')!r}; se esperaba {ZONA} (UTC-3)")
    tiempos, valores = datos["hourly"].get("time"), datos["hourly"].get("temperature_2m")
    if not isinstance(tiempos, list) or not isinstance(valores, list):
        raise invalida("faltan las listas 'time' o 'temperature_2m'")
    if len(tiempos) != len(valores):
        raise invalida(f"{len(tiempos)} horas y {len(valores)} temperaturas")
    try:
        indice = pd.DatetimeIndex(pd.to_datetime(tiempos, format="%Y-%m-%dT%H:%M"))
    except (ValueError, TypeError):
        raise invalida("hay horas con formato inválido")
    esperado = pd.date_range(pd.Timestamp(fecha) - pd.Timedelta(days=1), periods=48, freq="h")
    if not indice.equals(esperado):
        raise invalida("las horas no son exactamente las 48 del día anterior y el pedido, únicas y ordenadas")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in valores):
        raise invalida("hay temperaturas faltantes o no numéricas")
    if not all(TEMP_MIN <= v <= TEMP_MAX for v in valores):
        raise invalida(f"hay temperaturas fuera de {TEMP_MIN} a {TEMP_MAX} °C")
    info = {"punto_de_grilla": [datos.get("latitude"), datos.get("longitude")], "zona_horaria": datos["timezone"]}
    return pd.Series(valores, index=indice, dtype=float), info


# ---------------------------------------------------------------------------
# Logica
# ---------------------------------------------------------------------------
def tipo_de_dia(fecha: date, es_feriado: bool) -> str:
    """Sabados, domingos y feriados indicados por el usuario son no habiles (mismo binario que CAMMESA)."""
    return "no_habil" if es_feriado or fecha.weekday() >= 5 else "habil"


def construir_variables(fecha: date, temperaturas: pd.Series, tipo: str, historial_completo: bool) -> pd.DataFrame:
    """Variables del modelo para las 24 horas, con la regla unica del modulo (md.clima_del_dia)."""
    try:
        clima = md.clima_del_dia(temperaturas, fecha, historial_completo=historial_completo)
    except ValueError as e:
        raise HTTPException(422, f"Temperaturas insuficientes: {e}")
    X = pd.DataFrame({
        "hora": clima.index.hour, "dia_semana": clima.index.dayofweek, "mes": clima.index.month,
        "es_habil": int(tipo == "habil"),
    }, index=clima.index)
    X[md.CLIMATICAS] = clima[md.CLIMATICAS]
    return X[md.VARIABLES]


def pronosticar(X: pd.DataFrame) -> pd.DataFrame:
    """Valor central y banda con la regla unica del modulo (md.banda_80)."""
    modelos, ajuste = estado["modelos"], estado["metadatos"]["banda"]["ajuste_conformal_mw"]
    central = modelos["central"].predict(X)
    inferior, superior = md.banda_80(central, modelos["q10"].predict(X), modelos["q90"].predict(X), ajuste)
    return pd.DataFrame({"central": central, "inferior": inferior, "superior": superior}, index=X.index)


def modelo_usado(consulta: datetime) -> ModeloUsado:
    meta = estado["metadatos"]
    corte = pd.Timestamp(meta["corte_de_informacion"])
    return ModeloUsado(
        id=meta["id"], fecha_objetivo=meta["fecha_objetivo"], corte_de_informacion=meta["corte_de_informacion"],
        ultimo_dato_de_entrenamiento=meta["datos_de_entrenamiento"]["hasta"],
        antiguedad_del_corte_dias=(consulta.date() - corte.date()).days,
        artefacto_generado_utc=meta["generado_utc"], artefacto_sha256=meta["artefacto"]["sha256"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/salud", summary="Estado de la API")
def salud():
    return {"estado": "ok", "modelo_cargado": "modelos" in estado, "modelo_id": estado.get("metadatos", {}).get("id")}


@app.get("/modelo", summary="Snapshot cargado: datos, configuración, entorno, antigüedad y resultados")
def modelo():
    consulta = ahora()
    return {
        "metadatos": estado["metadatos"],
        "antiguedad": modelo_usado(consulta).model_dump(),
        "resultados_de_experimentos": estado["resultados"] or (
            "No hay resultados registrados para este artefacto exacto: las métricas publicadas corresponden a otro snapshot."),
        "entorno_actual": {"scikit-learn": sklearn.__version__, "python": sys.version.split()[0]},
    }


@app.post("/pronostico", response_model=RespuestaPronostico, summary="Pronóstico de las 24 horas de un día")
def pronostico(pedido: PedidoPronostico):
    consulta = ahora()
    inicio = pd.Timestamp(pedido.fecha)
    tipo = tipo_de_dia(pedido.fecha, pedido.es_feriado)
    advertencias = [estado["metadatos"]["advertencias"][0]]
    if tipo == "habil":
        advertencias.append("Se tomó la fecha como día hábil. Si es feriado o día no laborable, indicarlo con `es_feriado`.")

    if pedido.temperaturas is None:
        modo, historial = "pronostico_actual", "completo"
        temperaturas, info = consultar_open_meteo(pedido.fecha)
        fuente = FuenteMeteorologica(
            proveedor="Open-Meteo Forecast API", modelo="ecmwf_ifs", corrida=None,
            detalle=f"Consultado al procesar el pedido; punto de grilla {info['punto_de_grilla']}. "
                    "Open-Meteo no informa en esta respuesta qué corrida del modelo usó.")
    else:
        modo = "simulacion_manual"
        historial = "incompleto" if pedido.historial_incompleto else "completo"
        horas_dia = pd.date_range(inicio, periods=24, freq="h")
        temperaturas = pd.Series(pedido.temperaturas, index=horas_dia)
        if pedido.temperaturas_dia_anterior is not None:
            anterior = pd.Series(pedido.temperaturas_dia_anterior, index=horas_dia - pd.Timedelta(days=1))
            temperaturas = pd.concat([anterior, temperaturas])
        fuente = FuenteMeteorologica(proveedor="usuario", modelo=None, corrida=None,
                                     detalle="Temperaturas enviadas en el pedido")

    horas_objetivo = pd.date_range(inicio, periods=24, freq="h").tz_localize(ZONA)
    fecha_pasada = horas_objetivo[0] < pd.Timestamp(consulta)
    horizonte = None
    if modo == "pronostico_actual":
        h = (horas_objetivo - pd.Timestamp(consulta)) / pd.Timedelta(hours=1)
        horizonte = {"primera_hora": round(float(h[0]), 1), "ultima_hora": round(float(h[-1]), 1)}

    if historial == "incompleto":
        relacion = ("Simulación aproximada con historial térmico incompleto: el promedio de 24 h de las primeras horas "
                    "usa solo horas del día. Las métricas publicadas no validan este modo.")
        advertencias.append("Historial térmico incompleto: modo aproximado, no evaluado. Para el modo evaluado, "
                            "mandar también temperaturas_dia_anterior.")
    elif modo == "simulacion_manual":
        relacion = ("Escenario con temperaturas dadas: usa las mismas variables y la misma banda que la evaluación, pero "
                    "el error depende de esas temperaturas; las métricas publicadas se midieron con pronósticos reales.")
    else:
        relacion = ("Pronóstico con el pronóstico meteorológico disponible al momento de la consulta. La evaluación "
                    "publicada emitía el día anterior a las 12:00 con corridas de ECMWF identificadas; esta consulta "
                    "puede tener otra anticipación y no identifica la corrida.")
    if modo == "pronostico_actual" and fecha_pasada:
        advertencias.append("La fecha (o parte de sus horas) ya pasó: Open-Meteo devuelve hoy su mejor dato para esas horas, "
                            "no el pronóstico que existía antes. Esto no es un backtesting; para eso ver el notebook 04.")

    usado = modelo_usado(consulta)
    if usado.antiguedad_del_corte_dias > DIAS_ANTIGUEDAD_AVISO:
        advertencias.append(f"El modelo se entrenó con datos hasta el {usado.ultimo_dato_de_entrenamiento} "
                            f"(corte hace {usado.antiguedad_del_corte_dias} días): si cambió el nivel de la demanda, "
                            "hay que reentrenarlo (src/entrenar_modelo.py).")
    if inicio < pd.Timestamp(usado.fecha_objetivo):
        advertencias.append("La fecha es anterior a la fecha objetivo del modelo: se entrenó con datos posteriores a "
                            "ese día, así que el resultado no sirve para evaluarlo.")

    X = construir_variables(pedido.fecha, temperaturas, tipo, historial_completo=(historial == "completo"))
    del_dia = temperaturas.loc[inicio:]
    minimo, maximo = estado["metadatos"]["rango_temperatura_entrenamiento_c"]
    if del_dia.min() < minimo or del_dia.max() > maximo:
        advertencias.append(f"Hay temperaturas fuera del rango visto en el entrenamiento ({minimo} a {maximo} °C): "
                            "el pronóstico es menos confiable.")

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
    banda = {"nivel_nominal": 0.80, "regla": estado["metadatos"]["banda"]["regla"],
             "nota": "La cobertura empírica observada fue menor que la nominal en varios meses (ver GET /modelo)."}
    alcance = Alcance(modo=modo, historial_termico=historial, consultado_en=consulta, horizonte_horas=horizonte,
                      fecha_pasada=bool(fecha_pasada), fuente_meteorologica=fuente, banda=banda,
                      relacion_con_la_evaluacion=relacion, modelo=usado)
    return RespuestaPronostico(fecha=pedido.fecha, tipo_dia=tipo, alcance=alcance, horas=horas,
                               resumen=resumen, advertencias=advertencias)
