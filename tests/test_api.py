"""
Pruebas de la API (api/main.py), del modulo compartido y del script de entrenamiento.

Uso (desde la carpeta del proyecto):
    python -m pytest tests -v

No necesitan internet: en las pruebas de Open-Meteo se reemplaza solo la llamada HTTP
(requests.get) por respuestas simuladas, y se ejecuta la validacion real de la API.
El reloj de la API se fija para que horizonte y antiguedad sean reproducibles.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import api.main as api           # noqa: E402
import entrenar_modelo as em     # noqa: E402
import modelo_demanda as md      # noqa: E402

ZONA = ZoneInfo("America/Argentina/Buenos_Aires")
RELOJ = datetime(2026, 8, 11, 12, 0, tzinfo=ZONA)     # consulta el día anterior a las 12:00
DIA = "2026-08-12"                                     # miércoles hábil

# curvas de temperatura de invierno razonables: mínima de madrugada, máxima a la tarde
INVIERNO = [9, 8.5, 8, 7.5, 7, 7, 7, 7.5, 8.5, 10, 11.5, 13, 14, 15, 15.5, 15.5, 15, 14, 12.5, 11.5, 11, 10.5, 10, 9.5]
INVIERNO_ANTERIOR = [t + 1 for t in INVIERNO]


@pytest.fixture(scope="module")
def cliente():
    with TestClient(api.app) as c:   # el "with" ejecuta la carga del snapshot
        yield c


@pytest.fixture(autouse=True)
def reloj_fijo(monkeypatch):
    monkeypatch.setattr(api, "ahora", lambda: RELOJ)


def pedir(cliente, **pedido):
    return cliente.post("/pronostico", json=pedido)


def manual(**extra):
    return {"fecha": DIA, "temperaturas": INVIERNO, "temperaturas_dia_anterior": INVIERNO_ANTERIOR, **extra}


# ---------------------------------------------------------------------------
# Respuestas simuladas de Open-Meteo
# ---------------------------------------------------------------------------
class RespuestaFalsa:
    def __init__(self, status=200, datos=None, no_json=False):
        self.status_code, self._datos, self._no_json = status, datos, no_json

    def json(self):
        if self._no_json:
            raise ValueError("no es JSON")
        return self._datos


def respuesta_valida(fecha=DIA, temperaturas=None):
    horas = pd.date_range(pd.Timestamp(fecha) - pd.Timedelta(days=1), periods=48, freq="h")
    return {
        "latitude": -34.551846, "longitude": -58.448944, "timezone": "America/Argentina/Buenos_Aires",
        "utc_offset_seconds": -10800,
        "hourly": {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in horas],
                   "temperature_2m": temperaturas if temperaturas is not None else INVIERNO_ANTERIOR + INVIERNO},
    }


def simular_http(monkeypatch, respuesta=None, excepcion=None):
    def falso_get(*args, **kwargs):
        if excepcion:
            raise excepcion
        return respuesta
    monkeypatch.setattr(api.requests, "get", falso_get)


# ---------------------------------------------------------------------------
# Endpoints básicos y snapshot
# ---------------------------------------------------------------------------
def test_salud(cliente):
    datos = cliente.get("/salud").json()
    assert datos == {"estado": "ok", "modelo_cargado": True, "modelo_id": "gb_calendario_clima_2026-08-01"}


def test_modelo_informa_corte_antiguedad_y_resultados_del_artefacto(cliente):
    datos = cliente.get("/modelo").json()
    meta = datos["metadatos"]
    assert meta["configuracion"]["variables"] == md.VARIABLES
    assert meta["configuracion"]["hiperparametros"] == md.PARAMS_GB
    assert meta["corte_de_informacion"] == "2026-07-31 11:00:00"
    assert meta["datos_de_entrenamiento"]["hasta"] == "2026-07-30 22:00:00"
    assert datos["antiguedad"]["antiguedad_del_corte_dias"] == 11            # del 31/07 al 11/08
    resultados = datos["resultados_de_experimentos"]
    assert resultados["artefacto_sha256"] == meta["artefacto"]["sha256"]


def test_no_carga_con_otra_version_de_scikit_learn(monkeypatch):
    monkeypatch.setattr(api.sklearn, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="scikit-learn"):
        api.cargar_snapshot(api.MODELO_POR_DEFECTO)


def test_no_carga_si_el_artefacto_no_coincide_con_su_huella(tmp_path, monkeypatch):
    origen = BASE / "models" / api.MODELO_POR_DEFECTO
    copia = tmp_path / "models" / "copia"
    shutil.copytree(origen, copia)
    meta = json.loads((copia / "metadatos.json").read_text(encoding="utf-8"))
    meta["artefacto"]["sha256"] = "0" * 64
    (copia / "metadatos.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(api, "BASE", tmp_path)
    with pytest.raises(RuntimeError, match="huella"):
        api.cargar_snapshot("copia")


def test_metricas_no_se_atribuyen_a_otro_artefacto(tmp_path, monkeypatch):
    origen = BASE / "models" / api.MODELO_POR_DEFECTO
    shutil.copytree(origen, tmp_path / "models" / api.MODELO_POR_DEFECTO)
    registro = json.loads((BASE / "models" / "resultados_experimentos.json").read_text(encoding="utf-8"))
    registro[api.MODELO_POR_DEFECTO]["artefacto_sha256"] = "f" * 64
    otro_registro = tmp_path / "registro.json"
    otro_registro.write_text(json.dumps(registro), encoding="utf-8")
    monkeypatch.setattr(api, "BASE", tmp_path)
    monkeypatch.setattr(api, "REGISTRO_RESULTADOS", otro_registro)
    assert api.cargar_snapshot(api.MODELO_POR_DEFECTO)["resultados"] is None


# ---------------------------------------------------------------------------
# Simulación manual
# ---------------------------------------------------------------------------
def test_simulacion_manual_con_historial_completo(cliente):
    r = pedir(cliente, **manual())
    assert r.status_code == 200
    datos = r.json()
    alcance = datos["alcance"]
    assert datos["tipo_dia"] == "habil"
    assert alcance["modo"] == "simulacion_manual" and alcance["historial_termico"] == "completo"
    assert alcance["fuente_meteorologica"]["proveedor"] == "usuario"
    assert alcance["horizonte_horas"] is None
    assert alcance["consultado_en"].endswith("-03:00")
    assert alcance["banda"]["nivel_nominal"] == 0.8
    assert [h["hora"] for h in datos["horas"]] == list(range(24))
    for h in datos["horas"]:
        assert h["banda_inferior_mwh"] <= h["demanda_mwh"] <= h["banda_superior_mwh"]
        assert 8_000 < h["demanda_mwh"] < 35_000


def test_sin_dia_anterior_se_rechaza_salvo_pedido_explicito(cliente):
    r = pedir(cliente, fecha=DIA, temperaturas=INVIERNO)
    assert r.status_code == 422 and "temperaturas_dia_anterior" in r.text


def test_historial_incompleto_explicito_se_identifica(cliente):
    datos = pedir(cliente, fecha=DIA, temperaturas=INVIERNO, historial_incompleto=True).json()
    assert datos["alcance"]["historial_termico"] == "incompleto"
    assert "no validan este modo" in datos["alcance"]["relacion_con_la_evaluacion"]
    assert any("Historial térmico incompleto" in a for a in datos["advertencias"])


def test_historial_incompleto_no_inventa_temperaturas(cliente):
    """Con historial incompleto, las primeras horas promedian solo las horas del día (no se completa la ventana)."""
    X = api.construir_variables(pd.Timestamp(DIA).date(), pd.Series(INVIERNO, index=pd.date_range(DIA, periods=24, freq="h")),
                                "habil", historial_completo=False)
    assert X["temp_24h"].iloc[0] == INVIERNO[0]
    assert X["temp_24h"].iloc[1] == pytest.approx(np.mean(INVIERNO[:2]))


def test_fin_de_semana_y_feriado_son_no_habiles(cliente):
    assert pedir(cliente, **manual(fecha="2026-08-16")).json()["tipo_dia"] == "no_habil"
    habil = pedir(cliente, **manual()).json()
    feriado = pedir(cliente, **manual(es_feriado=True)).json()
    assert feriado["tipo_dia"] == "no_habil"
    assert feriado["resumen"]["energia_del_dia_mwh"] < habil["resumen"]["energia_del_dia_mwh"]


@pytest.mark.parametrize("pedido", [
    {"fecha": DIA, "temperaturas": INVIERNO[:23], "historial_incompleto": True},                 # faltan horas
    {"fecha": DIA, "temperaturas": [80] * 24, "historial_incompleto": True},                    # temperatura imposible
    {"fecha": DIA, "temperaturas_dia_anterior": INVIERNO},                                      # anterior sin el día
    {"fecha": DIA, "historial_incompleto": True},                                               # incompleto sin temperaturas
    {"fecha": DIA, "temperaturas": INVIERNO, "temperaturas_dia_anterior": INVIERNO, "historial_incompleto": True},
    {"fecha": "12/08/2026", "temperaturas": INVIERNO, "temperaturas_dia_anterior": INVIERNO},   # formato de fecha
])
def test_pedidos_invalidos(cliente, pedido):
    assert pedir(cliente, **pedido).status_code == 422


def test_temperatura_fuera_de_rango_de_entrenamiento_avisa(cliente):
    datos = pedir(cliente, fecha="2027-01-15", temperaturas=[42] * 24, temperaturas_dia_anterior=[40] * 24).json()
    assert any("fuera del rango" in a for a in datos["advertencias"])


def test_avisa_antiguedad_y_fecha_anterior_al_modelo(cliente, monkeypatch):
    monkeypatch.setattr(api, "ahora", lambda: datetime(2026, 10, 29, 9, 0, tzinfo=ZONA))
    datos = pedir(cliente, **manual(fecha="2026-05-06")).json()
    assert datos["alcance"]["modelo"]["antiguedad_del_corte_dias"] == 90
    assert any("corte hace 90 días" in a for a in datos["advertencias"])
    assert any("anterior a la fecha objetivo" in a for a in datos["advertencias"])


# ---------------------------------------------------------------------------
# Pronóstico actual con Open-Meteo (HTTP simulado, validación real)
# ---------------------------------------------------------------------------
def test_pronostico_actual_identifica_alcance(cliente, monkeypatch):
    simular_http(monkeypatch, RespuestaFalsa(datos=respuesta_valida()))
    datos = pedir(cliente, fecha=DIA).json()
    alcance = datos["alcance"]
    assert alcance["modo"] == "pronostico_actual" and alcance["historial_termico"] == "completo"
    assert alcance["fuente_meteorologica"]["modelo"] == "ecmwf_ifs"
    assert alcance["fuente_meteorologica"]["corrida"] is None              # el proveedor no la informa: no se inventa
    assert alcance["horizonte_horas"] == {"primera_hora": 12.0, "ultima_hora": 35.0}
    assert alcance["fecha_pasada"] is False


def test_pronostico_actual_sobre_fecha_pasada_no_es_backtesting(cliente, monkeypatch):
    simular_http(monkeypatch, RespuestaFalsa(datos=respuesta_valida("2026-08-05")))
    datos = pedir(cliente, fecha="2026-08-05").json()
    assert datos["alcance"]["fecha_pasada"] is True
    assert any("no es un backtesting" in a for a in datos["advertencias"])


def test_open_meteo_y_manual_dan_lo_mismo_con_las_mismas_temperaturas(cliente, monkeypatch):
    simular_http(monkeypatch, RespuestaFalsa(datos=respuesta_valida()))
    via_open_meteo = pedir(cliente, fecha=DIA).json()["horas"]
    via_manual = pedir(cliente, **manual()).json()["horas"]
    assert via_open_meteo == via_manual


def _modificar(cambio):
    datos = respuesta_valida()
    cambio(datos)
    return datos


def _duplicar_hora(d):
    d["hourly"]["time"][5] = d["hourly"]["time"][4]


def _desordenar(d):
    t = d["hourly"]["time"]
    t[3], t[4] = t[4], t[3]


@pytest.mark.parametrize("respuesta", [
    RespuestaFalsa(no_json=True),                                                                     # no es JSON
    RespuestaFalsa(datos={"latitude": -34.55}),                                                       # sin 'hourly'
    RespuestaFalsa(datos=[1, 2, 3]),                                                                  # estructura inesperada
    RespuestaFalsa(datos=_modificar(_duplicar_hora)),                                                 # hora duplicada
    RespuestaFalsa(datos=_modificar(_desordenar)),                                                    # horas desordenadas
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"].update(time=d["hourly"]["time"][:47],
                                                                  temperature_2m=d["hourly"]["temperature_2m"][:47]))),
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["temperature_2m"].pop())),                 # longitudes distintas
    RespuestaFalsa(datos=_modificar(lambda d: d.update(timezone="GMT", utc_offset_seconds=0))),      # otra zona horaria
    RespuestaFalsa(datos=respuesta_valida("2026-08-13")),                                             # otras fechas
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["temperature_2m"].__setitem__(10, None))),  # temperatura nula
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["temperature_2m"].__setitem__(10, "12"))),  # no numérica
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["temperature_2m"].__setitem__(10, True))),  # booleano
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["temperature_2m"].__setitem__(10, 80.0))),  # imposible
    RespuestaFalsa(datos=_modificar(lambda d: d["hourly"]["time"].__setitem__(0, "ayer"))),           # hora ilegible
])
def test_respuestas_invalidas_del_proveedor_son_502(cliente, monkeypatch, respuesta):
    simular_http(monkeypatch, respuesta)
    r = pedir(cliente, fecha=DIA)
    assert r.status_code == 502, r.text


@pytest.mark.parametrize("respuesta, excepcion, esperado", [
    (RespuestaFalsa(400, {"error": True, "reason": "Parameter 'start_date' is out of allowed range from 2026-07-28 to 2026-11-13"}),
     None, 422),                                                                    # la fecha pedida no tiene pronóstico
    (RespuestaFalsa(400, {"error": True, "reason": "Cannot initialize WeatherVariable"}), None, 502),
    (RespuestaFalsa(500, {}), None, 503),
    (RespuestaFalsa(429, {}), None, 503),
    (RespuestaFalsa(404, {}), None, 502),
    (None, requests.exceptions.ConnectionError(), 503),
    (None, requests.exceptions.Timeout(), 503),
])
def test_fallas_del_proveedor_se_distinguen_del_pedido_invalido(cliente, monkeypatch, respuesta, excepcion, esperado):
    simular_http(monkeypatch, respuesta, excepcion)
    assert pedir(cliente, fecha=DIA).status_code == esperado


# ---------------------------------------------------------------------------
# Equivalencia con la evaluación
# ---------------------------------------------------------------------------
def test_central_y_banda_coinciden_con_el_pipeline_evaluado(cliente):
    """Con las temperaturas de las corridas disponibles al emitir (agosto de 2026), la API devuelve exactamente
    el valor central y los extremos que da el pipeline del notebook 04 con el mismo snapshot."""
    pron = pd.read_csv(BASE / "data" / "processed" / "temperatura_pronosticada_agosto_2026.csv",
                       parse_dates=["timestamp", "dia_objetivo"])
    dia = pd.Timestamp(DIA)
    pron_dia = pron[pron["dia_objetivo"] == dia]
    serie = pron_dia.set_index("timestamp")["temperatura_pron_c"]          # D-1 01:00 -> D 23:00
    anterior = [float(serie.iloc[0])] + serie.loc[:dia - pd.Timedelta(hours=1)].tolist()   # la de D-1 00:00 no se usa
    datos = pedir(cliente, fecha=DIA, temperaturas=serie.loc[dia:].tolist(), temperaturas_dia_anterior=anterior).json()

    X = pd.DataFrame(index=pd.date_range(dia, periods=24, freq="h"))
    X["hora"], X["dia_semana"], X["mes"], X["es_habil"] = X.index.hour, X.index.dayofweek, X.index.month, 1
    X[md.CLIMATICAS] = md.clima_al_emitir(pron_dia)[md.CLIMATICAS].values
    modelos, ajuste = api.estado["modelos"], api.estado["metadatos"]["banda"]["ajuste_conformal_mw"]
    central = modelos["central"].predict(X[md.VARIABLES])
    inferior, superior = md.banda_80(central, modelos["q10"].predict(X[md.VARIABLES]), modelos["q90"].predict(X[md.VARIABLES]), ajuste)
    assert np.allclose([h["demanda_mwh"] for h in datos["horas"]], central, atol=0.05)
    assert np.allclose([h["banda_inferior_mwh"] for h in datos["horas"]], inferior, atol=0.05)
    assert np.allclose([h["banda_superior_mwh"] for h in datos["horas"]], superior, atol=0.05)


def test_regla_de_banda_contiene_al_central():
    central = np.array([100.0, 100.0, 100.0])
    inferior, superior = md.banda_80(central, q10=[90.0, 105.0, 90.0], q90=[110.0, 120.0, 95.0], ajuste=2.0)
    assert inferior.tolist() == [88.0, 100.0, 88.0] and superior.tolist() == [112.0, 122.0, 100.0]


# ---------------------------------------------------------------------------
# Módulo y entrenamiento
# ---------------------------------------------------------------------------
def test_clima_del_dia_exige_historial_completo():
    dia = pd.Timestamp(DIA)
    solo_dia = pd.Series(INVIERNO, index=pd.date_range(dia, periods=24, freq="h"))
    with pytest.raises(ValueError, match="faltan"):
        md.clima_del_dia(solo_dia, dia)
    horas = md.horas_con_historial(dia)
    duplicada = pd.Series(range(len(horas) + 1), index=horas.append(horas[:1]), dtype=float)
    with pytest.raises(ValueError, match="repetidas"):
        md.clima_del_dia(duplicada, dia)


def test_entrenamiento_valida_cobertura_de_los_datos():
    df, pron = em.cargar_datos()
    with pytest.raises(em.DatosInsuficientes, match="hacen falta datos"):
        em.validar_cobertura(df, pron, pd.Timestamp("2026-10-01"))
    em.validar_cobertura(df, pron, pd.Timestamp("2026-08-01"))   # el snapshot evaluado sí cumple
