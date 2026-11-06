"""
Pruebas de la demo con Streamlit (demo/streamlit_app.py), con el probador oficial de Streamlit (AppTest).

Se saltean si Streamlit no está instalado (las dependencias de la demo están en demo/requirements.txt).
No necesitan internet: la consulta a Open-Meteo se reemplaza por respuestas simuladas a nivel HTTP,
y se ejecuta la lógica real de la API.

Uso (desde la carpeta del proyecto, con las dependencias de la demo):
    python -m pytest tests -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest   # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import api.main as api   # noqa: E402
sys.path.insert(0, str(BASE / "demo"))
from caso_historico import pedido_historico   # noqa: E402

APP = str(BASE / "demo" / "streamlit_app.py")
MODO_HISTORICO = "Caso histórico validado — 12/08/2026"


class RespuestaFalsa:
    def __init__(self, status=200, datos=None):
        self.status_code, self._datos = status, datos

    def json(self):
        return self._datos


def respuesta_para(params):
    """Respuesta válida de Open-Meteo para las fechas pedidas (48 h)."""
    horas = pd.date_range(params["start_date"], periods=48, freq="h")
    curva = [12, 11.5, 11, 10.5, 10, 10, 10, 10.5, 11.5, 13, 14.5, 16, 17, 18, 18.5, 18.5, 18, 17, 15.5, 14.5, 14, 13.5, 13, 12.5]
    return {"latitude": -34.551846, "longitude": -58.448944, "timezone": "America/Argentina/Buenos_Aires",
            "utc_offset_seconds": -10800,
            "hourly": {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in horas], "temperature_2m": curva * 2}}


def abrir():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def textos(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(w.value for w in at.warning)


def test_pagina_inicial_sin_pronostico():
    at = abrir()
    assert at.title[0].value == "Forecast de demanda eléctrica horaria del MEM"
    assert "MAPE test 4,1 %" in at.info[0].value and "MAPE agosto de 2026 4,0 %" in at.info[0].value
    assert "no el error del pronóstico mostrado" in at.info[0].value
    assert "modelos" in api.estado   # el snapshot quedó cargado (st.cache_resource)


def test_modo_actual_hace_una_sola_consulta(monkeypatch):
    llamadas = []

    def falso_get(url, params=None, **kwargs):
        llamadas.append(params)
        return RespuestaFalsa(datos=respuesta_para(params))
    monkeypatch.setattr(api.requests, "get", falso_get)

    at = abrir()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(llamadas) == 1                          # la demo no vuelve a pedir la temperatura
    assert [m.label for m in at.metric] == ["Demanda máxima (MWh)", "Hora del pico", "Demanda mínima (MWh)",
                                            "Energía del día (GWh)", "Temp. mín / máx (°C)"]
    assert "pronostico_actual" in textos(at)
    assert "no informada por el proveedor" in textos(at)
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 24   # tabla de variables usadas


def test_caso_historico_validado():
    at = abrir()
    at.radio[0].set_value(MODO_HISTORICO).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    texto = textos(at)
    assert MODO_HISTORICO in texto and "no un pronóstico actual" in texto
    assert "simulacion_manual" in texto and "historial térmico completo" in texto
    assert "MAPE del 12/08/2026" in texto
    assert "2026-08-11 00:00 y 2026-08-11 06:00" in texto      # corridas registradas, no "no informada"
    assert "no informada por el proveedor" not in texto
    assert at.subheader[0].value.startswith("Miércoles 12/08/2026")


def test_caso_historico_reproduce_el_pipeline_evaluado():
    """La demo usa las temperaturas de la corrida registrada y da lo mismo que la API con esas temperaturas."""
    pedido, real = pedido_historico()
    assert len(pedido.temperaturas) == 24 and len(pedido.temperaturas_dia_anterior) == 24 and len(real) == 24
    respuesta, variables = api.calcular_pronostico(pedido)
    assert respuesta.alcance.modo == "simulacion_manual" and respuesta.alcance.historial_termico == "completo"
    # el valor de las 00:00 del día anterior no interviene: cambiarlo no cambia el pronóstico
    otro = pedido.model_copy(update={"temperaturas_dia_anterior": [0.0] + pedido.temperaturas_dia_anterior[1:]})
    assert api.calcular_pronostico(otro)[0].horas == respuesta.horas


@pytest.mark.parametrize("falla, mensaje", [
    (requests.exceptions.ConnectionError(), "Open-Meteo no responde"),
    (RespuestaFalsa(200, {"latitude": -34.55}), "Open-Meteo respondió datos inválidos"),
])
def test_error_del_proveedor_se_muestra_claro(monkeypatch, falla, mensaje):
    def falso_get(*args, **kwargs):
        if isinstance(falla, Exception):
            raise falla
        return falla
    monkeypatch.setattr(api.requests, "get", falso_get)
    at = abrir()
    at.button[0].click().run()
    assert not at.exception
    assert mensaje in at.error[0].value
    assert MODO_HISTORICO in at.info[-1].value          # sugiere el caso histórico para seguir sin conexión
    assert len(at.metric) == 0
