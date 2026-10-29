"""
Pruebas de la API (api/main.py).

Uso (desde la carpeta del proyecto):
    python -m pytest tests -v

No necesitan internet: la consulta a Open-Meteo se reemplaza por una funcion falsa.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import api.main as api          # noqa: E402
import modelo_demanda as md     # noqa: E402

# una curva de temperatura de invierno razonable: minima de madrugada, maxima a la tarde
TEMPERATURAS_INVIERNO = [9, 8.5, 8, 7.5, 7, 7, 7, 7.5, 8.5, 10, 11.5, 13, 14, 15, 15.5, 15.5, 15, 14, 12.5, 11.5, 11, 10.5, 10, 9.5]


@pytest.fixture(scope="module")
def cliente():
    with TestClient(api.app) as c:   # el "with" ejecuta la carga del modelo
        yield c


def pedir(cliente, **pedido):
    return cliente.post("/pronostico", json=pedido)


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json() == {"estado": "ok", "modelo_cargado": True}


def test_modelo_describe_variables_y_banda(cliente):
    meta = cliente.get("/modelo").json()
    assert meta["variables"] == md.VARIABLES
    assert meta["banda"]["nivel"] == 0.8 and meta["banda"]["ajuste_conformal_mw"] > 0


def test_pronostico_con_temperaturas_manuales(cliente):
    r = pedir(cliente, fecha="2026-08-12", temperaturas=TEMPERATURAS_INVIERNO)
    assert r.status_code == 200
    datos = r.json()
    assert datos["tipo_dia"] == "habil"            # miércoles
    assert datos["fuente_temperatura"] == "Temperaturas enviadas en el pedido"
    assert [h["hora"] for h in datos["horas"]] == list(range(24))
    for h in datos["horas"]:
        assert h["banda_inferior_mwh"] <= h["demanda_mwh"] <= h["banda_superior_mwh"]
        assert 8_000 < h["demanda_mwh"] < 35_000   # rango plausible del MEM
    assert datos["resumen"]["demanda_maxima_mwh"] == max(h["demanda_mwh"] for h in datos["horas"])


def test_fin_de_semana_y_feriado_son_no_habiles(cliente):
    domingo = pedir(cliente, fecha="2026-08-16", temperaturas=TEMPERATURAS_INVIERNO).json()
    assert domingo["tipo_dia"] == "no_habil"
    habil = pedir(cliente, fecha="2026-08-12", temperaturas=TEMPERATURAS_INVIERNO).json()
    feriado = pedir(cliente, fecha="2026-08-12", es_feriado=True, temperaturas=TEMPERATURAS_INVIERNO).json()
    assert feriado["tipo_dia"] == "no_habil"
    # con el mismo clima, un feriado demanda menos energía que un día hábil
    assert feriado["resumen"]["energia_del_dia_mwh"] < habil["resumen"]["energia_del_dia_mwh"]


@pytest.mark.parametrize("pedido", [
    {"fecha": "2026-08-12", "temperaturas": TEMPERATURAS_INVIERNO[:23]},                       # faltan horas
    {"fecha": "2026-08-12", "temperaturas": [80] * 24},                                        # temperatura imposible
    {"fecha": "2026-08-12", "temperaturas_dia_anterior": TEMPERATURAS_INVIERNO},               # anterior sin el día
    {"fecha": "12/08/2026", "temperaturas": TEMPERATURAS_INVIERNO},                            # formato de fecha
])
def test_pedidos_invalidos(cliente, pedido):
    assert pedir(cliente, **pedido).status_code == 422


def test_temperatura_fuera_de_rango_avisa(cliente):
    datos = pedir(cliente, fecha="2026-01-15", temperaturas=[42] * 24).json()
    assert any("fuera del rango" in a for a in datos["advertencias"])


def test_usa_open_meteo_si_no_hay_temperaturas(cliente, monkeypatch):
    def falso(fecha):
        horas = pd.date_range(pd.Timestamp(fecha) - pd.Timedelta(days=1), periods=48, freq="h")
        return pd.Series(TEMPERATURAS_INVIERNO * 2, index=horas)
    monkeypatch.setattr(api, "temperaturas_open_meteo", falso)
    datos = pedir(cliente, fecha="2026-08-12").json()
    assert datos["fuente_temperatura"].startswith("Open-Meteo")
    assert len(datos["horas"]) == 24


def test_fecha_sin_pronostico_de_open_meteo(cliente, monkeypatch):
    def falla(fecha):
        raise HTTPException(422, "Open-Meteo no tiene pronóstico para esa fecha")
    monkeypatch.setattr(api, "temperaturas_open_meteo", falla)
    assert pedir(cliente, fecha="2030-01-01").status_code == 422


def test_coincide_con_el_modelo_evaluado_en_agosto(cliente):
    """Con las temperaturas disponibles al emitir, la API arma las mismas variables que el pipeline del notebook 04."""
    pron = pd.read_csv(BASE / "data" / "processed" / "temperatura_pronosticada_agosto_2026.csv",
                       parse_dates=["timestamp", "dia_objetivo"])
    dia = pd.Timestamp("2026-08-12")
    pron_dia = pron[pron["dia_objetivo"] == dia].set_index("timestamp")["temperatura_pron_c"]
    anterior = [float(pron_dia.iloc[0])] + pron_dia.loc[:dia - pd.Timedelta(hours=1)].tolist()   # D-1 00:00 no se usa
    del_dia = pron_dia.loc[dia:].tolist()
    datos = pedir(cliente, fecha="2026-08-12", temperaturas=del_dia, temperaturas_dia_anterior=anterior).json()

    # mismo cálculo, directo con el módulo del proyecto
    X = pd.DataFrame(index=pd.date_range(dia, periods=24, freq="h"))
    X["hora"], X["dia_semana"], X["mes"], X["es_habil"] = X.index.hour, X.index.dayofweek, X.index.month, 1
    clima = md.clima_al_emitir(pron[pron["dia_objetivo"] == dia])
    X[md.CLIMATICAS] = clima[md.CLIMATICAS].values
    esperado = api.estado["modelos"]["central"].predict(X[md.VARIABLES])
    assert np.allclose([h["demanda_mwh"] for h in datos["horas"]], esperado, atol=0.1)
