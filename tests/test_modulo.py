"""
Pruebas del modulo compartido (src/modelo_demanda.py) y de la validacion del entrenamiento:
meses parciales, controles de la calibracion y fechas objetivo admitidas o rechazadas.

Uso (desde la carpeta del proyecto):
    python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import entrenar_modelo as em     # noqa: E402
import modelo_demanda as md      # noqa: E402


@pytest.fixture(scope="module")
def datos():
    return em.cargar_datos()   # serie con variables y temperatura historica, y pronosticos al emitir


def filas(df, desde, hasta):
    mascara = (df["timestamp"] >= desde) & (df["timestamp"] <= hasta)
    return df.loc[mascara, md.VARIABLES], df.loc[mascara, "timestamp"]


# ---------------------------------------------------------------------------
# Meses parciales
# ---------------------------------------------------------------------------
def test_reentrenar_cubre_meses_parciales(datos):
    """Un periodo del 02/05 al 02/06 tiene que predecir todas sus horas (antes quedaban 720 de mayo sin prediccion)."""
    df, _ = datos
    X, t = filas(df, "2026-05-02 00:00", "2026-06-02 23:00")
    salida = md.reentrenar_mensual(df, X, t, cuantiles=())
    assert len(salida) == 32 * 24
    assert salida["pred"].notna().all()


def test_mes_parcial_usa_el_modelo_del_dia_1(datos):
    """Las horas de mayo de un periodo que empieza el 02/05 se predicen igual que con el mes completo:
    con el modelo entrenado al corte del 01/05 (30/04 a las 11:00), anterior a todas ellas."""
    df, _ = datos
    X_parcial, t_parcial = filas(df, "2026-05-02 00:00", "2026-05-31 23:00")
    X_completo, t_completo = filas(df, "2026-05-01 00:00", "2026-05-31 23:00")
    parcial = md.reentrenar_mensual(df, X_parcial, t_parcial, cuantiles=())
    completo = md.reentrenar_mensual(df, X_completo, t_completo, cuantiles=())
    assert np.array_equal(parcial["pred"].values, completo.loc[parcial.index, "pred"].values)


# ---------------------------------------------------------------------------
# Calibración: nada se omite en silencio
# ---------------------------------------------------------------------------
def historia_sintetica(desde, hasta, semilla=0):
    horas = pd.date_range(desde, hasta, freq="h")
    return pd.DataFrame({"timestamp": horas, "puntaje": np.random.default_rng(semilla).normal(-200, 300, len(horas))})


def test_calibrar_con_historia_completa():
    historia = historia_sintetica("2026-02-01", "2026-05-31 23:00")
    ajuste = md.calibrar(historia, [pd.Timestamp("2026-05-01")])
    en_mayo = historia["timestamp"] >= "2026-05-01"
    assert ajuste[en_mayo].notna().all() and ajuste[~en_mayo].isna().all()


def test_calibrar_desde_mitad_de_mes():
    historia = historia_sintetica("2026-02-01", "2026-05-31 23:00")
    ajuste = md.calibrar(historia, [pd.Timestamp("2026-05-15")])
    assert ajuste[historia["timestamp"] >= "2026-05-15"].notna().all()


def test_calibrar_se_detiene_sin_historia_suficiente():
    historia = historia_sintetica("2026-04-01", "2026-05-31 23:00")   # 1 mes de los 3 que pide la ventana
    with pytest.raises(ValueError, match="historia insuficiente"):
        md.calibrar(historia, [pd.Timestamp("2026-05-01")])


def test_calibrar_se_detiene_con_puntajes_faltantes():
    historia = historia_sintetica("2026-02-01", "2026-05-31 23:00")
    historia.loc[100, "puntaje"] = np.nan
    with pytest.raises(ValueError, match="faltan 1 puntajes"):
        md.calibrar(historia, [pd.Timestamp("2026-05-01")])


# ---------------------------------------------------------------------------
# Fechas objetivo del entrenamiento
# ---------------------------------------------------------------------------
def test_fecha_objetivo_admitida(datos):
    df, pron = datos
    em.validar_cobertura(df, pron, pd.Timestamp("2026-08-01"))   # el snapshot evaluado


@pytest.mark.parametrize("fecha, motivo", [
    ("2026-08-02", "faltan pronosticos"),          # la calibracion necesitaria el pronostico del 01/08
    ("2026-10-01", "hacen falta datos"),           # la serie no llega al corte
    ("2026-07-15", "incluye horas del periodo"),   # la serie ya contiene el periodo a pronosticar
    ("2022-06-01", "no alcanza para una ventana"), # no hay 24 meses de historia
])
def test_fechas_objetivo_rechazadas(datos, fecha, motivo):
    df, pron = datos
    with pytest.raises(em.DatosInsuficientes, match=motivo):
        em.validar_cobertura(df, pron, pd.Timestamp(fecha))
