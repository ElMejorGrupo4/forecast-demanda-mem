"""
Caso historico validado de la demo: 12/08/2026.

Arma el pedido para api.calcular_pronostico con las temperaturas de la corrida de ECMWF disponible al
emitir (11/08/2026 12:00), registradas en data/processed/temperatura_pronosticada_agosto_2026.csv, y
devuelve la demanda real de ese dia que informo CAMMESA. No inventa datos: pasa por el modo de
simulacion manual de la API, con historial termico completo.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import api.main as api   # noqa: E402

DIA_HISTORICO = date(2026, 8, 12)
ARCHIVO_TEMPERATURA = BASE / "data" / "processed" / "temperatura_pronosticada_agosto_2026.csv"
ARCHIVO_DEMANDA = BASE / "data" / "processed" / "demanda_horaria_agosto_2026.csv"


def pedido_historico() -> tuple[api.PedidoPronostico, pd.Series]:
    """Pedido del caso historico y demanda real (24 horas) de ese dia."""
    dia = pd.Timestamp(DIA_HISTORICO)
    pron = pd.read_csv(ARCHIVO_TEMPERATURA, parse_dates=["timestamp", "dia_objetivo"])
    serie = pron[pron["dia_objetivo"] == dia].set_index("timestamp")["temperatura_pron_c"]   # D-1 01:00 -> D 23:00
    previas = serie.loc[:dia - pd.Timedelta(hours=1)].tolist()                                # 23 horas de D-1
    # La API pide 24 temperaturas del dia anterior, pero el promedio de 24 h solo usa de 01:00 a 23:00
    # (md.clima_del_dia): el valor de las 00:00 no interviene. Se completa con el de las 01:00.
    pedido = api.PedidoPronostico(fecha=DIA_HISTORICO, es_feriado=False,
                                  temperaturas=serie.loc[dia:].tolist(),
                                  temperaturas_dia_anterior=[previas[0]] + previas)
    demanda = pd.read_csv(ARCHIVO_DEMANDA, parse_dates=["timestamp"]).set_index("timestamp")["demanda_mwh"]
    return pedido, demanda.loc[dia:dia + pd.Timedelta(hours=23)]


def corridas_historicas() -> list[str]:
    """Corridas de ECMWF (hora UTC de inicio) de las que salen las temperaturas del caso historico."""
    pron = pd.read_csv(ARCHIVO_TEMPERATURA, parse_dates=["dia_objetivo"])
    return sorted(pron.loc[pron["dia_objetivo"] == pd.Timestamp(DIA_HISTORICO), "corrida_utc"].unique())
