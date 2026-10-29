"""
Entrena el modelo que sirve la API (api/main.py) y lo guarda en models/.

Usa exactamente la configuracion fijada (src/modelo_demanda.py):
  - Gradient Boosting de calendario y clima con los hiperparametros elegidos
    en el notebook 02.
  - Reentrenamiento con los ultimos 24 meses disponibles al corte de la
    primera emision posterior a los datos (el 31/07/2026 a las 11:00, para
    pronosticar desde el 01/08/2026). Es el mismo modelo que se evaluo en
    agosto de 2026 en el notebook 04.
  - Banda del 80 %: modelos de los cuantiles 10 y 90 con la misma regla, mas
    el ajuste conformal calculado con los puntajes de los ultimos 3 meses
    conocidos al corte (mayo a julio de 2026).

Salidas:
    models/modelo_demanda.joblib   modelos central, q10 y q90
    models/modelo_demanda.json     metadatos: datos usados, parametros, ajuste conformal, metricas y advertencias

Uso:
    python src/entrenar_modelo.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_demanda as md   # noqa: E402

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
SALIDA = BASE / "models"
PRIMER_DIA = pd.Timestamp("2026-08-01")   # primer dia que pronostica el modelo guardado


def main() -> None:
    serie = pd.read_csv(DATA / "demanda_horaria_mem.csv", parse_dates=["timestamp"])
    temperatura = pd.read_csv(DATA / "temperatura_horaria_ba.csv", parse_dates=["timestamp"])
    df = md.armar_variables(serie.merge(temperatura, on="timestamp", how="left", validate="one_to_one"))
    assert df["timestamp"].max() < PRIMER_DIA, "los datos no pueden llegar al periodo que se pronostica"

    # filas disponibles al corte del 31/07/2026 11:00, ultimos 24 meses
    disponible = (md.fila_para_entrenar(df["timestamp"], PRIMER_DIA)
                  & (df["timestamp"] >= PRIMER_DIA - pd.DateOffset(months=md.VENTANA_MESES)))
    X, y = df.loc[disponible, md.VARIABLES], df.loc[disponible, "demanda_mwh"]
    print(f"Entrenando con {len(X):,} horas: {df.loc[disponible, 'timestamp'].min()} -> {df.loc[disponible, 'timestamp'].max()}")
    modelos = {"central": md.entrenar(X, y)}
    for q in md.CUANTILES:
        modelos[f"q{round(q * 100)}"] = md.entrenar(X, y, q)

    # ajuste conformal: puntajes de mayo a julio de 2026 (fuera de muestra, con la temperatura disponible al emitir)
    pron = pd.read_csv(DATA / "temperatura_pronosticada_ba.csv", parse_dates=["timestamp", "dia_objetivo"])
    clima = md.clima_al_emitir(pron)
    previos = (df["timestamp"] >= PRIMER_DIA - pd.DateOffset(months=md.CALIBRACION_MESES)) & (df["timestamp"] < PRIMER_DIA)
    t_prev = df.loc[previos, "timestamp"]
    X_prev = df.loc[previos, md.VARIABLES].copy()
    X_prev[md.CLIMATICAS] = clima.loc[t_prev.values, md.CLIMATICAS].values
    reent = md.reentrenar_mensual(df, X_prev, t_prev)
    puntajes = pd.DataFrame({
        "timestamp": t_prev.values,
        "real": df.loc[previos, "demanda_mwh"].values,
        "q10": reent["q10"].values, "q90": reent["q90"].values,
    })
    puntajes["puntaje"] = np.maximum(puntajes["q10"] - puntajes["real"], puntajes["real"] - puntajes["q90"])
    calibracion = pd.concat([puntajes, pd.DataFrame({"timestamp": [PRIMER_DIA]})], ignore_index=True)
    ajuste = float(md.calibrar(calibracion, [PRIMER_DIA]).iloc[-1])
    print(f"Ajuste conformal de la banda: {ajuste:,.0f} MW")

    SALIDA.mkdir(exist_ok=True)
    joblib.dump(modelos, SALIDA / "modelo_demanda.joblib")
    metadatos = {
        "modelo": "Gradient Boosting de calendario y clima (HistGradientBoostingRegressor)",
        "variables": md.VARIABLES,
        "hiperparametros": md.PARAMS_GB,
        "temperatura_de_confort_c": md.TEMP_CONFORT,
        "rango_temperatura_entrenamiento_c": [float(X["temperatura_c"].min()), float(X["temperatura_c"].max())],
        "datos_de_entrenamiento": {
            "desde": str(df.loc[disponible, "timestamp"].min()),
            "hasta": str(df.loc[disponible, "timestamp"].max()),
            "horas": int(len(X)),
            "regla": f"últimos {md.VENTANA_MESES} meses disponibles al corte de la emisión del {PRIMER_DIA.date()}",
        },
        "banda": {
            "nivel": 0.80,
            "cuantiles": list(md.CUANTILES),
            "ajuste_conformal_mw": round(ajuste, 1),
            "calibracion": f"percentil 80 de los puntajes de los últimos {md.CALIBRACION_MESES} meses conocidos al corte",
        },
        "metricas": {
            "agosto_2026_periodo_nuevo": {"MAPE_%": 3.98, "WAPE_%": 4.06, "cobertura_banda_%": 72.4,
                                          "fuente": "notebook 04 (evaluación pre-registrada)"},
            "test_ene2025_jul2026": {"MAPE_%": 4.08, "WAPE_%": 4.18, "cobertura_banda_%": 79.1,
                                     "fuente": "notebook 02 (test reutilizado en las revisiones)"},
        },
        "temperatura": "Open-Meteo, modelo ECMWF IFS 9 km, coordenadas de Aeroparque (-34.56, -58.42)",
        "advertencias": [
            "La banda del 80 % no está bien calibrada en todos los meses: en agosto de 2026 cubrió el 72 % de las horas "
            "y en el test hubo 5 de 19 meses por debajo del 75 %.",
            "El modelo tiende a quedarse corto en los máximos diarios y en las horas de demanda alta.",
            "Describe asociaciones entre demanda, calendario y temperatura; no relaciones causales.",
            "Usa la temperatura de una sola ciudad (Buenos Aires) para la demanda de todo el país.",
        ],
        "version_scikit_learn": sklearn.__version__,
        "entrenado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }
    (SALIDA / "modelo_demanda.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: models/modelo_demanda.joblib y models/modelo_demanda.json")


if __name__ == "__main__":
    main()
