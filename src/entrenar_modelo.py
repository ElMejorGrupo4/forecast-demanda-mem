"""
Entrena un modelo para la API (api/main.py) y lo guarda como snapshot en models/<id>/.

Aplica exactamente la configuracion fijada (src/modelo_demanda.py):
  - Gradient Boosting de calendario y clima con los hiperparametros del notebook 02.
  - Entrenamiento con los ultimos 24 meses disponibles al corte de la emision de la
    FECHA OBJETIVO (primer dia que pronostica el modelo): el dia anterior a las 11:00,
    con las reglas de disponibilidad de la seccion 4 del notebook 02.
  - Banda del 80 % (nivel nominal): cuantiles 10 y 90 con la misma regla, ajuste conformal
    con los puntajes de los ultimos 3 meses conocidos al corte, y la regla de banda unica
    del modulo (md.banda_80).

El snapshot evaluado es gb_calendario_clima_2026-08-01 (fecha objetivo 01/08/2026): es el
modelo que se evaluo en agosto de 2026 en el notebook 04. Nunca se sobrescribe un snapshot.

Antes de entrenar se valida que los datos alcancen: la serie tiene que llegar al corte, la
ventana de 24 meses tiene que estar completa y tiene que haber pronosticos al emitir para los
3 meses de calibracion. Si no alcanzan, el script se detiene y dice que falta.

Este script NO escribe metricas: las metricas son resultados de un experimento sobre un
artefacto concreto y estan en models/resultados_experimentos.json, atadas a su huella sha256.

Salidas:
    models/<id>/modelo.joblib      modelos central, q10 y q90
    models/<id>/metadatos.json     configuracion, datos usados (con huellas), entorno, corte y advertencias

La fecha objetivo puede ser cualquier dia: si la ventana de calibracion empieza a mitad de
mes, md.reentrenar_mensual cubre ese mes parcial con el modelo de su dia 1 y predice solo las
filas pedidas. Si faltan predicciones o puntajes, el proceso se detiene.

Reproducibilidad: hay dos niveles distintos.
  - numerica: el modelo regenerado da exactamente las mismas predicciones;
  - binaria: el archivo regenerado tiene el mismo sha256.
La binaria se verifico solo en el entorno original (Python 3.12.6, Windows, requirements.txt).
En otro entorno el sha256 puede cambiar aunque las predicciones sean identicas; por eso las
metricas quedan atadas al sha256 del artefacto exacto que se evaluo. --comparar-con informa
los dos niveles por separado.

Uso:
    python src/entrenar_modelo.py --fecha-objetivo 2026-08-01
    python src/entrenar_modelo.py --fecha-objetivo 2026-08-01 --destino <carpeta> --comparar-con gb_calendario_clima_2026-08-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
MODELS = BASE / "models"
ARCHIVOS_DATOS = {
    "demanda": DATA / "demanda_horaria_mem.csv",
    "temperatura_historica": DATA / "temperatura_horaria_ba.csv",
    "temperatura_al_emitir": DATA / "temperatura_pronosticada_ba.csv",
}


class DatosInsuficientes(Exception):
    """Los datos no alcanzan para entrenar con esa fecha objetivo sin violar las reglas."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def commit_del_codigo() -> dict:
    """Commit de git del codigo con el que se entrena (y si habia cambios sin commitear)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True, check=True).stdout.strip()
        sucio = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "api"], cwd=BASE,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "cambios_sin_commitear_en_src_o_api": sucio}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "cambios_sin_commitear_en_src_o_api": None}


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    serie = pd.read_csv(ARCHIVOS_DATOS["demanda"], parse_dates=["timestamp"])
    temperatura = pd.read_csv(ARCHIVOS_DATOS["temperatura_historica"], parse_dates=["timestamp"])
    df = md.armar_variables(serie.merge(temperatura, on="timestamp", how="left", validate="one_to_one"))
    pron = pd.read_csv(ARCHIVOS_DATOS["temperatura_al_emitir"], parse_dates=["timestamp", "dia_objetivo"])
    return df, pron


def validar_cobertura(df: pd.DataFrame, pron: pd.DataFrame, fecha: pd.Timestamp) -> None:
    """Comprueba que los datos alcancen para la fecha objetivo; si no, explica que falta."""
    corte = md.corte(fecha)
    ultima_utilizable = corte - md.DURACION - md.DEMORA_TEMPERATURA   # ultima hora que se puede usar para entrenar
    if df["timestamp"].max() < ultima_utilizable:
        raise DatosInsuficientes(
            f"la serie termina el {df['timestamp'].max()}, pero para la fecha objetivo {fecha.date()} hacen falta datos "
            f"hasta el {ultima_utilizable} (corte {corte}). Actualizar la serie y la temperatura historica antes de entrenar.")
    if df["timestamp"].min() > fecha - pd.DateOffset(months=md.VENTANA_MESES):
        raise DatosInsuficientes(f"la serie empieza el {df['timestamp'].min()}: no alcanza para una ventana de {md.VENTANA_MESES} meses")
    if df["timestamp"].max() >= fecha:
        raise DatosInsuficientes("la serie incluye horas del periodo a pronosticar: elegir una fecha objetivo posterior a los datos")
    inicio_cal = fecha - pd.DateOffset(months=md.CALIBRACION_MESES)
    dias_cal = pd.date_range(inicio_cal, fecha - pd.Timedelta(days=1), freq="D")
    faltan = dias_cal.difference(pd.DatetimeIndex(pron["dia_objetivo"].unique()))
    if len(faltan):
        raise DatosInsuficientes(
            f"faltan pronosticos al emitir para {len(faltan)} dias de calibracion ({faltan[0].date()} -> {faltan[-1].date()}). "
            "Correr src/descargar_temperatura_pronosticada.py para ese periodo.")


def entrenar(fecha: pd.Timestamp, df: pd.DataFrame, pron: pd.DataFrame):
    validar_cobertura(df, pron, fecha)
    disponible = (md.fila_para_entrenar(df["timestamp"], fecha)
                  & (df["timestamp"] >= fecha - pd.DateOffset(months=md.VENTANA_MESES)))
    X, y = df.loc[disponible, md.VARIABLES], df.loc[disponible, "demanda_mwh"]
    if X.isna().any().any() or y.isna().any():
        raise DatosInsuficientes("hay valores faltantes en la ventana de entrenamiento")
    modelos = {"central": md.entrenar(X, y)}
    for q in md.CUANTILES:
        modelos[f"q{round(q * 100)}"] = md.entrenar(X, y, q)

    # ajuste conformal: puntajes fuera de muestra de los ultimos 3 meses, con la temperatura disponible al emitir
    clima = md.clima_al_emitir(pron)
    previos = (df["timestamp"] >= fecha - pd.DateOffset(months=md.CALIBRACION_MESES)) & (df["timestamp"] < fecha)
    t_prev = df.loc[previos, "timestamp"]
    X_prev = df.loc[previos, md.VARIABLES].copy()
    X_prev[md.CLIMATICAS] = clima.loc[t_prev.values, md.CLIMATICAS].values
    reent = md.reentrenar_mensual(df, X_prev, t_prev)
    puntajes = pd.DataFrame({"timestamp": t_prev.values, "real": df.loc[previos, "demanda_mwh"].values,
                             "q10": reent["q10"].values, "q90": reent["q90"].values})
    puntajes["puntaje"] = np.maximum(puntajes["q10"] - puntajes["real"], puntajes["real"] - puntajes["q90"])
    calibracion = pd.concat([puntajes, pd.DataFrame({"timestamp": [fecha]})], ignore_index=True)
    ajuste = float(md.calibrar(calibracion, [fecha]).iloc[-1])
    return modelos, ajuste, df.loc[disponible, "timestamp"], X, X_prev


def comparar_con_snapshot(modelos: dict, artefacto_nuevo: Path, snapshot_id: str, X_comparacion: pd.DataFrame) -> dict:
    """Reproduccion numerica (predicciones) y binaria (sha256) frente a un snapshot existente."""
    original = MODELS / snapshot_id / "modelo.joblib"
    if not original.exists():
        raise SystemExit(f"no existe el snapshot {snapshot_id} para comparar")
    previos = joblib.load(original)
    diferencias = {k: float(np.max(np.abs(modelos[k].predict(X_comparacion) - previos[k].predict(X_comparacion))))
                   for k in modelos}
    return {
        "horas_comparadas": int(len(X_comparacion)),
        "predicciones_identicas": all(d == 0.0 for d in diferencias.values()),
        "diferencia_maxima_mwh": diferencias,
        "sha256_identico": sha256(artefacto_nuevo) == sha256(original),
    }


def metadatos(identificador, fecha, ajuste, t_ent, X, artefacto: Path) -> dict:
    return {
        "id": identificador,
        "modelo": "Gradient Boosting de calendario y clima (HistGradientBoostingRegressor)",
        "fecha_objetivo": str(fecha.date()),
        "emision_de_la_fecha_objetivo": str(fecha - pd.Timedelta(days=1) + md.EMISION),
        "corte_de_informacion": str(md.corte(fecha)),
        "datos_de_entrenamiento": {
            "desde": str(t_ent.min()), "hasta": str(t_ent.max()), "horas": int(len(t_ent)),
            "regla": f"últimos {md.VENTANA_MESES} meses disponibles al corte (reglas de disponibilidad del notebook 02, sección 4)",
        },
        "rango_temperatura_entrenamiento_c": [float(X["temperatura_c"].min()), float(X["temperatura_c"].max())],
        "banda": {
            "nivel_nominal": 0.80,
            "cuantiles": list(md.CUANTILES),
            "ajuste_conformal_mw": round(ajuste, 1),
            "calibracion": f"percentil 80 de los puntajes de los últimos {md.CALIBRACION_MESES} meses conocidos al corte",
            "regla": "md.banda_80: cuantiles 10 y 90 ± ajuste conformal; si la banda no contiene al valor central, se extiende hasta él",
            "aclaracion": "80 % es el nivel nominal; la cobertura empírica observada fue menor en varios meses (ver resultados_experimentos.json)",
        },
        "configuracion": {
            "variables": md.VARIABLES,
            "hiperparametros": md.PARAMS_GB,
            "temperatura_de_confort_c": md.TEMP_CONFORT,
            "ventana_meses": md.VENTANA_MESES,
            "calibracion_meses": md.CALIBRACION_MESES,
            "emision": "día anterior a las 12:00", "margen_para_entrenar_y_emitir_h": md.MARGEN / pd.Timedelta(hours=1),
            "demora_supuesta_temperatura_historica_h": md.DEMORA_TEMPERATURA / pd.Timedelta(hours=1),
        },
        "datos": {
            nombre: {"archivo": str(ruta.relative_to(BASE)).replace("\\", "/"), "sha256": sha256(ruta)}
            for nombre, ruta in ARCHIVOS_DATOS.items()
        },
        "temperatura": "Open-Meteo, modelo ECMWF IFS 9 km, coordenadas de Aeroparque (-34.56, -58.42)",
        "entorno": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__,
        },
        "codigo": commit_del_codigo(),
        "artefacto": {"archivo": f"models/{identificador}/modelo.joblib", "sha256": sha256(artefacto)},
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "metricas": "Este archivo no incluye métricas. Las métricas son resultados de experimentos sobre un artefacto "
                    "concreto: ver models/resultados_experimentos.json (válidas solo si coincide el sha256 del artefacto).",
        "advertencias": [
            "El 80 % de la banda es su nivel nominal: en las evaluaciones la cobertura empírica fue menor en varios meses.",
            "El modelo tiende a quedarse corto en los máximos diarios y en las horas de demanda alta.",
            "Describe asociaciones entre demanda, calendario y temperatura; no relaciones causales.",
            "Usa la temperatura de una sola ciudad (Buenos Aires) para la demanda de todo el país.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un snapshot del modelo de la API.")
    parser.add_argument("--fecha-objetivo", required=True, help="primer día que pronostica el modelo (AAAA-MM-DD)")
    parser.add_argument("--destino", help="carpeta de salida (por defecto models/<id>)")
    parser.add_argument("--comparar-con", help="id de un snapshot existente: informa reproduccion numerica y binaria")
    args = parser.parse_args()

    fecha = pd.Timestamp(args.fecha_objetivo)
    identificador = f"gb_calendario_clima_{fecha.date()}"
    destino = Path(args.destino) if args.destino else MODELS / identificador
    if (destino / "modelo.joblib").exists():
        raise SystemExit(f"ya existe {destino}: los snapshots no se sobrescriben")

    df, pron = cargar_datos()
    try:
        modelos, ajuste, t_ent, X, X_prev = entrenar(fecha, df, pron)
    except DatosInsuficientes as e:
        raise SystemExit(f"No se puede entrenar: {e}")
    print(f"Entrenado con {len(t_ent):,} horas: {t_ent.min()} -> {t_ent.max()} | ajuste conformal {ajuste:,.0f} MW")

    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelos, destino / "modelo.joblib")
    meta = metadatos(identificador, fecha, ajuste, t_ent, X, destino / "modelo.joblib")
    (destino / "metadatos.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {destino}")
    if args.comparar_con:
        X_comparacion = pd.concat([X, X_prev])
        resultado = comparar_con_snapshot(modelos, destino / "modelo.joblib", args.comparar_con, X_comparacion)
        print(f"Comparacion con {args.comparar_con} sobre {resultado['horas_comparadas']:,} horas:")
        print(f"  reproduccion numerica (predicciones identicas): {resultado['predicciones_identicas']} "
              f"| diferencia maxima: {resultado['diferencia_maxima_mwh']}")
        print(f"  reproduccion binaria (mismo sha256): {resultado['sha256_identico']}")


if __name__ == "__main__":
    main()
