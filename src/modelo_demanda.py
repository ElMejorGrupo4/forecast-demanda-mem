"""
Reglas y modelos del pronostico de demanda, tal como quedaron fijados en el
notebook 02 (revision final). Los usa el notebook 04 para evaluar un periodo
nuevo con exactamente la misma configuracion.

Nada de este modulo se ajusta mirando el periodo a evaluar: los
hiperparametros, las variables, la ventana de reentrenamiento y la
calibracion salen de las selecciones hechas en el notebook 02 con
validacion dentro de train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Configuracion fijada (notebook 02)
# ---------------------------------------------------------------------------
TEMP_CONFORT = 18
CATEGORICAS = ["hora", "dia_semana", "mes"]
CLIMATICAS = ["temperatura_c", "temp_24h", "hdd", "cdd"]
NUMERICAS = ["es_habil"] + CLIMATICAS
VARIABLES = CATEGORICAS + NUMERICAS
REZAGOS = ["demanda_hace_48h", "demanda_hace_168h"]   # variante elegida en validacion (seccion 13)

# Gradient Boosting: mejores hiperparametros de la busqueda con validacion temporal (seccion 9)
PARAMS_GB = {
    "min_samples_leaf": 100, "max_iter": 200, "max_depth": 10, "learning_rate": 0.05,
    "l2_regularization": 1.0, "early_stopping": False, "random_state": 42,
}
FIN_TRAIN = pd.Timestamp("2025-01-01")
VENTANA_MESES = 24            # reentrenamiento mensual elegido en la validacion de 2024 (seccion 11.4)
CALIBRACION_MESES = 3         # calibracion conformal elegida en la validacion de 2024 (seccion 11.5)
CUANTILES = (0.1, 0.9)        # banda del 80 %

# Reglas de disponibilidad (seccion 4). Las demoras son SUPUESTOS documentados.
EMISION = pd.Timedelta(hours=12)              # el pronostico de D se emite D-1 a las 12:00
MARGEN = pd.Timedelta(hours=1)                # tiempo reservado para entrenar y emitir
DURACION = pd.Timedelta(hours=1)              # cada fila representa el intervalo [t, t + 1 h)
DEMORA_DEMANDA = pd.Timedelta(0)              # supuesto: medicion en tiempo real
DEMORA_TEMPERATURA = pd.Timedelta(hours=12)   # supuesto conservador para la serie historica de ECMWF IFS


def corte(dia):
    """Momento hasta el que tiene que estar disponible todo lo que se usa para pronosticar el dia `dia`."""
    return dia - pd.Timedelta(days=1) + EMISION - MARGEN


def demanda_conocida(timestamps, dia):
    """True para las horas cuya demanda ya se conocia al corte."""
    return timestamps + DURACION + DEMORA_DEMANDA <= corte(dia)


def fila_para_entrenar(timestamps, dia):
    """True para las horas que se pueden usar para entrenar: demanda y temperatura historica disponibles."""
    return demanda_conocida(timestamps, dia) & (timestamps + DURACION + DEMORA_TEMPERATURA <= corte(dia))


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
def variables_temperatura(temperatura: pd.Series) -> pd.DataFrame:
    """Las 4 variables climaticas a partir de una serie horaria de temperatura."""
    temp_24h = temperatura.rolling(24, min_periods=1).mean()
    return pd.DataFrame({
        "temperatura_c": temperatura,
        "temp_24h": temp_24h,
        "hdd": (TEMP_CONFORT - temp_24h).clip(lower=0),
        "cdd": (temp_24h - TEMP_CONFORT).clip(lower=0),
    })


def armar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calendario, clima con la temperatura historica y rezagos de demanda (df con grilla horaria completa)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert (df["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all(), "la grilla horaria tiene huecos"
    df["hora"] = df["timestamp"].dt.hour
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    df["mes"] = df["timestamp"].dt.month
    df["es_habil"] = (df["tipo_dia"] == "habil").astype(int)
    clima = variables_temperatura(df["temperatura_c"])
    df[clima.columns] = clima
    for k in (48, 168):
        df[f"demanda_hace_{k}h"] = df["demanda_mwh"].shift(k)
    return df


HORAS_PREVIAS = 23   # horas del dia anterior que necesita la ventana de 24 h de la primera hora del dia


def horas_con_historial(dia) -> pd.DatetimeIndex:
    """Las 47 horas que hacen falta para el clima del dia: D-1 01:00 -> D 23:00."""
    dia = pd.Timestamp(dia)
    return pd.date_range(dia - pd.Timedelta(hours=HORAS_PREVIAS), dia + pd.Timedelta(hours=23), freq="h")


def clima_del_dia(temperaturas: pd.Series, dia, historial_completo: bool = True) -> pd.DataFrame:
    """Variables climaticas de las 24 horas del dia `dia`. Es la regla unica que usan la evaluacion y la API.

    historial_completo=True (el modo evaluado): `temperaturas` tiene que cubrir, sin huecos ni duplicados,
    las 47 horas de D-1 01:00 a D 23:00. Asi todas las ventanas de 24 h estan completas.
    historial_completo=False (simulacion aproximada, no evaluada): `temperaturas` trae solo las 24 horas
    de D; el promedio de 24 h de las primeras horas usa solo las horas del dia. No se inventan temperaturas.
    """
    dia = pd.Timestamp(dia)
    if not temperaturas.index.is_unique:
        raise ValueError("hay horas repetidas en las temperaturas")
    if historial_completo:
        necesarias = horas_con_historial(dia)
        faltan = necesarias.difference(temperaturas.index)
        if len(faltan):
            raise ValueError(f"faltan {len(faltan)} horas para el historial completo (desde {faltan[0]})")
        serie = temperaturas.reindex(necesarias)
    else:
        serie = temperaturas.reindex(pd.date_range(dia, periods=24, freq="h"))
    if serie.isna().any():
        raise ValueError("hay temperaturas faltantes")
    return variables_temperatura(serie).loc[dia:]


def banda_80(central, q10, q90, ajuste: float):
    """Regla unica de la banda del 80 % (nivel nominal) que se evalua y se sirve.

    1. Cuantiles 10 y 90 ampliados por el ajuste conformal.
    2. Los modelos de los cuantiles y el central se entrenan por separado y pueden cruzarse: si la banda no
       contiene al valor central, se extiende hasta el. Una banda que no contiene su propio pronostico no
       se puede comunicar; el efecto sobre la cobertura se mide en el notebook 04.
    """
    central = np.asarray(central, dtype=float)
    inferior = np.minimum(np.asarray(q10, dtype=float) - ajuste, central)
    superior = np.maximum(np.asarray(q90, dtype=float) + ajuste, central)
    return inferior, superior


def clima_al_emitir(pronostico: pd.DataFrame) -> pd.DataFrame:
    """Variables climaticas de cada dia con las corridas disponibles al emitir (47 horas por dia -> 24)."""
    partes = []
    for dia, pron_dia in pronostico.groupby("dia_objetivo"):
        pron_dia = pron_dia.set_index("timestamp")
        clima_dia = clima_del_dia(pron_dia["temperatura_pron_c"], dia)
        clima_dia["horas_desde_emision"] = pron_dia.loc[dia:, "horas_desde_emision"]
        partes.append(clima_dia)
    disponible = pd.to_datetime(pronostico["disponible_supuesta_utc"]) - pd.Timedelta(hours=3)
    assert (disponible <= pronostico["dia_objetivo"].map(corte)).all(), \
        "con la demora supuesta, se uso una corrida no disponible al corte"
    return pd.concat(partes)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def entrenar(X, y, cuantil=None):
    if cuantil is None:
        return HistGradientBoostingRegressor(**PARAMS_GB).fit(X, y)
    return HistGradientBoostingRegressor(loss="quantile", quantile=cuantil, **PARAMS_GB).fit(X, y)


def reentrenar_mensual(df, X_pred, t_pred, ventana_meses=VENTANA_MESES, cuantiles=CUANTILES):
    """Igual que en el notebook 02: un modelo por mes, entrenado con lo disponible al corte del dia 1.

    Soporta periodos que empiezan o terminan a mitad de mes: el recorrido arranca el dia 1 del mes de la
    primera fila pedida, cada mes usa el modelo entrenado al corte de su dia 1 (anterior a todas sus filas)
    y solo se predicen las filas pedidas. Si alguna fila queda sin prediccion, se detiene con un error.
    """
    salida = pd.DataFrame(index=X_pred.index, dtype=float)
    primer_mes = t_pred.min().to_period("M").to_timestamp()
    for inicio in pd.date_range(primer_mes, t_pred.max(), freq="MS"):
        fin = inicio + pd.offsets.MonthBegin(1)
        en_mes = ((t_pred >= inicio) & (t_pred < fin)).values
        if not en_mes.any():
            continue
        disponible = fila_para_entrenar(df["timestamp"], inicio)
        if ventana_meses is not None:
            disponible &= df["timestamp"] >= inicio - pd.DateOffset(months=ventana_meses)
        assert (df.loc[disponible, "timestamp"] + DURACION + DEMORA_TEMPERATURA <= corte(inicio)).all()
        X_ent, y_ent = df.loc[disponible, VARIABLES], df.loc[disponible, "demanda_mwh"]
        assert X_ent.notna().all().all(), "hay variables faltantes en el entrenamiento"
        salida.loc[en_mes, "pred"] = entrenar(X_ent, y_ent).predict(X_pred[en_mes])
        for q in cuantiles:
            salida.loc[en_mes, f"q{round(q * 100)}"] = entrenar(X_ent, y_ent, q).predict(X_pred[en_mes])
    faltan = salida.isna().any(axis=1)
    if faltan.any():
        raise ValueError(f"reentrenar_mensual dejo {int(faltan.sum())} filas sin prediccion "
                         f"(desde {pd.Timestamp(t_pred[faltan.values].min())})")
    return salida


COBERTURA_MINIMA_CALIBRACION = 0.95   # tolera las horas excluidas a proposito por el corte de emision


def calibrar(historia: pd.DataFrame, meses, meses_historia=CALIBRACION_MESES) -> pd.Series:
    """Ajuste conformal de cada mes con los puntajes ya conocidos al corte de su primera emision.

    Se detiene con un error (no omite en silencio) si en la ventana hay puntajes faltantes o si la
    historia cubre menos del 95 % de las horas que deberia tener la ventana.
    """
    ajuste = pd.Series(np.nan, index=historia.index)
    for inicio in meses:
        inicio = pd.Timestamp(inicio)
        desde = inicio - pd.DateOffset(months=meses_historia)
        previos = demanda_conocida(historia["timestamp"], inicio) & (historia["timestamp"] >= desde)
        puntajes = historia.loc[previos, "puntaje"]
        if puntajes.isna().any():
            raise ValueError(f"faltan {int(puntajes.isna().sum())} puntajes en la ventana de calibracion de {inicio.date()}")
        esperadas = int((corte(inicio) - DURACION - DEMORA_DEMANDA - desde) / pd.Timedelta(hours=1)) + 1
        if len(puntajes) < COBERTURA_MINIMA_CALIBRACION * esperadas:
            raise ValueError(f"historia insuficiente para calibrar {inicio.date()}: {len(puntajes)} de {esperadas} horas "
                             f"de los {meses_historia} meses anteriores (desde {desde.date()})")
        en_mes = (historia["timestamp"] >= inicio) & (historia["timestamp"] < inicio + pd.offsets.MonthBegin(1))
        ajuste[en_mes] = puntajes.quantile(0.80)
    return ajuste


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def evaluar(nombre, y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "modelo": nombre,
        "MAPE_%": mean_absolute_percentage_error(y_real, y_pred) * 100,
        "WAPE_%": np.abs(y_real - y_pred).sum() / y_real.sum() * 100,
        "RMSE_MWh": root_mean_squared_error(y_real, y_pred),
        "MAE_MWh": mean_absolute_error(y_real, y_pred),
        "sesgo_MWh": (y_real - y_pred).mean(),       # > 0: subestima
    }


def evaluar_intervalo(nombre, real: pd.Series, inferior: pd.Series, superior: pd.Series):
    dentro = (real >= inferior) & (real <= superior)
    return {
        "intervalo": nombre,
        "cobertura_%": dentro.mean() * 100,
        "ancho_medio_MW": (superior - inferior).mean(),
        "real_arriba_%": (real > superior).mean() * 100,
        "real_abajo_%": (real < inferior).mean() * 100,
        "interval_score_MW": ((superior - inferior) + 2 / 0.2 * (inferior - real).clip(lower=0)
                              + 2 / 0.2 * (real - superior).clip(lower=0)).mean(),
        "dias_completos_cubiertos_%": dentro.groupby(dentro.index.normalize()).all().mean() * 100,
    }


def evaluar_picos(real: pd.Series, pred: pd.Series, umbral_alta: float):
    alta = real > umbral_alta
    por_dia = pd.DataFrame({"real": real, "pred": pred})
    dias = por_dia.groupby(por_dia.index.normalize()).agg(
        max_real=("real", "max"), max_pred=("pred", "max"),
        hora_real=("real", lambda s: s.idxmax().hour), hora_pred=("pred", lambda s: s.idxmax().hour),
    )
    error_max = (dias["max_real"] - dias["max_pred"]) / dias["max_real"] * 100
    return {
        "horas_demanda_alta": int(alta.sum()),
        "MAPE_horas_alta_%": (mean_absolute_percentage_error(real[alta], pred[alta]) * 100) if alta.any() else np.nan,
        "MAPE_maximo_diario_%": error_max.abs().mean(),
        "sesgo_maximo_diario_%": error_max.mean(),
        "dias_que_subestima_maximo_%": (error_max > 0).mean() * 100,
        "hora_del_maximo_a_1h_%": ((dias["hora_real"] - dias["hora_pred"]).abs() <= 1).mean() * 100,
    }
