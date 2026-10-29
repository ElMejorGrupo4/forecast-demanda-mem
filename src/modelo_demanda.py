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


def clima_al_emitir(pronostico: pd.DataFrame) -> pd.DataFrame:
    """Variables climaticas de cada dia con las corridas disponibles al emitir (47 horas por dia -> 24)."""
    partes = []
    for dia, pron_dia in pronostico.groupby("dia_objetivo"):
        pron_dia = pron_dia.set_index("timestamp")
        clima_dia = variables_temperatura(pron_dia["temperatura_pron_c"]).loc[dia:]
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
    """Igual que en el notebook 02: un modelo por mes, entrenado con lo disponible al corte del dia 1."""
    salida = pd.DataFrame(index=X_pred.index, dtype=float)
    for inicio in pd.date_range(t_pred.min(), t_pred.max(), freq="MS"):
        fin = inicio + pd.offsets.MonthBegin(1)
        disponible = fila_para_entrenar(df["timestamp"], inicio)
        if ventana_meses is not None:
            disponible &= df["timestamp"] >= inicio - pd.DateOffset(months=ventana_meses)
        assert (df.loc[disponible, "timestamp"] + DURACION + DEMORA_TEMPERATURA <= corte(inicio)).all()
        X_ent, y_ent = df.loc[disponible, VARIABLES], df.loc[disponible, "demanda_mwh"]
        assert X_ent.notna().all().all(), "hay variables faltantes en el entrenamiento"
        en_mes = ((t_pred >= inicio) & (t_pred < fin)).values
        salida.loc[en_mes, "pred"] = entrenar(X_ent, y_ent).predict(X_pred[en_mes])
        for q in cuantiles:
            salida.loc[en_mes, f"q{round(q * 100)}"] = entrenar(X_ent, y_ent, q).predict(X_pred[en_mes])
    return salida


def calibrar(historia: pd.DataFrame, meses, meses_historia=CALIBRACION_MESES) -> pd.Series:
    """Ajuste conformal de cada mes con los puntajes ya conocidos al corte de su primera emision."""
    ajuste = pd.Series(np.nan, index=historia.index)
    for inicio in meses:
        previos = (demanda_conocida(historia["timestamp"], inicio)
                   & (historia["timestamp"] >= inicio - pd.DateOffset(months=meses_historia)))
        en_mes = (historia["timestamp"] >= inicio) & (historia["timestamp"] < inicio + pd.offsets.MonthBegin(1))
        ajuste[en_mes] = historia.loc[previos, "puntaje"].quantile(0.80)
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
