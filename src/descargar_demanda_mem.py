"""
Construye la serie horaria de demanda del MEM (total pais) desde 2021 hasta
el ultimo mes publicado, encadenando dos planillas oficiales de CAMMESA.

Fuentes (ambas en cammesaweb.cammesa.com, seccion Informes y Estadisticas):

  A. "Demanda Horaria por Tipo"
       Total pais, paso horario, ene-2023 -> ultimo mes cerrado.
       Trae la demanda separada en Distribuidores y Grandes Usuarios.

  B. "Planilla Demanda Horaria por Regiones"
       Total pais + 9 regiones, paso horario, ene-2021 -> dic-2023.

Ambas se solapan en 2023 y son identicas (verificado: diferencia maxima
0.04%). Para el total pais se usa B en 2021-2022 y A en 2023 en adelante,
porque A se actualiza todos los meses. La B ademas se guarda desagregada por
region para el EDA.

Convencion horaria de CAMMESA: HORA = 1..24, donde la hora h representa el
intervalo (h-1):00 a h:00. Se convierte a timestamp = fecha + (h-1) horas.
Argentina no tiene horario de verano desde 2009, asi que cada dia tiene
exactamente 24 horas y no hay que tratar cambios de huso.

Tipo de dia: la planilla A solo distingue habil / no habil y tiene errores
(por ejemplo, marca como habiles el Jueves y Viernes Santo de 2023). Se
compara contra las dos fuentes de CAMMESA que clasifican en 7 tipos de dia
(la planilla B y la base diaria "Base Demanda Diaria 2017 2026.xlsx", que se
baja a mano, ver README). Si difieren, prevalece la clasificacion detallada.

Uso:
    python src/descargar_demanda_mem.py               # usa las planillas ya bajadas
    python src/descargar_demanda_mem.py --actualizar  # vuelve a bajarlas (meses nuevos)

Salidas:
    data/raw/cammesa_demanda_horaria_tipo_2023_2026.xlsx       (A, tal cual)
    data/raw/cammesa_demanda_horaria_regiones_2021_2023.xlsx   (B, tal cual)
    data/processed/demanda_horaria_mem.csv                     (serie total pais)
    data/processed/demanda_horaria_regiones_2021_2023.csv      (formato largo)
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

# wpdmdl es el id del archivo en el gestor de descargas del sitio de CAMMESA.
URL_TIPO = "https://cammesaweb.cammesa.com/download/demanda-horaria/?wpdmdl=41424"
URL_REGIONES = "https://cammesaweb.cammesa.com/?wpdmdl=47560"

XLSX_TIPO = RAW / "cammesa_demanda_horaria_tipo_2023_2026.xlsx"
XLSX_REGIONES = RAW / "cammesa_demanda_horaria_regiones_2021_2023.xlsx"
# base diaria de CAMMESA (descarga manual, la usa tambien construir_dataset_regional.py)
XLSX_DIARIA = RAW / "Base Demanda Diaria 2017 2026.xlsx"

REGIONES = {
    "BUENOS AIRES": "Buenos Aires",
    "CENTRO": "Centro",
    "COMAHUE": "Comahue",
    "CUYO": "Cuyo",
    "GRAN BS.AS.": "GBA",
    "LITORAL": "Litoral",
    "NORESTE": "NEA",
    "NOROESTE": "NOA",
    "PATAGONICA": "Patagonia",
}


def descargar(url: str, destino: Path, actualizar: bool = False) -> None:
    if destino.exists() and not actualizar:
        print(f"  ya existe {destino.name}, no se vuelve a bajar (usar --actualizar para bajarla de nuevo)")
        return
    print(f"  bajando {destino.name} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    contenido = r.content
    # La planilla por regiones viene comprimida en un zip con un unico xlsx.
    if contenido[:2] == b"PK" and "zip" in r.headers.get("content-type", ""):
        with zipfile.ZipFile(BytesIO(contenido)) as z:
            nombre = [n for n in z.namelist() if n.endswith(".xlsx")][0]
            contenido = z.read(nombre)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(contenido)


def a_timestamp(fecha: pd.Series, hora: pd.Series) -> pd.Series:
    return pd.to_datetime(fecha) + pd.to_timedelta(hora.astype(int) - 1, unit="h")


def leer_tipo(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=3)
    df.columns = [
        "anio", "mes", "n_mes", "n_dia", "tipo_dia", "dia_nombre",
        "fecha", "hora", "demanda_gu_mwh", "demanda_dist_mwh", "demanda_mwh",
    ]
    df = df.dropna(subset=["hora", "demanda_mwh"])
    df["timestamp"] = a_timestamp(df["fecha"], df["hora"])
    # CAMMESA escribe "Hábil" / "No Hábil" con mayusculas inconsistentes.
    df["tipo_dia"] = df["tipo_dia"].str.strip().str.lower().map({"hábil": "habil", "no hábil": "no_habil"})
    return df[["timestamp", "demanda_mwh", "demanda_dist_mwh", "demanda_gu_mwh", "tipo_dia"]]


def leer_regiones(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=11).iloc[:, :15]
    df.columns = ["anio", "mes", "fecha", "tipo_dia_detalle", "hora"] + list(REGIONES) + ["TOTAL"]
    df = df.dropna(subset=["hora", "TOTAL"])
    df["timestamp"] = a_timestamp(df["fecha"], df["hora"])
    # Esta planilla clasifica en 7 tipos (lunes habiles, ..., sabado, domingo o
    # feriado). Se reduce al mismo binario que la planilla A.
    df["tipo_dia"] = a_binario(df["tipo_dia_detalle"])
    return df


def a_binario(tipo_detallado: pd.Series) -> pd.Series:
    """Reduce los 7 tipos de dia de CAMMESA a habil / no habil.

    "Sabado o Semilaborable" y "Domingo o Feriado" son no habiles; "Lunes habiles" ... "Viernes habiles", habiles.
    """
    return tipo_detallado.str.contains("Domingo|Sabado", case=False).map({True: "no_habil", False: "habil"})


def leer_tipo_dia_diario(path: Path) -> pd.Series:
    """Tipo de dia (habil / no habil) de la base diaria de CAMMESA, indexado por fecha."""
    df = pd.read_excel(path, sheet_name="Datos Región", header=4, engine="calamine")
    df = df.dropna(subset=["Fecha", "Tipo día"])
    return pd.Series(a_binario(df["Tipo día"]).values, index=pd.to_datetime(df["Fecha"]), name="diaria")


def reconciliar_tipo_dia(mem: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """Compara el tipo de dia de la serie con las fuentes de 7 tipos y corrige las diferencias.

    Referencia: la base diaria donde exista (2017 -> mitad de 2026) y la planilla B en el resto.
    Si la base diaria y la planilla B no coinciden entre si, el script se detiene: no hay
    un criterio claro para decidir.
    """
    dia = mem["timestamp"].dt.normalize()
    ref = pd.DataFrame({"serie": mem.groupby(dia)["tipo_dia"].first()})
    ref = ref.join(reg.groupby(reg["timestamp"].dt.normalize())["tipo_dia"].first().rename("regiones"))
    if XLSX_DIARIA.exists():
        ref = ref.join(leer_tipo_dia_diario(XLSX_DIARIA))
    else:
        print(f"  AVISO: falta {XLSX_DIARIA.name}; el tipo de dia se valida solo contra la planilla B")
        ref["diaria"] = pd.NA

    ambas = ref["regiones"].notna() & ref["diaria"].notna()
    assert (ref.loc[ambas, "regiones"] == ref.loc[ambas, "diaria"]).all(), "la planilla B y la base diaria no coinciden"

    ref["referencia"] = ref["diaria"].fillna(ref["regiones"])
    distintos = ref[ref["referencia"].notna() & (ref["serie"] != ref["referencia"])]
    print(f"  tipo de dia: {ref['referencia'].notna().sum():,} dias comparados, {len(distintos)} corregidos")
    for fecha, fila in distintos.iterrows():
        print(f"    {fecha.date()} ({fecha.day_name()}): {fila['serie']} -> {fila['referencia']}")

    mem = mem.copy()
    corregido = dia.map(distintos["referencia"])
    mem["tipo_dia"] = corregido.fillna(mem["tipo_dia"])
    return mem


def main() -> None:
    actualizar = "--actualizar" in sys.argv
    print("Descarga de planillas CAMMESA")
    descargar(URL_TIPO, XLSX_TIPO, actualizar)
    descargar(URL_REGIONES, XLSX_REGIONES, actualizar)

    print("\nLectura")
    tipo = leer_tipo(XLSX_TIPO)
    reg = leer_regiones(XLSX_REGIONES)
    print(f"  A (por tipo):     {len(tipo):,} horas, {tipo.timestamp.min()} -> {tipo.timestamp.max()}")
    print(f"  B (por regiones): {len(reg):,} horas, {reg.timestamp.min()} -> {reg.timestamp.max()}")

    # --- verificacion del solape: si no coinciden, no se pueden encadenar ---
    solape = tipo.merge(reg[["timestamp", "TOTAL"]], on="timestamp")
    diff_pct = ((solape["demanda_mwh"] - solape["TOTAL"]) / solape["TOTAL"]).abs().max() * 100
    print(f"  solape: {len(solape):,} horas, diferencia maxima {diff_pct:.3f}%")
    assert diff_pct < 0.5, "las dos planillas no coinciden en el periodo comun"

    # --- serie total pais: B hasta donde empieza A, A en adelante ---
    inicio_a = tipo["timestamp"].min()
    previo = reg.loc[reg["timestamp"] < inicio_a, ["timestamp", "TOTAL", "tipo_dia"]].rename(columns={"TOTAL": "demanda_mwh"})
    previo["fuente"] = "regiones_2021_2023"
    tipo["fuente"] = "por_tipo_2023_2026"
    mem = pd.concat([previo, tipo], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    mem = mem[["timestamp", "demanda_mwh", "demanda_dist_mwh", "demanda_gu_mwh", "tipo_dia", "fuente"]]

    # --- tipo de dia: se valida contra las fuentes de 7 tipos y se corrige ---
    mem = reconciliar_tipo_dia(mem, reg)

    # --- grilla horaria completa, sin duplicados ni nulos (antes se verificaba solo en el EDA) ---
    grilla = pd.date_range(mem["timestamp"].min(), mem["timestamp"].max(), freq="h")
    assert mem["timestamp"].is_unique, "hay horas duplicadas"
    assert len(mem) == len(grilla) and (mem["timestamp"].values == grilla.values).all(), "faltan horas en la serie"
    assert mem["demanda_mwh"].notna().all() and (mem["demanda_mwh"] > 0).all(), "demanda nula o no positiva"

    # --- serie regional en formato largo (una fila por region y hora) ---
    reg_largo = reg.melt(
        id_vars=["timestamp", "tipo_dia"], value_vars=list(REGIONES),
        var_name="region", value_name="demanda_mwh",
    )
    reg_largo["region"] = reg_largo["region"].map(REGIONES)
    reg_largo = reg_largo.sort_values(["region", "timestamp"]).reset_index(drop=True)

    assert mem["tipo_dia"].notna().all(), "quedaron tipos de dia sin mapear"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    mem.to_csv(PROCESSED / "demanda_horaria_mem.csv", index=False, float_format="%.3f")
    reg_largo.to_csv(PROCESSED / "demanda_horaria_regiones_2021_2023.csv", index=False, float_format="%.3f")

    print("\nGuardado")
    print(f"  demanda_horaria_mem.csv: {len(mem):,} horas, {mem.timestamp.min()} -> {mem.timestamp.max()}")
    print(f"  demanda_horaria_regiones_2021_2023.csv: {len(reg_largo):,} filas")


if __name__ == "__main__":
    main()
