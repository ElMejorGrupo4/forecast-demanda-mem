"""
Prepara los datos del periodo nuevo de evaluacion: agosto de 2026.

Este script y el notebook 04 se fijaron en un commit ANTES de descargar la
demanda de agosto (pre-registro). Solo prepara datos: no calcula metricas
ni muestra estadisticas de la demanda de agosto.

Pasos:
  1. Baja la planilla "Demanda Horaria por Tipo" actualizada de CAMMESA a
     data/raw/periodo_nuevo/ (no pisa la planilla usada en el resto del trabajo).
  2. Verifica que agosto de 2026 este completo (744 horas).
  3. Verifica que, en el periodo comun (2023 -> julio de 2026), la planilla
     nueva coincida con la serie versionada: si CAMMESA reviso datos viejos,
     se reporta.
     DESVIACION DEL PRE-REGISTRO (documentada antes de calcular metricas):
     la version pre-registrada se detenia con diferencias mayores a 0,5 %.
     La planilla nueva trae junio de 2026 revisado (450 horas, maximo 0,55 %,
     +0,11 % en la media del mes). Ahora las revisiones se guardan en
     data/processed/revisiones_cammesa_2026.csv para un analisis de
     sensibilidad, y el script se detiene solo si alguna supera el 1 %.
  4. Tipo de dia de agosto: misma regla que descargar_demanda_mem.py. Si la
     base diaria de CAMMESA (7 tipos) difiere de la planilla horaria,
     prevalece la base diaria. Las diferencias se informan y se guardan.
  5. Arma la temperatura disponible al emitir cada dia de agosto con las
     mismas reglas que el test (corridas de ECMWF IFS con 6 h de demora de
     publicacion supuesta).

Salidas:
    data/processed/demanda_horaria_agosto_2026.csv
        columnas: timestamp, demanda_mwh, tipo_dia, tipo_dia_planilla
    data/processed/temperatura_pronosticada_agosto_2026.csv   (mismo formato que la del test)

Uso:
    python src/preparar_agosto_2026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descargar_demanda_mem as dm                      # noqa: E402
import descargar_temperatura_pronosticada as tp         # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PLANILLA_NUEVA = BASE / "data" / "raw" / "periodo_nuevo" / "cammesa_demanda_horaria_tipo_actualizada.xlsx"
SERIE = BASE / "data" / "processed" / "demanda_horaria_mem.csv"
SALIDA_DEMANDA = BASE / "data" / "processed" / "demanda_horaria_agosto_2026.csv"
SALIDA_TEMPERATURA = BASE / "data" / "processed" / "temperatura_pronosticada_agosto_2026.csv"
SALIDA_REVISIONES = BASE / "data" / "processed" / "revisiones_cammesa_2026.csv"

INICIO = pd.Timestamp("2026-08-01")
FIN = pd.Timestamp("2026-08-31 23:00")


def main() -> None:
    print("1. Planilla actualizada de CAMMESA")
    dm.descargar(dm.URL_TIPO, PLANILLA_NUEVA, actualizar="--actualizar" in sys.argv)
    nueva = dm.leer_tipo(PLANILLA_NUEVA)
    print(f"   cubre {nueva.timestamp.min()} -> {nueva.timestamp.max()}")

    print("2. Cobertura de agosto de 2026")
    agosto = nueva[(nueva.timestamp >= INICIO) & (nueva.timestamp <= FIN)].sort_values("timestamp").reset_index(drop=True)
    if len(agosto) != 744:
        raise SystemExit(f"agosto de 2026 no esta completo en la planilla ({len(agosto)} de 744 horas)")
    grilla = pd.date_range(INICIO, FIN, freq="h")
    assert (agosto["timestamp"].values == grilla.values).all(), "faltan o sobran horas en agosto"
    assert agosto["demanda_mwh"].notna().all() and (agosto["demanda_mwh"] > 0).all(), "demanda nula o no positiva"
    print("   744 horas, grilla completa, sin nulos")

    print("3. Periodo comun con la serie versionada")
    serie = pd.read_csv(SERIE, parse_dates=["timestamp"])
    comun = serie.merge(nueva[["timestamp", "demanda_mwh"]], on="timestamp", suffixes=("", "_nueva"))
    dif = ((comun["demanda_mwh_nueva"] - comun["demanda_mwh"]) / comun["demanda_mwh"]).abs() * 100
    print(f"   {len(comun):,} horas comunes | diferencia maxima {dif.max():.4f} % | horas con diferencia > 0,01 %: {(dif > 0.01).sum()}")
    revisadas = comun.loc[dif > 0.01, ["timestamp", "demanda_mwh", "demanda_mwh_nueva"]].rename(
        columns={"demanda_mwh": "demanda_versionada", "demanda_mwh_nueva": "demanda_revisada"})
    if len(revisadas):
        print(f"   revisiones por mes: {revisadas.groupby(revisadas.timestamp.dt.to_period('M')).size().to_dict()}")
    revisadas.to_csv(SALIDA_REVISIONES, index=False, float_format="%.3f")
    print(f"   guardado {SALIDA_REVISIONES.relative_to(BASE)} (para el analisis de sensibilidad)")
    assert dif.max() < 1.0, "CAMMESA reviso datos viejos en mas de 1 %: revisar antes de seguir"

    print("4. Tipo de dia de agosto")
    agosto["tipo_dia_planilla"] = agosto["tipo_dia"]
    if dm.XLSX_DIARIA.exists():
        diaria = dm.leer_tipo_dia_diario(dm.XLSX_DIARIA)
        dia = agosto["timestamp"].dt.normalize()
        referencia = dia.map(diaria)
        assert referencia.notna().all(), "la base diaria no cubre todo agosto"
        distintos = agosto.loc[referencia != agosto["tipo_dia"], "timestamp"].dt.normalize().unique()
        agosto["tipo_dia"] = referencia.values
        print(f"   {len(distintos)} dias corregidos con la base diaria: {[str(d.date()) for d in distintos]}")
    else:
        print(f"   AVISO: falta {dm.XLSX_DIARIA.name}; se usa el tipo de dia de la planilla horaria sin verificar")

    SALIDA_DEMANDA.parent.mkdir(parents=True, exist_ok=True)
    agosto[["timestamp", "demanda_mwh", "tipo_dia", "tipo_dia_planilla"]].to_csv(
        SALIDA_DEMANDA, index=False, float_format="%.3f")
    print(f"   guardado {SALIDA_DEMANDA.relative_to(BASE)}")

    print("5. Temperatura disponible al emitir cada dia de agosto")
    pron = pd.concat([tp.pronostico_del_dia(d) for d in pd.date_range(INICIO, FIN.normalize(), freq="D")], ignore_index=True)
    assert (pron.groupby("dia_objetivo").size() == 47).all() and pron["temperatura_pron_c"].notna().all()
    corte_local = pd.to_datetime(pron["dia_objetivo"]) - pd.Timedelta(days=1) + tp.HORA_EMISION - tp.MARGEN_EMISION
    assert (pd.to_datetime(pron["disponible_supuesta_utc"]) + tp.UTC_LOCAL <= corte_local).all(), \
        "con la demora supuesta, se uso una corrida no disponible al corte"
    combinaciones = pron.groupby("dia_objetivo")["corrida_utc"].agg(lambda s: " + ".join(sorted({x[-5:] for x in s})))
    print(f"   corridas por dia (hora UTC): {combinaciones.value_counts().to_dict()}")
    pron.to_csv(SALIDA_TEMPERATURA, index=False, float_format="%.1f")
    print(f"   guardado {SALIDA_TEMPERATURA.relative_to(BASE)}")


if __name__ == "__main__":
    main()
