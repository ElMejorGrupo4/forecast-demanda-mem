"""
Demo con Streamlit del forecast de demanda electrica horaria del MEM (total del pais).

No recalcula nada por su cuenta: llama a api.main.calcular_pronostico, la misma funcion que usa el
endpoint POST /pronostico de la API. Asi usa el mismo snapshot, la misma validacion de Open-Meteo,
las mismas variables (md.clima_del_dia), la misma banda (md.banda_80), el mismo bloque `alcance`
y las mismas advertencias.

Modos:
    - Pronostico actual: la temperatura sale del pronostico de ECMWF IFS en Open-Meteo (una sola consulta).
    - Caso historico validado (12/08/2026): reproduccion con las temperaturas de la corrida de ECMWF
      disponible al emitir, registradas en el repositorio. Pasa por el modo de simulacion manual de la API
      y se compara con la demanda real de CAMMESA de ese dia. No es un pronostico actual.

Uso (desde la carpeta del proyecto):
    pip install -r demo/requirements.txt
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from fastapi import HTTPException

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
import api.main as api            # noqa: E402  (logica unica de la API)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caso_historico import corridas_historicas, pedido_historico   # noqa: E402
from estilo import AZUL, AZUL_MARINO, CELESTE, CSS, VERDE_AGUA, configurar_grafico   # noqa: E402  (solo apariencia)

MODO_ACTUAL = "Pronóstico actual (Open-Meteo)"
MODO_HISTORICO = "Caso histórico validado — 12/08/2026"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MILES = "replace(format(datum.value, ',.0f'), ',', '.')"   # ejes con punto de miles, como el resto de la pagina


def numero(x: float, decimales: int = 0) -> str:
    """Formato argentino: punto para miles, coma para decimales."""
    texto = f"{x:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource
def cargar_snapshot() -> dict:
    """Carga el snapshot una sola vez por proceso, con los controles de la API (huella y version)."""
    return api.cargar_snapshot(os.environ.get("MODELO_ID", api.MODELO_POR_DEFECTO))


def ejecutar(pedido: api.PedidoPronostico) -> dict:
    """Corre la logica de la API; los errores del proveedor o del pedido se devuelven, no se esconden."""
    try:
        respuesta, variables = api.calcular_pronostico(pedido)
        return {"respuesta": respuesta, "variables": variables, "error": None}
    except HTTPException as e:
        return {"respuesta": None, "variables": None, "error": e}


def mostrar_error(error: HTTPException) -> None:
    if error.status_code == 422:
        st.error(f"**El pedido no es válido.** {error.detail}")
    elif error.status_code == 502:
        st.error(f"**Open-Meteo respondió datos inválidos** y no se usaron. {error.detail}")
    else:
        st.error(f"**Open-Meteo no responde en este momento.** {error.detail}")
    if error.status_code in (502, 503):
        st.info(f"Para seguir con la demo sin conexión, elegí «{MODO_HISTORICO}» en el panel lateral.")


def grafico_demanda(tabla: pd.DataFrame, real: pd.Series | None) -> alt.Chart:
    banda = alt.Chart(tabla).mark_area(opacity=0.25, color=CELESTE).encode(
        x=alt.X("hora:Q", title="Hora", scale=alt.Scale(domain=[0, 23], nice=False), axis=alt.Axis(tickCount=12)),
        y=alt.Y("banda_inferior_mwh:Q", title="MWh", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=MILES)),
        y2="banda_superior_mwh:Q",
        tooltip=[alt.Tooltip("hora:Q", title="Hora"),
                 alt.Tooltip("banda_inferior_mwh:Q", title="Banda inferior (MWh)", format=".0f"),
                 alt.Tooltip("banda_superior_mwh:Q", title="Banda superior (MWh)", format=".0f")],
    )
    lineas = tabla[["hora", "demanda_mwh"]].rename(columns={"demanda_mwh": "MWh"}).assign(serie="Pronóstico central")
    if real is not None:
        lineas = pd.concat([lineas, pd.DataFrame({"hora": range(24), "MWh": real.values, "serie": "Demanda real (CAMMESA)"})])
    series = ["Pronóstico central"] + (["Demanda real (CAMMESA)"] if real is not None else [])
    linea = alt.Chart(lineas).mark_line(point=True, strokeWidth=3).encode(
        x="hora:Q", y=alt.Y("MWh:Q", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=MILES)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=series, range=[AZUL_MARINO, VERDE_AGUA][:len(series)]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("hora:Q", title="Hora"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("MWh:Q", format=".0f")],
    )
    return configurar_grafico((banda + linea).properties(height=380))


def grafico_temperatura(variables: pd.DataFrame) -> alt.Chart:
    datos = pd.DataFrame({
        "hora": list(range(24)) * 2,
        "°C": list(variables["temperatura_c"]) + list(variables["temp_24h"]),
        "serie": ["Temperatura de la hora"] * 24 + ["Promedio de 24 h"] * 24,
    })
    return configurar_grafico(alt.Chart(datos).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("hora:Q", title="Hora", scale=alt.Scale(domain=[0, 23], nice=False), axis=alt.Axis(tickCount=12)),
        y=alt.Y("°C:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=["Temperatura de la hora", "Promedio de 24 h"],
                                                                range=[CELESTE, AZUL]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("hora:Q", title="Hora"), alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("°C:Q", format=".1f")],
    ).properties(height=260))


def pie() -> None:
    """Pie de página con la línea de degradado de las diapositivas."""
    with st.container(key="pie"):
        st.caption("Programa Energía Digital 2026 · Fundación YPF + Jump Educación")


def tabla_variables(variables: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Hora": variables["hora"].values,
        "Día de semana": [DIAS[d] for d in variables["dia_semana"]],
        "Mes": variables["mes"].values,
        "Día hábil": ["sí" if h else "no" for h in variables["es_habil"]],
        "Temperatura (°C)": variables["temperatura_c"].round(1).values,
        "Media móvil 24 h (°C)": variables["temp_24h"].round(2).values,
        "HDD (base 18 °C)": variables["hdd"].round(2).values,
        "CDD (base 18 °C)": variables["cdd"].round(2).values,
    })


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Forecast de demanda del MEM", page_icon="⚡", layout="wide")
st.html(CSS)                           # estilo de la presentación (solo apariencia)
api.estado.update(cargar_snapshot())   # api.calcular_pronostico usa el snapshot cargado

with st.sidebar:
    st.header("Pronóstico del día")
    st.caption("Total del MEM · modelo de calendario y clima")
    modo = st.radio("Temperatura", [MODO_ACTUAL, MODO_HISTORICO])
    hoy = date.today()
    if modo == MODO_ACTUAL:
        fecha = st.date_input("Fecha", value=hoy + timedelta(days=1), min_value=hoy - timedelta(days=90),
                              max_value=hoy + timedelta(days=14), format="DD/MM/YYYY")
        feriado = st.checkbox("Es feriado o día no laborable",
                              help="Sábados y domingos ya se toman como no hábiles.")
    else:
        st.info("Reproducción histórica con las temperaturas de la corrida de ECMWF disponible al emitir "
                "(11/08/2026, 12:00), registradas en el repositorio. No es un pronóstico actual.")
    if st.button("Pronosticar", type="primary", width="stretch"):
        if modo == MODO_ACTUAL:
            resultado = ejecutar(api.PedidoPronostico(fecha=fecha, es_feriado=feriado))
            resultado["real"] = None
        else:
            pedido, real = pedido_historico()
            resultado = ejecutar(pedido)
            resultado["real"] = real
        resultado["modo"] = modo
        st.session_state["resultado"] = resultado

with st.container(key="portada"):     # franja con el degradado de la portada de la presentación
    st.markdown('<p class="volanta">Proyecto final · Data Science</p>', unsafe_allow_html=True)
    st.title("Forecast de demanda eléctrica horaria del MEM", anchor=False)
    st.markdown("Pronóstico de las próximas 24 horas a partir del calendario y la temperatura")
    st.caption("Juan Serrano · Rodrigo Duarte")
st.info("**Métricas históricas del modelo** (evaluaciones pasadas, no el error del pronóstico mostrado): "
        "MAPE test 4,1 % · MAPE agosto de 2026 4,0 %")

resultado = st.session_state.get("resultado")
if resultado is None:
    st.write("Elegí una fecha y apretá **Pronosticar**.")
    pie()
    st.stop()
if resultado["error"] is not None:
    mostrar_error(resultado["error"])
    pie()
    st.stop()

respuesta, variables = resultado["respuesta"], resultado["variables"]
alcance = respuesta.alcance
horas = pd.DataFrame([h.model_dump() for h in respuesta.horas])
tipo = "hábil" if respuesta.tipo_dia == "habil" else "no hábil"

st.subheader(f"{DIAS[respuesta.fecha.weekday()].capitalize()} {respuesta.fecha:%d/%m/%Y} · día {tipo}")
if resultado["modo"] == MODO_HISTORICO:
    st.warning(f"**{MODO_HISTORICO}.** Reproducción histórica, no un pronóstico actual: se compara con la demanda "
               "real que informó CAMMESA ese día.")

with st.container(key="metricas"):     # tarjetas en dos filas (3 + 2) para que los números no se corten
    c1, c2, c3 = st.columns(3)
    c1.metric("Demanda máxima (MWh)", numero(respuesta.resumen.demanda_maxima_mwh))
    c2.metric("Hora del pico", f"{respuesta.resumen.hora_de_la_maxima} h")
    c3.metric("Demanda mínima (MWh)", numero(horas["demanda_mwh"].min()))
    c4, c5 = st.columns(2)
    c4.metric("Energía del día (GWh)", numero(respuesta.resumen.energia_del_dia_mwh / 1000, 1))
    c5.metric("Temp. mín / máx (°C)", f"{numero(horas['temperatura_c'].min(), 1)} / {numero(horas['temperatura_c'].max(), 1)}")

st.markdown("#### Demanda horaria: pronóstico central y banda nominal del 80 %")
st.altair_chart(grafico_demanda(horas, resultado["real"]), width="stretch")
st.caption("La banda tiene un nivel **nominal** del 80 %: en las evaluaciones su cobertura real fue menor en "
           "varios meses (72,45 % en agosto de 2026).")
if resultado["real"] is not None:
    error = (abs(resultado["real"].values - horas["demanda_mwh"].values) / resultado["real"].values).mean() * 100
    st.markdown(f"Error de este día frente a la demanda real de CAMMESA: **{numero(error, 2)} %** (MAPE del 12/08/2026).")

st.markdown("#### Temperatura horaria")
st.altair_chart(grafico_temperatura(variables), width="stretch")

with st.expander("Variables usadas por el modelo (las mismas 24 filas con las que se predijo)"):
    st.dataframe(tabla_variables(variables), hide_index=True, width="stretch")
    st.caption("HDD = max(18 − media 24 h, 0) y CDD = max(media 24 h − 18, 0): grados de calefacción y refrigeración.")

st.markdown("#### Alcance de la respuesta")
izq, der = st.columns(2)
with izq:
    fuente = alcance.fuente_meteorologica
    if resultado["modo"] == MODO_ACTUAL:
        origen = (f"- **Fuente meteorológica:** {fuente.proveedor}" + (f", modelo {fuente.modelo}" if fuente.modelo else "")
                  + f"\n- **Corrida:** {fuente.corrida or 'no informada por el proveedor'}\n")
    else:
        # la API recibe estas temperaturas como simulacion manual; la corrida sale del archivo registrado
        origen = ("- **Fuente meteorológica:** corridas de ECMWF IFS disponibles al emitir, registradas en el repositorio "
                  "(enviadas a la API como simulación manual)\n"
                  f"- **Corridas (inicio, UTC):** {' y '.join(corridas_historicas())}\n")
    st.markdown(
        f"- **Modo:** {alcance.modo} · historial térmico {alcance.historial_termico}\n"
        + origen +
        f"- **Consultado:** {alcance.consultado_en:%d/%m/%Y %H:%M} (hora de Argentina)"
        + (f"\n- **Horizonte:** {numero(alcance.horizonte_horas['primera_hora'], 1)} a "
           f"{numero(alcance.horizonte_horas['ultima_hora'], 1)} h desde la consulta" if alcance.horizonte_horas else "")
    )
with der:
    modelo = alcance.modelo
    st.markdown(
        f"- **Modelo:** {modelo.id}\n"
        f"- **Corte de información:** {modelo.corte_de_informacion}\n"
        f"- **Último dato de entrenamiento:** {modelo.ultimo_dato_de_entrenamiento}\n"
        f"- **Antigüedad del corte:** {modelo.antiguedad_del_corte_dias} días\n"
        f"- **Huella del artefacto:** `{modelo.artefacto_sha256[:12]}…`"
    )
st.caption(alcance.relacion_con_la_evaluacion)

st.markdown("#### Advertencias de la API")
for advertencia in respuesta.advertencias:
    st.warning(advertencia)

pie()
