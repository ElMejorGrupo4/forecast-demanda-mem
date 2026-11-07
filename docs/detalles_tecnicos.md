# Detalles técnicos

Complemento del [README](../README.md): el detalle de los datos, las reglas de evaluación, las revisiones, el diseño de la API y de la demo. Los resultados y las conclusiones están en el README.

**Contenido**

1. [Datos](#datos)
2. [Qué se conoce al emitir](#qué-se-conoce-al-emitir)
3. [Revisión antes de la entrega final](#revisión-antes-de-la-entrega-final)
4. [Protocolo de la evaluación de agosto de 2026](#protocolo-de-la-evaluación-de-agosto-de-2026)
5. [Diseño de la API (FastAPI)](#diseño-de-la-api-fastapi)
6. [Demo con Streamlit](#demo-con-streamlit)
7. [Reproducción: datos crudos y procedencia](#reproducción-datos-crudos-y-procedencia)
8. [Nota sobre feriados y tipo de día](#nota-sobre-feriados-y-tipo-de-día)

## Datos

| Fuente | Contenido | Período | Uso |
|---|---|---|---|
| CAMMESA — "Demanda Horaria por Tipo" | Demanda real neta total del MEM, paso horario, desagregada en Distribuidores y Grandes Usuarios | ene-2023 → jul-2026 | Base del modelo |
| CAMMESA — "Demanda Horaria por Regiones" | Total MEM + 9 regiones, paso horario | ene-2021 → dic-2023 | Base del modelo (2021–2022) y análisis regional |
| Open-Meteo — *Historical Weather API*, modelo ECMWF IFS (9 km), coordenadas de Aeroparque | Temperatura horaria histórica de Buenos Aires (serie armada con las primeras horas de cada corrida del modelo, no mediciones de estación) | ene-2021 → jul-2026 | Variable climática del modelo (entrenamiento) |
| Open-Meteo — *Single Runs API*, modelo ECMWF IFS (9 km), coordenadas de Aeroparque | Para cada día del test, la temperatura horaria de las corridas disponibles antes de emitir el pronóstico (00 y 06 UTC del día anterior), según una demora de publicación supuesta de 6 h | ene-2025 → jul-2026 | Variable climática del modelo (test) |
| CAMMESA (demanda diaria por región) + SMN (clima diario) — `dataset_demanda_por_region.csv` | Demanda media diaria de las 9 regiones con temperatura, humedad, viento, presión, etc. de una estación representativa del SMN por región | ene-2017 → jun-2026 | EDA regional (no es insumo del modelo horario) |

Las dos planillas de CAMMESA coinciden hora a hora en el año que se solapan (2023, diferencia máxima 0,04 %), por lo que se encadenan en una única serie continua de **48.912 horas (2021-01-01 → 2026-07-31), sin huecos ni nulos**.

El dataset regional diario se validó contra esa serie: sumando las 9 regiones por día coincide con el promedio diario de la serie horaria (diferencia máxima 0,38 %).

**Alcance elegido:** total país (MEM). Es la serie que CAMMESA efectivamente pronostica para el despacho, y es la única con más de 5 años de historia horaria continua y actualización mensual.

### Dataset regional diario con clima

Lo arma `src/construir_dataset_regional.py` a partir de tres archivos crudos, que van en `data/raw/`:

| Archivo | Procedencia | Contenido |
|---|---|---|
| `Base Demanda Diaria 2017 2026.xlsx` | CAMMESA — [Comportamiento de la demanda de energía eléctrica en el MEM](https://cammesaweb.cammesa.com/2026/09/18/covid-19-comportamiento-de-la-demanda-de-energia-electrica-en-el-mem/) | Hoja "Datos Región": demanda neta diaria por región en MW medios, tipo de día y temperatura de referencia de GBA |
| `smn_datos_meteorologicos_1991_2020.xlsx` | SMN — [descarga de datos](https://ws2.smn.gob.ar/descarga-de-datos) | Datos diarios por estación 1991–2020 (se usa desde 2017) |
| `smn_datos_meteorologicos_desde_2021.lst` | SMN — [descarga de datos](https://ws2.smn.gob.ar/descarga-de-datos) | Datos diarios por estación desde 2021, en texto separado por tabuladores |

Transformación, resumida:

1. **Clima.** Se leen los dos archivos del SMN. El `.lst` se lee **respetando los tabuladores**: un dato faltante es un campo vacío entre dos tabs, y separarlo por "cualquier espacio" (como hacía la primera versión) corre las columnas. Se eliminan las filas de encabezado que el archivo repite, se limpian espacios y `S/D` pasa a NaN. En `PRECIP`, según el diccionario de datos del SMN, una celda vacía es un día sin lluvia y se carga como 0 mm. Fechas a `datetime` y variables a numéricas.
2. **Una estación representativa seleccionada por región** (ver criterio abajo).
3. **Demanda.** La hoja "Datos Región" pasa de formato ancho (una columna por región) a largo (una fila por día y región), dentro de 2017-01-01 → 2026-06-30 (fin del dato del SMN).
4. **Unión** por fecha y región con *left merge*: se conservan todas las filas de demanda, y si faltara clima quedaría NaN y se reportaría.
5. **Validaciones:** 9 regiones, 3.468 días por región, 31.212 filas, sin duplicados de fecha + región, demanda positiva, TMAX ≥ TMIN, humedad 0–100, heliofanía 0–24, rangos físicos de temperatura, presión y viento, y comparación de medianas antes y después de 2021 para detectar columnas corridas. Si alguna falla, el script se detiene sin guardar.

**Criterio de estaciones por región.** Se usa siempre la misma estación representativa seleccionada por región. No se promedian "las estaciones que midieron ese día": la composición cambiaría con el tiempo y mezclaría climas distintos según qué estaciones reportaron.

| Región CAMMESA | Estación SMN |
|---|---|
| GRAN BS.AS. | AEROPARQUE AERO |
| BUENOS AIRES (resto de la provincia) | LA PLATA AERO |
| LITORAL | ROSARIO AERO |
| CENTRO | CORDOBA AERO |
| NOROESTE | TUCUMAN AERO |
| NORESTE | RESISTENCIA AERO |
| CUYO | MENDOZA AERO |
| COMAHUE | NEUQUEN AERO |
| PATAGONICA | COMODORO RIVADAVIA AERO |

**Limitación:** cada estación es una **proxy** del clima de su región. Regiones extensas como Patagonia, NOA o Comahue abarcan climas muy distintos y una sola estación no los representa a todos. Es una limitación espacial del dataset.

**Uso:** este dataset diario se usa para el **EDA regional**. El futuro modelo horario va a necesitar temperatura **horaria**; repetir la `TMEDIA` del día en las 24 horas perdería la variación dentro del día, que es la que mueve la curva de carga.

Las 9 estaciones tienen los 3.468 días del período y `TMEDIA` no tiene faltantes, así que no hicieron falta estaciones de respaldo. Los faltantes que hay (`HELIOF` y algunos `TMAX`/`TMIN`) quedan como NaN, sin imputar. Para GBA se evaluó la temperatura de referencia de CAMMESA: está muy correlacionada con Aeroparque (0,986), pero Aeroparque explica un poco mejor la demanda de GBA y mantiene la misma fuente para las 9 regiones.

## Qué se conoce al emitir

Se supone que el pronóstico del día D se emite el día D-1 a las 12:00, con una hora reservada para entrenar y emitir (corte de información: D-1 11:00). Todo el notebook respeta esas reglas, y hay controles automáticos que las verifican **bajo los supuestos de demora**. Lo registrado en las fuentes son la hora de cada dato y la hora de inicio de cada corrida; las demoras de publicación son supuestas, porque no están registradas:

- **Demanda:** cada fila representa el intervalo de *t* a *t* + 1 h. Se supone medición en tiempo real, así que una hora se conoce cuando termina el intervalo.
- **Temperatura de entrenamiento:** la serie histórica de ECMWF IFS. Se supone publicada a más tardar 12 h después.
- **Temperatura del test:** sale solo de corridas de ECMWF cuyo inicio más 6 h de demora supuesta es anterior al corte, incluidas las horas previas a la emisión que entran en el promedio de 24 h.
- **Modelos que usan estas reglas:** el entrenamiento inicial (hasta el 30/12/2024 22:00), el reentrenamiento mensual, los modelos cuantiles y la calibración.

## Revisión antes de la entrega final

Una revisión crítica del trabajo encontró errores y afirmaciones demasiado fuertes. Las correcciones principales:

| Hallazgo | Corrección |
|---|---|
| El test usaba el pronóstico del tiempo hecho 24 h antes de **cada** hora: cada hora salía de una corrida distinta, algunas posteriores a la emisión | Se fija la emisión en D-1 12:00 y se usa una sola corrida de ECMWF por día (*Single Runs API* de Open-Meteo). El MAPE casi no cambia (4,06 % → 4,07 %), pero ahora el experimento coincide con la operación |
| Cuatro días con el tipo de día mal etiquetado en la planilla horaria de CAMMESA: Jueves y Viernes Santo 2023 y los feriados puente del 23/03/2026 y del 10/07/2026 | `descargar_demanda_mem.py` compara el tipo de día con las fuentes de CAMMESA que distinguen 7 tipos (planilla regional y base diaria) y corrige las diferencias |
| El Gradient Boosting separaba una validación interna al azar | `early_stopping=False` |
| "+4 % de crecimiento en 2026" comparaba enero–julio contra un año completo | Comparando períodos equivalentes, el crecimiento es de 0,9 % |
| Se presentaba el ancho de la banda como "la reserva necesaria", y se decía que WAPE ≈ MAPE probaba un buen desempeño en los picos | Redacción corregida; se agregaron cobertura por mes, *interval score* y métricas específicas de picos |
| El ejemplo "hábil de invierno" del notebook 03 era el 9 de julio, feriado | Se cambió el ejemplo y se agregó un control |

**Segunda revisión:**

| Hallazgo | Corrección |
|---|---|
| El reentrenamiento mensual y la calibración usaban datos hasta el final del mes anterior, aunque el día 1 se pronostica el mediodía anterior | Corte único por emisión y disponibilidad de cada dato (demanda, temperatura, corridas), aplicado al entrenamiento inicial, al reentrenamiento, a los cuantiles y a la calibración, con controles automáticos de esas reglas bajo los supuestos de demora. El MAPE casi no cambia (4,07 % → 4,08 %) |
| La temperatura histórica se describía como ERA5 (5 días de demora) y se usaba como "conocida" para las horas previas a la emisión | Se comprobó que es ECMWF IFS (idéntica hora por hora a `models=ecmwf_ifs`), que Open-Meteo actualiza sin demora. Se fija el modelo y se registra la procedencia en `.meta.json`. En el test, las horas previas a la emisión también salen de corridas disponibles antes del corte (con 6 h de demora de publicación supuesta) |
| Sin el Excel diario, `descargar_demanda_mem.py` perdía dos correcciones de calendario y terminaba sin error | Tabla de correcciones versionada; con el Excel, el script verifica que coincida y se detiene si aparecen fechas nuevas. Se comprobó que con y sin el Excel la serie queda idéntica |
| La ventana de calibración de los intervalos estaba fijada a mano | Se elige con validación en 2024 entre 3, 6 y 12 meses y un ajuste por hora del día |
| Se afirmaba que los árboles no pueden predecir por encima del máximo de entrenamiento | Vale para el Random Forest, no para el Gradient Boosting; se mide qué pasa en este proyecto |
| El error meteorológico se atribuía a la anticipación, y el modelo de calendario y clima se presentaba como "explicativo" | Se aclara que anticipación, hora y estación van juntas, y que los modelos describen asociaciones, no causas |

## Protocolo de la evaluación de agosto de 2026

- **Pre-registro:** antes de descargar la demanda de agosto se guardó un commit (`40e41df`) que fija los hiperparámetros, las variables, el reentrenamiento (24 meses, al corte del 31/07 a las 11:00), la calibración (3 meses) y las métricas a reportar.
- **Control previo:** el código fijado (`src/modelo_demanda.py`) reproduce exactamente los resultados del notebook 02 en el test conocido.
- **Desviación documentada:** CAMMESA revisó junio de 2026 (+0,11 % en la media del mes) y el control de datos se detuvo. Antes de calcular métricas se documentó la desviación en otro commit y se declaró un análisis de sensibilidad. El resultado principal usa el historial versionado.

## Diseño de la API (FastAPI)

`api/main.py` sirve un **snapshot** del modelo de calendario y clima para pronosticar las 24 horas de un día. Por defecto carga `gb_calendario_clima_2026-08-01`, el modelo evaluado en agosto de 2026 en el notebook 04. Con la variable de entorno `MODELO_ID` se puede elegir otro snapshot.

| Endpoint | Qué hace |
|---|---|
| `GET /salud` | Estado de la API y snapshot cargado |
| `GET /modelo` | Metadatos del snapshot (configuración, datos usados con su huella, entorno, corte de información), su antigüedad y los resultados de experimentos registrados **para ese artefacto exacto** |
| `POST /pronostico` | Pronóstico de un día: demanda de cada hora, banda de nivel nominal 80 %, resumen, bloque `alcance` y advertencias |

**Modos de `POST /pronostico`**

| Modo | Pedido | Historial térmico | Relación con la evaluación |
|---|---|---|---|
| Pronóstico actual | `{"fecha": "2026-11-05"}` | Completo: la API consulta el pronóstico de ECMWF IFS en Open-Meteo para el día pedido y el anterior | Mismas variables y modelo, pero con el pronóstico disponible **al momento de la consulta**. Open-Meteo no informa la corrida usada |
| Simulación manual | `temperaturas` (24) + `temperaturas_dia_anterior` (24) | Completo: el modo evaluado | Mismas variables y banda; el error depende de las temperaturas dadas |
| Simulación aproximada | `temperaturas` (24) + `"historial_incompleto": true` | **Incompleto**: el promedio de 24 h de las primeras horas usa solo horas del día | **No validada** por las métricas publicadas; la respuesta lo indica |

Si se mandan temperaturas sin las del día anterior y sin `historial_incompleto: true`, el pedido se rechaza (422): no se completan ni se inventan temperaturas.

**Bloque `alcance` de cada respuesta**

- `modo` e `historial_termico`.
- `consultado_en`: momento de la consulta, con zona horaria.
- `horizonte_horas`: horas entre la consulta y la primera y la última hora pedidas (solo en el pronóstico actual).
- `fecha_pasada`: si la fecha ya pasó. En ese caso, una consulta a Open-Meteo **no es un backtesting**, porque devuelve el mejor dato de hoy, no el pronóstico de entonces. Para eso está el notebook 04.
- `fuente_meteorologica`: proveedor, modelo y corrida. La corrida queda vacía cuando el proveedor no la informa; no se inventa.
- `banda`: nivel nominal y regla.
- `relacion_con_la_evaluacion`: qué tan comparable es la respuesta con la evaluación publicada.
- `modelo`: snapshot, corte de información, último dato de entrenamiento, **antigüedad del corte** y huella del artefacto.

**Errores**

| Código | Cuándo |
|---|---|
| 422 | Pedido inválido: fecha mal escrita, cantidad de temperaturas, valores imposibles, combinación de campos o fecha fuera del rango del pronóstico de Open-Meteo |
| 502 | Open-Meteo respondió algo inválido, incluido un rechazo 400 con cuerpo mal formado (no es JSON, una lista, `null` o un objeto sin un motivo de texto): no es JSON, falta la estructura, longitudes distintas, horas duplicadas, desordenadas o de otra fecha, otra zona horaria, temperaturas nulas, no numéricas o imposibles |
| 503 | Open-Meteo no responde, está caído o limita los pedidos |

**Decisiones de diseño**

- **Modelo servido.** El Gradient Boosting de calendario y clima. Solo necesita la fecha y la temperatura; la variante con rezagos exigiría mandar 48 valores de demanda pasada.
- **Reglas únicas.** El clima de un día (`md.clima_del_dia`) y la banda (`md.banda_80`) salen del mismo módulo que usa la evaluación. Hay pruebas que verifican que, con las mismas temperaturas históricas, la API devuelve exactamente el valor central y los extremos del pipeline evaluado.
- **Banda.** Cuantiles 10 y 90 con el ajuste conformal (234 MW). Si los modelos se cruzan y la banda no contiene al valor central, se extiende hasta él. Esa extensión cambia 3 horas de agosto de 2026 y 44 del test, sin cambiar la cobertura: 72,45 % y 79,09 %, medidas en el anexo del notebook 04. El **80 % es el nivel nominal**; la **cobertura empírica** fue menor en varios meses, y la API no la promete.
- **Snapshots trazables.** `src/entrenar_modelo.py --fecha-objetivo AAAA-MM-DD` entrena con las reglas de disponibilidad: los últimos 24 meses conocidos al corte del día anterior a las 11:00. Antes de entrenar valida que los datos lleguen al corte y que haya pronósticos para calibrar; si no, se detiene y dice qué falta. Cada snapshot va a su propia carpeta, nunca sobrescribe otro y registra:
  - la configuración;
  - los datos usados, con su huella sha256;
  - el entorno, el commit del código y la huella del artefacto.

  **Reproducibilidad, en dos niveles:**

  - **Numérica:** el modelo regenerado da exactamente las mismas predicciones que el snapshot. Se verificó en el entorno original y, según la auditoría, también en otro entorno: con las dependencias fijadas y Python 3.12.14, las predicciones del modelo central y de los cuantiles 10 y 90 coincidieron en las 744 horas de agosto.
  - **Binaria (mismo sha256):** verificada **solo en el entorno original** (Python 3.12.6, Windows, `requirements.txt`). En el entorno de la auditoría el archivo regenerado tuvo otro sha256. No se determinó la causa de esa diferencia.

  Por eso las métricas siguen atadas al sha256 del artefacto exacto que se evaluó: un archivo regenerado con otra huella no las hereda, aunque sus predicciones sean iguales. `--comparar-con` informa los dos niveles por separado.
- **Fechas objetivo y meses parciales.** La fecha objetivo puede ser cualquier día. Si un período empieza a mitad de mes, `md.reentrenar_mensual` usa para ese mes el modelo entrenado al corte de su día 1, que es anterior a todas sus horas, y predice solo las filas pedidas. Si queda alguna fila sin predicción, o si la calibración encuentra puntajes faltantes o menos del 95 % de la historia de su ventana, el proceso se detiene con un mensaje claro en lugar de seguir con menos datos.
- **Métricas atadas al artefacto.** Están en `models/resultados_experimentos.json`, separadas del script de entrenamiento, con el experimento del que salen. La API las muestra solo si la huella del modelo cargado coincide: un snapshot nuevo no hereda métricas que no se le midieron.
- **Antigüedad.** El snapshot evaluado se generó en octubre de 2026, pero su último dato de entrenamiento es del **30/07/2026 a las 22:00**. La API informa el corte y la antigüedad en cada respuesta, y avisa cuando supera los 30 días.
- **Entorno.** `requirements.txt` fija las versiones exactas. La API se niega a arrancar si scikit-learn no coincide con la versión registrada en el snapshot, o si el archivo no coincide con su huella.
- **Tipo de día.** Sábados y domingos son no hábiles; los feriados los indica el usuario con `es_feriado`.

**Qué se comprobó y qué no**

- **Localmente, con 70 pruebas automáticas (sin internet), incluidas las de la demo con Streamlit:**
  - endpoints y modos;
  - validaciones de pedido;
  - respuestas inválidas y fallas de Open-Meteo, simuladas a nivel HTTP con la validación real, incluidos rechazos 400 con cuerpo mal formado;
  - equivalencia con el pipeline evaluado;
  - verificación de huella y de versión;
  - validación de cobertura del entrenamiento y fechas objetivo admitidas o rechazadas;
  - meses parciales en el reentrenamiento y controles de la calibración.
- **Con el código final**, el snapshot regenerado da las mismas predicciones (central, cuantil 10 y cuantil 90) y la misma banda en las 744 horas de agosto de 2026, con el mismo ajuste conformal (233,9 MW). En el entorno original también coincide su sha256.
- **Con consultas reales a Open-Meteo:** un día futuro, uno pasado y uno fuera de rango.
- **Qué no se comprobó:**
  - el comportamiento en un despliegue (servidor público, concurrencia, disponibilidad);
  - el desempeño de un snapshot nuevo hasta que se lo evalúe en un período posterior a su fecha objetivo.

## Demo con Streamlit

`demo/streamlit_app.py` es una interfaz para mostrar el modelo en la presentación. **No tiene lógica propia de pronóstico**: llama a `api.main.calcular_pronostico`, la misma función que usa el endpoint `POST /pronostico`. Por eso usa el mismo snapshot, la misma validación de Open-Meteo, las mismas variables, la misma banda, el mismo bloque `alcance` y las mismas advertencias. En el modo actual hace **una sola** consulta a Open-Meteo, y la tabla de variables que muestra es exactamente la que usó el modelo.

- **Alcance:** total del MEM; no hay selector regional, porque el modelo horario no pronostica por región.
- **Entrada:** fecha y si es feriado.
- **Salida:**
  - demanda máxima y hora del pico, demanda mínima, energía del día y temperatura mínima y máxima;
  - gráfico horario del pronóstico central con la **banda nominal del 80 %**;
  - gráfico de temperatura;
  - tabla de variables: hora, día de semana, mes, día hábil, temperatura, media móvil de 24 h, HDD y CDD con base 18 °C;
  - fuente meteorológica, modelo, antigüedad y advertencias de la API.
- **Métricas históricas:** "MAPE test 4,1 %" y "MAPE agosto de 2026 4,0 %" se muestran rotuladas como evaluaciones pasadas, no como el error del pronóstico mostrado.
- **Caso histórico validado — 12/08/2026:** una reproducción histórica, no un pronóstico actual. Usa las temperaturas de las corridas de ECMWF disponibles al emitir (00 y 06 UTC del 11/08/2026), registradas en el repositorio. Pasa por el modo de simulación manual de la API y se compara con la demanda real de CAMMESA de ese día. Sirve también para mostrar la demo sin conexión.
- **Errores:** si Open-Meteo no responde o responde datos inválidos, la página lo dice con claridad y sugiere el caso histórico. Si la fecha no tiene pronóstico, explica por qué.
- **No incluye:** SHAP, explicaciones con LLM ni afirmaciones causales.
- **Estilo:** los colores y tipografías de la presentación, en `.streamlit/config.toml` y `demo/estilo.py`. Solo cambian la apariencia.

**Ejecución local**, desde la carpeta del proyecto, en un entorno aparte:

```bash
python -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1      # Windows PowerShell (en Linux o macOS: source .venv-demo/bin/activate)
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

**Publicada en Streamlit Community Cloud:** https://forecast-demanda-mem.streamlit.app

- **Por qué funciona ahí:** el repo es público, el snapshot (2,2 MB) está versionado, `demo/requirements.txt` fija las versiones y Open-Meteo es accesible y gratuito para uso no comercial.
- **Configuración de la app:** rama `main`, archivo principal `demo/streamlit_app.py`, Python 3.12. Se actualiza sola con cada cambio en `main`.
- **Requisito:** scikit-learn tiene que quedar en 1.9.0, porque la API no carga el snapshot con otra versión.
- **Permisos en GitHub:** Streamlit necesita acceso a la organización y que la organización tenga habilitadas las *deploy keys* (Settings → Member privileges → Deploy keys).
- **Riesgo:** la app gratuita se suspende si nadie la usa y tarda en volver. Conviene abrirla antes de presentar, o usar el caso histórico.

## Reproducción: datos crudos y procedencia

`data/raw/` no se versiona en git (archivos pesados). `descargar_demanda_mem.py` baja sus planillas solo; para `construir_dataset_regional.py` hay que descargar los tres archivos crudos de las fuentes de arriba y copiarlos en `data/raw/` con esos nombres.

La base diaria (`Base Demanda Diaria 2017 2026.xlsx`) también interviene en la serie horaria: `descargar_demanda_mem.py` la usa para verificar el tipo de día. Las correcciones que resultan están versionadas en `data/reference/correcciones_tipo_dia.csv`, así que la serie se reproduce igual sin el Excel. Con el Excel, el script además comprueba que las correcciones sigan siendo las mismas, y se detiene si aparecen fechas nuevas sin registrar.

Los dos scripts de temperatura fijan el modelo (`ecmwf_ifs`) y guardan junto a cada CSV un `.meta.json` con la consulta, el punto de grilla y la fecha de descarga.

## Nota sobre feriados y tipo de día

CAMMESA clasifica cada día como *hábil* o *no hábil* (sábados, domingos y feriados) en sus propias planillas. Se adopta esa clasificación en lugar de construir un calendario de feriados propio, porque refleja el criterio que usa el operador para programar el despacho.

La planilla horaria solo distingue esos dos tipos y tiene errores. Por eso `descargar_demanda_mem.py` la compara con las dos fuentes de CAMMESA que clasifican en 7 tipos de día: la planilla regional (2021–2023) y la base diaria (2017–2026). Si difieren, prevalece la clasificación detallada, y el script se detiene si esas dos fuentes se contradicen entre sí. Se corrigieron 4 días, todos feriados o días no laborables con demanda de feriado:

| Fecha | Motivo | Planilla horaria | Corregido |
|---|---|---|---|
| 06/04/2023 | Jueves Santo (día no laborable) | hábil | no hábil |
| 07/04/2023 | Viernes Santo (feriado) | hábil | no hábil |
| 23/03/2026 | Día no laborable, puente previo al 24 de marzo | hábil | no hábil |
| 10/07/2026 | Día no laborable, puente posterior al 9 de julio | hábil | no hábil |

La tabla está versionada en `data/reference/correcciones_tipo_dia.csv`.
