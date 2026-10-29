# Forecast de demanda eléctrica del MEM argentino

Proyecto final del curso de Data Science — Programa Energía Digital 2026 (Fundación YPF + Jump Educación).

## El problema

CAMMESA, el operador del Mercado Eléctrico Mayorista (MEM), tiene que garantizar en todo momento que la generación iguale a la demanda: la electricidad no se almacena a escala de red. Para eso programa el despacho del día siguiente en base a un pronóstico de demanda. Cuanto mejor es ese pronóstico, menos reserva térmica hay que mantener encendida "por las dudas", con el consiguiente ahorro económico y de emisiones.

**Objetivo del modelo:** predecir la demanda eléctrica horaria del MEM (MWh) para las próximas 24 horas, a partir de variables de calendario (hora, día de semana, mes, tipo de día) y climáticas (temperatura).

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

## Estructura del repositorio

```
forecast-demanda-mem/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # descargas sin modificar (no versionadas en git)
│   ├── processed/    # datasets construidos (serie horaria MEM, temperaturas, regional horaria 2021-23, regional diaria con clima) y su procedencia (*.meta.json)
│   └── reference/    # correcciones de tipo de día reconciliadas con las fuentes de CAMMESA (versionada)
├── notebooks/        # numerados por etapa
│   ├── 01_eda_inicial.ipynb          # EDA (pre-entrega 2)
│   ├── 02_modelo_supervisado.ipynb   # regresión y clasificación (pre-entrega 3, ampliado en la 4)
│   └── 03_clustering.ipynb           # días típicos y días atípicos: K-means, DBSCAN, PCA (pre-entrega 4)
├── reports/
│   └── figures/      # gráficos exportados por los notebooks
└── src/
    ├── descargar_demanda_mem.py   # baja las planillas CAMMESA y arma la serie 2021-2026
    ├── descargar_temperatura.py   # baja la temperatura horaria histórica de Buenos Aires, ECMWF IFS (Open-Meteo)
    ├── descargar_temperatura_pronosticada.py  # baja la corrida de ECMWF disponible al emitir cada día del test (Open-Meteo)
    └── construir_dataset_regional.py  # arma el dataset regional diario con clima (CAMMESA + SMN)
```

## Cómo reproducir

```bash
pip install -r requirements.txt
python src/descargar_demanda_mem.py        # genera data/processed/demanda_horaria_mem.csv (--actualizar para bajar meses nuevos)
python src/construir_dataset_regional.py   # genera data/processed/dataset_demanda_por_region.csv
python src/descargar_temperatura.py        # genera data/processed/temperatura_horaria_ba.csv
python src/descargar_temperatura_pronosticada.py  # genera data/processed/temperatura_pronosticada_ba.csv
jupyter notebook notebooks/01_eda_inicial.ipynb
jupyter notebook notebooks/02_modelo_supervisado.ipynb
jupyter notebook notebooks/03_clustering.ipynb
```

`data/raw/` no se versiona en git (archivos pesados). `descargar_demanda_mem.py` baja sus planillas solo; para `construir_dataset_regional.py` hay que descargar los tres archivos crudos de las fuentes de arriba y copiarlos en `data/raw/` con esos nombres.

La base diaria (`Base Demanda Diaria 2017 2026.xlsx`) también interviene en la serie horaria: `descargar_demanda_mem.py` la usa para verificar el tipo de día. Las correcciones que resultan están versionadas en `data/reference/correcciones_tipo_dia.csv`, así que la serie se reproduce igual sin el Excel. Con el Excel, el script además comprueba que las correcciones sigan siendo las mismas, y se detiene si aparecen fechas nuevas sin registrar.

Los dos scripts de temperatura fijan el modelo (`ecmwf_ifs`) y guardan junto a cada CSV un `.meta.json` con la consulta, el punto de grilla y la fecha de descarga.

## Principales hallazgos del EDA

- La demanda va de ~9.300 a ~28.100 MW (media ~16.000), con tres estacionalidades superpuestas: diaria (valle 3–6 h), semanal (hábil > sábado > domingo) y anual con **dos picos**, verano (ene–feb) e invierno (jun–jul).
- **La hora del pico depende de la estación:** a la tarde (14–16 h) en verano, a la noche (20–21 h) en invierno. Esta interacción hora × temporada justifica probar modelos no lineales.
- Todas las horas récord de la serie son tardes hábiles de febrero (olas de calor). Se conservan: son los eventos que más importa acertar.
- GBA + Litoral + Buenos Aires concentran el 61 % de la demanda nacional, por lo que la temperatura de la región pampeana es la variable climática natural para el total país.
- La demanda tiene forma de **U** con la temperatura (mínimo en ~19 °C de temperatura media en GBA). La correlación lineal casi no la detecta, lo que refuerza el uso de modelos no lineales o de variables de grados-día.
- Las 11 variables climáticas del SMN quedaron alineadas y son consistentes en 2017–2026, pero no todas están listas para modelar: `HELIOF`, `TMAX` y `TMIN` tienen faltantes.

## Modelo supervisado (pre-entregas 3 y 4, revisado)

Notebook `02_modelo_supervisado.ipynb`. Predice la demanda horaria del MEM del día siguiente solo con calendario (hora, día de semana, mes, hábil/no hábil) y temperatura horaria de Buenos Aires, transformada en grados de calefacción y refrigeración con base en 18 °C, la zona de confort observada en los datos. Entrenamiento 2021–2024 y test ene-2025 → jul-2026, siempre por tiempo. Los hiperparámetros se eligen con validación cruzada temporal (`TimeSeriesSplit`), sin validaciones internas al azar (`early_stopping=False`).

**Qué se conoce al emitir.** Se supone que el pronóstico del día D se emite el día D-1 a las 12:00, con una hora reservada para entrenar y emitir (corte de información: D-1 11:00). Todo el notebook respeta esas reglas, y hay controles automáticos que las verifican **bajo los supuestos de demora**. Lo registrado en las fuentes son la hora de cada dato y la hora de inicio de cada corrida; las demoras de publicación son supuestas, porque no están registradas:

- **Demanda:** cada fila representa el intervalo de *t* a *t* + 1 h. Se supone medición en tiempo real, así que una hora se conoce cuando termina el intervalo.
- **Temperatura de entrenamiento:** la serie histórica de ECMWF IFS. Se supone publicada a más tardar 12 h después.
- **Temperatura del test:** sale solo de corridas de ECMWF cuyo inicio más 6 h de demora supuesta es anterior al corte, incluidas las horas previas a la emisión que entran en el promedio de 24 h.
- **Modelos que usan estas reglas:** el entrenamiento inicial (hasta el 30/12/2024 22:00), el reentrenamiento mensual, los modelos cuantiles y la calibración.

| Modelo (regresión) | MAPE validación | MAPE test | WAPE test |
|---|---|---|---|
| Referencia: misma hora de la semana anterior | — | 9,1 % | 9,2 % |
| Referencia: perfil promedio | — | 7,6 % | 7,9 % |
| Regresión lineal | 6,3 % | 6,1 % | 6,2 % |
| Random Forest | 4,8 % | 4,3 % | 4,4 % |
| **Gradient Boosting (modelo de calendario y clima elegido)** | **4,5 %** | **4,1 %** | **4,2 %** |
| Gradient Boosting reentrenado cada mes (últimos 24 meses) | — | 4,1 % | 4,2 % |
| Variante con demanda de hace 48 h y 168 h | 4,1 % | 3,6 % | 3,7 % |

El WAPE (suma de errores sobre suma de demanda) pondera más las horas de mayor demanda. Con la temperatura histórica el Gradient Boosting daría 3,98 %: usar el pronóstico le suma 0,10 puntos. El pronóstico se aparta 0,7 °C en promedio de la serie histórica de ECMWF IFS; esa diferencia **no** es un error medido contra estaciones meteorológicas.

Clasificación de **horas de demanda alta** (sobre el percentil 90 de train): el Random Forest anticipa el 91,6 % con una precisión del 63 % (AUC-PR 0,870) y supera a la regresión logística.

**Análisis complementarios:**

- **Picos.** El modelo ubica bien la hora del máximo diario (88 % de los días a ±1 h), pero tiende a quedarse corto en su magnitud: en el 59 % de los días subestima el máximo, y en las horas de demanda alta el MAPE es 5,6 %. No es una imposibilidad matemática: el Gradient Boosting suma correcciones y puede predecir por encima del máximo de entrenamiento, aunque en este test nunca lo hizo (máximo predicho: 25.771 MWh, contra 27.433 MWh en train).
- **Días de calor extremo** (máxima por encima del percentil 95 de train, 30,9 °C). El error es un 24 % mayor (MAPE 4,9 % contra 4,0 %), sin sesgo claro: son días más inciertos.
- **Reentrenamiento mensual** con los últimos 24 meses (ventana elegida validando sobre 2024). Reduce el sesgo de 232 a 65 MWh y el de 2026 de 2,4 % a 1,0 %, pero no el error total.
- **Intervalos de predicción del 80 %.** Salen de regresión cuantil y se calibran con un método conformal elegido en una validación sobre 2024: ajuste único con los últimos 3 meses. Esa validación casi no distingue entre ventanas de 3, 6 y 12 meses ni frente a un ajuste por hora. Cubren el 79,1 % de las horas en promedio, pero **ese promedio no garantiza buena cobertura mensual**: según el mes van del 57 % al 93 %, en 5 de 19 meses quedan por debajo del 75 %, y ninguna alternativa probada lo corrige. Cuantifican la incertidumbre de la demanda; no equivalen a la reserva del sistema.
- **Demanda reciente.** Sumar la demanda de hace 48 y 168 horas, disponible al emitir, baja el MAPE a 3,6 %. Las dos versiones describen asociaciones predictivas, no causas. La variante se eligió con validación en train, pero el test ya había sido usado para explorarla, así que no es una prueba independiente.

**Limitaciones:**

- **Supuestos de operación.** La hora de emisión, la hora reservada para entrenar, la medición de la demanda en tiempo real y las demoras de publicación de la temperatura son supuestos documentados, no el procedimiento oficial de CAMMESA.
- **Temperatura de modelo.** Tanto la temperatura histórica como la pronosticada son del modelo ECMWF IFS, no mediciones de estación. Los errores de temperatura que se reportan son diferencias entre esas dos series.
- **Subestimación de 2026.** La causa no está del todo identificada: comparando enero–julio de cada año, la demanda creció solo 0,9 %.
- **Una sola ciudad.** La temperatura de Buenos Aires no captura las olas de calor nacionales.
- **Calendario binario.** Hábil / no hábil no distingue vísperas, puentes ni vacaciones.
- **Intervalos.** Hay subcobertura en meses puntuales: febrero, junio y noviembre de 2025 y enero y julio de 2026 quedan por debajo del 75 %.
- **Período de test reutilizado.** Las revisiones se evaluaron sobre el mismo período de test; una confirmación independiente necesita meses posteriores a julio de 2026.

### Revisión antes de la entrega final

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

## Clustering: días típicos y atípicos (pre-entrega 4)

Notebook `03_clustering.ipynb`. Cada día (2.038, de 2021 a jul-2026) se describe con sus 24 demandas horarias divididas por el promedio del día, para agrupar por **forma** de la curva y no por nivel. El tipo de día, el mes y la temperatura no entran al clustering: se usan después para interpretar.

**K-means (k = 4)**, con k elegido por codo y silueta. Encuentra cuatro perfiles de carga con sentido físico:

| Grupo | Días | Forma de la curva |
|---|---|---|
| Hábil de invierno | 847 | Dos picos: mañana (~11 h) y noche (~20 h) |
| Hábil de verano | 480 | Pico de tarde (14–16 h), por el aire acondicionado |
| No hábil de invierno | 428 | Sin pico de mañana; pico de noche muy marcado |
| No hábil de verano | 283 | La más plana; incluye días hábiles de vacaciones de enero y febrero |

- **PCA:** dos componentes explican el 79 % de la variación. Una es el "pico de la tarde" (eje verano ↔ invierno) y la otra la "madrugada alta, mañana baja" (eje no hábil ↔ hábil).
- **Los grupos los marca más la temperatura que el calendario:** los meses de transición se reparten según haga calor o no.
- **DBSCAN** no encuentra grupos separados: los días forman un continuo (silueta de K-means ~0,37). Usado como detector de anomalías, marca un 4 % de días atípicos: fiestas de fin de año, carnavales, Semana Santa, días de calor extremo y el **apagón del 1/3/2023**, que detectó sin información externa.

## Integrantes

- Juan Serrano
- Rodrigo Duarte

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
