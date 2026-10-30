# Clasificador

Clasificación de quejas del CFPB para predecir `Product` a partir de
`Consumer complaint narrative`.

- [EDA/EDA.ipynb](EDA/EDA.ipynb): análisis con tablas y gráficas guardadas.
- [EDA/reglas_negocio.py](EDA/reglas_negocio.py): reglas de armonización y comprobaciones.
- [Baseline/Baseline.ipynb](Baseline/Baseline.ipynb): referencia mayoritaria y TF-IDF con regresión logística, sin equilibrado.
- [Baseline/baseline.py](Baseline/baseline.py): carga, particiones, entrenamiento y evaluación.
- [Modelado/Ponderacion_clases.ipynb](Modelado/Ponderacion_clases.ipynb): DEC-07, comparación con ponderación por clase.
- [Modelado/Equilibrio_alternativas.ipynb](Modelado/Equilibrio_alternativas.ipynb): pesos intermedios y reducción de Credit reporting.
- [Modelado/Comparacion_modelos.ipynb](Modelado/Comparacion_modelos.ipynb): modelos clásicos sobre el corte completo y piloto de MiniLM y DistilBERT, con matrices de confusión y errores frecuentes.

Los documentos de decisiones y conclusiones se entregan por separado.

## Ejecutar el EDA

Verificado con Python 3.13. Desde la raíz del proyecto, instala las dependencias
en el entorno que utilizarás como kernel:

```bash
python -m pip install -r EDA/requirements-notebook.txt
```

El CSV no se incluye en el repositorio. Debe conservar esta ubicación respecto a la raíz:

```text
cfpb_complaints_2023-06-01_2024-05-31/cfpb_complaints_2023-06-01_2024-05-31.csv
```

Abre `EDA/EDA.ipynb` en VS Code con la extensión Jupyter, selecciona el entorno
Python y pulsa **Ejecutar todo**. El directorio de trabajo puede ser la raíz o `EDA/`.

También puedes ejecutar y guardar las salidas desde la raíz con:

```bash
jupyter-execute EDA/EDA.ipynb --inplace --timeout=900
```

El análisis procesa el corte completo y puede tardar varios minutos. Los resultados
quedan en el notebook; no modifica el CSV original ni exporta archivos de resultados.

## Ejecutar el baseline

Utiliza el mismo CSV y Python 3.13. Instala las dependencias y ejecuta desde la raíz:

```bash
python -m pip install -r Baseline/requirements-baseline.txt
jupyter-execute Baseline/Baseline.ipynb --inplace --timeout=900
```

También puedes abrir el notebook en VS Code y pulsar **Ejecutar todo**. Entrena
con el 70 % de las filas y reserva un 15 % para validación y un 15 % para prueba,
con estratificación y semilla 42. No aplica equilibrado ni elimina textos repetidos;
publica el solapamiento entre particiones y las métricas globales, por clase y empresa.
Los parámetros del baseline están fijados; las mejoras se seleccionarán en validación.
El entrenamiento puede tardar varios minutos. Las salidas quedan en el notebook.

## DEC-07: ponderación por clase

Con las mismas dependencias del baseline:

```bash
jupyter-execute Modelado/Ponderacion_clases.ipynb --inplace --timeout=900
```

Este notebook utiliza las funciones comunes de `Baseline/baseline.py` para cargar
datos, crear particiones, construir el pipeline y calcular métricas. El experimento
está en `Modelado/ponderacion.py` y cambia solo `class_weight` de `None` a `balanced`.
Compara en validación con el mismo TF-IDF y las mismas filas, sin remuestreo.
El conjunto de prueba se reserva y el notebook original del baseline se conserva.

Para comparar también pesos intermedios y submuestreo de Credit reporting:

```bash
jupyter-execute Modelado/Equilibrio_alternativas.ipynb --inplace --timeout=1800
```

Reutiliza `Modelado/ponderacion.py`. Solo reduce la clase dominante en entrenamiento
hasta el tamaño de la segunda clase, con semilla 42; ajusta su TF-IDF sobre esa muestra.
Validación conserva sus filas y distribución. Los resultados quedan en el notebook.

La referencia aprobada para comparar nuevos modelos es TF-IDF + regresión logística
con pesos intermedios α=0,25 (normalizados a media 1), ejecutada en
`Modelado/Equilibrio_alternativas.ipynb`. El baseline original sin pesos se conserva.

## Comparar modelos

Los apartados clásicos usan las dependencias del baseline. Para ejecutar también
el piloto neuronal en CPU, instala las dependencias completas antes de ejecutar:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r Modelado/requirements-semanticos.txt
jupyter-execute Modelado/Comparacion_modelos.ipynb --inplace --timeout=7200
```

Reutiliza carga, particiones, pesos y métricas de `Baseline/baseline.py`.
`Modelado/comparacion.py` prepara el TF-IDF compartido y permite añadir clasificadores
a la comparación. Cada ajuste tiene su propia celda; las tablas y gráficas se generan
desde los resultados en memoria. Las matrices muestran recuentos y porcentajes por
clase real; las tablas ordenan los errores fuera de la diagonal.

Los apartados 13–21 utilizan `Modelado/semanticos.py`. El piloto en CPU usa
11.000 filas de entrenamiento y 5.500 de validación, estratificadas
dentro de las particiones originales. Reentrena las referencias sobre esas mismas filas.
Los modelos neuronales reciben hasta 128 tokens del inicio y final de cada narrativa;
el notebook informa del recorte. MiniLM permanece congelado; DistilBERT se ajusta durante
una época. Estas métricas se comparan dentro del piloto, separadas del corte completo.
Los pesos preentrenados se descargan en la caché de Hugging Face durante el primer uso.
La ejecución en CPU puede superar una hora. La comparación neuronal completa queda
para GPU, revisando tamaños, contexto y presupuesto de ajuste. No se exportan datos ni modelos.
