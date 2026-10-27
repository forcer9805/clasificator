# Clasificador

Análisis exploratorio de quejas del CFPB para predecir `Product` a partir de
`Consumer complaint narrative`.

- [EDA/EDA.ipynb](EDA/EDA.ipynb): análisis con tablas y gráficas guardadas.
- [EDA/reglas_negocio.py](EDA/reglas_negocio.py): reglas de armonización y comprobaciones.

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
