"""Comparación extensible de clasificadores; evaluación solo en validación."""
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from baseline import crear_pipeline, metricas, pesos_por_clase, TEXT, RANDOM_STATE

REFERENCE = 'Regresión logística · α=0,25'
SVM = 'LinearSVC · α=0,25'
NB = 'ComplementNB · sin pesos'
NB_WEIGHTED = 'ComplementNB · α=0,25'


def preparar_comparacion(data, parts):
    """Un único vocabulario ajustado con entrenamiento para todos los TF-IDF."""
    train = data.iloc[parts['Entrenamiento']]
    validation = data.iloc[parts['Validación']]
    assert set(train.index).isdisjoint(validation.index)
    labels = np.sort(train.Product.unique())
    weights = pesos_por_clase(train.Product, .25)
    vectorizer = crear_pipeline().named_steps['tfidf']
    start = perf_counter()
    with threadpool_limits(limits=4):
        x_train = vectorizer.fit_transform(train[TEXT])
        x_validation = vectorizer.transform(validation[TEXT])
    seconds = perf_counter() - start
    leaders = train.Company.value_counts().head(3).index.tolist()
    return {'train': train, 'validation': validation, 'labels': labels,
            'pesos': weights, 'vectorizer': vectorizer, 'x_train': x_train,
            'x_validation': x_validation, 'vectorizacion_segundos': seconds,
            'leaders': leaders, 'resultados': {}, 'prueba_evaluada': False}


def catalogo_modelos(context):
    """Configuraciones iniciales fijadas, sin búsqueda de hiperparámetros."""
    weights = context['pesos'].to_dict()
    return {
        REFERENCE: (crear_pipeline(class_weight=weights).named_steps['classifier'], False),
        SVM: (LinearSVC(C=1., class_weight=weights, dual='auto', tol=1e-4,
                        max_iter=3000, random_state=RANDOM_STATE), False),
        NB: (ComplementNB(alpha=1., norm=False), False),
        NB_WEIGHTED: (ComplementNB(alpha=1., norm=False), True),
    }


def entrenar_modelo(context, name, estimator, ponderar_filas=False):
    """Añade o reemplaza un resultado en memoria, sin exportar ni usar prueba.

    ComplementNB recibe pesos por fila; alpha=1 de ComplementNB es suavizado,
    independiente del exponente 0,25 utilizado para calcular esos pesos.
    """
    train, validation, labels = context['train'], context['validation'], context['labels']
    fit_kwargs = {}
    if ponderar_filas:
        fit_kwargs['sample_weight'] = train.Product.map(context['pesos']).to_numpy()
        np.testing.assert_allclose(fit_kwargs['sample_weight'].mean(), 1.)
    start = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
        warnings.simplefilter('always')
        estimator.fit(context['x_train'], train.Product, **fit_kwargs)
    seconds = perf_counter() - start
    np.testing.assert_array_equal(estimator.classes_, labels)
    start = perf_counter()
    with threadpool_limits(limits=4):
        pred = estimator.predict(context['x_validation'])
    prediction_seconds = perf_counter() - start
    return registrar_predicciones(context, name, pred, estimator, seconds,
                                   prediction_seconds, [str(w.message) for w in captured])


def registrar_predicciones(context, name, pred, estimator, seconds, prediction_seconds, avisos=()):
    """Evaluación común para modelos TF-IDF y neuronales, siempre en validación."""
    train, validation, labels = context['train'], context['validation'], context['labels']
    pred = pd.Series(pred, index=validation.index)
    assert len(pred) == len(validation) and pred.isin(labels).all()
    report = pd.DataFrame(classification_report(validation.Product, pred, labels=labels,
                           output_dict=True, zero_division=0)).T.loc[labels]
    groups = validation.Company.where(validation.Company.isin(context['leaders']), 'Resto de empresas')
    companies = []
    for group in context['leaders'] + ['Resto de empresas']:
        mask = groups.eq(group)
        companies.append({'Grupo': group, 'Filas': int(mask.sum()),
            'Clases presentes': validation.loc[mask, 'Product'].nunique(),
            **metricas(validation.loc[mask, 'Product'], pred.loc[mask], labels)})
    shared = validation.normalized_hash.isin(set(train.normalized_hash))
    seen_rows = []
    for seen in [True, False]:
        mask = shared.eq(seen)
        if mask.any():
            seen_rows.append({'Texto visto en entrenamiento': seen, 'Filas': int(mask.sum()),
                **metricas(validation.loc[mask, 'Product'], pred.loc[mask], labels)})
    result = {'modelo': estimator, 'predicciones': pred, 'reporte': report,
        'metricas': metricas(validation.Product, pred, labels),
        'matriz': confusion_matrix(validation.Product, pred, labels=labels, normalize='true'),
        'matriz_recuentos': confusion_matrix(validation.Product, pred, labels=labels),
        'empresas': pd.DataFrame(companies).set_index('Grupo'),
        'textos': pd.DataFrame(seen_rows),
        'configuracion': estimator.get_params() if hasattr(estimator, 'get_params') else {},
        'tiempos': {'Ajuste (s)': seconds, 'Predicción validación (s)': prediction_seconds,
                   'Iteraciones': int(np.max(estimator.n_iter_)) if hasattr(estimator, 'n_iter_') else None,
                   'Avisos': ' | '.join(avisos) or 'Sin avisos'}}
    context['resultados'][name] = result
    return pd.Series({**result['metricas'], **result['tiempos']}, name=name).to_frame()


def errores_frecuentes(context, name):
    """Pares real/predicha ordenados por número de errores, sin la diagonal."""
    labels = context['labels']
    matrix = context['resultados'][name]['matriz_recuentos']
    total_errors = int(matrix.sum() - np.trace(matrix))
    rows = []
    for i, real in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and matrix[i,j]:
                rows.append({'Clase real': real, 'Predicción incorrecta': predicted,
                    'Errores': int(matrix[i,j]), '% de la clase real': 100*matrix[i,j]/matrix[i].sum(),
                    '% de todos los errores': 100*matrix[i,j]/total_errors})
    columns = ['Clase real','Predicción incorrecta','Errores','% de la clase real','% de todos los errores']
    return pd.DataFrame(rows, columns=columns).sort_values('Errores', ascending=False).reset_index(drop=True)


def mostrar_matrices(context, name):
    """Recuentos y porcentajes por clase real para distinguir volumen y cobertura."""
    from textwrap import fill
    import matplotlib.pyplot as plt
    labels = context['labels']
    result = context['resultados'][name]
    fig, axes = plt.subplots(1,2,figsize=(25,11))
    for ax, key, title in zip(axes, ['matriz_recuentos','matriz'], ['Número de quejas','Proporción de cada clase real']):
        matrix = result[key]
        image = ax.imshow(matrix,cmap='Blues',vmin=0, vmax=matrix.max() if key=='matriz_recuentos' else 1)
        ax.set_xticks(range(len(labels)),[fill(p,22) for p in labels],rotation=60,ha='right',fontsize=7)
        ax.set_yticks(range(len(labels)),[fill(p,27) for p in labels],fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i,j] > 0 and (key=='matriz_recuentos' or matrix[i,j]>=.005):
                    value = str(matrix[i,j]) if key=='matriz_recuentos' else f'{matrix[i,j]:.1%}'
                    ax.text(j,i,value,ha='center',va='center',fontsize=7,
                            color='white' if matrix[i,j]>.5*image.get_clim()[1] else '#19313f')
        ax.set(title=title,xlabel='Predicción',ylabel='Clase real')
        fig.colorbar(image,ax=ax,fraction=.03,pad=.02)
    fig.suptitle(f'{name} · validación',fontsize=15)
    plt.tight_layout(); plt.show()


def resumir_comparacion(context):
    results = context['resultados']
    scores = pd.DataFrame({name: result['metricas'] for name, result in results.items()}).T
    scores.index.name = 'Modelo'
    return {
        'metricas': scores,
        'cambios': scores.subtract(scores.loc[REFERENCE]),
        'tiempos': pd.DataFrame({name: r['tiempos'] for name, r in results.items()}).T,
        'f1_clases': pd.DataFrame({name: r['reporte']['f1-score'] for name, r in results.items()}),
        'empresas': pd.concat({name: r['empresas'] for name, r in results.items()}, names=['Modelo']),
        'textos': pd.concat({name: r['textos'] for name, r in results.items()}, names=['Modelo']),
    }
