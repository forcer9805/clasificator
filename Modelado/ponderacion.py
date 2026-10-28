"""Comparaciones de equilibrio en validación; devuelve resultados sin exportar."""
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from baseline import crear_pipeline, metricas, pesos_por_clase, TEXT, RANDOM_STATE

BASE = 'Sin ponderación'
WEIGHTED = 'Ponderación por clase'
UNDERSAMPLED = 'Submuestreo Credit reporting'


def comparar_ponderacion(data, parts, ampliado=False):
    """Compara pesos; ampliado añade exponentes intermedios y submuestreo.

    Los pesos intermedios tienen media 1 para mantener la escala de C.
    El submuestreo ajusta su propio TF-IDF usando solo las filas retenidas.
    """
    train = data.iloc[parts['Entrenamiento']]
    validation = data.iloc[parts['Validación']]
    labels = np.sort(train.Product.unique())
    assert len(labels) == 11
    pipelines = {BASE: crear_pipeline(), WEIGHTED: crear_pipeline(class_weight='balanced')}
    first_params = pipelines[BASE].named_steps['classifier'].get_params()
    second_params = pipelines[WEIGHTED].named_steps['classifier'].get_params()
    assert {key for key in first_params if first_params[key] != second_params[key]} == {'class_weight'}

    # n_train / (n_classes * n_train_of_class); solo estadísticas de entrenamiento.
    weights = pd.Series(compute_class_weight(class_weight='balanced', classes=labels, y=train.Product),
                        index=labels, name='Peso')
    counts = train.Product.value_counts().reindex(labels)
    weight_table = pd.DataFrame({'Filas en entrenamiento': counts, 'Peso': weights})
    weight_table['Peso total de la clase'] = counts * weights
    weight_table.index.name = 'Product'
    np.testing.assert_allclose(weight_table['Peso total de la clase'], len(train)/len(labels))
    np.testing.assert_allclose((counts*weights).sum(), len(train))
    if ampliado:
        for alpha in [.25, .5]:
            # Mantener peso medio 1 evita cambiar la escala global de regularización.
            intermediate = pesos_por_clase(train.Product, alpha)
            name = f'Ponderación α={alpha:g}'
            pipelines[name] = crear_pipeline(class_weight=intermediate.to_dict())
            weight_table[name] = intermediate
        pipelines[UNDERSAMPLED] = crear_pipeline()

    selected = np.arange(len(train))
    if ampliado:
        majority = counts.idxmax()
        assert majority.startswith('Credit reporting')
        cap = int(counts.drop(majority).max())
        majority_positions = np.flatnonzero(train.Product.eq(majority).to_numpy())
        retained = np.random.default_rng(RANDOM_STATE).choice(majority_positions, cap, replace=False)
        selected = np.sort(np.concatenate([np.flatnonzero(train.Product.ne(majority).to_numpy()), retained]))
        assert len(np.unique(selected)) == len(selected)
        assert train.iloc[selected].Product.value_counts()[majority] == cap
        pd.testing.assert_series_equal(train.iloc[selected].Product.value_counts().drop(majority).sort_index(),
                                       counts.drop(majority).sort_index(), check_names=False)
    sample_counts = pd.DataFrame({'Original': counts,
        'Submuestreo': train.iloc[selected].Product.value_counts().reindex(labels)})

    start = perf_counter()
    vectorizer = pipelines[BASE].named_steps['tfidf']
    with threadpool_limits(limits=4):
        x_train = vectorizer.fit_transform(train[TEXT])
        x_validation = vectorizer.transform(validation[TEXT])
    vector_seconds = perf_counter() - start
    # El mismo vocabulario y matrices para ambos; prueba no se transforma ni evalúa.
    for name, pipeline in pipelines.items():
        if name != UNDERSAMPLED:
            pipeline.set_params(tfidf=vectorizer)
    assert pipelines[BASE].named_steps['tfidf'] is pipelines[WEIGHTED].named_steps['tfidf']

    reports, predictions, rows, fit_rows, matrices = {}, {}, [], [], {}
    for name, pipeline in pipelines.items():
        fit_train, fit_validation, y_train = x_train, x_validation, train.Product
        current_vector_seconds = vector_seconds
        if name == UNDERSAMPLED:
            start = perf_counter()
            with threadpool_limits(limits=4):
                fit_train = pipeline.named_steps['tfidf'].fit_transform(train.iloc[selected][TEXT])
                fit_validation = pipeline.named_steps['tfidf'].transform(validation[TEXT])
            current_vector_seconds = perf_counter() - start
            y_train = train.iloc[selected].Product
        print(f'Ajustando {name}: {len(y_train):,} filas', flush=True)
        classifier = pipeline.named_steps['classifier']
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter('always', ConvergenceWarning)
            classifier.fit(fit_train, y_train)
        seconds = perf_counter() - start
        np.testing.assert_array_equal(classifier.classes_, labels)
        with threadpool_limits(limits=4):
            pred = classifier.predict(fit_validation)
        predictions[name] = pd.Series(pred, index=validation.index, name='Predicción')
        rows.append({'Modelo': name, **metricas(validation.Product, pred, labels)})
        fit_rows.append({'Modelo': name, 'Filas entrenamiento': len(y_train),
                         'Vectorización (s)': current_vector_seconds,
                         'Vocabulario': len(pipeline.named_steps['tfidf'].vocabulary_),
                         'Ajuste clasificador (s)': seconds,
                         'Iteraciones': int(classifier.n_iter_.max()),
                         'Avisos': ' | '.join(str(w.message) for w in captured) or 'Sin avisos'})
        print(f'{name}: {seconds:.1f} s; macro-F1={rows[-1]["Macro-F1"]:.4f}', flush=True)
        reports[name] = pd.DataFrame(classification_report(validation.Product, pred, labels=labels,
                                    output_dict=True, zero_division=0)).T.loc[labels]
        matrices[name] = confusion_matrix(validation.Product, pred, labels=labels, normalize='true')

    scores = pd.DataFrame(rows).set_index('Modelo')
    per_class = pd.DataFrame(index=labels)
    for name in pipelines:
        for metric in ['precision','recall','f1-score']:
            per_class[f'{metric} · {name}'] = reports[name][metric]
    per_class['Soporte validación'] = reports[BASE]['support'].astype(int)
    for metric in ['precision','recall','f1-score']:
        per_class[f'Cambio {metric}'] = reports[WEIGHTED][metric] - reports[BASE][metric]
    per_class.index.name = 'Product'

    leaders = train.Company.value_counts().head(3).index.tolist()
    groups = validation.Company.where(validation.Company.isin(leaders), 'Resto de empresas')
    company_rows = []
    for group in leaders+['Resto de empresas']:
        mask = groups.eq(group)
        for name, pred in predictions.items():
            company_rows.append({'Empresa / grupo': group, 'Modelo': name,
                'Filas': int(mask.sum()), 'Clases presentes': validation.loc[mask,'Product'].nunique(),
                **metricas(validation.loc[mask,'Product'], pred.loc[mask], labels)})

    # Diagnóstico, sin excluir ni reasignar observaciones.
    seen_rows = []
    for name, pred in predictions.items():
        actual_train = train.iloc[selected] if name == UNDERSAMPLED else train
        seen = validation.normalized_hash.isin(set(actual_train.normalized_hash))
        for is_seen in [True,False]:
            mask = seen.eq(is_seen)
            seen_rows.append({'Texto visto en entrenamiento': is_seen, 'Modelo': name,
                'Filas': int(mask.sum()), 'Clases presentes': validation.loc[mask,'Product'].nunique(),
                **metricas(validation.loc[mask,'Product'], pred.loc[mask], labels)})

    return {'pipelines': pipelines, 'pesos': weight_table, 'metricas': scores,
            'cambios': scores.subtract(scores.loc[BASE]), 'muestra': sample_counts,
            'cambios_globales': (scores.loc[WEIGHTED]-scores.loc[BASE]).rename('Cambio absoluto'),
            'clases': per_class, 'reportes': reports, 'empresas': pd.DataFrame(company_rows),
            'textos_vistos': pd.DataFrame(seen_rows), 'ajustes': pd.DataFrame(fit_rows),
            'matrices': matrices, 'labels': labels, 'vectorizacion_segundos': vector_seconds,
            'vocabulario': len(vectorizer.vocabulary_), 'filas_entrenamiento': len(train),
            'filas_validacion': len(validation), 'prueba_evaluada': False}
