"""Baseline sin equilibrado. Solo devuelve resultados en memoria; no exporta archivos."""
from pathlib import Path
from time import perf_counter
import hashlib
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

RANDOM_STATE = 42
TEXT = 'Consumer complaint narrative'
MODEL_NAME = 'TF-IDF + regresión logística'


def normalizar_texto(text):
    return re.sub(r'\s+', ' ', str(text).lower()).strip()


def cargar_datos(root):
    root = Path(root)
    if str(root / 'EDA') not in sys.path:
        sys.path.insert(0, str(root / 'EDA'))
    from reglas_negocio import reglas_aprobadas

    source = root / 'cfpb_complaints_2023-06-01_2024-05-31/cfpb_complaints_2023-06-01_2024-05-31.csv'
    columns = ['Complaint ID', 'Date received', 'Company', 'Product', 'Sub-product', TEXT]
    chunks = []
    for chunk in pd.read_csv(source, usecols=columns, dtype=str, chunksize=25_000):
        if chunk[TEXT].isna().any():
            raise ValueError('Hay narrativas ausentes: revisar el corte antes de entrenar.')
        normalized = chunk[TEXT].map(normalizar_texto)
        if normalized.eq('').any():
            raise ValueError('Hay narrativas vacías: revisar el corte antes de entrenar.')
        chunk['normalized_hash'] = pd.util.hash_pandas_object(normalized, index=False).to_numpy()
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)
    del chunks
    data = data.sort_values('Complaint ID', key=lambda col: col.astype('int64')).reset_index(drop=True)
    assert data['Complaint ID'].is_unique
    assert pd.to_datetime(data['Date received']).between('2023-06-01', '2024-05-31').all()
    assert len(data) == 582_489
    data['Product_original'] = data.Product
    mapping = reglas_aprobadas().set_index(['Producto original', 'Subproducto'])['Producto armonizado']
    targets = mapping.reindex(pd.MultiIndex.from_frame(data[['Product', 'Sub-product']])).to_numpy()
    mask = pd.notna(targets)
    data['Product'] = data.Product.where(~mask, targets)
    assert data.Product.nunique() == 11 and int(mask.sum()) == 96_869
    assert data.normalized_hash.nunique() == 364_650
    return data


def crear_particiones(data):
    indices = np.arange(len(data))
    train_validation, test = train_test_split(indices, test_size=.15, random_state=RANDOM_STATE,
                                             stratify=data.Product)
    train, validation = train_test_split(train_validation, test_size=.15/.85, random_state=RANDOM_STATE,
                                        stratify=data.iloc[train_validation].Product)
    parts = {'Entrenamiento': train, 'Validación': validation, 'Prueba': test}
    combined = np.concatenate(list(parts.values()))
    assert len(combined) == len(data) and np.unique(combined).size == len(data)
    for index in parts.values():
        assert data.iloc[index].Product.nunique() == 11
    return parts


def auditar_particiones(data, parts):
    summary, fingerprints = [], {}
    for name, indices in parts.items():
        subset = data.iloc[indices]
        summary.append({'Partición': name, 'Filas': len(indices), '% del corte': 100*len(indices)/len(data),
                        'Clases': subset.Product.nunique(), 'Textos distintos': subset.normalized_hash.nunique(),
                        'Fecha mínima': subset['Date received'].min(), 'Fecha máxima': subset['Date received'].max()})
        ids = '\n'.join(sorted(subset['Complaint ID']))
        fingerprints[name] = hashlib.sha256(ids.encode()).hexdigest()
    support = pd.DataFrame({name: data.iloc[index].Product.value_counts() for name,index in parts.items()}).fillna(0).astype(int)
    support.index.name = 'Product'
    train_hashes = set(data.iloc[parts['Entrenamiento']].normalized_hash)
    validation_hashes = set(data.iloc[parts['Validación']].normalized_hash)
    overlap = []
    for name, reference_name, reference in [('Validación', 'Entrenamiento', train_hashes),
                                            ('Prueba', 'Entrenamiento', train_hashes),
                                            ('Prueba', 'Entrenamiento + validación', train_hashes | validation_hashes)]:
        subset = data.iloc[parts[name]]
        seen = subset.normalized_hash.isin(reference)
        overlap.append({'Partición': name, 'Referencia': reference_name,
                        'Filas con texto compartido': int(seen.sum()),
                        '% de filas con texto compartido': 100*seen.mean(),
                        'Textos distintos compartidos': int(subset.loc[seen,'normalized_hash'].nunique())})
    return {'resumen': pd.DataFrame(summary).set_index('Partición'), 'soporte': support,
            'solapamiento': pd.DataFrame(overlap), 'huellas': pd.Series(fingerprints, name='SHA-256 de IDs de cada partición')}


def metricas(y_true, y_pred, labels):
    return {'Accuracy': accuracy_score(y_true,y_pred),
            'Macro-F1': f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0),
            'F1 ponderada': f1_score(y_true,y_pred,labels=labels,average='weighted',zero_division=0)}


def pesos_por_clase(y, alpha=.25):
    """Pesos balanced ** alpha, con peso medio 1 en las filas de entrenamiento."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha debe estar entre 0 y 1.')
    labels = np.sort(pd.Series(y).unique())
    weights = pd.Series(compute_class_weight('balanced', classes=labels, y=y), index=labels).pow(alpha)
    counts = pd.Series(y).value_counts().reindex(labels)
    weights /= (counts * weights).sum() / len(y)
    np.testing.assert_allclose((counts * weights).sum(), len(y))
    return weights.rename('Peso')


def crear_pipeline(class_weight=None):
    """Constructor común; class_weight=None conserva la referencia original."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=normalizar_texto, lowercase=False,
                                ngram_range=(1,2), min_df=3, max_features=80_000,
                                sublinear_tf=True, dtype=np.float32)),
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs', tol=1e-3,
                                          max_iter=300, class_weight=class_weight, random_state=RANDOM_STATE)),
    ])


def ejecutar_baseline(data, parts):
    train = data.iloc[parts['Entrenamiento']]
    labels = np.sort(data.Product.unique())
    # El vocabulario y el IDF se aprenden exclusivamente de entrenamiento.
    pipeline = crear_pipeline()
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(np.zeros((len(train),1)), train.Product)
    start = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(train[TEXT], train.Product)
    fit_seconds = perf_counter() - start
    warning_messages = [str(w.message) for w in captured]
    classifier = pipeline.named_steps['classifier']
    assert classifier.class_weight is None
    results, predictions, timings = [], {}, []
    for name in ['Validación','Prueba']:
        subset = data.iloc[parts[name]]
        dummy_prediction = dummy.predict(np.zeros((len(subset),1)))
        results.append({'Modelo': 'Clase mayoritaria', 'Partición': name,
                        **metricas(subset.Product,dummy_prediction,labels)})
        start = perf_counter()
        with threadpool_limits(limits=4):
            probabilities = pipeline.predict_proba(subset[TEXT])
        elapsed = perf_counter() - start
        prediction = classifier.classes_[probabilities.argmax(axis=1)]
        results.append({'Modelo': MODEL_NAME, 'Partición': name, **metricas(subset.Product,prediction,labels)})
        predictions[name] = pd.DataFrame({'Real': subset.Product.to_numpy(), 'Predicción': prediction,
            'Score máximo sin calibrar': probabilities.max(axis=1)}, index=subset.index)
        timings.append({'Partición': name, 'Filas': len(subset), 'Inferencia de lote (s)': elapsed,
                        'ms/fila (lote)': 1000*elapsed/len(subset)})
    train_hashes = set(train.normalized_hash)
    leaders = train.Company.value_counts().head(3).index.tolist()
    company_metrics, seen_metrics = [], []
    for name in ['Validación','Prueba']:
        subset = data.iloc[parts[name]]
        pred = predictions[name]
        company_groups = subset.Company.where(subset.Company.isin(leaders), 'Resto de empresas')
        for group in leaders+['Resto de empresas']:
            mask = company_groups.eq(group)
            company_metrics.append({'Partición': name, 'Grupo': group, 'Filas': int(mask.sum()),
                                    **metricas(subset.loc[mask,'Product'],pred.loc[mask,'Predicción'],labels)})
        for seen in [True,False]:
            mask = subset.normalized_hash.isin(train_hashes).eq(seen)
            if mask.any():
                seen_metrics.append({'Partición': name, 'Texto visto en entrenamiento': seen,
                    'Filas': int(mask.sum()), 'Clases presentes': subset.loc[mask,'Product'].nunique(),
                    **metricas(subset.loc[mask,'Product'],pred.loc[mask,'Predicción'],labels)})
    validation_pred = predictions['Validación']
    report = pd.DataFrame(classification_report(validation_pred.Real,validation_pred['Predicción'],
                          labels=labels,output_dict=True,zero_division=0)).T
    test_pred = predictions['Prueba']
    test_report = pd.DataFrame(classification_report(test_pred.Real,test_pred['Predicción'],
                               labels=labels,output_dict=True,zero_division=0)).T
    errors = validation_pred.loc[validation_pred.Real.ne(validation_pred['Predicción'])].join(
        data[['Complaint ID','Company',TEXT]])
    errors['Texto visto en entrenamiento'] = data.loc[errors.index,'normalized_hash'].isin(train_hashes)
    errors['Extracto'] = errors[TEXT].str.slice(0,250)
    errors = errors.drop(columns=TEXT)
    frequent_confusions = errors.groupby(['Real','Predicción']).size().rename('Errores').sort_values(ascending=False)
    inference_per_row = timings[0]['Inferencia de lote (s)']/timings[0]['Filas']
    return {'pipeline': pipeline, 'dummy': dummy, 'labels': labels, 'metricas': pd.DataFrame(results),
            'reporte_validacion': report, 'reporte_prueba': test_report,
            'matriz_validacion': confusion_matrix(validation_pred.Real,validation_pred['Predicción'],labels=labels,normalize='true'),
            'empresas': pd.DataFrame(company_metrics), 'vistos_no_vistos': pd.DataFrame(seen_metrics),
            'predicciones': predictions, 'errores_validacion': errors,
            'confusiones_validacion': frequent_confusions,
            'tiempos': pd.DataFrame(timings), 'entrenamiento_segundos': fit_seconds,
            'avisos': warning_messages, 'iteraciones': int(classifier.n_iter_.max()),
            'vocabulario': len(pipeline.named_steps['tfidf'].vocabulary_),
            'estimacion_500k_segundos': inference_per_row*500_000}
