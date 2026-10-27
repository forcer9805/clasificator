import pandas as pd

MAPPING_VERSION = '2026-10-27-v1'

def reglas_aprobadas():
    credit_old = 'Credit reporting, credit repair services, or other personal consumer reports'
    cards_old = 'Credit card or prepaid card'
    loan_old = 'Payday loan, title loan, or personal loan'
    rows = []
    def add(old, subs, new):
        rows.extend((old, sub, new) for sub in subs)
    add(credit_old, ['Credit reporting', 'Other personal consumer report'],
        'Credit reporting or other personal consumer reports')
    add(credit_old, ['Credit repair services'], 'Debt or credit management')
    add(cards_old, ['General-purpose credit card or charge card', 'Store credit card'], 'Credit card')
    add(cards_old, ['General-purpose prepaid card', 'Government benefit card', 'Gift card',
                    'Payroll card', 'Student prepaid card'], 'Prepaid card')
    add(loan_old, ['Installment loan', 'Personal line of credit', 'Payday loan', 'Title loan', 'Pawn loan'],
        'Payday loan, title loan, personal loan, or advance loan')
    add('Money transfer, virtual currency, or money service', ['Debt settlement'], 'Debt or credit management')
    return pd.DataFrame(rows, columns=['Producto original', 'Subproducto', 'Producto armonizado'])


def perfil_conflictos(data, hash_col, label_col):
    group = data.groupby(hash_col).agg(filas=('Complaint ID', 'size'), etiquetas=(label_col, 'nunique'))
    repeated = group[group.filas > 1]
    conflicts = group[group.etiquetas > 1]
    return group, {
        'Textos distintos': len(group),
        'Filas adicionales repetidas': int((repeated.filas - 1).sum()),
        'Grupos con varias etiquetas': len(conflicts),
        'Filas en grupos con varias etiquetas': int(conflicts.filas.sum()),
    }


def aplicar_reglas_negocio(original):
    data = original.copy()
    n = len(data)
    assert data['Complaint ID'].is_unique
    data['Product_original'] = data['Product']
    rules = reglas_aprobadas()
    mapping = rules.set_index(['Producto original', 'Subproducto'])['Producto armonizado']
    assert len(mapping) == 16 and mapping.index.is_unique
    pairs = pd.MultiIndex.from_frame(data[['Product_original', 'Sub-product']])
    candidates = mapping.reindex(pairs).to_numpy()
    mapped = pd.notna(candidates)
    data['Product'] = data['Product'].where(~mapped, candidates)
    data['taxonomy_version'] = MAPPING_VERSION

    # Las tres etiquetas retiradas se sustituyen en todo el corte, también el día 25.
    # El cuarto producto de origen conserva el resto de sus subproductos.
    discontinued = ['Credit reporting, credit repair services, or other personal consumer reports',
                    'Credit card or prepaid card', 'Payday loan, title loan, or personal loan']
    assert not data['Product'].isin(discontinued).any()
    assert len(data) == n and data['Complaint ID'].equals(original['Complaint ID'])
    assert data['text_hash'].equals(original['text_hash'])
    assert data['normalized_hash'].equals(original['normalized_hash'])
    assert data['Product'].nunique() == 11

    transition = pd.Timestamp('2023-08-25')
    audit = []
    for _, rule in rules.iterrows():
        mask = original['Product'].eq(rule['Producto original']) & original['Sub-product'].eq(rule['Subproducto'])
        assert mask.any(), rule.to_dict()
        assert data.loc[mask, 'Product'].eq(rule['Producto armonizado']).all()
        audit.append({**rule.to_dict(), 'Filas sustituidas': int(mask.sum()),
                      'Antes del 25/08': int((mask & original.date.lt(transition)).sum()),
                      'El 25/08': int((mask & original.date.eq(transition)).sum()),
                      'Después del 25/08': int((mask & original.date.gt(transition)).sum())})
    audit = pd.DataFrame(audit)
    assert audit['Filas sustituidas'].sum() == mapped.sum()
    distribution = data['Product'].value_counts().rename_axis('Product').to_frame('Quejas')
    distribution['% del total'] = 100 * distribution.Quejas / n
    assert distribution.Quejas.sum() == n

    company_counts = data.Company.value_counts()
    leaders = company_counts.head(3).index.tolist()
    group_order = leaders + ['Resto de empresas (agregado)']
    data['company_group'] = data.Company.where(data.Company.isin(leaders), group_order[-1])
    concentration = data.groupby('company_group').agg(
        Filas=('Complaint ID', 'size'), Textos_normalizados_distintos=('normalized_hash', 'nunique'))
    concentration['% de filas'] = 100 * concentration.Filas / n
    concentration['Repeticiones adicionales dentro del grupo'] = concentration.Filas - concentration.Textos_normalizados_distintos
    concentration['% repetición interna'] = 100 * concentration['Repeticiones adicionales dentro del grupo'] / concentration.Filas

    # Cada texto global pesa uno. Si aparece en k empresas, cada empresa recibe 1/k.
    # Así los totales son aditivos y no se cuenta un texto global varias veces.
    text_company = data[['normalized_hash', 'Company', 'company_group']].drop_duplicates()
    company_multiplicity = text_company.groupby('normalized_hash').Company.transform('size')
    text_company['Peso de texto único'] = 1.0 / company_multiplicity
    weighted = text_company.groupby('company_group')['Peso de texto único'].sum()
    concentration['Textos únicos (peso fraccional entre empresas)'] = weighted
    global_unique = data.normalized_hash.nunique()
    assert abs(weighted.sum() - global_unique) < 1e-6
    concentration['% de textos únicos (peso fraccional)'] = 100 * weighted / global_unique
    concentration = concentration.reindex(group_order)

    by_product = data.groupby(['company_group', 'Product']).agg(
        Filas=('Complaint ID', 'size'), Textos_normalizados_distintos=('normalized_hash', 'nunique'))
    by_product['% filas dentro del grupo'] = 100 * by_product.Filas / by_product.groupby(level=0).Filas.transform('sum')
    assert by_product.Filas.sum() == n

    profiles, groups = {}, {}
    stages = [
        ('Etiquetas originales · texto original', 'text_hash', 'Product_original'),
        ('Etiquetas originales · texto normalizado', 'normalized_hash', 'Product_original'),
        ('Etiquetas armonizadas · texto original', 'text_hash', 'Product'),
        ('Etiquetas armonizadas · texto normalizado', 'normalized_hash', 'Product'),
    ]
    for label, text_col, product_col in stages:
        groups[label], profiles[label] = perfil_conflictos(data, text_col, product_col)
    conflict_table = pd.DataFrame(profiles).T
    conflict_table['% filas en grupos con varias etiquetas'] = 100 * conflict_table['Filas en grupos con varias etiquetas'] / n
    old = groups[stages[1][0]]
    new = groups[stages[3][0]]
    assert old.index.equals(new.index)
    resolved = old.etiquetas.gt(1) & new.etiquetas.eq(1)
    persistent = old.etiquetas.gt(1) & new.etiquetas.gt(1)
    introduced = old.etiquetas.eq(1) & new.etiquetas.gt(1)
    changes = pd.DataFrame([
        {'Resultado tras armonizar (texto normalizado)': name,
         'Grupos': int(mask.sum()), 'Filas': int(new.loc[mask, 'filas'].sum())}
        for name, mask in [('Conflicto resuelto', resolved), ('Conflicto persistente', persistent),
                           ('Conflicto introducido por desagregar categorías', introduced)]
    ])
    assert int(resolved.sum() + persistent.sum()) == profiles[stages[1][0]]['Grupos con varias etiquetas']
    assert int(persistent.sum() + introduced.sum()) == profiles[stages[3][0]]['Grupos con varias etiquetas']

    conflict_hashes = new.index[new.etiquetas.gt(1)]
    selected = data.loc[data.normalized_hash.isin(conflict_hashes)]
    examples = selected.groupby('normalized_hash').agg(
        Filas=('Complaint ID', 'size'), Empresas=('Company', 'nunique'),
        Etiquetas_originales=('Product_original', lambda x: ' | '.join(sorted(x.unique()))),
        Etiquetas_armonizadas=('Product', lambda x: ' | '.join(sorted(x.unique()))),
        IDs_ejemplo=('Complaint ID', lambda x: ', '.join(x.head(3))),
    ).sort_values('Filas', ascending=False).head(10)
    introduced_examples = data.loc[data.normalized_hash.isin(new.index[introduced]),
        ['Complaint ID', 'Product_original', 'Product', 'Sub-product', 'Company', 'Date received', 'normalized_hash']].copy()

    summary = {
        'mapping_version': MAPPING_VERSION, 'rows_before': n, 'rows_after': len(data),
        'products_before': int(original.Product.nunique()), 'products_after': int(data.Product.nunique()),
        'rules': len(rules), 'rules_observed': len(audit), 'rows_recoded': int(mapped.sum()),
        'transition_day_rows': int(original.date.eq(transition).sum()),
        'transition_day_rows_recoded': int((mapped & original.date.eq(transition)).sum()),
        'recoded_after_transition_day': int((mapped & original.date.gt(transition)).sum()),
        'unique_normalized_texts': int(global_unique), 'no_rows_removed': True,
        'leaders': leaders,
        'top3_row_share_pct': float(concentration.iloc[:3]['% de filas'].sum()),
        'top3_unique_text_weighted_share_pct': float(concentration.iloc[:3]['% de textos únicos (peso fraccional)'].sum()),
        'conflicts': {k: {key: int(value) for key, value in val.items()} for k,val in profiles.items()},
        'conflicts_resolved_groups': int(resolved.sum()), 'conflicts_persistent_groups': int(persistent.sum()),
        'conflicts_introduced_groups': int(introduced.sum()),
        'notes': ['Text hashes are pandas 64-bit hashes; residual collision risk.',
                  'Within-group unique counts are not additive across groups.',
                  'Weighted shares allocate each global text equally among its distinct companies.',
                  'Conflicting labels are flagged, not corrected or excluded.'],
    }
    tables = {'reglas_taxonomia': rules, 'auditoria_taxonomia': audit,
                 'distribucion_productos': distribution, 'concentracion_empresas': concentration,
                 'productos_por_grupo_empresa': by_product, 'conflictos_etiquetas': conflict_table,
                 'cambios_conflictos': changes, 'ejemplos_conflictos': examples,
                 'conflictos_introducidos': introduced_examples}
    return {'datos': data, 'resumen': summary, **tables}
