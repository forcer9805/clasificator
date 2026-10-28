"""Piloto reproducible de MiniLM y DistilBERT; sin exportar datos ni modelos."""
import gc
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from baseline import RANDOM_STATE, TEXT, normalizar_texto, crear_pipeline
from comparacion import registrar_predicciones

MINILM = 'MiniLM + regresión logística · α=0,25'
DISTIL = 'DistilBERT ajustado · α=0,25'
MINILM_ID = 'sentence-transformers/all-MiniLM-L6-v2'
DISTIL_ID = 'distilbert/distilbert-base-uncased'
REVISIONS = {
    MINILM_ID: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    DISTIL_ID: '12040accade4e8a0f71eabdb258fecc2e7e948be',
}


def seleccionar_piloto(data, parts, n_train=11000, n_validation=5500):
    """Muestreo estratificado independiente dentro de cada partición existente."""
    selected = {}
    for name, size in [('Entrenamiento',n_train),('Validación',n_validation)]:
        pool = parts[name]
        if size is None or size >= len(pool):
            selected[name] = np.array(pool,copy=True)
        else:
            selected[name], _ = train_test_split(pool,train_size=size,random_state=RANDOM_STATE,
                                                stratify=data.iloc[pool].Product)
        assert set(selected[name]).issubset(set(pool))
        assert data.iloc[selected[name]].Product.nunique() == 11
    assert set(selected['Entrenamiento']).isdisjoint(selected['Validación'])
    assert set(np.concatenate(list(selected.values()))).isdisjoint(parts['Prueba'])
    return selected


def _inicializar():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def tokenizar_narrativas(texts, tokenizer, max_tokens=128):
    """Secuencias completas si caben; en otro caso inicio [SEP] final.

    El recorte es explícito y se audita. No elimina observaciones de la muestra.
    """
    if max_tokens < 8 or max_tokens > 512:
        raise ValueError('max_tokens debe estar entre 8 y 512.')
    tokens, lengths, clipped = [], [], 0
    values = [normalizar_texto(t) for t in texts]
    for start in range(0,len(values),128):
        encoded = tokenizer(values[start:start+128],add_special_tokens=False,truncation=False,
                            return_attention_mask=False,return_token_type_ids=False,verbose=False)
        for ids in encoded['input_ids']:
            lengths.append(len(ids))
            if len(ids)+tokenizer.num_special_tokens_to_add(pair=False) <= max_tokens:
                sequence = tokenizer.build_inputs_with_special_tokens(ids)
            else:
                clipped += 1
                capacity = max_tokens-tokenizer.num_special_tokens_to_add(pair=True)
                first = capacity//2
                sequence = tokenizer.build_inputs_with_special_tokens(ids[:first],ids[-(capacity-first):])
            assert len(sequence) <= max_tokens
            tokens.append({'input_ids':sequence,'attention_mask':[1]*len(sequence)})
    stats = {'Filas':len(tokens),'Con recorte':clipped,'% con recorte':100*clipped/len(tokens),
             'Tokens originales medios':float(np.mean(lengths)),
             'Tokens originales p95':float(np.percentile(lengths,95)), 'Máximo de entrada':max_tokens}
    return tokens, stats


def _preparar_textos(context, model_id, max_tokens):
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=REVISIONS[model_id])
    x_train, train_stats = tokenizar_narrativas(context['train'][TEXT],tokenizer,max_tokens)
    x_val, val_stats = tokenizar_narrativas(context['validation'][TEXT],tokenizer,max_tokens)
    stats = pd.DataFrame([train_stats,val_stats],index=['Entrenamiento','Validación'])
    return tokenizer, x_train, x_val, stats, perf_counter()-start


def _encode(model, examples, tokenizer, device, batch_size):
    loader = DataLoader(examples,batch_size=batch_size,shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer,return_tensors='pt'))
    encoded = []
    model.eval()
    with torch.inference_mode():
        for step,batch in enumerate(loader,1):
            batch = {k:v.to(device) for k,v in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).expand_as(hidden).float()
            mean = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            encoded.append(F.normalize(mean,p=2,dim=1).cpu().numpy())
            if step % 100 == 0:
                print(f'MiniLM: {step}/{len(loader)} lotes codificados',flush=True)
    return np.concatenate(encoded).astype(np.float32)


def entrenar_minilm(context, max_tokens=128, batch_size=32):
    device = _inicializar()
    tokenizer, texts_train, texts_val, stats, token_seconds = _preparar_textos(context,MINILM_ID,max_tokens)
    start = perf_counter()
    encoder = AutoModel.from_pretrained(MINILM_ID,revision=REVISIONS[MINILM_ID]).to(device)
    encoder.requires_grad_(False)
    loading_seconds = perf_counter()-start
    start = perf_counter()
    print('MiniLM: codificando entrenamiento',flush=True)
    x_train = _encode(encoder,texts_train,tokenizer,device,batch_size)
    encode_train_seconds = perf_counter()-start
    start = perf_counter()
    print('MiniLM: codificando validación',flush=True)
    x_val = _encode(encoder,texts_val,tokenizer,device,batch_size)
    encode_val_seconds = perf_counter()-start
    classifier = crear_pipeline(class_weight=context['pesos'].to_dict()).named_steps['classifier']
    start = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
        warnings.simplefilter('always')
        classifier.fit(x_train,context['train'].Product)
    seconds = perf_counter()-start
    start = perf_counter()
    with threadpool_limits(limits=4):
        pred = classifier.predict(x_val)
    prediction_seconds = perf_counter()-start
    summary = registrar_predicciones(context,MINILM,pred,classifier,seconds,prediction_seconds,
                                     [str(w.message) for w in captured])
    result = context['resultados'][MINILM]
    result['recorte'] = stats
    result['configuracion'].update({'Modelo base':MINILM_ID,'Revisión':encoder.config._commit_hash,
        'Tokens máximos':max_tokens,'Dispositivo':str(device),'Encoder ajustado':False})
    result['costes_neuronales'] = {'Tokenización (s)':token_seconds,'Carga modelo (s)':loading_seconds,
        'Codificación entrenamiento (s)':encode_train_seconds,'Codificación validación (s)':encode_val_seconds,
        'Ajuste clasificador (s)':seconds,'Predicción clasificador (s)':prediction_seconds}
    del encoder,x_train,x_val
    gc.collect()
    return summary


def entrenar_distilbert(context, max_tokens=128, batch_size=16, epochs=1, learning_rate=2e-5):
    device = _inicializar()
    tokenizer, texts_train, texts_val, stats, token_seconds = _preparar_textos(context,DISTIL_ID,max_tokens)
    labels = context['labels']
    lookup = {str(label):i for i,label in enumerate(labels)}
    train_items = [{**item,'labels':lookup[str(label)]} for item,label in zip(texts_train,context['train'].Product)]
    collator = DataCollatorWithPadding(tokenizer,return_tensors='pt')
    train_loader = DataLoader(train_items,batch_size=batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),collate_fn=collator)
    val_loader = DataLoader(texts_val,batch_size=batch_size,shuffle=False,collate_fn=collator)
    start = perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(DISTIL_ID,revision=REVISIONS[DISTIL_ID],num_labels=len(labels),
        id2label={i:str(label) for i,label in enumerate(labels)},label2id=lookup).to(device)
    loading_seconds = perf_counter()-start
    weights = torch.tensor(context['pesos'].reindex(labels).to_numpy(),dtype=torch.float32,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.01)
    total_steps = len(train_loader)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(.1*total_steps),
                                               num_training_steps=total_steps)
    start = perf_counter()
    history = []
    model.train()
    for epoch in range(epochs):
        loss_total, examples = 0., 0
        for step,batch in enumerate(train_loader,1):
            batch = {k:v.to(device) for k,v in batch.items()}
            target = batch.pop('labels')
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            losses = F.cross_entropy(logits,target,reduction='none')
            loss = (losses*weights[target]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); scheduler.step()
            loss_total += loss.item()*len(target); examples += len(target)
            if step % 100 == 0:
                print(f'DistilBERT época {epoch+1}: {step}/{len(train_loader)} lotes',flush=True)
        history.append({'Época':epoch+1,'Filas':examples,'Pérdida ponderada':loss_total/examples})
    seconds = perf_counter()-start
    print('DistilBERT: entrenamiento completado; evaluando validación',flush=True)
    start = perf_counter()
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            predictions.extend(model(**batch).logits.argmax(1).cpu().tolist())
    prediction_seconds = perf_counter()-start
    pred = labels[np.array(predictions)]
    summary = registrar_predicciones(context,DISTIL,pred,model,seconds,prediction_seconds)
    result = context['resultados'][DISTIL]
    result['modelo'] = None  # Evitar retener pesos y optimizador cuando ya hay predicciones.
    result['recorte'] = stats
    result['historia'] = pd.DataFrame(history)
    result['configuracion'] = {'Modelo base':DISTIL_ID,'Revisión':model.config._commit_hash,
        'Tokens máximos':max_tokens,'Lote':batch_size,'Épocas':epochs,'Learning rate':learning_rate,
        'Dispositivo':str(device),'Encoder ajustado':True,'Peso de clase':'α=0,25, media 1',
        'Pérdida':'Media por lote de entropía cruzada multiplicada por peso de clase'}
    result['costes_neuronales'] = {'Tokenización (s)':token_seconds,'Carga modelo (s)':loading_seconds,
        'Ajuste red (s)':seconds,'Predicción validación (s)':prediction_seconds}
    del model,optimizer,scheduler
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return summary
