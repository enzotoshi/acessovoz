"""Treino supervisionado, avaliação em teste separado e exportação de evidências."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import DATASET, RESULTADOS, COMANDOS, META_POR_CLASSE, SEED, SR, N_MFCC
from audio import caracteristicas

def csv_salvar(caminho, linhas):
    if not linhas: return
    with open(caminho, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)

def carregar_dados(pasta=DATASET):
    X, y, arquivos, contagem, vistos = [], [], [], [], {}
    for rotulo, classe in enumerate(COMANDOS):
        caminhos = sorted((Path(pasta) / classe).glob('*.wav'))
        if len(caminhos) < META_POR_CLASSE:
            raise ValueError(f'{classe}: {len(caminhos)} áudios. Grave ao menos {META_POR_CLASSE}.')
        contagem.append({'classe': classe, 'quantidade': len(caminhos)})
        for p in caminhos:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in vistos:
                raise ValueError(f'Arquivo duplicado: {p} e {vistos[digest]}. Grave outra amostra.')
            vistos[digest] = str(p)
            x, _ = caracteristicas(p)
            X.append(x); y.append(rotulo); arquivos.append(str(p.relative_to(pasta)))
    return np.asarray(X), np.asarray(y), arquivos, contagem

def dividir(y, arquivos=None, pasta=DATASET):
    manifesto = Path(pasta) / 'manifesto.csv'
    if manifesto.exists():
        if arquivos is None: raise ValueError('Manifesto exige a lista dos arquivos.')
        with manifesto.open(encoding='utf-8-sig', newline='') as f:
            registros = list(csv.DictReader(f))
        mapa = {r['arquivo']: r for r in registros}
        if len(mapa) != len(registros) or set(mapa) != set(arquivos):
            raise ValueError('Manifesto deve corresponder exatamente aos WAVs do dataset.')
        grupos = {nome: np.array([i for i,a in enumerate(arquivos)
                  if mapa[a]['conjunto'] == nome], dtype=int)
                  for nome in ('treino', 'validacao', 'teste')}
        if sum(map(len,grupos.values())) != len(y): raise ValueError('Conjunto inválido no manifesto.')
        for nome, idxs in grupos.items():
            if set(y[idxs]) != set(range(len(COMANDOS))):
                raise ValueError('Faltam classes em '+nome)
        vozes = {nome:{mapa[arquivos[i]]['perfil'] for i in ids}
                 for nome,ids in grupos.items()}
        if (vozes['treino'] & vozes['teste'] or vozes['treino'] & vozes['validacao']
                or vozes['validacao'] & vozes['teste']):
            raise ValueError('Perfis de síntese compartilhados entre conjuntos.')
        return grupos['treino'], grupos['validacao'], grupos['teste']
    idx = np.arange(len(y))
    tr, resto = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    va, te = train_test_split(resto, test_size=0.5, stratify=y[resto], random_state=SEED)
    return tr, va, te

def treinar(pasta=DATASET, saida=RESULTADOS, epocas=100):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    saida = Path(saida); saida.mkdir(parents=True, exist_ok=True)
    X, y, arquivos, contagem = carregar_dados(pasta)
    tr, va, te = dividir(y, arquivos, pasta)
    # A normalização aprende SOMENTE no treino. Validação e teste não entram no adapt.
    normalizar = tf.keras.layers.Normalization(axis=-1)
    normalizar.adapt(X[tr])
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)), normalizar,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(COMANDOS), activation='softmax'),
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = modelo.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epocas,
                     batch_size=16, verbose=2,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=12, restore_best_weights=True)])
    loss, acc = modelo.evaluate(X[te], y[te], verbose=0)
    probs = modelo.predict(X[te], verbose=0); pred = probs.argmax(axis=1)
    classes = list(COMANDOS)
    relatorio = classification_report(y[te], pred, labels=range(len(classes)),
                                      target_names=classes, zero_division=0)
    modelo.save(saida / 'modelo.keras')
    config = {'classes': classes, 'frases': COMANDOS, 'sr': SR, 'n_mfcc': N_MFCC,
              'features': 'media_desvio_3_blocos', 'seed': SEED,
              'python': platform.python_version(), 'tensorflow': tf.__version__,
              'tipo_audio': 'fala sintetizada' if (Path(pasta)/'manifesto.csv').exists() else 'áudios fornecidos',
              'divisao': 'perfis de sintese do manifesto' if (Path(pasta)/'manifesto.csv').exists() else 'aleatoria estratificada'}
    (saida / 'config_modelo.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    metricas = {'acuracia_teste': float(acc), 'perda_teste': float(loss),
                'treino': len(tr), 'validacao': len(va), 'teste': len(te),
                'epocas_executadas': len(hist.history['loss']),
                'origem_avaliacao': 'fala sintetizada' if (Path(pasta)/'manifesto.csv').exists() else 'áudios fornecidos'}
    (saida / 'metricas.json').write_text(json.dumps(metricas, indent=2), encoding='utf-8')
    (saida / 'relatorio.txt').write_text(f'Origem: {metricas["origem_avaliacao"]}\nAcurácia de teste: {acc:.2%}\nPerda: {loss:.4f}\n\n'+relatorio,
                                       encoding='utf-8')
    csv_salvar(saida / 'contagem_dataset.csv', contagem)
    csv_salvar(saida / 'divisao_dados.csv', [
        {'arquivo': arquivos[i], 'classe': classes[y[i]], 'conjunto': nome}
        for nome, indices in [('treino', tr), ('validacao', va), ('teste', te)] for i in indices])
    csv_salvar(saida / 'previsoes_teste.csv', [
        {'arquivo': arquivos[i], 'real': classes[y[i]], 'previsto': classes[pred[j]],
         'escore_softmax': float(probs[j].max()), 'acertou': bool(y[i] == pred[j])}
        for j, i in enumerate(te)])
    csv_salvar(saida / 'historico.csv', [dict(epoca=i+1, **{k: float(v[i]) for k,v in hist.history.items()})
                                            for i in range(len(hist.history['loss']))])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, chave, titulo in zip(axs, ['accuracy', 'loss'], ['Acurácia', 'Perda']):
        ax.plot(hist.history[chave], label='Treino')
        ax.plot(hist.history['val_'+chave], label='Validação')
        ax.set(xlabel='Época (índice a partir de zero)', title=titulo); ax.legend()
    fig.tight_layout(); fig.savefig(saida / 'treinamento.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(confusion_matrix(y[te], pred, labels=range(len(classes))),
                           display_labels=list(COMANDOS.values())).plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title('Matriz de confusão — teste de fala sintetizada' if (Path(pasta)/'manifesto.csv').exists() else 'Matriz de confusão — teste')
    fig.tight_layout(); fig.savefig(saida / 'matriz_confusao.png', dpi=160); plt.close(fig)
    _, mfcc = caracteristicas(Path(pasta) / arquivos[tr[0]])
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mfcc, origin='lower', aspect='auto', extent=[0, 3, 0, N_MFCC])
    ax.set(xlabel='Tempo aproximado (s)', ylabel='Coeficiente MFCC', title=arquivos[tr[0]])
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(saida / 'mfcc.png', dpi=160); plt.close(fig)
    print((saida / 'relatorio.txt').read_text(encoding='utf-8'))
    print('Resultados salvos em:', saida)
    return metricas

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, default=DATASET)
    ap.add_argument('--saida', type=Path, default=RESULTADOS)
    ap.add_argument('--epocas', type=int, default=100)
    a = ap.parse_args()
    if a.epocas < 1: ap.error('epocas deve ser positivo')
    treinar(a.dataset, a.saida, a.epocas)
