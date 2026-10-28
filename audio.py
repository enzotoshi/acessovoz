"""Mesma extração de características no treino e na previsão."""
import numpy as np
import librosa
from config import SR, SEGUNDOS, N_MFCC

def caracteristicas(caminho):
    # Limite de 10 s para detectar, em vez de truncar silenciosamente, frases longas.
    y, _ = librosa.load(str(caminho), sr=SR, mono=True, duration=10)
    if y.size < int(0.25 * SR) or not np.isfinite(y).all():
        raise ValueError(f'Áudio vazio, muito curto ou inválido: {caminho}')
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 0.001:
        raise ValueError(f'Silêncio ou volume muito baixo: {caminho}')
    # Mesmo ganho de pico utilizado na geração dos áudios de treino.
    y = y * (0.85 / max(float(np.max(np.abs(y))), 1e-8))
    # Remove silêncio das bordas. Não remove pausas internas.
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) > SEGUNDOS * SR:
        raise ValueError(f'Fala maior que {SEGUNDOS} s. Regrave: {caminho}')
    y = librosa.util.fix_length(y, size=SEGUNDOS * SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                               n_fft=512, hop_length=160, n_mels=40)
    # Resumo de 20 coeficientes em três partes temporais, preservando alguma ordem.
    partes = np.array_split(mfcc, 3, axis=1)
    vetor = np.concatenate([np.concatenate([p.mean(axis=1), p.std(axis=1)])
                             for p in partes]).astype('float32')
    return vetor, mfcc
