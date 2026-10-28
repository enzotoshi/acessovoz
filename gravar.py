"""Gravação local. No Colab, use o gravador do notebook."""
import argparse
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import DATASET, COMANDOS, SR, SEGUNDOS, META_POR_CLASSE

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pessoa', required=True, help='Identificador anônimo, ex.: p01')
    ap.add_argument('--quantidade', type=int, default=10, help='Por comando e por pessoa')
    a = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', a.pessoa) or a.quantidade < 1:
        ap.error('Use identificador simples e quantidade positiva.')
    for classe, frase in COMANDOS.items():
        pasta = DATASET / classe
        pasta.mkdir(parents=True, exist_ok=True)
        existentes = len(list(pasta.glob('*.wav')))
        print(f'\n{frase.upper()} — {existentes}/{META_POR_CLASSE} gravações na pasta')
        for i in range(a.quantidade):
            while True:
                input(f'Enter para gravar "{frase}" ({i+1}/{a.quantidade})...')
                print('GRAVANDO: fale agora.')
                y = sd.rec(int(SEGUNDOS * SR), samplerate=SR, channels=1, dtype='float32')
                sd.wait()
                if np.sqrt(np.mean(y ** 2)) < 0.001:
                    print('Volume muito baixo. Tente novamente.'); continue
                sd.play(y, SR); sd.wait()
                if input('Enter para salvar; r para repetir: ').strip().lower() == 'r':
                    continue
                numero = 1
                while (pasta / f'{a.pessoa}_{numero:03d}.wav').exists(): numero += 1
                destino = pasta / f'{a.pessoa}_{numero:03d}.wav'
                sf.write(destino, y, SR, subtype='PCM_16')
                print('Salvo:', destino.name)
                break
    print('Coleta concluída. Não copie um áudio para criar novas amostras.')
if __name__ == '__main__':
    main()
