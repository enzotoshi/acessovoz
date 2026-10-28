"""Carrega modelo salvo e classifica um áudio novo, sem transcrição externa."""
import json
import numpy as np
from config import RESULTADOS, COMANDOS
from audio import caracteristicas

class Reconhecedor:
    def __init__(self, pasta=RESULTADOS):
        import tensorflow as tf
        from pathlib import Path
        pasta = Path(pasta)
        cfg = json.loads((pasta / 'config_modelo.json').read_text(encoding='utf-8'))
        if cfg['classes'] != list(COMANDOS):
            raise ValueError('Classes incompatíveis. Use o código do mesmo treinamento.')
        self.classes = cfg['classes']
        from compatibilidade_keras import carregar_modelo
        self.modelo = carregar_modelo(pasta / 'modelo.keras')

    def prever(self, arquivo, limiar=0.70):
        x, _ = caracteristicas(arquivo)
        p = self.modelo.predict(x[np.newaxis], verbose=0)[0]
        i = int(p.argmax()); escore = float(p[i])
        return {'classe': self.classes[i], 'frase': COMANDOS[self.classes[i]],
                'escore': escore, 'aceito': escore >= limiar and self.classes[i] != 'desconhecido',
                'pontuacoes': {classe: float(valor) for classe, valor in zip(self.classes, p)}}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('arquivo')
    a = ap.parse_args()
    print(Reconhecedor().prever(a.arquivo))
