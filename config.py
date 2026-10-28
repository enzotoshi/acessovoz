"""Configuração comum à coleta, treinamento e demonstração."""
from pathlib import Path
BASE = Path(__file__).resolve().parent
DATASET = BASE / 'dataset'
RESULTADOS = BASE / 'resultados'
COMANDOS = {
    'ouvir_paragrafo': 'ouvir parágrafo',
    'proximo_paragrafo': 'próximo parágrafo',
    'paragrafo_anterior': 'parágrafo anterior',
    'ativar_contraste': 'ativar contraste',
    'abrir_ajuda': 'abrir ajuda',
    'proxima_pagina': 'próxima página',
    'pagina_anterior': 'página anterior',
    'desconhecido': 'comando desconhecido',
}
SR = 16000
SEGUNDOS = 3
N_MFCC = 20
SEED = 42
META_POR_CLASSE = 20
