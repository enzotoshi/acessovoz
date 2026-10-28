"""Gera uma nova cópia do dataset com voz neural natural."""
import argparse,asyncio
from pathlib import Path
from config import BASE
from gerar_dataset_natural import gerar
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--saida',type=Path,default=BASE/'nova_geracao_neural')
    asyncio.run(gerar(ap.parse_args().saida))
