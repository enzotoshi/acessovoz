"""Compatibilidade de inicializadores entre versões do Keras, sem mudar pesos."""
import json
import zipfile
from io import BytesIO
from pathlib import Path
import tempfile


def modelo_compativel(origem):
    """Devolve bytes .keras sem opções de eixos nulas de GlorotUniform."""
    origem = Path(origem)
    with zipfile.ZipFile(origem) as z:
        config = json.loads(z.read('config.json'))
        def limpar(obj):
            if isinstance(obj, dict):
                if obj.get('class_name') == 'GlorotUniform':
                    cfg = obj.get('config', {})
                    for chave in ('input_axes', 'output_axes'):
                        if chave in cfg and cfg[chave] is None:
                            del cfg[chave]
                if obj.get('class_name') == 'Dense':
                    cfg = obj.get('config', {})
                    if cfg.get('quantization_config', 'ausente') is None:
                        del cfg['quantization_config']
                for valor in obj.values():
                    limpar(valor)
            elif isinstance(obj, list):
                for valor in obj:
                    limpar(valor)
        limpar(config)
        saida = BytesIO()
        with zipfile.ZipFile(saida, 'w', zipfile.ZIP_DEFLATED) as novo:
            for item in z.infolist():
                dados = json.dumps(config).encode('utf-8') if item.filename == 'config.json' else z.read(item.filename)
                novo.writestr(item, dados)
    return saida.getvalue()


def carregar_modelo(arquivo):
    import tensorflow as tf
    # A inferência não precisa restaurar o otimizador de treinamento.
    with tempfile.TemporaryDirectory(prefix='acessovoz_keras_') as tmp:
        compativel = Path(tmp)/'modelo.keras'
        compativel.write_bytes(modelo_compativel(arquivo))
        return tf.keras.models.load_model(compativel, compile=False)
