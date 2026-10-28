# AcessoVoz — livro 3D

Frontend Three.js e servidor Python para controlar um livro com voz ou texto. Projeto de Enzo Toshiyuki Costa e Guilherme Sousa dos Santos.

## Abrir pelo GitHub Pages

A versão `docs/` funciona inteiramente no navegador, inclusive microfone, MFCCs e inferência neural com os pesos treinados. Não precisa instalar Python nem baixar o repositório.

Para ativar uma vez: Settings → Pages → Deploy from a branch → main → /docs → Save. Depois abra https://enzotoshi.github.io/acessovoz/.

A gravação é processada no navegador; não é enviada a um servidor. A versão JavaScript preserva os pesos e a extração de características do modelo Python. O reamostrador de áudio do navegador pode produzir pequenas diferenças. As pontuações não garantem acerto e a classe desconhecido não rejeita todos os sons possíveis.

## Executar localmente (alternativa)

Use Python 3.11 ou 3.12. Abra o terminal na pasta do repositório:

```bash
python -m venv .venv
```

Ative o ambiente: Windows: `.venv\Scripts\activate`. macOS/Linux: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements_frontend.txt
python web.py
```

Abra http://localhost:8000. Mantenha o terminal aberto. O modelo treinado está incluído. Não abra frontend/index.html diretamente.

## Comandos

- ouvir parágrafo: lê a página pela voz do navegador.
- próximo parágrafo e parágrafo anterior: viram as páginas.
- ativar contraste: alterna o contraste.
- abrir ajuda: apresenta os comandos.

Arraste o livro para mudar o ângulo. Digite um comando, grave três segundos pelo microfone ou envie um WAV de até três segundos. Permita o microfone no navegador quando solicitado.

Texto usa correspondência normalizada dos cinco comandos, sem porcentagem neural. Áudio usa Librosa/MFCCs e rede neural TensorFlow de seis classes. Desconhecido ou pontuação abaixo de 70% não executa ações. As pontuações softmax não são garantia de acerto.

## Avaliação

Modelo treinado com 120 WAVs sintetizados: cinco comandos com 20 exemplos cada e 20 frases de desconhecido. Divisão: 72 treino, 24 validação e 24 teste. Resultado: 23/24 acertos (95,83%); desconhecido: 3/4. Um desconhecido foi classificado como abrir ajuda. Vozes sintéticas compartilhadas entre os conjuntos: o resultado não mede fala humana. Ensaie o microfone no computador da apresentação.

As evidências estão em resultados/. O repositório contém os arquivos necessários para executar o site, incluindo o modelo. Áudios brutos de treinamento e slides não são necessários para a execução e não estão neste repositório. Para gerar novamente os dados, instale também `edge-tts` e execute `python gerar_dataset_natural.py --saida dados_gerados`; depois use `python treinar.py --dataset dados_gerados/dataset`. A geração precisa de internet e pode produzir resultados de treinamento diferentes.

Three.js e sua licença MIT estão em frontend/vendor/. O servidor inicia apenas em localhost. Este repositório não configura hospedagem pública.
