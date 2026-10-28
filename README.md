# AcessoVoz — livro 3D acessível

Abra **https://enzotoshi.github.io/acessovoz/**. Funciona no navegador, sem instalar Python ou baixar arquivos. Microfone requer permissão no navegador.

Enzo Toshiyuki Costa e Guilherme Sousa dos Santos.

## Comandos

| Comando | Alternativas escritas | Ação |
|---|---|---|
| ouvir parágrafo | ouvir, ler, ler texto | Narrar o trecho selecionado |
| próximo parágrafo | avançar parágrafo | Destacar o próximo parágrafo |
| parágrafo anterior | voltar parágrafo | Destacar o parágrafo anterior |
| próxima página | próximo, avançar | Virar a página |
| página anterior | voltar, anterior | Voltar uma página |
| ativar contraste | contraste, alto contraste | Alternar contraste |
| abrir ajuda | ajuda, comandos | Mostrar ajuda |

As ações de página e parágrafo são independentes. A entrada escrita aceita alternativas definidas, sem inventar pontuação neural. A voz usa MFCCs e os pesos da rede neural treinada de oito classes: sete ações e desconhecido. Formas curtas de voz também foram incluídas no treinamento. Sons desconhecidos ou pontuação abaixo de 70% não executam ações. A rejeição não é universal e a pontuação softmax não é garantia de acerto.

## Livro e narradora

O leitor contém somente um trecho das páginas 7 a 9 do PDF de **Harry Potter e a Ordem da Fênix**, fornecido pelo usuário, com autoria de **J. K. Rowling**. Não contém o livro completo. O trecho tem narração neural **pt-BR-FranciscaNeural** pelo Edge TTS, em 15 arquivos MP3 correspondentes aos trechos de leitura. O comando ouvir inicia a narração; o player permite pausar e retomar. A opção continuar leitura passa aos próximos trechos. A gravação de comando interrompe a narração.

O usuário pode carregar outro TXT; textos personalizados usam a voz disponível no navegador, pois não há servidor online de síntese.

## Modelo e avaliação

220 WAVs sintéticos. Divisão por perfil: 136 treino, 42 validação e 42 teste. Resultado desta revisão: **41/42 acertos (97,62%)**. Desconhecido: 3/4, com um erro para ouvir parágrafo. As vozes sintéticas são compartilhadas entre os conjuntos. Isso não mede reconhecimento de pessoas reais. As formas curtas e as frases completas foram incluídas; a rede ainda pode confundir comandos humanos.

`docs/modelo.json` contém os mesmos pesos do modelo Keras em resultados/. MFCCs e inferência são executados em JavaScript. Testados contra Python nos WAVs: diferença máxima de características abaixo de 0,0002. O reamostrador do navegador pode produzir pequenas diferenças em gravações novas.

## Publicação

A raiz redireciona para docs/. O Pages pode publicar main /root ou main /docs. Three.js está em docs/vendor/, com licença MIT.

## Servidor Python opcional

A versão principal não precisa de servidor. Para servir a cópia local: instale requirements_frontend.txt e execute python web.py, depois abra localhost:8000. O modelo treinado está incluído. Áudios brutos de treino não estão no repositório; estão no ZIP completo de entrega.
