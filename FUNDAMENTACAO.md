# Fundamentação teórica — AcessoVoz

## O que é uma rede neural artificial?

Uma rede neural artificial é um modelo matemático formado por unidades chamadas neurônios artificiais, organizadas em camadas. Cada unidade combina números de entrada com pesos e um termo de ajuste chamado viés. Uma função de ativação permite aprender relações mais complexas. Durante o treinamento, os pesos são ajustados para reduzir os erros nas previsões. A inspiração no cérebro é limitada: a rede não pensa nem compreende as palavras como uma pessoa.

No AcessoVoz, a entrada é uma representação numérica do áudio. As camadas intermediárias procuram padrões que ajudam a distinguir sete ações e exemplos desconhecidos. A saída contém oito escores, um por classe. A rede aprende a classificar exemplos, e não a transcrever qualquer fala.

## Como funciona o aprendizado supervisionado?

No aprendizado supervisionado, cada exemplo recebe um rótulo correto. Um áudio de alguém dizendo “ouvir parágrafo”, por exemplo, é colocado na pasta correspondente. A rede faz uma previsão; a função de perda mede a diferença em relação ao rótulo. A retropropagação calcula como os pesos contribuem para esse erro e o otimizador os atualiza.

Uma época corresponde a uma passagem pelos dados de treino. O conjunto de validação acompanha a aprendizagem e ajuda a decidir quando parar. O conjunto de teste fica separado para medir o desempenho final em exemplos que não ajustaram os pesos. Rótulos errados ensinam associações erradas.

## O que são MFCCs e como representam sons?

MFCC significa coeficientes cepstrais de frequência de Mel. Eles resumem características da distribuição de energia do som entre frequências, em pequenos intervalos de tempo. Para calculá-los, o áudio é analisado em janelas, passa por uma representação espectral e por filtros na escala Mel, relacionados à percepção humana de frequência. Em seguida, aplicam-se logaritmo e transformada discreta do cosseno para obter os coeficientes.

O resultado é uma matriz: coeficientes nas linhas e instantes nas colunas. Não é uma lista de palavras e não contém o significado da frase. Ela representa aspectos acústicos úteis para distinguir padrões de fala. O projeto calcula 20 coeficientes com Librosa e resume sua média e desvio em três partes temporais. Esse resumo reduz a entrada para 120 números, mas perde detalhes: é uma escolha didática simples.

## Como avaliar?

Acurácia mede a proporção de comandos classificados corretamente. A matriz de confusão mostra quais comandos são confundidos: na diagonal ficam os acertos; fora dela, os erros. Precisão, recall e F1 complementam a acurácia por classe.

Compare as curvas de treino e validação. Treino melhorando enquanto validação piora pode indicar sobreajuste: o modelo aprende peculiaridades das gravações em vez de padrões que generalizem. Um conjunto pequeno, ruído, falas semelhantes, diferenças entre microfones e rótulos incorretos podem causar erros.

A qualidade da aprendizagem depende dos exemplos disponíveis. Uma rede treinada com fala sintetizada pode ter dificuldade com a fala de uma pessoa real. Testar áudios novos e discutir os erros faz parte do trabalho, mesmo quando a acurácia parece boa.

## Aplicação e possibilidades futuras

A ideia é permitir controlar um leitor sem depender apenas de cliques ou digitação. Na demonstração, comandos navegam entre parágrafos, leem um trecho, alternam contraste e abrem ajuda. No futuro, seria possível importar documentos, ampliar os dados, incluir uma classe para falas desconhecidas e testar o protótipo com usuários, respeitando suas necessidades e consentimento. Essas possibilidades não foram implementadas ou comprovadas como resultado deste trabalho.

## Referências

- TensorFlow: https://www.tensorflow.org/tutorials/audio/simple_audio
- TensorFlow/Keras: https://www.tensorflow.org/guide/keras/training_with_built_in_methods
- Librosa: https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html


## Dataset desta versão com voz neural natural

O dataset contém 220 WAVs sintetizados em português brasileiro pelo serviço Microsoft Edge TTS via edge-tts. Usamos Francisca e Antonio, com pequenas variações de velocidade e sem modificar o tom original. Cada combinação de voz e velocidade forma um perfil. São 20 perfis artificiais, e nenhuma pessoa gravou o dataset.

Cada perfil pertence a um único conjunto. Treino e teste usam Francisca e Antonio em velocidades diferentes, e a validação também compartilha essas vozes em velocidades distintas. Portanto, não há separação completa por voz: os resultados continuam limitados ao motor sintético e não demonstram desempenho humano. Os arquivos extras de demonstração usam Francisca na velocidade +2%.

O programa conserva os MP3s originais e cria WAVs de 16 kHz para usar o mesmo processamento de MFCCs. A qualidade da síntese mudou e o modelo foi retreinado com os novos WAVs. A captura humana no Colab registra tentativas separadas, sem modificar as métricas sintéticas.

Fonte: https://github.com/rany2/edge-tts

## Versão atual

Sete ações independentes e desconhecido. Comandos curtos também aparecem no treinamento. A navegação por parágrafos destaca trechos; a navegação por páginas vira o livro. O site executa os MFCCs e a inferência neural em JavaScript com os pesos treinados em TensorFlow, sem servidor externo. A narração neural Francisca usa MP3s previamente gerados do trecho fornecido.
