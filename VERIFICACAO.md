# Verificação

120 WAVs únicos conferidos pelo SHA-256 do manifesto. Modelo retreinado com seis classes. Teste sintético: 23/24 (95,83%). Desconhecido: 3/4, com um erro para abrir ajuda. Arquivos e resultados estão em resultados/.

API validada com comandos escritos conhecidos e desconhecidos, WAVs e rejeição de arquivo inválido. Navegação, contraste, ajuda e rejeição de texto conferidos em navegador com WebGL. A captura de um microfone físico deve ser ensaiada no computador da apresentação.

Entrada escrita usa correspondência normalizada de frases. A entrada de áudio usa MFCCs e rede neural. Desconhecido ou score abaixo de 0,70 não executa ações. A rejeição não é universal.
