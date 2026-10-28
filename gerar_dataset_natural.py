"""Gera voz neural pt-BR com o serviço online do Edge via edge-tts."""
import argparse,asyncio,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import edge_tts
from config import BASE,COMANDOS,SR,SEGUNDOS

# Mantém a validação TLS e acrescenta uma CA configurada pelo ambiente, se houver.
if os.environ.get('SSL_CERT_FILE'):
    from edge_tts import communicate
    communicate._SSL_CTX.load_verify_locations(cafile=os.environ['SSL_CERT_FILE'])

VOZES=['pt-BR-FranciscaNeural','pt-BR-AntonioNeural']

def perfis():
    result=[]
    for voz in VOZES[:2]:
        for taxa in [-12,-8,-4,0,4,8]:
            result.append(dict(perfil=f'neural_{len(result)+1:02d}',conjunto='treino',voz=voz,taxa_percentual=taxa))
    for voz in VOZES:
        for taxa in [-10,10]:
            result.append(dict(perfil=f'neural_{len(result)+1:02d}',conjunto='validacao',voz=voz,taxa_percentual=taxa))
    for voz in VOZES[:2]:
        for taxa in [-6,6]:
            result.append(dict(perfil=f'neural_{len(result)+1:02d}',conjunto='teste',voz=voz,taxa_percentual=taxa))
    return result

def converter(mp3,wav):
    y,_=librosa.load(mp3,sr=SR,mono=True)
    if len(y)<int(.2*SR):raise ValueError('Síntese vazia ou curta demais: '+str(mp3))
    y,_=librosa.effects.trim(y,top_db=40)
    if len(y)>int((SEGUNDOS-.22)*SR):raise ValueError('A frase não cabe no limite de 3 segundos: '+str(mp3))
    pico=float(np.max(np.abs(y)))
    if pico<.001:raise ValueError('Síntese em silêncio: '+str(mp3))
    # Mesmo ganho de pico em todos os arquivos, sem alterar o tom da voz.
    y=y*(.85/pico)
    y=np.pad(y,(int(.10*SR),int(.12*SR)))
    y=librosa.util.fix_length(y,size=SR*SEGUNDOS)
    wav.parent.mkdir(parents=True,exist_ok=True)
    sf.write(wav,y,SR,subtype='PCM_16')

async def gerar(pasta,retomar=False):
    pasta=Path(pasta);jobs=[]
    frases_desconhecidas=['qual é a previsão','ligar televisão','comprar pão','hoje está quente','tocar uma música','fechar janela','quero tomar café','onde fica a escola','desligar computador','aumentar volume','enviar mensagem','quanto custa isso','vamos passear','marcar consulta','mostrar fotografia','acender a luz','pedir uma pizza','o cachorro correu','buscar meu celular','amanhã vou viajar']
    for classe,frase in COMANDOS.items():
        for i,perfil in enumerate(perfis()):
            jobs.append((classe,frases_desconhecidas[i] if classe=="desconhecido" else frase,perfil,pasta/'dataset'/classe/f'{perfil["perfil"]}.wav',
                pasta/'originais_naturais'/classe/f'{perfil["perfil"]}.mp3'))
        jobs.append((classe,frase,dict(perfil='demonstracao',conjunto='demo',voz=VOZES[0],taxa_percentual=2),
                     pasta/'exemplos_novos'/f'{classe}.wav',pasta/'previas_naturais'/f'{classe}.mp3'))
    if not retomar and any(wav.exists() for _,_,_,wav,_ in jobs):
        raise FileExistsError('A pasta de saída já contém WAVs. Escolha uma pasta vazia.')
    sem=asyncio.Semaphore(2);rows=[];done=0
    async def job(classe,frase,perfil,wav,mp3):
        nonlocal done
        async with sem:
            mp3.parent.mkdir(parents=True,exist_ok=True)
            if not (retomar and wav.exists() and mp3.exists() and mp3.stat().st_size>500):
                for tentativa in range(3):
                    try:
                        await edge_tts.Communicate(frase+'.',voice=perfil['voz'],
                            rate=f'{perfil["taxa_percentual"]:+d}%',pitch='+0Hz',connect_timeout=15,receive_timeout=30).save(str(mp3))
                        await asyncio.to_thread(converter,mp3,wav)
                        break
                    except (edge_tts.exceptions.NoAudioReceived,asyncio.TimeoutError):
                        if tentativa==2:raise
                        await asyncio.sleep(5*(tentativa+1))
            if perfil['conjunto']!='demo':
                rows.append(dict(arquivo=wav.relative_to(pasta/'dataset').as_posix(),classe=classe,frase=frase,
                    origem='Microsoft_Edge_voz_neural',**perfil,
                    variante=perfil['perfil'],taxa_hz=SR,duracao_s=SEGUNDOS,
                    sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                    mp3_origem=mp3.relative_to(pasta).as_posix()))
            await asyncio.sleep(0.5)
            done+=1
            if done%10==0 or done==len(jobs):print(f'Áudios concluídos: {done}/{len(jobs)}',flush=True)
    await asyncio.gather(*(job(*j) for j in jobs))
    rows.sort(key=lambda r:(list(COMANDOS).index(r['classe']),r['perfil']))
    manifesto=pasta/'dataset'/'manifesto.csv'
    with manifesto.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (pasta/'dataset'/'origem.json').write_text(json.dumps(dict(
        tipo='100% sintetizado com voz neural',motor='Microsoft Edge TTS via edge-tts',
        idioma='pt-BR',audios_dataset=120,comandos=5,classes=6,perfis=20,vozes=VOZES,pessoas_gravadas=0,
        divisao='72 treino, 24 validação, 24 teste por perfil de síntese',
        limite='Os três conjuntos compartilham duas vozes e motor, com perfis de velocidade separados. Não houve avaliação humana.',
        exemplos_novos='5 sínteses adicionais com Francisca e velocidade +2%, fora das métricas de teste.',
        mp3='Originais do serviço preservados em originais_naturais e previas_naturais.',
        fonte='https://github.com/rany2/edge-tts'),ensure_ascii=False,indent=2),encoding='utf-8')
    return pasta

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--saida',type=Path,required=True);ap.add_argument('--retomar',action='store_true')
    args=ap.parse_args();asyncio.run(gerar(args.saida,args.retomar))
