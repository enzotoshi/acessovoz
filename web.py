"""Frontend local 3D com classificação neural de WAVs."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import tempfile,threading,unicodedata
from fastapi import FastAPI,UploadFile,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from config import BASE,COMANDOS
from prever import Reconhecedor
app=FastAPI();modelo=None;lock=threading.Lock()
def obter():
 global modelo
 if modelo is None:modelo=Reconhecedor()
 return modelo
class Texto(BaseModel):
 texto:str=Field(max_length=200)
def normalizar(s):return ' '.join(''.join(c for c in unicodedata.normalize('NFD',s.lower()) if unicodedata.category(c)!='Mn' and (c.isalnum() or c.isspace())).split())
@app.get('/api/status')
def status():
 with lock:r=obter()
 return {'pronto':True,'classes':r.classes}
@app.post('/api/texto')
def texto(d:Texto):
 c=next((k for k,v in COMANDOS.items() if k!='desconhecido' and normalizar(v)==normalizar(d.texto)), 'desconhecido')
 return {'classe':c,'frase':COMANDOS[c],'aceito':c!='desconhecido','escore':None,'origem':'texto: correspondência do comando','pontuacoes':{}}
@app.post('/api/audio')
async def audio(arquivo:UploadFile):
 dados=await arquivo.read(2_000_001)
 if len(dados)>2_000_000:raise HTTPException(413,'Áudio muito grande. Use um WAV curto.')
 if dados[:4]!=b'RIFF' or dados[8:12]!=b'WAVE':raise HTTPException(400,'Envie um arquivo WAV de até 3 segundos.')
 try:
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'audio.wav';p.write_bytes(dados)
   with lock:r=obter().prever(p)
  r['aceito']=r['aceito'] and r['classe']!='desconhecido'
  r['origem']='rede neural: MFCC + seis classes'
  return r
 except ValueError as e:raise HTTPException(400,str(e))
@app.get('/')
def inicio():return FileResponse(BASE/'frontend/index.html')
app.mount('/static',StaticFiles(directory=BASE/'frontend'),name='static')
if __name__=='__main__':
 import uvicorn
 print('Abra http://localhost:8000 no navegador.')
 uvicorn.run(app,host='127.0.0.1',port=8000)
