import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.data_generator import load_data
from src.ml_models import detect_anomalies, predict_failures
from src.complaint_analyzer import classify_complaint
from src.rag import retrieve
from src.agent import build_agent
from src.voice import transcribe_audio, text_to_speech
from src.voice_assistant import process_voice_request
from src.database import save_incident, audit

app=FastAPI(title="TelecomAI Enterprise API",version="2.0")
network,equipment,_=load_data()

class Complaint(BaseModel):
    complaint:str

class Issue(BaseModel):
    issue:str
    tower_id:str="TWR-1024"

@app.get("/")
def root():
    return {"name":"TelecomAI Enterprise","version":"2.0","status":"running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "voice_transcription": "local"
    }

@app.get("/network/anomalies")
def network_anomalies():
    return detect_anomalies(network).query("is_anomaly==True").tail(100).to_dict("records")

@app.get("/equipment/risks")
def equipment_risks():
    return predict_failures(equipment).head(100).to_dict("records")

@app.post("/complaints/analyze")
def complaint(x:Complaint):
    return classify_complaint(x.complaint)

@app.get("/rag/search")
def rag(q:str):
    return retrieve(q)

@app.post("/agent/troubleshoot")
def troubleshoot(x:Issue):
    a=detect_anomalies(network)
    r=a[a.tower_id==x.tower_id].sort_values("timestamp").tail(1)
    an=r.iloc[0].to_dict() if not r.empty else {}
    result=build_agent().invoke({"issue":x.issue,"tower_id":x.tower_id,"anomaly":an})
    audit("api-user","api","agent_troubleshoot",x.issue[:500])
    return result
@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    data = await file.read()

    text = transcribe_audio(
        data,
        file.filename or "voice.webm"
    )

    return {"text": text}
@app.post("/voice/respond")
async def voice_respond(x: Issue):

    a = detect_anomalies(network)

    r = (
        a[a.tower_id == x.tower_id]
        .sort_values("timestamp")
        .tail(1)
    )

    an = r.iloc[0].to_dict() if not r.empty else {}

    result, response = process_voice_request(
        x.issue,
        audience="employee",
        tower_id=x.tower_id,
        anomaly=an
    )

    audio = text_to_speech(response)

    return {
        "response": response,
        "audio_bytes": len(audio)
    }