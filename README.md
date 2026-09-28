# TelecomAI — AI-Powered Telecom Network Intelligence

This is a VS Code-ready MVP based on `Telecom_AI_Project_Proposal.docx`.

## Features
- Synthetic telecom KPI data generation
- Isolation Forest network anomaly detection
- Random Forest equipment failure prediction
- Customer complaint classification
- TF-IDF RAG knowledge retrieval
- Optional Gemini explanations
- LangGraph troubleshooting workflow
- Streamlit command-center dashboard
- FastAPI backend
- SQLite incident storage

## Run in VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

API (second terminal):
```powershell
uvicorn api:app --reload
```

Optional Gemini:
1. Copy `.env.example` to `.env`
2. Add `GOOGLE_API_KEY=...`
3. Keep the model in `GEMINI_MODEL` as configured or change it.

The app works without Gemini by using a local fallback explanation.

## Architecture
Data -> Processing -> ML anomaly/failure models -> RAG -> LangGraph agent -> recommendations -> dashboard

Human approval is intentionally required before operational action. This demo never changes a real telecom network.


## Voice Assistant

The upgraded dashboard includes a **browser-based voice assistant**:
- 🎙️ speech-to-text using the browser Speech Recognition API
- 🤖 telecom complaint classification
- 📚 RAG knowledge retrieval
- 🔊 text-to-speech response using Speech Synthesis
- works without an external speech API in the demo

Use Chrome or Edge for best browser support.

## Recommended demo

1. Open **Executive Dashboard**
2. Show anomaly and equipment-risk KPIs
3. Open **Network Intelligence**
4. Show **Predictive Maintenance**
5. Use **Customer AI** with a spoken complaint
6. Ask the **RAG Knowledge Assistant** a troubleshooting question
7. Run the **LangGraph Agent**
8. Finish with **Voice Assistant** and demonstrate spoken response

## Testing

```powershell
pytest
```

## Enterprise upgrade

See `ENTERPRISE_UPGRADE.md`.

The upgraded version adds:
- microphone voice input
- OpenAI transcription + text-to-speech
- role-based access control
- audit logging
- SQLAlchemy persistence with optional PostgreSQL
- FastAPI voice endpoints
- Docker deployment
- Incident Center

Run tests:

```powershell
pytest
```

Run with Docker:

```powershell
docker compose up --build
```

## Customer + Employee Voice Contact Center

The voice assistant now has two modes:

- **Customer mode:** listens to a customer complaint and gives an empathetic, safe spoken response without exposing internal network details.
- **Employee mode:** listens to a support/NOC request and returns a spoken operational assessment with diagnosis, RAG evidence and recommended action.

See `VOICE_CONTACT_CENTER.md`.
