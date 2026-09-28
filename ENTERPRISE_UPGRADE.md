# Enterprise Upgrade

The project now includes a more realistic enterprise architecture.

## Added

### 1. Voice AI
- Microphone input in Streamlit
- OpenAI audio transcription
- Telecom-specific transcription prompt
- AI troubleshooting pipeline
- OpenAI text-to-speech response
- FastAPI voice endpoints

The current OpenAI Python SDK exposes audio transcription and speech methods through `client.audio.transcriptions.create(...)` and `client.audio.speech.create(...)`. citeturn0search0turn0search2

### 2. RBAC
Roles:
- `admin`
- `noc_engineer`
- `support`
- `viewer`

Access is enforced at the Streamlit workspace level.

### 3. Auditability
Incident records and audit events are stored through SQLAlchemy.

SQLite is the default for local development. Set `DATABASE_URL` to use PostgreSQL in a deployment.

### 4. Deployment
A Dockerfile and docker-compose configuration are included.

## Voice setup

Create `.env`:

```env
OPENAI_API_KEY=your_key
OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=cedar
TELECOM_ADMIN_PASSWORD=your_demo_password
```

Run:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Sign in with the configured demo password.

## Production architecture

```text
Browser / Mobile
      |
      v
Authentication + RBAC
      |
      +----------------------+
      |                      |
      v                      v
Voice Gateway           Web Dashboard
      |                      |
      v                      v
Speech-to-Text         Telecom APIs
      |
      v
Intent / LangGraph
      |
      +--------+---------+
      |        |         |
      v        v         v
    RAG       ML       KPI/Events
      |        |         |
      +--------+---------+
               |
               v
        Recommendation
               |
               v
        Human Approval
               |
               v
      Authorized OSS/BSS
```

## Important safety boundary

The AI generates diagnosis and recommendations. It does not directly modify network configuration. A real deployment should put an authenticated approval service between AI recommendations and OSS/BSS/network-control systems.
