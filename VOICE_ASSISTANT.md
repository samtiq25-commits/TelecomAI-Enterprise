# Voice Assistant

TelecomAI includes a browser-based voice assistant.

## How it works

1. The user clicks **🎙️ Speak**.
2. Chrome/Edge speech recognition converts speech to text.
3. The text is passed to the Streamlit application.
4. TelecomAI classifies the complaint and retrieves relevant telecom SOP content.
5. The response can be read aloud using the browser Speech Synthesis API.

## Why browser voice?

This keeps the MVP simple and avoids requiring a separate paid speech-to-text service.

## Browser support

Use a recent Chrome or Edge browser. If speech recognition is unavailable, the app will display a message and the text fields can still be used normally.

## Production upgrade path

For a production telecom deployment, replace browser-only recognition with an authenticated speech service, add speaker/session authentication, redact PII, log consent, and enforce role-based access controls.

Never let the voice assistant directly execute network configuration changes. Keep the flow:

Voice → Intent → Retrieval/Agent → Recommendation → Human approval → Authorized system.
