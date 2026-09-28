
import os
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()


# ----------------------------------------
# LOCAL SPEECH RECOGNITION
# ----------------------------------------

@st.cache_resource
def get_local_whisper_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio_local(audio_file):
    """
    Transcribe audio locally without OpenAI API credits.
    Supports multilingual speech, including Urdu and English.
    """

    model = get_local_whisper_model()

    audio_file.seek(0)
    audio_bytes = audio_file.read()

    # Preserve the actual audio format when possible.
    filename = getattr(audio_file, "name", "voice.webm")
    suffix = Path(filename).suffix or ".webm"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            language=None,
            initial_prompt=(
                "Telecom network support conversation. "
                "Telecom terms include internet, tower, "
                "signal, latency, packet loss, 4G, 5G, "
                "dropped calls, bandwidth and throughput. "
                "The speaker may use English or Urdu."
            ),
        )
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return transcript.strip()

    finally:
        Path(temp_path).unlink(missing_ok=True)


def transcribe_audio(
    audio_bytes: bytes,
    filename="voice.webm"
):
    """
    Existing interface preserved.
    Uses local Faster-Whisper instead of OpenAI.
    """

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    return transcribe_audio_local(audio_file)


# ----------------------------------------
# LOCAL TEXT-TO-SPEECH
# ----------------------------------------

def text_to_speech(text: str) -> bytes:
    """
    Generate spoken audio locally using Windows
    speech synthesis. Returns WAV audio bytes.
    """

    import pyttsx3

    if not text or not text.strip():
        return b""

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_audio:
            temp_path = temp_audio.name

        engine = pyttsx3.init()
        engine.setProperty("rate", 165)

        engine.save_to_file(
            text[:4096],
            temp_path
        )

        engine.runAndWait()
        engine.stop()

        with open(temp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        return audio_bytes

    finally:
        if temp_path and os.path.exists(temp_path):
            Path(temp_path).unlink(missing_ok=True)