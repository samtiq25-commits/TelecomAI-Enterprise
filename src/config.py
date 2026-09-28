from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
KB_DIR = ROOT / "knowledge_base"

NETWORK_CSV = DATA_DIR / "network_data.csv"
EQUIPMENT_CSV = DATA_DIR / "equipment_data.csv"
CUSTOMER_CSV = DATA_DIR / "customer_complaints.csv"
NETWORK_EVENTS_CSV = DATA_DIR / "network_events.csv"

ANOMALY_MODEL = MODEL_DIR / "anomaly_model.joblib"
FAILURE_MODEL = MODEL_DIR / "failure_model.joblib"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

for p in (DATA_DIR, MODEL_DIR, KB_DIR):
    p.mkdir(exist_ok=True)