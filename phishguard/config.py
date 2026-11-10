"""Project paths and shared settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DB_PATH = ROOT / "phishguard.db"

URL_MODEL_PATH = MODELS_DIR / "url_model.joblib"
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Verdict thresholds on the phishing probability (0..1)
PHISHING_THRESHOLD = 0.70
SUSPICIOUS_THRESHOLD = 0.40

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
