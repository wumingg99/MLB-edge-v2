import os

from dotenv import load_dotenv


load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLV_TELEGRAM_TOKEN = os.getenv("CLV_TELEGRAM_TOKEN")
CLV_TELEGRAM_CHAT_ID = os.getenv("CLV_TELEGRAM_CHAT_ID", "-1004395816920")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")
SHEETS_URL = os.getenv("SHEETS_URL")
SHEETS_SECRET = os.getenv("SHEETS_SECRET")
SHEET_NAME = os.getenv("SHEET_NAME", "predictions_v3")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# V3 selection thresholds. Confidence alone never triggers a bet.
MIN_EXPECTED_VALUE = float(os.getenv("MIN_EXPECTED_VALUE", "0.025"))
MIN_PROBABILITY_EDGE = float(os.getenv("MIN_PROBABILITY_EDGE", "0.02"))
MIN_MODEL_AGREEMENT = float(os.getenv("MIN_MODEL_AGREEMENT", "0.60"))
MAX_ENSEMBLE_STD_TOTAL = float(
    os.getenv("MAX_ENSEMBLE_STD_TOTAL", "1.25")
)
MAX_ENSEMBLE_STD_MARGIN = float(
    os.getenv("MAX_ENSEMBLE_STD_MARGIN", "1.50")
)
MIN_DATA_QUALITY = float(os.getenv("MIN_DATA_QUALITY", "0.65"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_BET_FRACTION = float(os.getenv("MAX_BET_FRACTION", "0.02"))

# Backward-compatible display aliases. Bet selection does not use these.
MIN_CONFIDENCE = 0.0
FLAG_ONLY_CONFIDENCE = 0.0
BET_CONFIDENCE = 0.0
FULL_BET_CONFIDENCE = 0.0
RL_MIN_CONFIDENCE = 0.0
MIN_MODELS_AGREE = 0
EDGE_THRESHOLD = 0.0
MC_STD_MAX = 999.0
MC_FORMULA_MAX_DIFF = 999.0
