import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")
SHEETS_URL = os.getenv("SHEETS_URL")
SHEETS_SECRET = os.getenv("SHEETS_SECRET")
SHEET_NAME = os.getenv("SHEET_NAME", "predictions_v2")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# V2 thresholds — tighter than v1
MIN_CONFIDENCE = 58.0          # Skip below this (v1 was 54)
FLAG_ONLY_CONFIDENCE = 63.0    # Flag but don't bet
BET_CONFIDENCE = 63.0          # Bet threshold
FULL_BET_CONFIDENCE = 68.0     # Full Kelly threshold
RL_MIN_CONFIDENCE = 65.0       # Run line minimum
MIN_MODELS_AGREE = 3           # Out of 5 (v2 has LightGBM)
EDGE_THRESHOLD = 1.5           # Minimum total gap
MC_STD_MAX = 4.0               # Max MC std dev (high = skip)
MC_FORMULA_MAX_DIFF = 1.5      # Max diff between formula and MC
