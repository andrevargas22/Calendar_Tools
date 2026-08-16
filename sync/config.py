"""
Configuration for the calendar sync application.
"""

import os
from datetime import datetime
from pathlib import Path

# Load .env file if present (local development)
_env_path = Path(__file__).resolve().parent.parent / '.env'
if _env_path.is_file():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                os.environ.setdefault(_key.strip(), _val.strip())

def _get_bool(env_name: str, default: bool = False) -> bool:
	val = os.environ.get(env_name)
	if val is None:
		return default
	return val.strip().lower() in {"1", "true", "yes", "y", "on"}

# --- Credentials & endpoints ---
TEAMS_ICS_URL = os.environ.get('TEAMS_ICS_URL')
GOOGLE_SERVICE_ACCOUNT_KEY = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
CALENDAR_ID = os.environ.get('GOOGLE_CALENDAR_ID_WORK')

# --- Timezone ---
TIMEZONE = 'America/Sao_Paulo'

# --- Sync window ---
START_HOUR = 7        # Starting hour for period
END_HOUR = 18         # Ending hour for period
DAYS_RANGE = 11       # Number of days ahead to sync
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', 30))

# --- Skip date ranges (vacation/absence periods) ---
_SKIP_RANGES_RAW = os.environ.get('SKIP_DATE_RANGES', '2026-05-25:2026-06-12')

# --- Cancel prefixes ---
CANCEL_PREFIX = os.environ.get('CANCEL_PREFIX', 'Cancelado:')
CANCEL_PREFIXES = tuple(
	p.strip() for p in os.environ.get(
		'CANCEL_PREFIXES',
		f"{CANCEL_PREFIX},Cancelado:,Canceled event:,Cancelled event:,Canceled:,Cancelled:"
	).split(',') if p.strip()
)

# --- Logging ---
LOG_MASK_TITLES = _get_bool('LOG_MASK_TITLES', True)

SKIP_DATE_RANGES = []
if _SKIP_RANGES_RAW.strip():
    for _pair in _SKIP_RANGES_RAW.split(','):
        _pair = _pair.strip()
        if ':' in _pair:
            _start_str, _end_str = _pair.split(':', 1)
            try:
                _s = datetime.strptime(_start_str.strip(), '%Y-%m-%d')
                _e = datetime.strptime(_end_str.strip(), '%Y-%m-%d')
                SKIP_DATE_RANGES.append((_s, _e))
            except ValueError:
                pass