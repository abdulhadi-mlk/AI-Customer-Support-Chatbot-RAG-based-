import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Resolve the project .env explicitly so the launch directory does not matter.
# The project file must win over a stale placeholder inherited by Streamlit.
load_dotenv(ENV_FILE, override=True)

PLACEHOLDER_VALUES = {
    "",
    "your_actual_google_gemini_api_key",
    "your_api_key_here",
    "your_google_api_key",
    "your-google-api-key",
    "your-gemini-api-key",
    "your_api_key",
    "YOUR_GOOGLE_API_KEY",
}


def get_google_api_key() -> str:
    """Return the configured Gemini key or raise a safe, actionable error."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    normalized_key = api_key.lower()
    if (
        normalized_key in {value.lower() for value in PLACEHOLDER_VALUES}
        or normalized_key.startswith(("your_", "your-"))
    ):
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. Add your actual Gemini API key "
            f"to {ENV_FILE} as GOOGLE_API_KEY=your_actual_google_gemini_api_key."
        )
    return api_key
