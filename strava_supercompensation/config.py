"""Configuration management for the Strava Supercompensation tool."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # Strava API
    STRAVA_CLIENT_ID: str = os.getenv("STRAVA_CLIENT_ID", "")
    STRAVA_CLIENT_SECRET: str = os.getenv("STRAVA_CLIENT_SECRET", "")
    STRAVA_REDIRECT_URI: str = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8000/callback")
    STRAVA_API_BASE_URL: str = "https://www.strava.com/api/v3"
    STRAVA_AUTH_BASE_URL: str = "https://www.strava.com/oauth"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./strava_supercompensation.db")

    # Model Parameters (Banister Model)
    FITNESS_DECAY_RATE: float = float(os.getenv("FITNESS_DECAY_RATE", "42"))  # days
    FATIGUE_DECAY_RATE: float = float(os.getenv("FATIGUE_DECAY_RATE", "7"))  # days
    FITNESS_MAGNITUDE: float = float(os.getenv("FITNESS_MAGNITUDE", "0.1"))  # Reduced to prevent overflow
    FATIGUE_MAGNITUDE: float = float(os.getenv("FATIGUE_MAGNITUDE", "0.15"))  # Reduced to prevent overflow

    # Training Recommendation Thresholds
    FATIGUE_HIGH_THRESHOLD: float = 0.5  # Fatigue/Fitness ratio above this = rest day
    FORM_HIGH_THRESHOLD: float = 5  # Form above this = hard training possible
    FORM_LOW_THRESHOLD: float = -5  # Form below this = easy only

    # Application
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PORT: int = int(os.getenv("PORT", "8000"))
    TOKEN_FILE: Path = Path.home() / ".strava_supercompensation" / "tokens.json"

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.STRAVA_CLIENT_ID or not cls.STRAVA_CLIENT_SECRET:
            raise ValueError(
                "Missing Strava API credentials. Please set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET"
            )
        return True

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure required directories exist."""
        cls.TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)


config = Config()