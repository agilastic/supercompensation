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

    # Multi-Sport Configuration
    ENABLED_SPORTS = {
        "Run": os.getenv("ENABLE_RUNNING", "true").lower() == "true",
        "Ride": os.getenv("ENABLE_CYCLING", "true").lower() == "true",
        "Hike": os.getenv("ENABLE_HIKING", "true").lower() == "true",
        "WeightTraining": os.getenv("ENABLE_WEIGHT_TRAINING", "true").lower() == "true",
        "Workout": os.getenv("ENABLE_WORKOUT", "true").lower() == "true",
        "Rowing": os.getenv("ENABLE_ROWING", "true").lower() == "true",
        "Swim": os.getenv("ENABLE_SWIMMING", "false").lower() == "true",
        "AlpineSki": os.getenv("ENABLE_SKIING", "false").lower() == "true",
        "Yoga": os.getenv("ENABLE_YOGA", "false").lower() == "true",
    }

    SPORT_PREFERENCES = {
        "Run": float(os.getenv("PREFERENCE_RUNNING", "0.30")),
        "Ride": float(os.getenv("PREFERENCE_CYCLING", "0.45")),
        "Hike": float(os.getenv("PREFERENCE_HIKING", "0.18")),
        "WeightTraining": float(os.getenv("PREFERENCE_WEIGHT_TRAINING", "0.05")),
        "Workout": float(os.getenv("PREFERENCE_WORKOUT", "0.02")),
        "Rowing": float(os.getenv("PREFERENCE_ROWING", "0.01")),
        "Swim": float(os.getenv("PREFERENCE_SWIMMING", "0.0")),
        "AlpineSki": float(os.getenv("PREFERENCE_SKIING", "0.0")),
        "Yoga": float(os.getenv("PREFERENCE_YOGA", "0.0")),
    }

    SPORT_RECOVERY_TIMES = {
        "Run": float(os.getenv("RECOVERY_TIME_RUNNING", "48")),
        "Ride": float(os.getenv("RECOVERY_TIME_CYCLING", "24")),
        "Hike": float(os.getenv("RECOVERY_TIME_HIKING", "24")),
        "WeightTraining": float(os.getenv("RECOVERY_TIME_WEIGHT_TRAINING", "48")),
        "Workout": float(os.getenv("RECOVERY_TIME_WORKOUT", "24")),
        "Rowing": float(os.getenv("RECOVERY_TIME_ROWING", "24")),
        "Swim": float(os.getenv("RECOVERY_TIME_SWIMMING", "24")),
        "AlpineSki": float(os.getenv("RECOVERY_TIME_SKIING", "48")),
        "Yoga": float(os.getenv("RECOVERY_TIME_YOGA", "12")),
    }

    @classmethod
    def get_enabled_sports(cls) -> list[str]:
        """Get list of enabled sports for recommendations."""
        return [sport for sport, enabled in cls.ENABLED_SPORTS.items() if enabled]

    @classmethod
    def get_sport_preference(cls, sport: str) -> float:
        """Get preference weight for a sport."""
        return cls.SPORT_PREFERENCES.get(sport, 0.1)

    @classmethod
    def get_sport_recovery_time(cls, sport: str) -> float:
        """Get recovery time for a sport in hours."""
        return cls.SPORT_RECOVERY_TIMES.get(sport, 24.0)

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