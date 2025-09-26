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

    # Physiological Bounds - prevent corrupted metrics (adjusted for serious cyclists)
    MAX_DAILY_LOAD: float = float(os.getenv("MAX_DAILY_LOAD", "1000"))  # Max for ultra-endurance events
    MAX_FITNESS: float = float(os.getenv("MAX_FITNESS", "400"))  # Elite athlete fitness levels
    MAX_FATIGUE: float = float(os.getenv("MAX_FATIGUE", "150"))  # High fatigue tolerance
    MAX_FORM: float = float(os.getenv("MAX_FORM", "100"))  # Extended form range

    # Training Recommendation Thresholds
    FATIGUE_HIGH_THRESHOLD: float = 0.5  # Fatigue/Fitness ratio above this = rest day
    FORM_HIGH_THRESHOLD: float = 5  # Form above this = hard training possible
    FORM_LOW_THRESHOLD: float = -5  # Form below this = easy only

    # Training Plan Configuration
    TRAINING_MAX_WEEKLY_HOURS: float = float(os.getenv("TRAINING_MAX_WEEKLY_HOURS", "14"))  # Maximum hours per week
    TRAINING_REST_DAYS: str = os.getenv("TRAINING_REST_DAYS", "0")  # Comma-separated days (0=Mon, 6=Sun)
    TRAINING_STRENGTH_DAYS: str = os.getenv("TRAINING_STRENGTH_DAYS", "1")  # Mandatory strength days (1=Tue)

    # Application
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PORT: int = int(os.getenv("PORT", "8000"))
    TOKEN_FILE: Path = Path.home() / ".strava_supercompensation" / "tokens.json"

    # Race Day Peaking System (Olympic Competition Periodization)
    TARGET_RACE_DATES: str = os.getenv("TARGET_RACE_DATES", "")
    PEAK_PERFORMANCE_DURATION: int = int(os.getenv("PEAK_PERFORMANCE_DURATION", "3"))
    TAPER_DURATION: int = int(os.getenv("TAPER_DURATION", "14"))
    SUPERCOMPENSATION_WINDOW: int = int(os.getenv("SUPERCOMPENSATION_WINDOW", "7"))
    PEAK_FORM_TARGET: float = float(os.getenv("PEAK_FORM_TARGET", "25"))
    RACE_READINESS_THRESHOLD: float = float(os.getenv("RACE_READINESS_THRESHOLD", "80"))

    # Sport-Specific Taper Durations
    TAPER_CONFIGS = {
        "Run": {
            5: int(os.getenv("TAPER_RUN_5K", "2")),
            10: int(os.getenv("TAPER_RUN_10K", "2")),
            21.1: int(os.getenv("TAPER_RUN_HALF", "3")),
            42.2: int(os.getenv("TAPER_RUN_MARATHON", "5")),
            100: int(os.getenv("TAPER_RUN_ULTRA", "5")),
        },
        "Ride": {
            100: int(os.getenv("TAPER_RIDE_SHORT", "2")),
            200: int(os.getenv("TAPER_RIDE_MEDIUM", "2")),
            300: int(os.getenv("TAPER_RIDE_LONG", "4")),
            1000: int(os.getenv("TAPER_RIDE_ULTRA", "4")),
        },
        "Hike": {
            50: int(os.getenv("TAPER_HIKE_SHORT", "1")),
            100: int(os.getenv("TAPER_HIKE_MEDIUM", "2")),
            200: int(os.getenv("TAPER_HIKE_LONG", "3")),
        }
    }

    # Taper Volume Reductions by Day
    TAPER_VOLUME_REDUCTIONS = {
        1: float(os.getenv("TAPER_VOLUME_DAY_1", "20")) / 100,
        2: float(os.getenv("TAPER_VOLUME_DAY_2", "30")) / 100,
        3: float(os.getenv("TAPER_VOLUME_DAY_3", "50")) / 100,
        4: float(os.getenv("TAPER_VOLUME_DAY_4", "60")) / 100,
        5: float(os.getenv("TAPER_VOLUME_DAY_5", "70")) / 100,
    }

    # Post-Race Recovery Periods
    POST_RACE_RECOVERY = {
        "Run": {
            5: int(os.getenv("RECOVERY_RUN_5K", "1")),
            10: int(os.getenv("RECOVERY_RUN_10K", "2")),
            21.1: int(os.getenv("RECOVERY_RUN_HALF", "3")),
            42.2: int(os.getenv("RECOVERY_RUN_MARATHON", "7")),
            50: int(os.getenv("RECOVERY_RUN_50K", "7")),
            70: int(os.getenv("RECOVERY_RUN_70K", "10")),
            100: int(os.getenv("RECOVERY_RUN_100K", "14")),
        },
        "Ride": {
            100: int(os.getenv("RECOVERY_RIDE_SHORT", "1")),
            200: int(os.getenv("RECOVERY_RIDE_MEDIUM", "2")),
            300: int(os.getenv("RECOVERY_RIDE_LONG", "3")),
            1000: int(os.getenv("RECOVERY_RIDE_ULTRA", "5")),
        },
        "Hike": {
            50: int(os.getenv("RECOVERY_HIKE_SHORT", "1")),
            100: int(os.getenv("RECOVERY_HIKE_MEDIUM", "3")),
            200: int(os.getenv("RECOVERY_HIKE_LONG", "5")),
        }
    }

    # Recovery Intensity Limits
    RECOVERY_INTENSITY_LIMITS = {
        1: float(os.getenv("RECOVERY_DAY_1_INTENSITY", "0")) / 100,
        2: float(os.getenv("RECOVERY_DAY_2_INTENSITY", "20")) / 100,
        3: float(os.getenv("RECOVERY_DAY_3_INTENSITY", "30")) / 100,
        7: float(os.getenv("RECOVERY_WEEK_1_INTENSITY", "50")) / 100,
        14: float(os.getenv("RECOVERY_WEEK_2_INTENSITY", "75")) / 100,
    }

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

    # Sport Load Multipliers (Olympic-Level Enhancement)
    # Adjusts TSS based on sport-specific systemic impact
    SPORT_LOAD_MULTIPLIERS = {
        "Run": float(os.getenv("LOAD_MULTIPLIER_RUNNING", "1.5")),
        "Ride": float(os.getenv("LOAD_MULTIPLIER_CYCLING", "1.0")),
        "Hike": float(os.getenv("LOAD_MULTIPLIER_HIKING", "1.2")),
        "WeightTraining": float(os.getenv("LOAD_MULTIPLIER_WEIGHT_TRAINING", "1.3")),
        "Workout": float(os.getenv("LOAD_MULTIPLIER_WORKOUT", "1.1")),
        "Rowing": float(os.getenv("LOAD_MULTIPLIER_ROWING", "0.9")),
        "Swim": float(os.getenv("LOAD_MULTIPLIER_SWIMMING", "0.8")),
        "AlpineSki": float(os.getenv("LOAD_MULTIPLIER_SKIING", "1.4")),
        "Yoga": float(os.getenv("LOAD_MULTIPLIER_YOGA", "0.5")),
    }

    # Duration Calculation Multipliers (TSS to minutes conversion)
    DURATION_MULTIPLIERS = {
        "REST": float(os.getenv("DURATION_MULTIPLIER_REST", "0.0")),
        "RECOVERY": float(os.getenv("DURATION_MULTIPLIER_RECOVERY", "1.2")),
        "AEROBIC": float(os.getenv("DURATION_MULTIPLIER_AEROBIC", "1.0")),
        "TEMPO": float(os.getenv("DURATION_MULTIPLIER_TEMPO", "0.85")),
        "THRESHOLD": float(os.getenv("DURATION_MULTIPLIER_THRESHOLD", "0.75")),
        "VO2MAX": float(os.getenv("DURATION_MULTIPLIER_VO2MAX", "0.65")),
        "NEUROMUSCULAR": float(os.getenv("DURATION_MULTIPLIER_NEUROMUSCULAR", "0.5")),
        "LONG": float(os.getenv("DURATION_MULTIPLIER_LONG", "1.4")),
        "INTERVALS": float(os.getenv("DURATION_MULTIPLIER_INTERVALS", "0.7")),
        "FARTLEK": float(os.getenv("DURATION_MULTIPLIER_FARTLEK", "0.9")),
    }

    # Recovery Time Multipliers (for calculating recovery needs)
    RECOVERY_MULTIPLIERS = {
        "REST": float(os.getenv("RECOVERY_MULTIPLIER_REST", "0.0")),
        "RECOVERY": float(os.getenv("RECOVERY_MULTIPLIER_RECOVERY", "0.5")),
        "AEROBIC": float(os.getenv("RECOVERY_MULTIPLIER_AEROBIC", "1.0")),
        "TEMPO": float(os.getenv("RECOVERY_MULTIPLIER_TEMPO", "1.2")),
        "THRESHOLD": float(os.getenv("RECOVERY_MULTIPLIER_THRESHOLD", "1.5")),
        "VO2MAX": float(os.getenv("RECOVERY_MULTIPLIER_VO2MAX", "2.0")),
        "NEUROMUSCULAR": float(os.getenv("RECOVERY_MULTIPLIER_NEUROMUSCULAR", "2.5")),
        "LONG": float(os.getenv("RECOVERY_MULTIPLIER_LONG", "1.3")),
        "INTERVALS": float(os.getenv("RECOVERY_MULTIPLIER_INTERVALS", "1.8")),
        "FARTLEK": float(os.getenv("RECOVERY_MULTIPLIER_FARTLEK", "1.0")),
    }

    # Adaptive Model Parameters
    ENABLE_ADAPTIVE_PARAMETERS: bool = os.getenv("ENABLE_ADAPTIVE_PARAMETERS", "true").lower() == "true"
    ADAPTATION_RATE: float = float(os.getenv("ADAPTATION_RATE", "0.05"))
    MIN_FITNESS_DECAY: float = float(os.getenv("MIN_FITNESS_DECAY", "20"))
    MAX_FITNESS_DECAY: float = float(os.getenv("MAX_FITNESS_DECAY", "60"))
    MIN_FATIGUE_DECAY: float = float(os.getenv("MIN_FATIGUE_DECAY", "3"))
    MAX_FATIGUE_DECAY: float = float(os.getenv("MAX_FATIGUE_DECAY", "14"))

    # Overtraining and Readiness Parameters
    OVERTRAINING_THRESHOLD: float = float(os.getenv("OVERTRAINING_THRESHOLD", "1.0"))
    READINESS_WEIGHT_HRV: float = float(os.getenv("READINESS_WEIGHT_HRV", "0.4"))
    READINESS_WEIGHT_SLEEP: float = float(os.getenv("READINESS_WEIGHT_SLEEP", "0.35"))
    READINESS_WEIGHT_STRESS: float = float(os.getenv("READINESS_WEIGHT_STRESS", "0.25"))

    # Environmental Optimal Ranges
    OPTIMAL_TEMP_MIN: float = float(os.getenv("OPTIMAL_TEMP_MIN", "10"))
    OPTIMAL_TEMP_MAX: float = float(os.getenv("OPTIMAL_TEMP_MAX", "20"))
    OPTIMAL_HUMIDITY_MIN: float = float(os.getenv("OPTIMAL_HUMIDITY_MIN", "40"))
    OPTIMAL_HUMIDITY_MAX: float = float(os.getenv("OPTIMAL_HUMIDITY_MAX", "60"))
    OPTIMAL_WIND_MAX: float = float(os.getenv("OPTIMAL_WIND_MAX", "15"))
    ALTITUDE_THRESHOLD: float = float(os.getenv("ALTITUDE_THRESHOLD", "1500"))

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
    def get_sport_load_multiplier(cls, sport: str) -> float:
        """Get load multiplier for sport-specific TSS adjustment."""
        return cls.SPORT_LOAD_MULTIPLIERS.get(sport, 1.0)

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

    @classmethod
    def get_race_dates(cls) -> list:
        """Parse and return list of target race dates."""
        from datetime import datetime

        if not cls.TARGET_RACE_DATES:
            return []

        race_dates = []
        for date_str in cls.TARGET_RACE_DATES.split(','):
            try:
                race_date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
                race_dates.append(race_date)
            except ValueError:
                continue

        return sorted(race_dates)

    @classmethod
    def get_race_priority(cls, race_date) -> str:
        """Get race priority level for a specific date."""
        date_str = race_date.strftime('%Y_%m_%d')
        priority_key = f"RACE_PRIORITY_{date_str}"
        return os.getenv(priority_key, "B_RACE")

    @classmethod
    def days_to_next_race(cls, current_date=None) -> int:
        """Calculate days until next target race."""
        from datetime import datetime

        if current_date is None:
            current_date = datetime.now()

        race_dates = cls.get_race_dates()
        upcoming_races = [race for race in race_dates if race > current_date]

        if not upcoming_races:
            return float('inf')  # No upcoming races

        next_race = min(upcoming_races)
        return (next_race - current_date).days

    @classmethod
    def get_race_details(cls, race_date):
        """Get race type and distance for a specific date."""
        date_str = race_date.strftime('%Y_%m_%d')
        race_type = os.getenv(f"RACE_TYPE_{date_str}", "Run")
        race_distance = float(os.getenv(f"RACE_DISTANCE_{date_str}", "10"))
        return race_type, race_distance

    @classmethod
    def get_taper_duration_for_race(cls, race_date) -> int:
        """Calculate sport and distance-specific taper duration."""
        race_type, race_distance = cls.get_race_details(race_date)

        # Get sport-specific taper configurations
        sport_tapers = cls.TAPER_CONFIGS.get(race_type, {})

        # Find appropriate taper duration based on distance
        taper_days = cls.TAPER_DURATION  # Default

        # Find the smallest distance threshold that's >= race distance
        for distance_threshold in sorted(sport_tapers.keys()):
            if race_distance <= distance_threshold:
                taper_days = sport_tapers[distance_threshold]
                break
        else:
            # Use the maximum taper duration for ultra distances
            if sport_tapers:
                taper_days = max(sport_tapers.values())

        return taper_days

    @classmethod
    def get_next_race_with_taper(cls, current_date=None):
        """Get next race date with its specific taper duration."""
        from datetime import datetime

        if current_date is None:
            current_date = datetime.now()

        race_dates = cls.get_race_dates()
        for race_date in race_dates:
            if race_date > current_date:
                taper_duration = cls.get_taper_duration_for_race(race_date)
                days_to_race = (race_date - current_date).days
                return race_date, days_to_race, taper_duration

        return None, float('inf'), 0

    @classmethod
    def is_in_taper_phase(cls, current_date=None) -> bool:
        """Check if currently in tapering phase before a race."""
        race_date, days_to_race, taper_duration = cls.get_next_race_with_taper(current_date)
        if race_date:
            return days_to_race <= taper_duration
        return False

    @classmethod
    def get_taper_volume_reduction(cls, days_to_race: int) -> float:
        """Get volume reduction factor for taper day."""
        return cls.TAPER_VOLUME_REDUCTIONS.get(days_to_race, 0.85)

    @classmethod
    def get_recovery_duration_for_race(cls, race_date) -> int:
        """Calculate post-race recovery duration based on sport and distance."""
        race_type, race_distance = cls.get_race_details(race_date)

        # Get sport-specific recovery configurations
        sport_recovery = cls.POST_RACE_RECOVERY.get(race_type, {})

        # Find appropriate recovery duration based on distance
        recovery_days = 3  # Default

        # Find the smallest distance threshold that's >= race distance
        for distance_threshold in sorted(sport_recovery.keys()):
            if race_distance <= distance_threshold:
                recovery_days = sport_recovery[distance_threshold]
                break
        else:
            # Use the maximum recovery duration for ultra distances
            if sport_recovery:
                recovery_days = max(sport_recovery.values())

        return recovery_days

    @classmethod
    def get_recovery_intensity_limit(cls, days_after_race: int) -> float:
        """Get intensity limit for post-race recovery day."""
        # Find the appropriate limit based on days after race
        for day_threshold in sorted(cls.RECOVERY_INTENSITY_LIMITS.keys()):
            if days_after_race <= day_threshold:
                return cls.RECOVERY_INTENSITY_LIMITS[day_threshold]
        return 1.0  # Full intensity after recovery period

    @classmethod
    def is_in_recovery_phase(cls, current_date=None) -> tuple:
        """Check if currently in post-race recovery phase.

        Returns:
            (is_recovering, days_since_race, recovery_duration, race_date)
        """
        from datetime import datetime

        if current_date is None:
            current_date = datetime.now()

        race_dates = cls.get_race_dates()

        # Check past races for recovery period
        for race_date in race_dates:
            if race_date <= current_date:
                days_since_race = (current_date - race_date).days
                recovery_duration = cls.get_recovery_duration_for_race(race_date)

                if days_since_race <= recovery_duration:
                    return True, days_since_race, recovery_duration, race_date

        return False, 0, 0, None

    @classmethod
    def get_training_rest_days(cls) -> list:
        """Parse and return list of rest days for training plan.

        Returns:
            List of integers representing rest days (0=Monday, 6=Sunday)
        """
        if not cls.TRAINING_REST_DAYS:
            return [0]  # Default to Monday rest day

        rest_days = []
        for day_str in cls.TRAINING_REST_DAYS.split(','):
            try:
                day = int(day_str.strip())
                if 0 <= day <= 6:  # Valid day of week
                    rest_days.append(day)
            except ValueError:
                continue

        return rest_days if rest_days else [0]  # Default to Monday if parsing fails


config = Config()