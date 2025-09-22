"""Database models for Strava activities and metrics."""

from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Activity(Base):
    """Strava activity model."""

    __tablename__ = "activities"

    id = Column(Integer, primary_key=True)
    strava_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(255))
    type = Column(String(50))  # Run, Ride, Swim, etc.
    workout_type = Column(Integer)  # 0=Default, 1=Race, 2=Long Run, 3=Workout, etc.
    start_date = Column(DateTime, nullable=False)
    distance = Column(Float)  # meters
    moving_time = Column(Integer)  # seconds
    elapsed_time = Column(Integer)  # seconds
    total_elevation_gain = Column(Float)  # meters
    average_heartrate = Column(Float)  # bpm
    max_heartrate = Column(Float)  # bpm
    average_speed = Column(Float)  # m/s
    max_speed = Column(Float)  # m/s
    average_cadence = Column(Float)  # rpm
    average_watts = Column(Float)  # watts
    kilojoules = Column(Float)
    suffer_score = Column(Integer)  # Strava's relative effort
    training_load = Column(Float)  # Calculated TSS or equivalent
    raw_data = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Activity(strava_id={self.strava_id}, name={self.name}, date={self.start_date})>"


class Metric(Base):
    """Daily calculated metrics."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False)
    fitness = Column(Float, nullable=False)  # CTL (Chronic Training Load)
    fatigue = Column(Float, nullable=False)  # ATL (Acute Training Load)
    form = Column(Float, nullable=False)  # TSB (Training Stress Balance)
    daily_load = Column(Float, default=0.0)  # Daily training stress
    recommendation = Column(String(50))  # REST, EASY, MODERATE, HARD
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Metric(date={self.date}, fitness={self.fitness:.1f}, fatigue={self.fatigue:.1f}, form={self.form:.1f})>"


class ModelState(Base):
    """Store model parameters and state."""

    __tablename__ = "model_state"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default")
    fitness_decay = Column(Float, nullable=False)
    fatigue_decay = Column(Float, nullable=False)
    fitness_magnitude = Column(Float, nullable=False)
    fatigue_magnitude = Column(Float, nullable=False)
    last_sync = Column(DateTime)
    last_calculation = Column(DateTime)
    additional_params = Column(Text)  # JSON for extensibility
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ModelState(user_id={self.user_id}, last_sync={self.last_sync})>"


class AuthToken(Base):
    """Store OAuth tokens."""

    __tablename__ = "auth_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    expires_at = Column(Integer, nullable=False)  # Unix timestamp
    athlete_id = Column(String(50))
    athlete_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        return datetime.utcnow().timestamp() >= self.expires_at

    def __repr__(self):
        return f"<AuthToken(user_id={self.user_id}, athlete_name={self.athlete_name})>"


class GarminAuthToken(Base):
    """Store Garmin OAuth1 tokens."""

    __tablename__ = "garmin_auth_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    access_token = Column(Text, nullable=False)
    access_token_secret = Column(Text, nullable=False)
    consumer_key = Column(String(255))
    consumer_secret = Column(String(255))
    garmin_user_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<GarminAuthToken(user_id={self.user_id}, garmin_user_id={self.garmin_user_id})>"


class HRVData(Base):
    """Store Heart Rate Variability data from Garmin."""

    __tablename__ = "hrv_data"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default")
    date = Column(DateTime, nullable=False)

    # HRV Metrics
    hrv_rmssd = Column(Float)  # Root Mean Square of Successive Differences (ms)
    hrv_score = Column(Float)  # Garmin HRV Score (0-100)
    hrv_status = Column(String(50))  # balanced, unbalanced, low, high

    # Additional HRV metrics if available
    stress_level = Column(Float)  # Stress level (0-100)
    stress_qualifier = Column(String(50))  # low, medium, high, overreaching
    recovery_advisor = Column(String(255))  # Recovery recommendations

    # Raw data for analysis
    baseline_low_upper = Column(Float)
    baseline_balanced_lower = Column(Float)
    baseline_balanced_upper = Column(Float)

    # Metadata
    measurement_timestamp = Column(DateTime)
    raw_data = Column(Text)  # JSON string for full response
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<HRVData(date={self.date}, hrv_score={self.hrv_score}, status={self.hrv_status})>"


class SleepData(Base):
    """Store sleep data from Garmin."""

    __tablename__ = "sleep_data"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default")
    date = Column(DateTime, nullable=False)  # Sleep date (night of sleep)

    # Sleep Duration
    total_sleep_time = Column(Integer)  # Total sleep in seconds
    deep_sleep_time = Column(Integer)  # Deep sleep in seconds
    light_sleep_time = Column(Integer)  # Light sleep in seconds
    rem_sleep_time = Column(Integer)  # REM sleep in seconds
    awake_time = Column(Integer)  # Time awake in seconds

    # Sleep Quality Metrics
    sleep_score = Column(Float)  # Overall sleep score (0-100)
    sleep_efficiency = Column(Float)  # Sleep efficiency percentage
    sleep_latency = Column(Integer)  # Time to fall asleep (seconds)

    # Sleep Timing
    bedtime = Column(DateTime)  # When went to bed
    sleep_start_time = Column(DateTime)  # When actually fell asleep
    sleep_end_time = Column(DateTime)  # When woke up
    wake_time = Column(DateTime)  # When got out of bed

    # Additional Metrics
    restlessness = Column(Float)  # Restlessness level
    interruptions = Column(Integer)  # Number of sleep interruptions
    average_heart_rate = Column(Float)  # Average HR during sleep
    lowest_heart_rate = Column(Float)  # Lowest HR during sleep

    # Sleep stages breakdown (percentages)
    deep_sleep_pct = Column(Float)
    light_sleep_pct = Column(Float)
    rem_sleep_pct = Column(Float)
    awake_pct = Column(Float)

    # Recovery indicators
    overnight_hrv = Column(Float)  # HRV during sleep
    sleep_stress = Column(Float)  # Stress level during sleep
    recovery_score = Column(Float)  # Recovery score from sleep

    # Metadata
    raw_data = Column(Text)  # JSON string for full response
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SleepData(date={self.date}, total_sleep={self.total_sleep_time/3600:.1f}h, score={self.sleep_score})>"


class WellnessData(Base):
    """Store daily wellness metrics from Garmin."""

    __tablename__ = "wellness_data"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default")
    date = Column(DateTime, nullable=False)

    # Stress Metrics
    stress_avg = Column(Float)  # Average stress level (0-100)
    stress_max = Column(Float)  # Maximum stress level
    stress_qualifier = Column(String(50))  # low, medium, high, unknown
    rest_stress_duration = Column(Integer)  # Time in rest stress (seconds)
    low_stress_duration = Column(Integer)  # Time in low stress (seconds)
    medium_stress_duration = Column(Integer)  # Time in medium stress (seconds)
    high_stress_duration = Column(Integer)  # Time in high stress (seconds)
    stress_duration = Column(Integer)  # Total stress monitoring time

    # Body Battery (if available)
    body_battery_charged = Column(Float)  # Body Battery charged
    body_battery_drained = Column(Float)  # Body Battery drained
    body_battery_highest = Column(Float)  # Highest Body Battery level
    body_battery_lowest = Column(Float)  # Lowest Body Battery level

    # Respiration
    avg_respiration = Column(Float)  # Average respiration rate
    max_respiration = Column(Float)  # Maximum respiration rate
    min_respiration = Column(Float)  # Minimum respiration rate

    # Pulse Ox (if available)
    avg_spo2 = Column(Float)  # Average blood oxygen saturation
    lowest_spo2 = Column(Float)  # Lowest SpO2 reading

    # Steps and Activity
    total_steps = Column(Integer)  # Total daily steps
    goal_steps = Column(Integer)  # Step goal
    calories_goal = Column(Float)  # Calorie goal
    calories_bmr = Column(Float)  # BMR calories
    calories_active = Column(Float)  # Active calories
    calories_consumed = Column(Float)  # Total calories consumed

    # Distance and Floors
    total_distance = Column(Float)  # Total distance in meters
    floors_ascended = Column(Float)  # Floors climbed
    floors_descended = Column(Float)  # Floors descended

    # Intensity Minutes
    moderate_intensity_minutes = Column(Integer)
    vigorous_intensity_minutes = Column(Integer)

    # Metadata
    raw_data = Column(Text)  # JSON string for full response
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<WellnessData(date={self.date}, stress_avg={self.stress_avg}, steps={self.total_steps})>"


class PeriodizationState(Base):
    """Track current periodization cycle and phase."""

    __tablename__ = "periodization_state"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default", unique=True)

    # Current cycle information
    cycle_start_date = Column(DateTime, nullable=False)  # When current cycle started
    cycle_length_weeks = Column(Integer, default=4)  # Total cycle length (e.g., 4 weeks for 3:1)
    current_week = Column(Integer, nullable=False)  # Current week in cycle (1-based)
    current_phase = Column(String(50), nullable=False)  # BUILD, PEAK, RECOVERY, etc.

    # Phase progression within cycle
    build_weeks = Column(Integer, default=3)  # Number of build weeks in cycle
    recovery_weeks = Column(Integer, default=1)  # Number of recovery weeks in cycle

    # Goal-oriented periodization (optional)
    goal_event_date = Column(DateTime)  # Target race/event date
    goal_type = Column(String(100))  # Type of goal (5k, marathon, century, etc.)
    weeks_to_goal = Column(Integer)  # Weeks remaining to goal

    # Training phase distribution (for longer-term planning)
    base_phase_end = Column(DateTime)  # When base phase ends
    build_phase_end = Column(DateTime)  # When build phase ends
    peak_phase_end = Column(DateTime)  # When peak phase ends
    taper_phase_end = Column(DateTime)  # When taper phase ends

    # Periodization model settings
    periodization_type = Column(String(50), default="undulating")  # linear, undulating, block, polarized
    auto_progression = Column(Boolean, default=True)  # Auto-advance phases

    # Performance tracking
    baseline_fitness = Column(Float)  # Fitness at cycle start
    target_fitness = Column(Float)  # Target fitness for cycle
    performance_trend = Column(String(50))  # improving, maintaining, declining

    # Metadata
    notes = Column(Text)  # User notes about current phase
    last_phase_change = Column(DateTime)  # When phase last changed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_current_cycle_day(self) -> int:
        """Get current day within the cycle (0-based)."""
        if not self.cycle_start_date:
            return 0
        days_elapsed = (datetime.utcnow() - self.cycle_start_date).days
        cycle_length_days = self.cycle_length_weeks * 7
        return days_elapsed % cycle_length_days

    def get_days_in_current_week(self) -> int:
        """Get days elapsed in current week (0-6)."""
        return self.get_current_cycle_day() % 7

    def should_advance_week(self) -> bool:
        """Check if we should advance to next week."""
        cycle_day = self.get_current_cycle_day()
        expected_week = (cycle_day // 7) + 1
        return expected_week > self.current_week

    def should_advance_cycle(self) -> bool:
        """Check if we should start a new cycle."""
        return self.get_current_cycle_day() >= (self.cycle_length_weeks * 7)

    def get_current_phase_type(self) -> str:
        """Determine current phase based on week in cycle."""
        if self.current_week <= self.build_weeks:
            return "BUILD" if self.current_week < self.build_weeks else "PEAK"
        else:
            return "RECOVERY"

    def __repr__(self):
        return f"<PeriodizationState(user_id={self.user_id}, week={self.current_week}, phase={self.current_phase})>"