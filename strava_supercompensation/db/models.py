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