"""Banister Impulse-Response model implementation for supercompensation analysis."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd

from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric, ModelState


class BanisterModel:
    """Implementation of Banister's Fitness-Fatigue model."""

    def __init__(
        self,
        fitness_decay: float = None,
        fatigue_decay: float = None,
        fitness_magnitude: float = None,
        fatigue_magnitude: float = None,
    ):
        """Initialize the Banister model with parameters.

        Args:
            fitness_decay: Time constant for fitness decay (days) - typically 42
            fatigue_decay: Time constant for fatigue decay (days) - typically 7
            fitness_magnitude: Magnitude factor for fitness response - typically 1.0
            fatigue_magnitude: Magnitude factor for fatigue response - typically 2.0
        """
        self.fitness_decay = fitness_decay or config.FITNESS_DECAY_RATE
        self.fatigue_decay = fatigue_decay or config.FATIGUE_DECAY_RATE
        self.fitness_magnitude = fitness_magnitude or config.FITNESS_MAGNITUDE
        self.fatigue_magnitude = fatigue_magnitude or config.FATIGUE_MAGNITUDE

    def impulse_response(self, training_loads: np.ndarray, days: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate fitness, fatigue, and form using impulse-response model.

        Args:
            training_loads: Array of daily training loads
            days: Array of days (time points)

        Returns:
            Tuple of (fitness, fatigue, form) arrays
        """
        n_days = len(days)
        fitness = np.zeros(n_days)
        fatigue = np.zeros(n_days)

        # Initialize with first day's load if present
        if len(training_loads) > 0 and training_loads[0] > 0:
            fitness[0] = training_loads[0] * self.fitness_magnitude
            fatigue[0] = training_loads[0] * self.fatigue_magnitude

        # Calculate cumulative impulse responses
        for i in range(1, n_days):
            # Decay from previous day
            fitness[i] = fitness[i-1] * np.exp(-1 / self.fitness_decay)
            fatigue[i] = fatigue[i-1] * np.exp(-1 / self.fatigue_decay)

            # Add today's training impulse
            if training_loads[i] > 0:
                fitness[i] += training_loads[i] * self.fitness_magnitude
                fatigue[i] += training_loads[i] * self.fatigue_magnitude

        # Calculate form (Training Stress Balance)
        form = fitness - fatigue

        return fitness, fatigue, form

    def predict_optimal_load(self, current_fitness: float, current_fatigue: float, days_ahead: int = 7) -> Dict[str, float]:
        """Predict optimal training load for upcoming days.

        Args:
            current_fitness: Current fitness level (CTL)
            current_fatigue: Current fatigue level (ATL)
            days_ahead: Number of days to predict

        Returns:
            Dictionary with recommended loads for different scenarios
        """
        recommendations = {}

        # Current form
        current_form = current_fitness - current_fatigue

        # Simulate different training scenarios
        scenarios = {
            "rest": 0,
            "easy": 30,
            "moderate": 60,
            "hard": 100,
            "very_hard": 150,
        }

        for scenario_name, load in scenarios.items():
            # Simulate this load for the next few days
            future_fitness = current_fitness
            future_fatigue = current_fatigue

            for day in range(days_ahead):
                future_fitness = future_fitness * np.exp(-1 / self.fitness_decay) + load * self.fitness_magnitude
                future_fatigue = future_fatigue * np.exp(-1 / self.fatigue_decay) + load * self.fatigue_magnitude

            future_form = future_fitness - future_fatigue
            recommendations[scenario_name] = {
                "load": load,
                "predicted_fitness": future_fitness,
                "predicted_fatigue": future_fatigue,
                "predicted_form": future_form,
                "form_change": future_form - current_form,
            }

        return recommendations


class SupercompensationAnalyzer:
    """Analyze training data using supercompensation principles."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.model = BanisterModel()
        self._load_model_state()

    def _load_model_state(self):
        """Load model parameters from database."""
        with self.db.get_session() as session:
            state = session.query(ModelState).filter_by(user_id=self.user_id).first()
            if state:
                self.model.fitness_decay = state.fitness_decay
                self.model.fatigue_decay = state.fatigue_decay
                self.model.fitness_magnitude = state.fitness_magnitude
                self.model.fatigue_magnitude = state.fatigue_magnitude

    def analyze(self, days_back: int = 90) -> pd.DataFrame:
        """Analyze training history and calculate metrics.

        Args:
            days_back: Number of days of history to analyze

        Returns:
            DataFrame with daily metrics
        """
        # Get activities from database
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        # Create daily training load series with dates (not datetimes)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', normalize=True)
        daily_loads = pd.Series(index=date_range, data=0.0)

        # Get and process activities within session context
        # Use datetime objects for filtering to include all activities on end_date
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        with self.db.get_session() as session:
            activities = session.query(Activity).filter(
                Activity.start_date >= start_datetime,
                Activity.start_date <= end_datetime
            ).order_by(Activity.start_date).all()

            # Aggregate activities by day within the session
            for activity in activities:
                # Use the date as a pandas Timestamp for consistent indexing
                activity_date = pd.Timestamp(activity.start_date.date())
                if activity.training_load:
                    if activity_date in daily_loads.index:
                        daily_loads[activity_date] += activity.training_load

        # Calculate fitness, fatigue, and form
        days = np.arange(len(daily_loads))
        loads = daily_loads.values
        fitness, fatigue, form = self.model.impulse_response(loads, days)

        # Create DataFrame
        df = pd.DataFrame({
            'date': daily_loads.index,
            'training_load': loads,
            'fitness': fitness,
            'fatigue': fatigue,
            'form': form,
        })

        # Save metrics to database
        self._save_metrics(df)

        return df

    def _save_metrics(self, df: pd.DataFrame):
        """Save calculated metrics to database."""
        with self.db.get_session() as session:
            for _, row in df.iterrows():
                # Check if metric for this date exists
                # Convert pandas Timestamp to datetime at start of day
                if hasattr(row['date'], 'to_pydatetime'):
                    metric_date = row['date'].to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    metric_date = datetime.combine(row['date'], datetime.min.time())

                # Use date range query to handle any time component issues
                date_start = metric_date
                date_end = metric_date + timedelta(days=1)
                existing = session.query(Metric).filter(
                    Metric.date >= date_start,
                    Metric.date < date_end
                ).first()

                if existing:
                    # Update existing metric
                    existing.fitness = row['fitness']
                    existing.fatigue = row['fatigue']
                    existing.form = row['form']
                    existing.daily_load = row['training_load']
                else:
                    # Create new metric
                    metric = Metric(
                        date=metric_date,  # This is already normalized to start of day
                        fitness=row['fitness'],
                        fatigue=row['fatigue'],
                        form=row['form'],
                        daily_load=row['training_load'],
                    )
                    session.add(metric)

            # Update model state
            state = session.query(ModelState).filter_by(user_id=self.user_id).first()
            if not state:
                state = ModelState(
                    user_id=self.user_id,
                    fitness_decay=self.model.fitness_decay,
                    fatigue_decay=self.model.fatigue_decay,
                    fitness_magnitude=self.model.fitness_magnitude,
                    fatigue_magnitude=self.model.fatigue_magnitude,
                )
                session.add(state)
            state.last_calculation = datetime.utcnow()
            session.commit()

    def get_current_state(self) -> Dict[str, float]:
        """Get current fitness, fatigue, and form values."""
        with self.db.get_session() as session:
            latest_metric = session.query(Metric).order_by(Metric.date.desc()).first()

            if latest_metric:
                return {
                    "date": latest_metric.date,
                    "fitness": latest_metric.fitness,
                    "fatigue": latest_metric.fatigue,
                    "form": latest_metric.form,
                    "daily_load": latest_metric.daily_load,
                    "recommendation": latest_metric.recommendation,
                }
            else:
                return {
                    "date": datetime.utcnow(),
                    "fitness": 0.0,
                    "fatigue": 0.0,
                    "form": 0.0,
                    "daily_load": 0.0,
                    "recommendation": "NO_DATA",
                }

    def get_metrics_history(self, days: int = 30) -> List[Dict]:
        """Get metrics history for visualization."""
        with self.db.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            metrics = session.query(Metric).filter(
                Metric.date >= cutoff_date
            ).order_by(Metric.date).all()

            return [
                {
                    "date": m.date.isoformat(),
                    "fitness": m.fitness,
                    "fatigue": m.fatigue,
                    "form": m.form,
                    "load": m.daily_load,
                    "recommendation": m.recommendation,
                }
                for m in metrics
            ]