"""Banister Impulse-Response model implementation for supercompensation analysis."""

import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import json

from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric, ModelState, AdaptiveModelParameters, PerformanceOutcome


class BanisterModel:
    """Implementation of Banister's Fitness-Fatigue model."""

    def __init__(
        self,
        fitness_decay: float = None,
        fatigue_decay: float = None,
        fitness_magnitude: float = None,
        fatigue_magnitude: float = None,
        user_id: str = "default",
        enable_adaptive: bool = None,
    ):
        """Initialize the Banister model with parameters.

        Args:
            fitness_decay: Time constant for fitness decay (days) - typically 42
            fatigue_decay: Time constant for fatigue decay (days) - typically 7
            fitness_magnitude: Magnitude factor for fitness response - typically 1.0
            fatigue_magnitude: Magnitude factor for fatigue response - typically 2.0
            user_id: User identifier for adaptive parameters
            enable_adaptive: Enable adaptive parameter learning
        """
        self.user_id = user_id
        self.enable_adaptive = enable_adaptive if enable_adaptive is not None else config.ENABLE_ADAPTIVE_PARAMETERS

        # Load adaptive parameters if enabled
        if self.enable_adaptive:
            self._load_adaptive_parameters()
        else:
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
        # Physiological bounds based on sports science research (configurable)
        MAX_DAILY_LOAD = config.MAX_DAILY_LOAD
        MAX_FITNESS = config.MAX_FITNESS
        MAX_FATIGUE = config.MAX_FATIGUE

        # Check for extreme loads and warn
        extreme_loads = training_loads > MAX_DAILY_LOAD
        if np.any(extreme_loads):
            extreme_count = np.sum(extreme_loads)
            max_extreme = np.max(training_loads[extreme_loads])
            logging.warning(f"Found {extreme_count} extreme training loads (max: {max_extreme:.1f}), capping at {MAX_DAILY_LOAD}")

        # Cap training loads at physiologically reasonable values
        training_loads = np.clip(training_loads, 0, MAX_DAILY_LOAD)

        n_days = len(days)
        fitness = np.zeros(n_days)
        fatigue = np.zeros(n_days)

        # Initialize with first day's load if present
        if len(training_loads) > 0 and training_loads[0] > 0:
            fitness[0] = training_loads[0] * self.fitness_magnitude
            fatigue[0] = training_loads[0] * self.fatigue_magnitude

        # Calculate cumulative impulse responses with bounds checking
        for i in range(1, n_days):
            # Decay from previous day
            fitness[i] = fitness[i-1] * np.exp(-1 / self.fitness_decay)
            fatigue[i] = fatigue[i-1] * np.exp(-1 / self.fatigue_decay)

            # Add today's training impulse
            if training_loads[i] > 0:
                fitness[i] += training_loads[i] * self.fitness_magnitude
                fatigue[i] += training_loads[i] * self.fatigue_magnitude

            # Apply physiological bounds to prevent corruption
            fitness[i] = min(fitness[i], MAX_FITNESS)
            fatigue[i] = min(fatigue[i], MAX_FATIGUE)

        # Calculate form (Training Stress Balance)
        form = fitness - fatigue

        # Cap form at physiologically reasonable bounds
        form = np.clip(form, -config.MAX_FORM, config.MAX_FORM)

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

    def _load_adaptive_parameters(self):
        """Load adaptive model parameters from database."""
        db = get_db()
        with db.get_session() as session:
            params = session.query(AdaptiveModelParameters).filter_by(user_id=self.user_id).first()

            if params:
                self.fitness_decay = params.fitness_decay
                self.fatigue_decay = params.fatigue_decay
                self.fitness_magnitude = params.fitness_magnitude
                self.fatigue_magnitude = params.fatigue_magnitude
            else:
                # Initialize with defaults and create record
                self.fitness_decay = config.FITNESS_DECAY_RATE
                self.fatigue_decay = config.FATIGUE_DECAY_RATE
                self.fitness_magnitude = config.FITNESS_MAGNITUDE
                self.fatigue_magnitude = config.FATIGUE_MAGNITUDE

                # Create adaptive parameters record
                new_params = AdaptiveModelParameters(
                    user_id=self.user_id,
                    fitness_decay=self.fitness_decay,
                    fatigue_decay=self.fatigue_decay,
                    fitness_magnitude=self.fitness_magnitude,
                    fatigue_magnitude=self.fatigue_magnitude,
                )
                session.add(new_params)
                session.commit()

    def adapt_parameters(self, performance_delta: float, perceived_effort: float, fatigue_level: float):
        """Adapt model parameters based on performance outcomes.

        Args:
            performance_delta: Difference between expected and actual performance (-1 to 1)
            perceived_effort: Subjective effort rating (1-10)
            fatigue_level: Current fatigue level (1-10)
        """
        if not self.enable_adaptive:
            return

        db = get_db()
        with db.get_session() as session:
            params = session.query(AdaptiveModelParameters).filter_by(user_id=self.user_id).first()

            if not params:
                return

            # Adaptation logic based on performance feedback
            adaptation_rate = config.ADAPTATION_RATE

            # If underperforming with high effort -> slower recovery
            if performance_delta < -0.2 and perceived_effort > 7:
                # Increase fatigue decay time (slower recovery)
                params.fatigue_decay = min(
                    params.fatigue_decay * (1 + adaptation_rate),
                    config.MAX_FATIGUE_DECAY
                )
                params.overtraining_incidents += 1

            # If overperforming with low effort -> faster adaptation
            elif performance_delta > 0.2 and perceived_effort < 5:
                # Decrease fitness decay time (better fitness retention)
                params.fitness_decay = max(
                    params.fitness_decay * (1 - adaptation_rate * 0.5),
                    config.MIN_FITNESS_DECAY
                )

            # If consistently fatigued -> adjust fatigue magnitude
            if fatigue_level > 7:
                params.fatigue_magnitude = min(
                    params.fatigue_magnitude * (1 + adaptation_rate * 0.3),
                    0.3  # Cap fatigue magnitude
                )

            # If never fatigued -> may need more load
            elif fatigue_level < 3 and performance_delta > 0:
                params.fatigue_magnitude = max(
                    params.fatigue_magnitude * (1 - adaptation_rate * 0.3),
                    0.05  # Minimum fatigue magnitude
                )
                params.undertraining_incidents += 1

            # Update parameters
            params.total_adaptations += 1
            params.last_adaptation = datetime.now(timezone.utc)
            params.recent_performance_trend = performance_delta

            # Update confidence based on consistency
            if abs(performance_delta) < 0.1:
                params.adaptation_confidence = min(params.adaptation_confidence * 1.05, 1.0)
            else:
                params.adaptation_confidence = max(params.adaptation_confidence * 0.95, 0.3)

            session.commit()

            # Update local parameters
            self.fitness_decay = params.fitness_decay
            self.fatigue_decay = params.fatigue_decay
            self.fitness_magnitude = params.fitness_magnitude
            self.fatigue_magnitude = params.fatigue_magnitude


class SupercompensationAnalyzer:
    """Analyze training data using supercompensation principles."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.model = BanisterModel(user_id=user_id, enable_adaptive=config.ENABLE_ADAPTIVE_PARAMETERS)
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
        end_date = datetime.now(timezone.utc).date()
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
                    # Apply sport-specific load multiplier
                    sport_multiplier = config.get_sport_load_multiplier(activity.type)
                    adjusted_load = activity.training_load * sport_multiplier

                    if activity_date in daily_loads.index:
                        daily_loads[activity_date] += adjusted_load

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
            state.last_calculation = datetime.now(timezone.utc)
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
                    "date": datetime.now(timezone.utc),
                    "fitness": 0.0,
                    "fatigue": 0.0,
                    "form": 0.0,
                    "daily_load": 0.0,
                    "recommendation": "NO_DATA",
                }

    def get_metrics_history(self, days: int = 30) -> List[Dict]:
        """Get metrics history for visualization."""
        with self.db.get_session() as session:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
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

    def track_performance_outcome(
        self,
        activity_id: str,
        recommended_type: str,
        recommended_load: float,
        perceived_effort: float,
        performance_quality: float = None,
        notes: str = None
    ):
        """Track performance outcome for model adaptation.

        Args:
            activity_id: Strava activity ID
            recommended_type: What was recommended (REST, EASY, etc.)
            recommended_load: Recommended training load
            perceived_effort: RPE 1-10
            performance_quality: Subjective quality 1-10
            notes: Optional notes about the session
        """
        with self.db.get_session() as session:
            # Get the activity
            activity = session.query(Activity).filter_by(strava_id=activity_id).first()
            if not activity:
                return

            # Get current model state
            current_state = self.get_current_state()

            # Calculate performance delta
            expected_load = recommended_load
            actual_load = activity.training_load * config.get_sport_load_multiplier(activity.type)
            load_compliance = 1.0 - abs(expected_load - actual_load) / (expected_load + 1e-5)

            # Estimate outcome score
            outcome_score = 0.5  # Neutral default
            if performance_quality:
                outcome_score = performance_quality / 10.0
            elif perceived_effort:
                # Infer from effort vs intensity
                if recommended_type in ["EASY", "RECOVERY"] and perceived_effort <= 4:
                    outcome_score = 0.8
                elif recommended_type in ["MODERATE"] and 4 <= perceived_effort <= 6:
                    outcome_score = 0.8
                elif recommended_type in ["HARD", "PEAK"] and perceived_effort >= 7:
                    outcome_score = 0.8
                else:
                    outcome_score = 0.4

            # Create performance outcome record
            outcome = PerformanceOutcome(
                user_id=self.user_id,
                activity_id=activity_id,
                date=activity.start_date,
                recommended_type=recommended_type,
                recommended_load=recommended_load,
                actual_type=activity.type,
                actual_load=actual_load,
                actual_sport=activity.type,
                perceived_effort=perceived_effort,
                performance_quality=performance_quality,
                compliance_score=load_compliance,
                outcome_score=outcome_score,
                fitness_at_recommendation=current_state.get("fitness", 0),
                fatigue_at_recommendation=current_state.get("fatigue", 0),
                form_at_recommendation=current_state.get("form", 0),
                notes=notes
            )
            session.add(outcome)
            session.commit()

            # Adapt model parameters if enabled
            if config.ENABLE_ADAPTIVE_PARAMETERS:
                performance_delta = outcome_score - 0.7  # Expected good outcome is 0.7
                self.model.adapt_parameters(
                    performance_delta=performance_delta,
                    perceived_effort=perceived_effort,
                    fatigue_level=current_state.get("fatigue", 0) / 10.0  # Normalize to 1-10
                )