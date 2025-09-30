"""
Performance Prediction Module

ML-based race performance prediction using historical training data.
Provides taper optimization and feature importance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import os
import warnings

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """
    Machine learning model for predicting race performance.

    Features:
    - Time series aware training
    - Hyperparameter optimization
    - Feature importance analysis
    - Taper strategy comparison
    """

    def __init__(self, db_session=None):
        """Initialize the performance predictor."""
        self.db = db_session
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_metadata = {}
        self.logger = logging.getLogger(__name__)

    def engineer_features(self,
                         target_date: datetime,
                         lookback_days: int = 90,
                         activities: List[Any] = None,
                         metrics: Any = None) -> Dict[str, float]:
        """
        Engineer features from training history for ML model.

        Args:
            target_date: Date to predict performance for
            lookback_days: Days of history to consider
            activities: Optional list of activities (for testing)
            metrics: Optional metrics data (for testing)

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # If no database session, return mock features for testing
        if self.db is None and activities is None:
            return self._get_mock_features()

        # Calculate date range
        end_date = target_date
        start_date = target_date - timedelta(days=lookback_days)

        # Training load features
        load_features = self._get_training_load_features(start_date, end_date, activities)
        features.update(load_features)

        # Volume and frequency features
        volume_features = self._get_volume_features(start_date, end_date, activities)
        features.update(volume_features)

        # Intensity distribution features
        intensity_features = self._get_intensity_distribution(start_date, end_date, activities)
        features.update(intensity_features)

        # Recovery and readiness features
        recovery_features = self._get_recovery_features(start_date, end_date, metrics)
        features.update(recovery_features)

        # Periodization features
        periodization_features = self._get_periodization_features(start_date, end_date)
        features.update(periodization_features)

        return features

    def _get_training_load_features(self,
                                   start_date: datetime,
                                   end_date: datetime,
                                   activities: List[Any] = None) -> Dict[str, float]:
        """Extract training load and fitness features."""
        features = {}

        if self.db is not None:
            # Query from database
            from ..db.models import Metric

            metrics = self.db.query(Metric).filter(
                Metric.date >= start_date,
                Metric.date <= end_date
            ).order_by(Metric.date).all()

            if metrics:
                ctl_values = [m.fitness for m in metrics if m.fitness]
                atl_values = [m.fatigue for m in metrics if m.fatigue]
                tsb_values = [m.form for m in metrics if m.form]
                load_values = [m.daily_load for m in metrics if m.daily_load]
            else:
                ctl_values = atl_values = tsb_values = load_values = []
        else:
            # Use provided activities or mock data
            ctl_values = [80, 82, 84, 85, 87, 89, 90]  # Mock CTL progression
            atl_values = [90, 88, 85, 83, 80, 78, 75]  # Mock ATL
            tsb_values = [-10, -6, -1, 2, 7, 11, 15]   # Mock TSB
            load_values = [100, 120, 80, 90, 110, 60, 40]  # Mock daily loads

        # Current fitness metrics
        features['ctl_current'] = ctl_values[-1] if ctl_values else 0
        features['atl_current'] = atl_values[-1] if atl_values else 0
        features['tsb_current'] = tsb_values[-1] if tsb_values else 0

        # Trends and changes
        if len(ctl_values) >= 7:
            features['ctl_7d_change'] = ctl_values[-1] - ctl_values[-7]
            features['atl_7d_change'] = atl_values[-1] - atl_values[-7]
            features['tsb_7d_avg'] = np.mean(tsb_values[-7:])
        else:
            features['ctl_7d_change'] = 0
            features['atl_7d_change'] = 0
            features['tsb_7d_avg'] = 0

        if len(ctl_values) >= 28:
            features['ctl_28d_change'] = ctl_values[-1] - ctl_values[-28]
            features['load_28d_avg'] = np.mean(load_values[-28:]) if load_values else 0
        else:
            features['ctl_28d_change'] = 0
            features['load_28d_avg'] = 0

        # Variability metrics
        if ctl_values and len(ctl_values) > 1:
            features['ctl_coefficient_variation'] = np.std(ctl_values) / (np.mean(ctl_values) + 0.001)
        else:
            features['ctl_coefficient_variation'] = 0

        # Peak values
        features['ctl_peak'] = max(ctl_values) if ctl_values else 0
        features['tsb_peak'] = max(tsb_values) if tsb_values else 0

        return features

    def _get_volume_features(self,
                            start_date: datetime,
                            end_date: datetime,
                            activities: List[Any] = None) -> Dict[str, float]:
        """Extract training volume and frequency features."""
        features = {}

        if self.db is not None and activities is None:
            # Query from database
            from ..db.models import Activity

            activities = self.db.query(Activity).filter(
                Activity.start_date >= start_date,
                Activity.start_date <= end_date
            ).all()

        if activities:
            # Calculate volume metrics
            total_hours = sum(a.elapsed_time / 3600 for a in activities if a.elapsed_time)
            total_distance = sum(a.distance / 1000 for a in activities if a.distance)
            activity_count = len(activities)

            # Weekly averages
            weeks = (end_date - start_date).days / 7
            features['weekly_hours_avg'] = total_hours / max(weeks, 1)
            features['weekly_distance_avg'] = total_distance / max(weeks, 1)
            features['weekly_sessions_avg'] = activity_count / max(weeks, 1)

            # Recent volume (last 14 days)
            recent_cutoff = end_date - timedelta(days=14)
            recent_activities = [a for a in activities if a.start_date >= recent_cutoff]

            if recent_activities:
                features['recent_hours'] = sum(a.elapsed_time / 3600 for a in recent_activities if a.elapsed_time)
                features['recent_distance'] = sum(a.distance / 1000 for a in recent_activities if a.distance)
                features['recent_sessions'] = len(recent_activities)
            else:
                features['recent_hours'] = 0
                features['recent_distance'] = 0
                features['recent_sessions'] = 0

        else:
            # Mock data for testing
            features['weekly_hours_avg'] = 8.5
            features['weekly_distance_avg'] = 60
            features['weekly_sessions_avg'] = 5
            features['recent_hours'] = 15
            features['recent_distance'] = 100
            features['recent_sessions'] = 8

        return features

    def _get_intensity_distribution(self,
                                   start_date: datetime,
                                   end_date: datetime,
                                   activities: List[Any] = None) -> Dict[str, float]:
        """Extract training intensity distribution features."""
        features = {}

        # Initialize intensity zones
        easy_hours = moderate_hours = hard_hours = 0

        if self.db is not None and activities is None:
            from ..db.models import Activity
            activities = self.db.query(Activity).filter(
                Activity.start_date >= start_date,
                Activity.start_date <= end_date
            ).all()

            for activity in activities:
                if activity.average_heartrate and activity.elapsed_time:
                    hours = activity.elapsed_time / 3600

                    # Estimate intensity based on HR (simplified)
                    if activity.average_heartrate < 140:
                        easy_hours += hours
                    elif activity.average_heartrate < 160:
                        moderate_hours += hours
                    else:
                        hard_hours += hours
        else:
            # Mock intensity distribution
            easy_hours = 25
            moderate_hours = 10
            hard_hours = 5

        total_hours = easy_hours + moderate_hours + hard_hours

        if total_hours > 0:
            features['intensity_easy_percent'] = (easy_hours / total_hours) * 100
            features['intensity_moderate_percent'] = (moderate_hours / total_hours) * 100
            features['intensity_hard_percent'] = (hard_hours / total_hours) * 100
        else:
            features['intensity_easy_percent'] = 80
            features['intensity_moderate_percent'] = 15
            features['intensity_hard_percent'] = 5

        # Polarization index (how polarized is training)
        features['polarization_index'] = features['intensity_easy_percent'] + features['intensity_hard_percent'] - features['intensity_moderate_percent']

        return features

    def _get_recovery_features(self,
                              start_date: datetime,
                              end_date: datetime,
                              metrics: Any = None) -> Dict[str, float]:
        """Extract recovery and readiness features."""
        features = {}

        if self.db is not None:
            # Query HRV data
            from ..db.models import HRVData
            hrv_data = self.db.query(HRVData).filter(
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date).all()

            if hrv_data:
                hrv_values = [h.hrv_rmssd for h in hrv_data if h.hrv_rmssd]
                if hrv_values:
                    features['hrv_avg'] = np.mean(hrv_values)
                    features['hrv_cv'] = np.std(hrv_values) / (np.mean(hrv_values) + 0.001)
                    features['hrv_trend'] = hrv_values[-1] - hrv_values[0] if len(hrv_values) > 1 else 0
                else:
                    features['hrv_avg'] = 50
                    features['hrv_cv'] = 0.1
                    features['hrv_trend'] = 0
            else:
                features['hrv_avg'] = 50
                features['hrv_cv'] = 0.1
                features['hrv_trend'] = 0

            # Query sleep data
            from ..db.models import SleepData
            sleep_data = self.db.query(SleepData).filter(
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).all()

            if sleep_data:
                sleep_scores = [s.sleep_score for s in sleep_data if s.sleep_score]
                features['sleep_avg'] = np.mean(sleep_scores) if sleep_scores else 75
            else:
                features['sleep_avg'] = 75
        else:
            # Mock recovery features
            features['hrv_avg'] = 55
            features['hrv_cv'] = 0.08
            features['hrv_trend'] = 2.5
            features['sleep_avg'] = 78

        return features

    def _get_periodization_features(self,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, float]:
        """Extract periodization and timing features."""
        features = {}

        # Days until target
        features['days_training'] = (end_date - start_date).days

        # Week of year (seasonality)
        features['week_of_year'] = end_date.isocalendar()[1]

        # Training age (mock for now)
        features['training_age_days'] = 365  # Would query first activity date

        return features

    def _get_mock_features(self) -> Dict[str, float]:
        """Return mock features for testing."""
        return {
            'ctl_current': 85,
            'atl_current': 75,
            'tsb_current': 10,
            'ctl_7d_change': 3,
            'atl_7d_change': -5,
            'tsb_7d_avg': 8,
            'ctl_28d_change': 12,
            'load_28d_avg': 90,
            'ctl_coefficient_variation': 0.15,
            'ctl_peak': 92,
            'tsb_peak': 15,
            'weekly_hours_avg': 8.5,
            'weekly_distance_avg': 60,
            'weekly_sessions_avg': 5,
            'recent_hours': 15,
            'recent_distance': 100,
            'recent_sessions': 8,
            'intensity_easy_percent': 75,
            'intensity_moderate_percent': 20,
            'intensity_hard_percent': 5,
            'polarization_index': 70,
            'hrv_avg': 55,
            'hrv_cv': 0.08,
            'hrv_trend': 2.5,
            'sleep_avg': 78,
            'days_training': 90,
            'week_of_year': 40,
            'training_age_days': 365
        }

    def prepare_training_dataset(self, min_races: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for model training from historical race results.

        Args:
            min_races: Minimum number of races required for training

        Returns:
            Features DataFrame and target Series
        """
        if self.db is None:
            # Return mock data for testing
            return self._get_mock_training_data()

        from ..db.models import Activity

        # Get race activities
        races = self.db.query(Activity).filter(
            Activity.type.in_(['Run', 'Ride']),
            Activity.elapsed_time.isnot(None),
            Activity.distance.isnot(None),
            Activity.distance > 5000  # At least 5km
        ).order_by(Activity.start_date).all()

        # Filter to likely races (high intensity)
        race_activities = []
        for activity in races:
            # Simple heuristic for race detection
            if activity.average_heartrate and activity.max_heartrate:
                intensity = activity.average_heartrate / activity.max_heartrate
                if intensity > 0.85:  # High intensity likely = race
                    race_activities.append(activity)

        if len(race_activities) < min_races:
            self.logger.warning(f"Only {len(race_activities)} races found, need {min_races}. Using mock data.")
            return self._get_mock_training_data()

        features_list = []
        targets = []

        for race in race_activities:
            # Engineer features for race day
            race_features = self.engineer_features(race.start_date)

            # Calculate performance metric (speed in km/h)
            speed_kmh = (race.distance / 1000) / (race.elapsed_time / 3600)

            features_list.append(race_features)
            targets.append(speed_kmh)

        features_df = pd.DataFrame(features_list)
        target_series = pd.Series(targets)

        return features_df, target_series

    def _get_mock_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate mock training data for testing."""
        # Create 10 mock race samples
        np.random.seed(42)
        n_samples = 10

        features_list = []
        targets = []

        for i in range(n_samples):
            features = self._get_mock_features()

            # Add some variation
            for key in features:
                features[key] *= np.random.uniform(0.8, 1.2)

            features_list.append(features)

            # Mock target (race speed in km/h)
            base_speed = 12  # 12 km/h = 5:00/km pace
            variation = np.random.uniform(-1, 1)
            targets.append(base_speed + variation)

        return pd.DataFrame(features_list), pd.Series(targets)

    def train_model(self,
                   optimize_hyperparams: bool = False,
                   model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train performance prediction model.

        Args:
            optimize_hyperparams: Whether to optimize hyperparameters
            model_type: Type of model to use

        Returns:
            Model metadata and performance metrics
        """
        # Prepare dataset
        X, y = self.prepare_training_dataset()

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Select model based on availability and preference
        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            base_model = xgb.XGBRegressor(random_state=42, n_estimators=100)
        elif model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(random_state=42, n_estimators=100)
        else:
            base_model = RandomForestRegressor(random_state=42, n_estimators=100)

        # Hyperparameter optimization if requested and available
        if optimize_hyperparams and OPTUNA_AVAILABLE:
            self.logger.info("Optimizing hyperparameters with Optuna...")
            best_params = self._optimize_hyperparameters(X, y, model_type)
            base_model.set_params(**best_params)

        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', self.scaler),
            ('model', base_model)
        ])

        # Train model
        self.model.fit(X, y)

        # Calculate validation metrics
        tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 3))
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error'
        )

        # Store metadata
        self.model_metadata = {
            'model_type': model_type,
            'mae': -cv_scores.mean(),
            'mae_std': cv_scores.std(),
            'r2_score': self.model.score(X, y),
            'feature_count': len(self.feature_names),
            'training_samples': len(X),
            'trained_date': datetime.now().isoformat(),
            'features': self.feature_names
        }

        self.logger.info(f"Model trained: MAE = {self.model_metadata['mae']:.2f} ± {self.model_metadata['mae_std']:.2f}")

        return self.model_metadata

    def _optimize_hyperparameters(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 model_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            if model_type == "xgboost" and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestRegressor(**params, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 3))
            scores = cross_val_score(
                pipeline, X, y,
                cv=tscv,
                scoring='neg_mean_absolute_error'
            )

            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        return study.best_params

    def predict_race_performance(self,
                                race_date: datetime,
                                race_distance_km: float) -> Dict[str, Any]:
        """
        Predict race performance for a given date and distance.

        Args:
            race_date: Date of the race
            race_distance_km: Race distance in kilometers

        Returns:
            Prediction results with confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Engineer features for race day
        race_features = self.engineer_features(race_date)

        # Convert to DataFrame
        feature_df = pd.DataFrame([race_features])[self.feature_names]

        # Predict base performance
        predicted_base_speed = self.model.predict(feature_df)[0]

        # Apply distance-based performance scaling using Riegel's formula and physiological constraints
        realistic_speed = self._apply_distance_scaling(predicted_base_speed, race_distance_km)

        # Calculate time from realistic speed
        predicted_time_hours = race_distance_km / realistic_speed
        predicted_time_seconds = predicted_time_hours * 3600

        # Format time
        hours = int(predicted_time_hours)
        minutes = int((predicted_time_hours - hours) * 60)
        seconds = int(((predicted_time_hours - hours) * 60 - minutes) * 60)

        # Calculate confidence (simplified - would use prediction intervals in production)
        confidence = min(0.95, 0.70 + (self.model_metadata.get('r2_score', 0.7) * 0.25))

        # Get feature importance if possible
        feature_importance = self._get_feature_importance(feature_df)

        return {
            'predicted_speed_kmh': realistic_speed,
            'predicted_time_hours': predicted_time_hours,
            'predicted_time_seconds': predicted_time_seconds,
            'predicted_time_formatted': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            'predicted_pace_per_km': f"{int(60/realistic_speed)}:{int((60/realistic_speed % 1) * 60):02d}",
            'confidence': confidence,
            'model_mae': self.model_metadata.get('mae', 0),
            'feature_importance': feature_importance
        }

    def _get_feature_importance(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance for the prediction."""
        importance_dict = {}

        try:
            # Get the model from pipeline
            model = self.model.named_steps['model']

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for feat, imp in zip(self.feature_names, importances):
                    importance_dict[feat] = float(imp)

                # Sort by importance and take top 5
                importance_dict = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])

        except Exception as e:
            self.logger.debug(f"Could not extract feature importance: {e}")

        return importance_dict

    def _apply_distance_scaling(self, base_speed_kmh: float, race_distance_km: float) -> float:
        """
        Apply distance-based performance scaling using Riegel's formula and physiological constraints.

        Args:
            base_speed_kmh: Base speed prediction from ML model
            race_distance_km: Target race distance

        Returns:
            Realistic speed adjusted for race distance
        """
        # Reference distance (10K) for scaling
        reference_distance = 10.0

        # If predicting 10K, return base speed with constraints
        if abs(race_distance_km - reference_distance) < 0.1:
            return self._apply_physiological_constraints(base_speed_kmh, race_distance_km)

        # Calculate reference 10K time from base speed
        reference_time_hours = reference_distance / base_speed_kmh
        reference_time_minutes = reference_time_hours * 60

        # Apply Riegel's formula: T₂ = T₁ × (D₂/D₁)^1.06
        riegel_exponent = 1.06
        distance_ratio = race_distance_km / reference_distance
        scaling_factor = distance_ratio ** riegel_exponent

        # Calculate target race time
        target_time_minutes = reference_time_minutes * scaling_factor
        target_time_hours = target_time_minutes / 60

        # Calculate speed from scaled time
        scaled_speed = race_distance_km / target_time_hours

        # Apply physiological constraints for realism
        realistic_speed = self._apply_physiological_constraints(scaled_speed, race_distance_km)

        # Apply conservative improvement limits for short-term predictions (4 weeks)
        # Limit improvements to realistic gains based on training adaptation
        conservative_speed = self._apply_improvement_limits(realistic_speed, race_distance_km)

        return conservative_speed

    def _apply_physiological_constraints(self, speed_kmh: float, distance_km: float) -> float:
        """
        Apply physiological constraints to ensure realistic performance predictions.

        Args:
            speed_kmh: Predicted speed
            distance_km: Race distance

        Returns:
            Speed adjusted for physiological realism
        """
        # Define realistic speed ranges for recreational athletes (km/h)
        # More conservative ranges based on actual performance data
        speed_constraints = {
            5.0: {'min': 9.0, 'max': 15.0},      # 5K: 20:00 - 33:20
            10.0: {'min': 8.0, 'max': 14.0},     # 10K: 42:51 - 75:00
            21.1: {'min': 7.0, 'max': 12.2},     # HM: 1:44 - 3:01 (conservative range)
            42.195: {'min': 6.0, 'max': 11.0},   # Marathon: 3:50 - 7:02
            50.0: {'min': 5.0, 'max': 9.5},      # 50K: 5:16 - 10:00
            75.0: {'min': 4.0, 'max': 8.0},      # 75K: 9:23 - 18:45
            100.0: {'min': 3.5, 'max': 7.0}      # 100K: 14:17 - 28:34
        }

        # Find closest distance for constraints
        closest_distance = min(speed_constraints.keys(),
                             key=lambda x: abs(x - distance_km))

        constraints = speed_constraints[closest_distance]

        # Apply constraints
        constrained_speed = max(constraints['min'],
                               min(constraints['max'], speed_kmh))

        return constrained_speed

    def _apply_improvement_limits(self, predicted_speed_kmh: float, distance_km: float) -> float:
        """
        Apply conservative improvement limits for short-term (4-week) predictions.

        Args:
            predicted_speed_kmh: Speed prediction from model
            distance_km: Race distance

        Returns:
            Speed with conservative improvement limits applied
        """
        # For HM distance, check against known recent performance
        if abs(distance_km - 21.1) < 0.1:  # Half Marathon
            # Recent HM time was 1:44 (1h 44min = 12.15 km/h)
            recent_hm_speed = 21.1 / (1 + 44/60)  # 12.15 km/h

            # Realistic 4-week improvement: 1-3% for recreational athletes
            max_improvement_factor = 1.03  # 3% improvement maximum
            max_realistic_speed = recent_hm_speed * max_improvement_factor

            # Use the more conservative of predicted vs realistic improvement
            return min(predicted_speed_kmh, max_realistic_speed)

        # For other distances, use actual performance data from user's activities
        # Query recent performance data from database for personalized baselines
        recent_performance_speed = self._get_recent_performance_baseline(distance_km)

        if recent_performance_speed:
            # Apply 2-3% improvement limit for 4-week prediction based on actual data
            max_improvement_factor = 1.03  # 3% improvement maximum
            max_realistic_speed = recent_performance_speed * max_improvement_factor

            # Return the more conservative prediction
            return min(predicted_speed_kmh, max_realistic_speed)
        else:
            # Fallback: if no recent data for this distance, allow model prediction
            # but cap at reasonable recreational athlete limits
            return predicted_speed_kmh

    def _get_recent_performance_baseline(self, target_distance_km: float) -> Optional[float]:
        """
        Get recent performance baseline from user's actual activities for the target distance.

        Args:
            target_distance_km: Target race distance

        Returns:
            Recent average speed for similar distances, or None if no data
        """
        if not self.db:
            return None

        try:
            from sqlalchemy import text
            from datetime import datetime, timedelta

            # Look for activities within similar distance range (±20%) from last 6 months
            distance_tolerance = 0.2  # 20% tolerance
            min_distance = target_distance_km * (1 - distance_tolerance)
            max_distance = target_distance_km * (1 + distance_tolerance)

            # Look back 6 months for recent performance
            cutoff_date = datetime.now() - timedelta(days=180)

            query = text("""
                SELECT distance, elapsed_time, activity_type
                FROM activities
                WHERE distance/1000 BETWEEN :min_dist AND :max_dist
                AND start_date >= :cutoff_date
                AND activity_type IN ('Run', 'Ride')
                AND elapsed_time > 0
                ORDER BY start_date DESC
                LIMIT 5
            """)

            result = self.db.execute(query, {
                'min_dist': min_distance,
                'max_dist': max_distance,
                'cutoff_date': cutoff_date
            }).fetchall()

            if not result:
                return None

            # Calculate average speed from recent similar-distance activities
            speeds = []
            for row in result:
                distance_m, elapsed_time_s, activity_type = row
                if distance_m and elapsed_time_s and elapsed_time_s > 0:
                    distance_km = distance_m / 1000
                    time_hours = elapsed_time_s / 3600
                    speed_kmh = distance_km / time_hours

                    # Only include reasonable speeds (filter out GPS errors)
                    if 3 <= speed_kmh <= 25:  # 3-25 km/h is reasonable range
                        speeds.append(speed_kmh)

            if speeds:
                # Return average of recent performances
                return sum(speeds) / len(speeds)

        except Exception as e:
            self.logger.debug(f"Could not query recent performance data: {e}")

        return None

    def optimize_taper(self,
                      race_date: datetime,
                      race_distance_km: float,
                      current_ctl: float = 80) -> Dict[str, Any]:
        """
        Compare different taper strategies and recommend optimal approach.

        Args:
            race_date: Date of the race
            race_distance_km: Race distance in kilometers
            current_ctl: Current chronic training load

        Returns:
            Taper strategy comparison and recommendations
        """
        strategies = {}

        # Define taper strategies with different TSB targets
        taper_configs = {
            'aggressive': {'target_tsb': 25, 'duration_days': 14, 'load_reduction': 0.5},
            'moderate': {'target_tsb': 15, 'duration_days': 10, 'load_reduction': 0.65},
            'minimal': {'target_tsb': 10, 'duration_days': 7, 'load_reduction': 0.75}
        }

        for name, config in taper_configs.items():
            # Simulate the taper effect on features
            taper_features = self.engineer_features(race_date)

            # Adjust features based on taper strategy
            taper_features['tsb_current'] = config['target_tsb']
            taper_features['atl_current'] = current_ctl * config['load_reduction']
            taper_features['recent_hours'] *= config['load_reduction']
            taper_features['recent_sessions'] *= config['load_reduction']

            # Predict with this taper
            feature_df = pd.DataFrame([taper_features])[self.feature_names]
            predicted_speed = self.model.predict(feature_df)[0]

            strategies[name] = {
                'predicted_speed_kmh': predicted_speed,
                'predicted_time_hours': race_distance_km / predicted_speed,
                'tsb_target': config['target_tsb'],
                'duration_days': config['duration_days'],
                'load_reduction': config['load_reduction']
            }

        # Find optimal strategy
        best_strategy = max(strategies.items(), key=lambda x: x[1]['predicted_speed_kmh'])
        worst_strategy = min(strategies.items(), key=lambda x: x[1]['predicted_speed_kmh'])

        return {
            'recommended_strategy': best_strategy[0],
            'all_strategies': strategies,
            'best_predicted_speed': best_strategy[1]['predicted_speed_kmh'],
            'improvement_over_worst': best_strategy[1]['predicted_speed_kmh'] - worst_strategy[1]['predicted_speed_kmh'],
            'optimal_tsb': best_strategy[1]['tsb_target'],
            'optimal_duration': best_strategy[1]['duration_days']
        }

    def save_model(self, filepath: str):
        """Save trained model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metadata': self.model_metadata
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_metadata = model_data['metadata']

        self.logger.info(f"Model loaded from {filepath}")

    def get_shap_values(self, X: pd.DataFrame) -> Optional[Any]:
        """
        Get SHAP values for model interpretability.

        Args:
            X: Features DataFrame

        Returns:
            SHAP values if available
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available for model interpretability")
            return None

        try:
            # Get the model from pipeline
            model = self.model.named_steps['model']

            # Create SHAP explainer
            explainer = shap.Explainer(model)

            # Calculate SHAP values
            shap_values = explainer(X)

            return shap_values

        except Exception as e:
            self.logger.error(f"Failed to calculate SHAP values: {e}")
            return None