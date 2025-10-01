"""
Enhanced ML Features for Training Analysis

This module provides advanced machine learning capabilities:
1. LSTM models for time-series prediction (CTL/ATL/TSB forecasting)
2. Isolation Forest for anomaly detection (overtraining, illness, data errors)
3. Bayesian regression for uncertainty quantification
4. Model comparison framework (Banister vs ML predictions)
5. Automated model retraining system

Author: AI Training System
Date: 2025-10-01
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import joblib
import json
from pathlib import Path
import warnings

# Optional deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    date: datetime
    metric: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str  # 'overtraining', 'illness', 'data_error', 'normal'
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TrajectoryPrediction:
    """Predicted trajectory for CTL/ATL/TSB."""
    dates: List[datetime]
    ctl_predicted: List[float]
    atl_predicted: List[float]
    tsb_predicted: List[float]
    ctl_confidence_lower: List[float]
    ctl_confidence_upper: List[float]
    atl_confidence_lower: List[float]
    atl_confidence_upper: List[float]
    model_confidence: float  # Overall model confidence 0-1
    forecast_horizon_days: int


@dataclass
class ModelComparison:
    """Comparison between Banister model and ML predictions."""
    metric: str
    banister_value: float
    ml_value: float
    ml_confidence_interval: Tuple[float, float]
    difference: float
    percent_difference: float
    recommendation: str


class LSTMTrajectoryPredictor:
    """
    LSTM-based predictor for CTL/ATL/TSB trajectories.

    Uses sequential neural network to predict future training stress values
    based on historical patterns and current state.
    """

    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 14):
        """
        Initialize LSTM predictor.

        Args:
            sequence_length: Number of historical days to use for prediction
            forecast_horizon: Number of days to forecast into the future
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = ['ctl', 'atl', 'tsb', 'daily_load', 'ramp_rate']
        self.is_trained = False
        self.training_history = None
        self.logger = logging.getLogger(__name__)

        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. LSTM predictions will use fallback method.")

    def _build_model(self, n_features: int) -> Optional[Any]:
        """Build LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            return None

        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(64, activation='tanh', return_sequences=True,
                       input_shape=(self.sequence_length, n_features)),
            layers.Dropout(0.2),

            # Second LSTM layer
            layers.LSTM(32, activation='tanh', return_sequences=False),
            layers.Dropout(0.2),

            # Dense layers for prediction
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),

            # Output layer: predict CTL, ATL, TSB for next day
            layers.Dense(3)  # [CTL, ATL, TSB]
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_sequences(self,
                         data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for LSTM training.

        Args:
            data: DataFrame with columns: ctl, atl, tsb, daily_load, ramp_rate

        Returns:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples, 3) for [CTL, ATL, TSB]
        """
        # Ensure required columns exist
        required_cols = ['ctl', 'atl', 'tsb']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Add derived features if not present
        if 'daily_load' not in data.columns:
            data['daily_load'] = data.get('training_load', 0)

        if 'ramp_rate' not in data.columns:
            # Calculate ramp rate (7-day change in CTL)
            data['ramp_rate'] = data['ctl'].diff(7).fillna(0)

        # Select feature columns
        feature_data = data[self.feature_names].values

        # Normalize features
        feature_data_scaled = self.scaler_X.fit_transform(feature_data)

        # Create sequences
        X, y = [], []
        for i in range(len(feature_data_scaled) - self.sequence_length):
            X.append(feature_data_scaled[i:i + self.sequence_length])
            # Target is next day's CTL, ATL, TSB
            y.append(feature_data_scaled[i + self.sequence_length, :3])

        return np.array(X), np.array(y)

    def train(self,
             training_data: pd.DataFrame,
             validation_split: float = 0.2,
             epochs: int = 100,
             batch_size: int = 32,
             early_stopping_patience: int = 15) -> Dict[str, Any]:
        """
        Train LSTM model on historical data.

        Args:
            training_data: Historical CTL/ATL/TSB data
            validation_split: Fraction of data for validation
            epochs: Maximum training epochs
            batch_size: Training batch size
            early_stopping_patience: Early stopping patience

        Returns:
            Training history and metrics
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Using mock training.")
            self.is_trained = True
            return {'status': 'mock', 'message': 'TensorFlow not installed'}

        try:
            # Prepare sequences
            X, y = self.prepare_sequences(training_data)

            if len(X) < 50:
                raise ValueError(f"Insufficient data: need at least 50 samples, got {len(X)}")

            # Build model
            n_features = X.shape[2]
            self.model = self._build_model(n_features)

            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

            # Train model
            self.training_history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            self.is_trained = True

            # Calculate final metrics
            final_loss = self.training_history.history['loss'][-1]
            final_val_loss = self.training_history.history['val_loss'][-1]
            final_mae = self.training_history.history['mae'][-1]

            return {
                'status': 'success',
                'final_loss': float(final_loss),
                'final_val_loss': float(final_val_loss),
                'final_mae': float(final_mae),
                'epochs_trained': len(self.training_history.history['loss']),
                'samples_trained': len(X)
            }

        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def predict_trajectory(self,
                          current_data: pd.DataFrame,
                          forecast_days: int = None) -> TrajectoryPrediction:
        """
        Predict future CTL/ATL/TSB trajectory.

        Args:
            current_data: Recent historical data (last sequence_length days)
            forecast_days: Days to forecast (default: self.forecast_horizon)

        Returns:
            TrajectoryPrediction with forecasted values and confidence intervals
        """
        if forecast_days is None:
            forecast_days = self.forecast_horizon

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Fallback if TensorFlow not available
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return self._fallback_prediction(current_data, forecast_days)

        try:
            # Prepare input sequence
            X, _ = self.prepare_sequences(current_data)
            last_sequence = X[-1:] if len(X) > 0 else self.scaler_X.transform(
                current_data[self.feature_names].tail(self.sequence_length).values
            ).reshape(1, self.sequence_length, len(self.feature_names))

            # Forecast iteratively
            predictions = []
            current_sequence = last_sequence.copy()

            for _ in range(forecast_days):
                # Predict next day
                next_pred = self.model.predict(current_sequence, verbose=0)[0]
                predictions.append(next_pred)

                # Update sequence for next prediction
                # Append prediction and remove oldest timestep
                new_row = np.concatenate([next_pred, [0, 0]])  # Add placeholder for load and ramp_rate
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],
                    new_row.reshape(1, 1, -1)
                ], axis=1)

            # Denormalize predictions
            predictions = np.array(predictions)
            # Pad predictions to match feature count for inverse transform
            predictions_padded = np.concatenate([
                predictions,
                np.zeros((len(predictions), len(self.feature_names) - 3))
            ], axis=1)
            predictions_denorm = self.scaler_X.inverse_transform(predictions_padded)[:, :3]

            # Extract CTL, ATL, TSB
            ctl_pred = predictions_denorm[:, 0].tolist()
            atl_pred = predictions_denorm[:, 1].tolist()
            tsb_pred = predictions_denorm[:, 2].tolist()

            # Calculate confidence intervals (simple uncertainty estimation)
            # Use ±5% as uncertainty (should be replaced with proper uncertainty quantification)
            uncertainty = 0.05
            ctl_lower = [v * (1 - uncertainty) for v in ctl_pred]
            ctl_upper = [v * (1 + uncertainty) for v in ctl_pred]
            atl_lower = [v * (1 - uncertainty) for v in atl_pred]
            atl_upper = [v * (1 + uncertainty) for v in atl_pred]

            # Generate future dates
            last_date = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else datetime.now()
            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

            return TrajectoryPrediction(
                dates=future_dates,
                ctl_predicted=ctl_pred,
                atl_predicted=atl_pred,
                tsb_predicted=tsb_pred,
                ctl_confidence_lower=ctl_lower,
                ctl_confidence_upper=ctl_upper,
                atl_confidence_lower=atl_lower,
                atl_confidence_upper=atl_upper,
                model_confidence=0.85,  # Mock confidence
                forecast_horizon_days=forecast_days
            )

        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            return self._fallback_prediction(current_data, forecast_days)

    def _fallback_prediction(self,
                            current_data: pd.DataFrame,
                            forecast_days: int) -> TrajectoryPrediction:
        """Fallback prediction using simple exponential decay."""
        last_ctl = current_data['ctl'].iloc[-1]
        last_atl = current_data['atl'].iloc[-1]
        last_tsb = current_data['tsb'].iloc[-1]

        # Simple decay model
        ctl_decay = 0.95  # CTL decays slower
        atl_decay = 0.85  # ATL decays faster

        ctl_pred = [last_ctl * (ctl_decay ** i) for i in range(1, forecast_days + 1)]
        atl_pred = [last_atl * (atl_decay ** i) for i in range(1, forecast_days + 1)]
        tsb_pred = [ctl - atl for ctl, atl in zip(ctl_pred, atl_pred)]

        last_date = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else datetime.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

        return TrajectoryPrediction(
            dates=future_dates,
            ctl_predicted=ctl_pred,
            atl_predicted=atl_pred,
            tsb_predicted=tsb_pred,
            ctl_confidence_lower=[v * 0.9 for v in ctl_pred],
            ctl_confidence_upper=[v * 1.1 for v in ctl_pred],
            atl_confidence_lower=[v * 0.9 for v in atl_pred],
            atl_confidence_upper=[v * 1.1 for v in atl_pred],
            model_confidence=0.6,  # Lower confidence for fallback
            forecast_horizon_days=forecast_days
        )

    def save(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model (use .keras format for compatibility with Keras 3+)
        if TENSORFLOW_AVAILABLE and self.model is not None:
            self.model.save(str(filepath.with_suffix('.keras')))

        # Save scalers and metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'metadata': metadata
        }, str(filepath.with_suffix('.pkl')))

        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model from disk."""
        filepath = Path(filepath)

        # Load model (support both .keras and legacy .h5 formats)
        if TENSORFLOW_AVAILABLE:
            keras_path = filepath.with_suffix('.keras')
            h5_path = filepath.with_suffix('.h5')
            if keras_path.exists():
                self.model = keras.models.load_model(str(keras_path))
            elif h5_path.exists():
                # Try to load legacy .h5, but it may fail with new Keras
                try:
                    self.model = keras.models.load_model(str(h5_path))
                except Exception as e:
                    self.logger.warning(f"Failed to load legacy .h5 model: {e}. Please retrain with --train-lstm")
                    self.model = None

        # Load scalers and metadata
        data = joblib.load(str(filepath.with_suffix('.pkl')))
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        metadata = data['metadata']

        self.sequence_length = metadata['sequence_length']
        self.forecast_horizon = metadata['forecast_horizon']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']

        self.logger.info(f"Model loaded from {filepath}")


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detection for training data.

    Detects:
    - Overtraining patterns
    - Illness onset
    - Data entry errors
    - Unusual physiological responses
    """

    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.05-0.15 typical)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.thresholds = {}
        self.logger = logging.getLogger(__name__)

    def fit(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit anomaly detector on historical data.

        Args:
            training_data: Historical training data with features

        Returns:
            Fitting statistics
        """
        # Select features for anomaly detection
        feature_cols = [
            'ctl', 'atl', 'tsb', 'daily_load', 'ramp_rate',
            'hrv_rmssd', 'resting_hr', 'sleep_score', 'stress_avg'
        ]

        # Use only available features
        self.feature_names = [col for col in feature_cols if col in training_data.columns]

        if len(self.feature_names) < 3:
            raise ValueError(f"Need at least 3 features, got {len(self.feature_names)}")

        # Prepare data
        X = training_data[self.feature_names].fillna(method='ffill').fillna(method='bfill')
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Calculate feature-specific thresholds
        for feature in self.feature_names:
            values = training_data[feature].dropna()
            self.thresholds[feature] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'q95': float(values.quantile(0.95)),
                'q05': float(values.quantile(0.05))
            }

        return {
            'status': 'success',
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'features': self.feature_names
        }

    def detect_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """
        Detect anomalies in data.

        Args:
            data: Data to check for anomalies

        Returns:
            List of detected anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")

        # Prepare data
        X = data[self.feature_names].fillna(method='ffill').fillna(method='bfill')
        X_scaled = self.scaler.transform(X)

        # Predict anomalies
        predictions = self.model.predict(X_scaled)  # 1=normal, -1=anomaly
        scores = self.model.score_samples(X_scaled)  # Lower=more anomalous

        # Analyze anomalies
        anomalies = []
        for idx, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly detected
                date = data.index[idx] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

                # Determine anomaly type and severity
                anomaly_type, severity, recommendations = self._classify_anomaly(
                    data.iloc[idx], score
                )

                # Find primary anomalous metric
                row_values = X_scaled[idx]
                anomaly_metric_idx = np.argmax(np.abs(row_values))
                anomaly_metric = self.feature_names[anomaly_metric_idx]

                anomalies.append(AnomalyResult(
                    date=date,
                    metric=anomaly_metric,
                    value=float(data.iloc[idx][anomaly_metric]),
                    anomaly_score=float(score),
                    is_anomaly=True,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    recommendations=recommendations
                ))

        return anomalies

    def _classify_anomaly(self,
                         row: pd.Series,
                         score: float) -> Tuple[str, str, List[str]]:
        """
        Classify anomaly type and generate recommendations.

        Returns:
            (anomaly_type, severity, recommendations)
        """
        # Severity based on anomaly score
        if score < -0.5:
            severity = 'critical'
        elif score < -0.3:
            severity = 'high'
        elif score < -0.1:
            severity = 'medium'
        else:
            severity = 'low'

        # Classify anomaly type
        anomaly_type = 'unknown'
        recommendations = []

        # Check for overtraining signals
        if 'tsb' in row and row['tsb'] < -30:
            if 'hrv_rmssd' in row and 'hrv_rmssd' in self.thresholds:
                hrv_threshold = self.thresholds['hrv_rmssd']['mean'] - 2 * self.thresholds['hrv_rmssd']['std']
                if row.get('hrv_rmssd', float('inf')) < hrv_threshold:
                    anomaly_type = 'Overtraining Risk - High fatigue with low HRV'
                    recommendations = [
                        "Severe overtraining risk detected",
                        "Take 2-3 complete rest days immediately",
                        "Monitor HRV closely - target return to baseline",
                        "Consider medical consultation if symptoms persist"
                    ]

        # Check for illness onset
        if anomaly_type == 'unknown' and 'resting_hr' in row and 'resting_hr' in self.thresholds:
            rhr_threshold = self.thresholds['resting_hr']['mean'] + 2 * self.thresholds['resting_hr']['std']
            if row.get('resting_hr', 0) > rhr_threshold:
                if 'hrv_rmssd' in row and row.get('hrv_rmssd', float('inf')) < self.thresholds.get('hrv_rmssd', {}).get('mean', 0):
                    anomaly_type = 'Possible Illness - Elevated resting heart rate'
                    recommendations = [
                        "Possible illness onset detected (elevated RHR + low HRV)",
                        "Skip high-intensity training for 24-48 hours",
                        "Monitor body temperature and symptoms",
                        "Increase hydration and sleep"
                    ]

        # Check for extreme training load spikes
        if anomaly_type == 'unknown' and 'daily_load' in row:
            daily_load = row.get('daily_load', 0)
            if daily_load > 500:
                anomaly_type = 'Data Error - Extremely high training load'
                recommendations = [
                    f"Unusually high training load ({daily_load:.0f} TSS) - verify data accuracy",
                    "Check for duplicate activities or incorrect duration",
                    "Review activity sync logs"
                ]
            elif 'daily_load' in self.thresholds and daily_load > self.thresholds['daily_load']['q95'] * 2:
                anomaly_type = f'Very High Training Load - {daily_load:.0f} TSS (>95th percentile)'
                recommendations = [
                    f"Training load ({daily_load:.0f} TSS) is significantly higher than normal",
                    "Ensure adequate recovery over next 48 hours",
                    "Monitor fatigue levels closely"
                ]

        # Check for ramp rate issues (training load progression)
        if anomaly_type == 'unknown' and 'ramp_rate' in row and 'ramp_rate' in self.thresholds:
            ramp_rate = row.get('ramp_rate', 0)
            daily_load = row.get('daily_load', 0)

            # Context: Is today a rest day or active day?
            is_rest_day = daily_load < 10

            if is_rest_day and ramp_rate < -50:
                anomaly_type = 'Rest Day After Heavy Training - Sharp load decrease'
                recommendations = [
                    f"Rest day following heavy training block (load dropped {abs(ramp_rate):.0f} TSS)",
                    "This is normal recovery - anomaly will resolve when you resume training",
                    "Use this time for adaptation and recovery"
                ]
            elif ramp_rate > self.thresholds['ramp_rate']['q95']:
                anomaly_type = f'Rapid Fitness Gain - Training ramping up quickly'
                recommendations = [
                    f"Training load increasing rapidly (+{ramp_rate:.0f} TSS/week)",
                    "Monitor for signs of overreaching",
                    "Consider scheduling recovery week soon"
                ]
            elif ramp_rate < self.thresholds['ramp_rate']['q05']:
                if is_rest_day:
                    anomaly_type = 'Rest/Recovery Day - Load reduction detected'
                    recommendations = [
                        "Low training load compared to recent trend",
                        "Normal if this is a planned recovery day",
                        "Anomaly will clear when training resumes"
                    ]
                else:
                    anomaly_type = f'Detraining Risk - Significant load decrease'
                    recommendations = [
                        f"Training load decreasing ({ramp_rate:.0f} TSS/week)",
                        "Fitness may decline if this continues",
                        "Resume normal training volume if no injury/illness"
                    ]

        # Check for extreme fatigue (TSB)
        if anomaly_type == 'unknown' and 'tsb' in row:
            tsb = row.get('tsb', 0)
            if tsb < -50:
                anomaly_type = f'Extreme Fatigue - TSB at {tsb:.0f} (very negative)'
                recommendations = [
                    f"Training Stress Balance ({tsb:.0f}) indicates severe fatigue",
                    "Immediate recovery period recommended",
                    "Reduce training volume by 50% for next 3-5 days"
                ]
            elif tsb > 25:
                anomaly_type = f'High Freshness - TSB at {tsb:.0f} (very positive)'
                recommendations = [
                    f"Very fresh/tapered state (TSB: {tsb:.0f})",
                    "Good time for high-intensity work or racing",
                    "Consider if this taper was intentional"
                ]

        # Default recommendations if type not determined
        if anomaly_type == 'unknown':
            anomaly_type = 'Unusual Training Pattern - Statistical outlier detected'
            recommendations = [
                "Training metrics outside normal range for your history",
                "Monitor closely for next 2-3 days",
                "Consider extra recovery if feeling fatigued"
            ]

        return anomaly_type, severity, recommendations


class BayesianPerformancePredictor:
    """
    Gaussian Process Regression for uncertainty quantification.

    Provides probabilistic predictions with confidence intervals.
    """

    def __init__(self):
        """Initialize Bayesian predictor."""
        # Define kernel: combination of RBF (smooth patterns) and White (noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=0.1,
            normalize_y=True
        )
        self.scaler_X = StandardScaler()
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Fit Gaussian Process model.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Fitting statistics
        """
        X_scaled = self.scaler_X.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return {
            'status': 'success',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'kernel': str(self.model.kernel_)
        }

    def predict_with_uncertainty(self,
                                X: pd.DataFrame,
                                confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty quantification.

        Args:
            X: Features for prediction
            confidence_level: Confidence level for intervals (0.95 = 95%)

        Returns:
            Dictionary with predictions, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler_X.transform(X)
        y_pred, y_std = self.model.predict(X_scaled, return_std=True)

        # Calculate confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        return {
            'predictions': y_pred,
            'std_dev': y_std,
            'lower_bound': y_pred - z_score * y_std,
            'upper_bound': y_pred + z_score * y_std,
            'confidence_level': confidence_level
        }


def compare_models(banister_prediction: float,
                  ml_prediction: float,
                  ml_confidence_interval: Tuple[float, float],
                  metric_name: str = "Performance") -> ModelComparison:
    """
    Compare Banister model vs ML prediction.

    Args:
        banister_prediction: Prediction from Banister model
        ml_prediction: Prediction from ML model
        ml_confidence_interval: (lower, upper) bounds
        metric_name: Name of metric being compared

    Returns:
        ModelComparison result
    """
    difference = ml_prediction - banister_prediction
    percent_diff = (difference / banister_prediction * 100) if banister_prediction != 0 else 0

    # Generate recommendation
    if abs(percent_diff) < 5:
        recommendation = "Models agree - high confidence in prediction"
    elif ml_confidence_interval[0] <= banister_prediction <= ml_confidence_interval[1]:
        recommendation = "Banister within ML confidence interval - both models valid"
    elif percent_diff > 10:
        recommendation = "ML predicts better performance - consider ML model"
    elif percent_diff < -10:
        recommendation = "ML predicts worse performance - monitor closely"
    else:
        recommendation = "Models differ moderately - use average or conservative estimate"

    return ModelComparison(
        metric=metric_name,
        banister_value=banister_prediction,
        ml_value=ml_prediction,
        ml_confidence_interval=ml_confidence_interval,
        difference=difference,
        percent_difference=percent_diff,
        recommendation=recommendation
    )


# Auto-retraining utilities
class ModelRetrainingScheduler:
    """
    Manages automatic model retraining schedule.
    """

    def __init__(self, models_dir: str = "models/ml_enhanced"):
        """Initialize retraining scheduler."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.metadata_file = self.models_dir / "retraining_metadata.json"

    def should_retrain(self,
                      model_name: str,
                      retrain_interval_days: int = 7) -> bool:
        """Check if model should be retrained."""
        metadata = self._load_metadata()

        if model_name not in metadata:
            return True  # Never trained

        last_trained = datetime.fromisoformat(metadata[model_name]['last_trained'])
        days_since_training = (datetime.now() - last_trained).days

        return days_since_training >= retrain_interval_days

    def mark_trained(self, model_name: str, metrics: Dict[str, Any]):
        """Mark model as trained with timestamp and metrics."""
        metadata = self._load_metadata()
        metadata[model_name] = {
            'last_trained': datetime.now().isoformat(),
            'metrics': metrics
        }
        self._save_metadata(metadata)

    def _load_metadata(self) -> Dict:
        """Load retraining metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict):
        """Save retraining metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
