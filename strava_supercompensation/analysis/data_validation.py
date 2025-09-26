"""Data validation utilities for detecting and handling anomalous data spikes.

This module implements physiologically-informed validation for all training
and biometric data to prevent corrupted metrics from affecting analysis.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation check."""
    is_valid: bool
    reason: Optional[str] = None
    suggested_value: Optional[float] = None
    confidence: float = 1.0  # 0.0 to 1.0


class DataValidator:
    """Comprehensive data validator for training and biometric data."""

    # Physiological bounds for validation
    PHYSIOLOGICAL_BOUNDS = {
        # Training metrics
        'fitness': {'min': 10, 'max': 400, 'daily_change_max': 15},
        'fatigue': {'min': 5, 'max': 250, 'daily_change_max': 50},
        'form': {'min': -80, 'max': 120, 'daily_change_max': 30},
        'training_load': {'min': 0, 'max': 1200, 'daily_change_max': 200},

        # Body composition (adult ranges)
        'weight_kg': {'min': 40, 'max': 200, 'daily_change_max': 2.0},
        'body_fat_percent': {'min': 3, 'max': 50, 'daily_change_max': 5.0},
        'muscle_mass_kg': {'min': 20, 'max': 100, 'daily_change_max': 1.0},
        'bone_mass_kg': {'min': 1.5, 'max': 6.0, 'daily_change_max': 0.2},
        'visceral_fat': {'min': 1, 'max': 30, 'daily_change_max': 3.0},

        # Heart rate metrics (bpm)
        'resting_hr': {'min': 30, 'max': 120, 'daily_change_max': 15},
        'max_hr': {'min': 120, 'max': 220, 'daily_change_max': 5},
        'hrv_rmssd': {'min': 5, 'max': 200, 'daily_change_max': 30},

        # Activity metrics
        'duration_minutes': {'min': 1, 'max': 1440, 'daily_change_max': 600},
        'distance_km': {'min': 0.1, 'max': 1000, 'daily_change_max': 300},
        'elevation_gain_m': {'min': 0, 'max': 8000, 'daily_change_max': 4000},
    }

    def __init__(self):
        """Initialize the data validator."""
        self.validation_history = {}

    def validate_metric(
        self,
        metric_name: str,
        value: float,
        previous_values: List[float] = None,
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """Validate a single metric value.

        Args:
            metric_name: Name of the metric being validated
            value: Current value to validate
            previous_values: Recent historical values for trend analysis
            context: Additional context (age, gender, sport type, etc.)

        Returns:
            ValidationResult with validation outcome
        """
        if metric_name not in self.PHYSIOLOGICAL_BOUNDS:
            # Unknown metric - perform basic sanity checks
            if np.isnan(value) or np.isinf(value):
                return ValidationResult(False, "Invalid numeric value")
            return ValidationResult(True)

        bounds = self.PHYSIOLOGICAL_BOUNDS[metric_name]

        # Basic range check
        if value < bounds['min'] or value > bounds['max']:
            return ValidationResult(
                False,
                f"Value {value} outside physiological range [{bounds['min']}, {bounds['max']}]",
                suggested_value=np.clip(value, bounds['min'], bounds['max'])
            )

        # Trend analysis if historical data available
        if previous_values and len(previous_values) > 0:
            trend_result = self._validate_trend(metric_name, value, previous_values, bounds)
            if not trend_result.is_valid:
                return trend_result

        # Context-specific validation
        if context:
            context_result = self._validate_context(metric_name, value, context)
            if not context_result.is_valid:
                return context_result

        return ValidationResult(True)

    def _validate_trend(
        self,
        metric_name: str,
        value: float,
        previous_values: List[float],
        bounds: Dict[str, float]
    ) -> ValidationResult:
        """Validate value against recent trends."""
        if not previous_values:
            return ValidationResult(True)

        # Check for sudden spikes
        recent_value = previous_values[-1]
        daily_change = abs(value - recent_value)

        if daily_change > bounds['daily_change_max']:
            # Calculate median of recent values as suggested alternative
            median_recent = np.median(previous_values[-7:])  # Last week

            return ValidationResult(
                False,
                f"Sudden change of {daily_change:.1f} exceeds maximum daily change {bounds['daily_change_max']:.1f}",
                suggested_value=median_recent,
                confidence=0.7
            )

        # Check for impossible consecutive changes
        if len(previous_values) >= 2:
            changes = np.diff(previous_values[-3:] + [value])
            if np.any(np.abs(changes) > bounds['daily_change_max'] * 1.5):
                return ValidationResult(
                    False,
                    "Multiple consecutive extreme changes detected",
                    suggested_value=np.median(previous_values[-7:]),
                    confidence=0.8
                )

        return ValidationResult(True)

    def _validate_context(
        self,
        metric_name: str,
        value: float,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Apply context-specific validation rules."""

        # Body composition context validation
        if metric_name == 'body_fat_percent':
            gender = context.get('gender', 'unknown')
            age = context.get('age', 30)

            # Gender-specific ranges
            if gender == 'male':
                if value < 3 or value > 25:  # Athletic male range
                    return ValidationResult(
                        False,
                        f"Body fat {value}% unusual for athletic male",
                        confidence=0.6
                    )
            elif gender == 'female':
                if value < 10 or value > 32:  # Athletic female range
                    return ValidationResult(
                        False,
                        f"Body fat {value}% unusual for athletic female",
                        confidence=0.6
                    )

        # Training load context validation
        if metric_name == 'training_load':
            sport_type = context.get('sport_type', 'unknown')
            duration = context.get('duration_minutes', 60)

            # TSS per minute validation
            if duration > 0:
                tss_per_minute = value / duration
                if tss_per_minute > 3.0:  # Very high intensity
                    return ValidationResult(
                        False,
                        f"TSS/minute ratio {tss_per_minute:.1f} extremely high",
                        confidence=0.8
                    )
                elif tss_per_minute < 0.3:  # Very low intensity
                    return ValidationResult(
                        False,
                        f"TSS/minute ratio {tss_per_minute:.1f} suspiciously low",
                        confidence=0.5
                    )

        return ValidationResult(True)

    def validate_batch(
        self,
        data: Dict[str, List[float]],
        context: Dict[str, Any] = None
    ) -> Dict[str, List[ValidationResult]]:
        """Validate a batch of data points for multiple metrics.

        Args:
            data: Dictionary mapping metric names to lists of values
            context: Shared context for all validations

        Returns:
            Dictionary mapping metric names to validation results
        """
        results = {}

        for metric_name, values in data.items():
            metric_results = []

            for i, value in enumerate(values):
                # Use previous values for trend analysis
                previous_values = values[max(0, i-14):i] if i > 0 else []

                result = self.validate_metric(
                    metric_name,
                    value,
                    previous_values,
                    context
                )
                metric_results.append(result)

            results[metric_name] = metric_results

        return results

    def clean_data(
        self,
        data: Dict[str, List[float]],
        strategy: str = 'interpolate'
    ) -> Dict[str, List[float]]:
        """Clean invalid data points using specified strategy.

        Args:
            data: Raw data dictionary
            strategy: Cleaning strategy ('interpolate', 'remove', 'replace')

        Returns:
            Cleaned data dictionary
        """
        validation_results = self.validate_batch(data)
        cleaned_data = {}

        for metric_name, values in data.items():
            if metric_name not in validation_results:
                cleaned_data[metric_name] = values
                continue

            results = validation_results[metric_name]
            cleaned_values = []

            for i, (value, result) in enumerate(zip(values, results)):
                if result.is_valid:
                    cleaned_values.append(value)
                else:
                    # Apply cleaning strategy
                    if strategy == 'remove':
                        continue  # Skip invalid values
                    elif strategy == 'replace' and result.suggested_value is not None:
                        cleaned_values.append(result.suggested_value)
                        logger.warning(f"Replaced {metric_name}[{i}] = {value} with {result.suggested_value}: {result.reason}")
                    elif strategy == 'interpolate' and len(cleaned_values) > 0:
                        # Simple interpolation with next valid value
                        next_valid_idx = self._find_next_valid(results, i)
                        if next_valid_idx is not None and next_valid_idx < len(values):
                            interpolated = (cleaned_values[-1] + values[next_valid_idx]) / 2
                            cleaned_values.append(interpolated)
                            logger.warning(f"Interpolated {metric_name}[{i}] = {interpolated}: {result.reason}")
                        elif result.suggested_value is not None:
                            cleaned_values.append(result.suggested_value)
                        else:
                            cleaned_values.append(cleaned_values[-1] if cleaned_values else 0)

            cleaned_data[metric_name] = cleaned_values

        return cleaned_data

    def _find_next_valid(self, results: List[ValidationResult], start_idx: int) -> Optional[int]:
        """Find the next valid data point index."""
        for i in range(start_idx + 1, len(results)):
            if results[i].is_valid:
                return i
        return None

    def generate_validation_report(
        self,
        validation_results: Dict[str, List[ValidationResult]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""

        total_points = sum(len(results) for results in validation_results.values())
        invalid_points = sum(
            sum(1 for r in results if not r.is_valid)
            for results in validation_results.values()
        )

        # Per-metric summary
        metric_summary = {}
        for metric_name, results in validation_results.items():
            invalid_count = sum(1 for r in results if not r.is_valid)
            metric_summary[metric_name] = {
                'total_points': len(results),
                'invalid_points': invalid_count,
                'validity_rate': (len(results) - invalid_count) / len(results) if results else 1.0,
                'common_issues': self._summarize_issues([r for r in results if not r.is_valid])
            }

        return {
            'overall_validity_rate': (total_points - invalid_points) / total_points if total_points > 0 else 1.0,
            'total_points_validated': total_points,
            'total_invalid_points': invalid_points,
            'metric_summary': metric_summary,
            'recommendations': self._generate_recommendations(metric_summary)
        }

    def _summarize_issues(self, invalid_results: List[ValidationResult]) -> List[str]:
        """Summarize common validation issues."""
        if not invalid_results:
            return []

        issue_counts = {}
        for result in invalid_results:
            if result.reason:
                # Extract issue type from reason
                if 'outside physiological range' in result.reason:
                    issue_counts['range_violations'] = issue_counts.get('range_violations', 0) + 1
                elif 'sudden change' in result.reason or 'extreme changes' in result.reason:
                    issue_counts['trend_anomalies'] = issue_counts.get('trend_anomalies', 0) + 1
                else:
                    issue_counts['other'] = issue_counts.get('other', 0) + 1

        return [f"{issue_type}: {count}" for issue_type, count in issue_counts.items()]

    def _generate_recommendations(self, metric_summary: Dict[str, Any]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []

        for metric_name, summary in metric_summary.items():
            validity_rate = summary['validity_rate']

            if validity_rate < 0.8:
                recommendations.append(
                    f"Check data source for {metric_name} - {validity_rate:.1%} validity rate is concerning"
                )

            if 'trend_anomalies' in str(summary.get('common_issues', [])):
                recommendations.append(
                    f"Review {metric_name} measurement procedures - frequent sudden changes detected"
                )

        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")

        return recommendations