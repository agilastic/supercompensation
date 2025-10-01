"""
Comprehensive Correlation Analysis for Wellness-Performance Relationships

This module implements:
1. Multivariate correlation matrices with statistical significance
2. Time-lagged correlation analysis for leading indicators
3. Visualization-ready correlation data structures
4. Automatic insight generation from correlation patterns
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..db import get_db
from ..db.models import (
    Activity, Metric, HRVData, SleepData, WellnessData,
    BodyComposition, BloodPressure
)
from ..config import config


@dataclass
class CorrelationResult:
    """Represents a correlation between two variables."""
    variable_1: str
    variable_2: str
    correlation: float  # Pearson correlation coefficient
    p_value: float
    n_samples: int
    is_significant: bool
    lag_days: int = 0  # For time-lagged correlations
    interpretation: str = ""

    @property
    def strength(self) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"

    @property
    def direction(self) -> str:
        """Direction of correlation."""
        return "positive" if self.correlation > 0 else "negative"


@dataclass
class LeadingIndicator:
    """Represents a metric that predicts future performance."""
    indicator_metric: str
    target_metric: str
    optimal_lag_days: int
    correlation: float
    p_value: float
    predictive_power: str
    actionable_insight: str


class CorrelationCategory(Enum):
    """Categories of correlation analysis."""
    WELLNESS_PERFORMANCE = "wellness_performance"
    RECOVERY_ADAPTATION = "recovery_adaptation"
    TRAINING_RESPONSE = "training_response"
    HEALTH_PERFORMANCE = "health_performance"
    ENVIRONMENTAL_IMPACT = "environmental_impact"


class WellnessPerformanceCorrelations:
    """
    Comprehensive correlation analysis for wellness-performance relationships.

    Analyzes relationships between:
    - HRV metrics vs performance outcomes
    - Sleep quality vs training adaptation
    - Body composition vs power output
    - Stress levels vs fatigue/recovery
    - Training load vs health markers
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.logger = logging.getLogger(__name__)

    def analyze_all_correlations(
        self,
        days_back: int = 90,
        min_samples: int = 10,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis across all metrics.

        Args:
            days_back: Number of days of historical data to analyze
            min_samples: Minimum number of samples required for correlation
            significance_level: P-value threshold for statistical significance

        Returns:
            Dictionary containing correlation matrices and insights
        """
        self.logger.info(f"Starting comprehensive correlation analysis ({days_back} days)")

        # Load all data
        data = self._load_integrated_dataset(days_back)

        if len(data) < min_samples:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {min_samples} days of data, found {len(data)}',
                'data_points': len(data)
            }

        # Calculate correlation matrices by category
        results = {
            'analysis_period': {
                'days_back': days_back,
                'start_date': (datetime.now() - timedelta(days=days_back)).date().isoformat(),
                'end_date': datetime.now().date().isoformat(),
                'data_points': len(data)
            },
            'correlation_matrices': {},
            'significant_correlations': [],
            'leading_indicators': [],
            'actionable_insights': [],
            'category_analyses': {}
        }

        # 1. Wellness-Performance Correlations
        wellness_perf = self._analyze_wellness_performance(data, min_samples, significance_level)
        results['category_analyses']['wellness_performance'] = wellness_perf
        results['significant_correlations'].extend(wellness_perf['significant'])

        # 2. Recovery-Adaptation Correlations
        recovery_adapt = self._analyze_recovery_adaptation(data, min_samples, significance_level)
        results['category_analyses']['recovery_adaptation'] = recovery_adapt
        results['significant_correlations'].extend(recovery_adapt['significant'])

        # 3. Training Response Correlations
        training_resp = self._analyze_training_response(data, min_samples, significance_level)
        results['category_analyses']['training_response'] = training_resp
        results['significant_correlations'].extend(training_resp['significant'])

        # 4. Health-Performance Correlations
        health_perf = self._analyze_health_performance(data, min_samples, significance_level)
        results['category_analyses']['health_performance'] = health_perf
        results['significant_correlations'].extend(health_perf['significant'])

        # 5. Time-Lagged Correlations (Leading Indicators)
        leading = self._analyze_leading_indicators(data, min_samples, significance_level)
        results['leading_indicators'] = leading

        # Generate full correlation matrix
        results['correlation_matrices']['full'] = self._build_correlation_matrix(data)

        # Generate actionable insights
        results['actionable_insights'] = self._generate_insights(results)

        # Summary statistics
        results['summary'] = self._generate_summary(results)

        self.logger.info(f"Found {len(results['significant_correlations'])} significant correlations")

        return results

    def _load_integrated_dataset(self, days_back: int) -> pd.DataFrame:
        """
        Load and integrate all data sources into a single DataFrame.

        Returns DataFrame with columns:
        - date
        - Training metrics: ctl, atl, tsb, daily_load
        - HRV metrics: hrv_rmssd, hrv_score, resting_hr
        - Sleep metrics: sleep_score, deep_sleep_pct, sleep_efficiency
        - Wellness metrics: stress_avg, body_battery
        - Body composition: weight, body_fat_pct, water_pct
        - Health metrics: systolic, diastolic, hr_rest
        - Performance metrics: weekly_distance, weekly_hours
        """
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({'date': date_range})

        with self.db.get_session() as session:
            # Load training metrics (CTL/ATL/TSB)
            metrics = session.query(Metric).filter(
                Metric.date >= start_date,
                Metric.date <= end_date
            ).all()

            metrics_df = pd.DataFrame([{
                'date': pd.Timestamp(m.date.date()),
                'ctl': m.fitness,
                'atl': m.fatigue,
                'tsb': m.form,
                'daily_load': m.daily_load
            } for m in metrics])

            if not metrics_df.empty:
                df = df.merge(metrics_df, on='date', how='left')

            # Load HRV data
            hrv_data = session.query(HRVData).filter(
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).all()

            hrv_df = pd.DataFrame([{
                'date': pd.Timestamp(h.date.date()),
                'hrv_rmssd': h.hrv_rmssd,
                'hrv_score': h.hrv_score,
                'resting_hr': h.resting_heart_rate,
                'stress_level': h.stress_level
            } for h in hrv_data])

            if not hrv_df.empty:
                df = df.merge(hrv_df, on='date', how='left')

            # Load sleep data
            sleep_data = session.query(SleepData).filter(
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).all()

            sleep_df = pd.DataFrame([{
                'date': pd.Timestamp(s.date.date()),
                'sleep_score': s.sleep_score,
                'total_sleep_hours': s.total_sleep_time / 3600 if s.total_sleep_time else None,
                'deep_sleep_pct': s.deep_sleep_pct,
                'rem_sleep_pct': s.rem_sleep_pct,
                'sleep_efficiency': s.sleep_efficiency,
                'overnight_hrv': s.overnight_hrv
            } for s in sleep_data])

            if not sleep_df.empty:
                df = df.merge(sleep_df, on='date', how='left')

            # Load wellness data
            wellness_data = session.query(WellnessData).filter(
                WellnessData.date >= start_date,
                WellnessData.date <= end_date
            ).all()

            wellness_df = pd.DataFrame([{
                'date': pd.Timestamp(w.date.date()),
                'stress_avg': w.stress_avg,
                'body_battery_charged': w.body_battery_charged,
                'body_battery_drained': w.body_battery_drained,
                'total_steps': w.total_steps
            } for w in wellness_data])

            if not wellness_df.empty:
                df = df.merge(wellness_df, on='date', how='left')

            # Load body composition
            body_comp = session.query(BodyComposition).filter(
                BodyComposition.date >= start_date,
                BodyComposition.date <= end_date
            ).all()

            body_df = pd.DataFrame([{
                'date': pd.Timestamp(b.date.date()),
                'weight_kg': b.weight_kg,
                'body_fat_pct': b.body_fat_percent,
                'water_pct': b.water_percent,
                'muscle_mass_pct': b.muscle_mass_percent,
                'bmr': b.basal_metabolism_kcal
            } for b in body_comp])

            if not body_df.empty:
                df = df.merge(body_df, on='date', how='left')

            # Load blood pressure
            bp_data = session.query(BloodPressure).filter(
                BloodPressure.date >= start_date,
                BloodPressure.date <= end_date
            ).all()

            bp_df = pd.DataFrame([{
                'date': pd.Timestamp(b.date.date()),
                'systolic': b.systolic,
                'diastolic': b.diastolic,
                'bp_heart_rate': b.heart_rate
            } for b in bp_data])

            if not bp_df.empty:
                df = df.merge(bp_df, on='date', how='left')

            # Calculate derived performance metrics
            activities = session.query(Activity).filter(
                Activity.start_date >= start_date,
                Activity.start_date <= end_date
            ).all()

            # Aggregate weekly performance metrics
            for idx, row in df.iterrows():
                week_start = row['date'] - timedelta(days=7)
                week_activities = [a for a in activities
                                  if week_start <= pd.Timestamp(a.start_date.date()) <= row['date']]

                if week_activities:
                    df.at[idx, 'weekly_distance_km'] = sum(
                        a.distance / 1000 for a in week_activities if a.distance
                    )
                    df.at[idx, 'weekly_hours'] = sum(
                        a.moving_time / 3600 for a in week_activities if a.moving_time
                    )
                    df.at[idx, 'weekly_sessions'] = len(week_activities)

                    # Average intensity (simplified)
                    hr_activities = [a for a in week_activities if a.average_heartrate]
                    if hr_activities:
                        df.at[idx, 'avg_weekly_hr'] = np.mean([a.average_heartrate for a in hr_activities])

        return df

    def _analyze_wellness_performance(
        self,
        data: pd.DataFrame,
        min_samples: int,
        sig_level: float
    ) -> Dict[str, Any]:
        """
        Analyze correlations between wellness metrics and performance.

        Key relationships:
        - HRV vs CTL/ATL/TSB
        - Sleep quality vs training load capacity
        - Stress levels vs performance metrics
        """
        correlations = []

        # HRV vs Training Metrics
        hrv_training_pairs = [
            ('hrv_rmssd', 'ctl', 'HRV RMSSD vs Fitness (CTL)'),
            ('hrv_rmssd', 'atl', 'HRV RMSSD vs Fatigue (ATL)'),
            ('hrv_rmssd', 'tsb', 'HRV RMSSD vs Form (TSB)'),
            ('hrv_score', 'ctl', 'HRV Score vs Fitness (CTL)'),
            ('hrv_score', 'daily_load', 'HRV Score vs Daily Training Load'),
            ('resting_hr', 'atl', 'Resting HR vs Fatigue (ATL)'),
            ('resting_hr', 'tsb', 'Resting HR vs Form (TSB)'),
        ]

        for var1, var2, description in hrv_training_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        # Sleep vs Performance
        sleep_perf_pairs = [
            ('sleep_score', 'ctl', 'Sleep Quality vs Fitness Capacity'),
            ('sleep_score', 'daily_load', 'Sleep Quality vs Training Load'),
            ('total_sleep_hours', 'atl', 'Sleep Duration vs Fatigue'),
            ('deep_sleep_pct', 'tsb', 'Deep Sleep % vs Form'),
            ('sleep_efficiency', 'ctl', 'Sleep Efficiency vs Fitness'),
        ]

        for var1, var2, description in sleep_perf_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        # Stress vs Performance
        stress_perf_pairs = [
            ('stress_avg', 'atl', 'Stress Level vs Fatigue'),
            ('stress_avg', 'tsb', 'Stress Level vs Form'),
            ('stress_level', 'daily_load', 'HRV Stress vs Training Load'),
        ]

        for var1, var2, description in stress_perf_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        return {
            'category': 'Wellness-Performance',
            'correlations': [self._corr_to_dict(c) for c in correlations],
            'significant': [self._corr_to_dict(c) for c in correlations if c.is_significant],
            'count': len(correlations),
            'significant_count': sum(1 for c in correlations if c.is_significant)
        }

    def _analyze_recovery_adaptation(
        self,
        data: pd.DataFrame,
        min_samples: int,
        sig_level: float
    ) -> Dict[str, Any]:
        """
        Analyze recovery-adaptation relationships.

        Key relationships:
        - HRV recovery vs training adaptation
        - Sleep quality vs CTL progression rate
        - Body battery vs fatigue clearance
        """
        correlations = []

        # Calculate CTL/ATL change rates (7-day)
        if 'ctl' in data.columns:
            data['ctl_7d_change'] = data['ctl'].diff(7)
            data['atl_7d_change'] = data['atl'].diff(7)

        recovery_pairs = [
            ('hrv_rmssd', 'ctl_7d_change', 'HRV vs Fitness Adaptation Rate'),
            ('sleep_score', 'ctl_7d_change', 'Sleep Quality vs Fitness Gains'),
            ('overnight_hrv', 'atl_7d_change', 'Overnight HRV vs Fatigue Recovery'),
            ('body_battery_charged', 'tsb', 'Body Battery Recovery vs Form'),
            ('deep_sleep_pct', 'ctl_7d_change', 'Deep Sleep vs Training Adaptation'),
        ]

        for var1, var2, description in recovery_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        return {
            'category': 'Recovery-Adaptation',
            'correlations': [self._corr_to_dict(c) for c in correlations],
            'significant': [self._corr_to_dict(c) for c in correlations if c.is_significant],
            'count': len(correlations),
            'significant_count': sum(1 for c in correlations if c.is_significant)
        }

    def _analyze_training_response(
        self,
        data: pd.DataFrame,
        min_samples: int,
        sig_level: float
    ) -> Dict[str, Any]:
        """
        Analyze training response patterns.

        Key relationships:
        - Training load vs HRV response
        - Intensity distribution vs adaptation
        - Volume vs recovery metrics
        """
        correlations = []

        training_pairs = [
            ('daily_load', 'hrv_rmssd', 'Training Load vs Next-Day HRV'),
            ('daily_load', 'sleep_score', 'Training Load vs Sleep Quality'),
            ('daily_load', 'resting_hr', 'Training Load vs Resting HR'),
            ('weekly_distance_km', 'hrv_score', 'Weekly Volume vs HRV Score'),
            ('weekly_hours', 'stress_avg', 'Weekly Training Hours vs Stress'),
        ]

        for var1, var2, description in training_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        return {
            'category': 'Training-Response',
            'correlations': [self._corr_to_dict(c) for c in correlations],
            'significant': [self._corr_to_dict(c) for c in correlations if c.is_significant],
            'count': len(correlations),
            'significant_count': sum(1 for c in correlations if c.is_significant)
        }

    def _analyze_health_performance(
        self,
        data: pd.DataFrame,
        min_samples: int,
        sig_level: float
    ) -> Dict[str, Any]:
        """
        Analyze health markers vs performance.

        Key relationships:
        - Body composition vs power output
        - Blood pressure vs training capacity
        - Weight changes vs performance
        """
        correlations = []

        health_pairs = [
            ('weight_kg', 'ctl', 'Body Weight vs Fitness'),
            ('body_fat_pct', 'weekly_distance_km', 'Body Fat % vs Weekly Volume'),
            ('water_pct', 'tsb', 'Hydration % vs Form'),
            ('systolic', 'resting_hr', 'Blood Pressure vs Resting HR'),
            ('bp_heart_rate', 'atl', 'BP Heart Rate vs Fatigue'),
            ('muscle_mass_pct', 'ctl', 'Muscle Mass % vs Fitness'),
        ]

        for var1, var2, description in health_pairs:
            if var1 in data.columns and var2 in data.columns:
                corr_result = self._calculate_correlation(data, var1, var2, min_samples, sig_level)
                if corr_result:
                    corr_result.interpretation = description
                    correlations.append(corr_result)

        return {
            'category': 'Health-Performance',
            'correlations': [self._corr_to_dict(c) for c in correlations],
            'significant': [self._corr_to_dict(c) for c in correlations if c.is_significant],
            'count': len(correlations),
            'significant_count': sum(1 for c in correlations if c.is_significant)
        }

    def _analyze_leading_indicators(
        self,
        data: pd.DataFrame,
        min_samples: int,
        sig_level: float,
        max_lag_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Identify time-lagged correlations to find leading indicators.

        Example: Does sleep quality 2 days ago predict today's performance?
        """
        leading_indicators = []

        # Define potential leading indicator -> target metric pairs
        indicator_target_pairs = [
            ('hrv_rmssd', 'tsb', 'HRV predicts future form'),
            ('sleep_score', 'daily_load', 'Sleep quality predicts training capacity'),
            ('stress_avg', 'atl', 'Stress predicts fatigue accumulation'),
            ('resting_hr', 'tsb', 'Resting HR predicts form changes'),
            ('deep_sleep_pct', 'ctl', 'Deep sleep predicts fitness gains'),
            ('body_battery_charged', 'daily_load', 'Body battery predicts training readiness'),
        ]

        for indicator, target, description in indicator_target_pairs:
            if indicator not in data.columns or target not in data.columns:
                continue

            best_lag = 0
            best_corr = 0
            best_pval = 1.0

            # Test different lag periods
            for lag in range(1, max_lag_days + 1):
                # Shift indicator back by lag days
                lagged_data = data[[indicator, target]].copy()
                lagged_data[f'{indicator}_lag{lag}'] = lagged_data[indicator].shift(lag)

                # Calculate correlation
                corr_result = self._calculate_correlation(
                    lagged_data,
                    f'{indicator}_lag{lag}',
                    target,
                    min_samples,
                    sig_level
                )

                if corr_result and abs(corr_result.correlation) > abs(best_corr):
                    best_lag = lag
                    best_corr = corr_result.correlation
                    best_pval = corr_result.p_value

            if abs(best_corr) > 0.2 and best_pval < sig_level:
                # Determine predictive power
                abs_corr = abs(best_corr)
                if abs_corr >= 0.6:
                    predictive_power = "strong"
                elif abs_corr >= 0.4:
                    predictive_power = "moderate"
                else:
                    predictive_power = "weak"

                # Generate actionable insight
                insight = self._generate_leading_indicator_insight(
                    indicator, target, best_lag, best_corr, description
                )

                leading_indicators.append({
                    'indicator_metric': indicator,
                    'target_metric': target,
                    'optimal_lag_days': best_lag,
                    'correlation': round(best_corr, 3),
                    'p_value': round(best_pval, 4),
                    'predictive_power': predictive_power,
                    'description': description,
                    'actionable_insight': insight
                })

        return sorted(leading_indicators, key=lambda x: abs(x['correlation']), reverse=True)

    def _calculate_correlation(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        min_samples: int,
        sig_level: float
    ) -> Optional[CorrelationResult]:
        """
        Calculate Pearson correlation with statistical significance testing.
        """
        # Remove rows with missing values
        clean_data = data[[var1, var2]].dropna()

        if len(clean_data) < min_samples:
            return None

        # Calculate Pearson correlation
        corr_coef, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])

        return CorrelationResult(
            variable_1=var1,
            variable_2=var2,
            correlation=corr_coef,
            p_value=p_value,
            n_samples=len(clean_data),
            is_significant=p_value < sig_level
        )

    def _build_correlation_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Build full correlation matrix for all numeric columns.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr(method='pearson')

        # Calculate p-values for each pair
        n = len(data)
        p_values = pd.DataFrame(
            np.ones((len(numeric_cols), len(numeric_cols))),
            index=numeric_cols,
            columns=numeric_cols
        )

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i != j:
                    clean_data = data[[col1, col2]].dropna()
                    if len(clean_data) >= 3:
                        _, pval = stats.pearsonr(clean_data[col1], clean_data[col2])
                        p_values.iloc[i, j] = pval

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'p_value_matrix': p_values.to_dict(),
            'variables': numeric_cols
        }

    def _generate_leading_indicator_insight(
        self,
        indicator: str,
        target: str,
        lag: int,
        corr: float,
        description: str
    ) -> str:
        """Generate actionable insight from leading indicator."""
        direction = "improves" if corr > 0 else "worsens"

        insights = {
            'hrv_rmssd': f"Monitor HRV closely - changes appear {lag} days before form shifts",
            'sleep_score': f"Poor sleep today predicts reduced training capacity in {lag} days",
            'stress_avg': f"Elevated stress is an early warning ({lag} days) of fatigue buildup",
            'resting_hr': f"Rising RHR predicts form decline {lag} days in advance",
            'deep_sleep_pct': f"Deep sleep quality today influences fitness gains {lag} days later",
            'body_battery_charged': f"Body battery recovery predicts training readiness {lag} days ahead"
        }

        return insights.get(indicator, f"{indicator} {direction} {target} with {lag}-day lag")

    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from correlation analysis."""
        insights = []

        # Analyze significant correlations
        all_significant = results['significant_correlations']

        # Find strongest correlations
        if all_significant:
            sorted_corrs = sorted(all_significant, key=lambda x: abs(x['correlation']), reverse=True)

            # Top 3 strongest correlations
            for i, corr in enumerate(sorted_corrs[:3], 1):
                var1 = corr['variable_1']
                var2 = corr['variable_2']
                strength = corr['strength']
                direction = corr['direction']

                insights.append(
                    f"#{i} Strongest relationship: {corr['interpretation']} "
                    f"({strength} {direction} correlation: {corr['correlation']:.2f})"
                )

        # Analyze leading indicators
        leading = results['leading_indicators']
        if leading:
            for indicator in leading[:2]:  # Top 2
                insights.append(
                    f"Leading indicator: {indicator['actionable_insight']}"
                )

        # Category-specific insights
        for category, analysis in results['category_analyses'].items():
            if analysis['significant_count'] > 0:
                insights.append(
                    f"{category.replace('_', ' ').title()}: "
                    f"{analysis['significant_count']}/{analysis['count']} "
                    f"correlations statistically significant"
                )

        return insights

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_correlations = sum(
            analysis['count']
            for analysis in results['category_analyses'].values()
        )

        total_significant = sum(
            analysis['significant_count']
            for analysis in results['category_analyses'].values()
        )

        return {
            'total_correlations_tested': total_correlations,
            'significant_correlations': total_significant,
            'significance_rate': round(total_significant / total_correlations * 100, 1) if total_correlations > 0 else 0,
            'leading_indicators_found': len(results['leading_indicators']),
            'data_quality_score': self._calculate_data_quality(results)
        }

    def _calculate_data_quality(self, results: Dict[str, Any]) -> str:
        """Calculate overall data quality score."""
        data_points = results['analysis_period']['data_points']

        if data_points >= 60:
            return "excellent"
        elif data_points >= 30:
            return "good"
        elif data_points >= 14:
            return "fair"
        else:
            return "limited"

    def _corr_to_dict(self, corr: CorrelationResult) -> Dict[str, Any]:
        """Convert CorrelationResult to dictionary."""
        return {
            'variable_1': str(corr.variable_1),
            'variable_2': str(corr.variable_2),
            'correlation': float(round(corr.correlation, 3)),
            'p_value': float(round(corr.p_value, 4)),
            'n_samples': int(corr.n_samples),
            'is_significant': bool(corr.is_significant),
            'strength': str(corr.strength),
            'direction': str(corr.direction),
            'interpretation': str(corr.interpretation),
            'lag_days': int(corr.lag_days)
        }


def analyze_correlations(days_back: int = 90, user_id: str = "default") -> Dict[str, Any]:
    """
    Convenience function to run comprehensive correlation analysis.

    Args:
        days_back: Number of days to analyze
        user_id: User identifier

    Returns:
        Complete correlation analysis results
    """
    analyzer = WellnessPerformanceCorrelations(user_id=user_id)
    return analyzer.analyze_all_correlations(days_back=days_back)
