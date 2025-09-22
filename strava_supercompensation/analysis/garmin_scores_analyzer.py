"""Garmin device scores analysis for training readiness assessment.

This analyzer takes direct scores from Garmin devices (HRV score, sleep score, stress level)
and calculates a simple weighted readiness score without baseline analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..db import get_db
from ..db.models import HRVData, SleepData, WellnessData


class GarminScoresAnalyzer:
    """Analyze Garmin device scores for training readiness assessment."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

    def get_readiness_score(self, date: datetime = None) -> Dict[str, any]:
        """Calculate training readiness score based on wellness metrics.

        Args:
            date: Date to analyze (defaults to today)

        Returns:
            Dictionary with readiness score and contributing factors
        """
        if date is None:
            date = datetime.utcnow()

        # Get latest wellness data within 2 days of target date
        date_window_start = date - timedelta(days=1)
        date_window_end = date + timedelta(days=1)

        with self.db.get_session() as session:
            # Get HRV data
            hrv_data = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= date_window_start,
                HRVData.date <= date_window_end
            ).order_by(HRVData.date.desc()).first()

            # Get sleep data (night before the target date)
            sleep_date_start = date - timedelta(days=1)
            sleep_date_end = date
            sleep_data = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= sleep_date_start,
                SleepData.date <= sleep_date_end
            ).order_by(SleepData.date.desc()).first()

            # Get wellness data
            wellness_data = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= date_window_start,
                WellnessData.date <= date_window_end
            ).order_by(WellnessData.date.desc()).first()

        # Calculate component scores (0-100 scale)
        components = {}
        overall_score = None

        # HRV Score (direct from Garmin, 0-100)
        if hrv_data and hrv_data.hrv_score is not None:
            components['hrv'] = {
                'score': hrv_data.hrv_score,
                'weight': 0.35,
                'status': hrv_data.hrv_status or 'unknown',
                'value': hrv_data.hrv_rmssd
            }

        # Sleep Score (direct from Garmin, 0-100)
        if sleep_data and sleep_data.sleep_score is not None:
            components['sleep'] = {
                'score': sleep_data.sleep_score,
                'weight': 0.30,
                'duration_hours': sleep_data.total_sleep_time / 3600 if sleep_data.total_sleep_time else None,
                'efficiency': sleep_data.sleep_efficiency
            }

        # Stress Score (inverted from Garmin, 0-100 -> 100-0)
        if wellness_data and wellness_data.stress_avg is not None:
            stress_score = max(0, 100 - wellness_data.stress_avg)  # Invert stress to readiness
            components['stress'] = {
                'score': stress_score,
                'weight': 0.20,
                'avg_stress': wellness_data.stress_avg,
                'qualifier': wellness_data.stress_qualifier
            }

        # Body Battery Score (use drained vs charged ratio)
        if wellness_data and wellness_data.body_battery_charged and wellness_data.body_battery_drained:
            if wellness_data.body_battery_drained > 0:
                battery_ratio = wellness_data.body_battery_charged / wellness_data.body_battery_drained
                # Convert ratio to 0-100 scale (optimal ratio around 1.0)
                battery_score = min(100, max(0, 50 + (battery_ratio - 1.0) * 50))
            else:
                battery_score = 100  # No drain is perfect

            components['body_battery'] = {
                'score': battery_score,
                'weight': 0.15,
                'charged': wellness_data.body_battery_charged,
                'drained': wellness_data.body_battery_drained,
                'highest': wellness_data.body_battery_highest,
                'lowest': wellness_data.body_battery_lowest
            }

        # Calculate weighted overall score if we have any components
        if components:
            total_weight = sum(comp['weight'] for comp in components.values())
            if total_weight > 0:
                weighted_sum = sum(comp['score'] * comp['weight'] for comp in components.values())
                overall_score = weighted_sum / total_weight

        return {
            'readiness_score': overall_score,
            'components': components,
            'date': date,
            'data_quality': self._assess_data_quality(components),
            'recommendations': self._generate_readiness_recommendations(overall_score, components)
        }

    def get_recovery_status(self, days: int = 7) -> Dict[str, any]:
        """Assess recovery status over recent days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with recovery analysis
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        with self.db.get_session() as session:
            # Get HRV trend
            hrv_records = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date).all()

            # Get sleep trend
            sleep_records = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).order_by(SleepData.date).all()

            # Get stress trend
            wellness_records = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= start_date,
                WellnessData.date <= end_date
            ).order_by(WellnessData.date).all()

        # Analyze trends
        trends = {}

        # HRV trend analysis
        if hrv_records and len(hrv_records) >= 3:
            hrv_scores = [r.hrv_score for r in hrv_records if r.hrv_score is not None]
            if len(hrv_scores) >= 3:
                # Linear trend analysis (simple slope)
                x = np.arange(len(hrv_scores))
                slope = np.polyfit(x, hrv_scores, 1)[0] if len(hrv_scores) > 1 else 0
                trends['hrv'] = {
                    'direction': 'improving' if slope > 0.5 else 'declining' if slope < -0.5 else 'stable',
                    'slope': slope,
                    'current': hrv_scores[-1],
                    'average': np.mean(hrv_scores),
                    'recent_average': np.mean(hrv_scores[-3:]) if len(hrv_scores) >= 3 else np.mean(hrv_scores)
                }

        # Sleep trend analysis
        if sleep_records and len(sleep_records) >= 3:
            sleep_scores = [r.sleep_score for r in sleep_records if r.sleep_score is not None]
            if len(sleep_scores) >= 3:
                x = np.arange(len(sleep_scores))
                slope = np.polyfit(x, sleep_scores, 1)[0] if len(sleep_scores) > 1 else 0
                trends['sleep'] = {
                    'direction': 'improving' if slope > 0.5 else 'declining' if slope < -0.5 else 'stable',
                    'slope': slope,
                    'current': sleep_scores[-1],
                    'average': np.mean(sleep_scores),
                    'recent_average': np.mean(sleep_scores[-3:]) if len(sleep_scores) >= 3 else np.mean(sleep_scores)
                }

        # Stress trend analysis (lower stress = better recovery)
        if wellness_records and len(wellness_records) >= 3:
            stress_values = [r.stress_avg for r in wellness_records if r.stress_avg is not None]
            if len(stress_values) >= 3:
                x = np.arange(len(stress_values))
                slope = np.polyfit(x, stress_values, 1)[0] if len(stress_values) > 1 else 0
                trends['stress'] = {
                    'direction': 'improving' if slope < -0.5 else 'declining' if slope > 0.5 else 'stable',
                    'slope': slope,
                    'current': stress_values[-1],
                    'average': np.mean(stress_values),
                    'recent_average': np.mean(stress_values[-3:]) if len(stress_values) >= 3 else np.mean(stress_values)
                }

        # Overall recovery assessment
        recovery_status = self._assess_overall_recovery(trends)

        return {
            'recovery_status': recovery_status,
            'trends': trends,
            'period_days': days,
            'data_points': {
                'hrv': len(hrv_records),
                'sleep': len(sleep_records),
                'wellness': len(wellness_records)
            }
        }

    def get_wellness_insights(self, days: int = 14) -> Dict[str, any]:
        """Generate wellness insights and patterns.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with wellness insights
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        insights = {
            'period_days': days,
            'patterns': {},
            'correlations': {},
            'recommendations': []
        }

        with self.db.get_session() as session:
            # Get all wellness data
            wellness_records = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= start_date,
                WellnessData.date <= end_date
            ).order_by(WellnessData.date).all()

            sleep_records = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).order_by(SleepData.date).all()

            hrv_records = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date).all()

        # Pattern analysis
        if wellness_records:
            stress_values = [r.stress_avg for r in wellness_records if r.stress_avg is not None]
            if stress_values:
                insights['patterns']['stress'] = {
                    'average': np.mean(stress_values),
                    'max': max(stress_values),
                    'min': min(stress_values),
                    'high_stress_days': len([s for s in stress_values if s > 50]),
                    'low_stress_days': len([s for s in stress_values if s < 25])
                }

        if sleep_records:
            sleep_durations = [r.total_sleep_time / 3600 for r in sleep_records if r.total_sleep_time]
            sleep_scores = [r.sleep_score for r in sleep_records if r.sleep_score is not None]

            if sleep_durations:
                insights['patterns']['sleep_duration'] = {
                    'average_hours': np.mean(sleep_durations),
                    'min_hours': min(sleep_durations),
                    'max_hours': max(sleep_durations),
                    'nights_under_7h': len([d for d in sleep_durations if d < 7]),
                    'nights_over_9h': len([d for d in sleep_durations if d > 9])
                }

            if sleep_scores:
                insights['patterns']['sleep_quality'] = {
                    'average_score': np.mean(sleep_scores),
                    'poor_nights': len([s for s in sleep_scores if s < 60]),
                    'excellent_nights': len([s for s in sleep_scores if s > 85])
                }

        # Generate recommendations based on patterns
        insights['recommendations'] = self._generate_wellness_recommendations(insights['patterns'])

        return insights

    def _assess_data_quality(self, components: Dict) -> str:
        """Assess quality of available wellness data."""
        component_count = len(components)

        if component_count >= 3:
            return 'excellent'
        elif component_count >= 2:
            return 'good'
        elif component_count >= 1:
            return 'limited'
        else:
            return 'insufficient'

    def _generate_readiness_recommendations(self, readiness_score: Optional[float], components: Dict) -> List[str]:
        """Generate training recommendations based on readiness score."""
        recommendations = []

        if readiness_score is None:
            recommendations.append("Insufficient wellness data for personalized recommendations")
            return recommendations

        # Overall readiness recommendations
        if readiness_score >= 80:
            recommendations.append("Excellent readiness - suitable for high intensity training")
        elif readiness_score >= 65:
            recommendations.append("Good readiness - moderate to high intensity training appropriate")
        elif readiness_score >= 50:
            recommendations.append("Moderate readiness - focus on easy to moderate intensity")
        elif readiness_score >= 35:
            recommendations.append("Low readiness - consider easy training or active recovery")
        else:
            recommendations.append("Poor readiness - prioritize rest and recovery")

        # Component-specific recommendations
        if 'hrv' in components and components['hrv']['score'] < 40:
            recommendations.append("Low HRV detected - reduce training intensity and focus on recovery")

        if 'sleep' in components and components['sleep']['score'] < 60:
            recommendations.append("Poor sleep quality - prioritize sleep hygiene and consider lighter training")

        if 'stress' in components and components['stress']['score'] < 40:  # Remember stress is inverted
            recommendations.append("High stress levels - incorporate stress management and reduce training load")

        if 'body_battery' in components and components['body_battery']['score'] < 50:
            recommendations.append("Low energy reserves - ensure adequate nutrition and rest")

        return recommendations

    def _assess_overall_recovery(self, trends: Dict) -> str:
        """Assess overall recovery status from trends."""
        if not trends:
            return 'insufficient_data'

        improving_count = 0
        declining_count = 0
        stable_count = 0

        for trend_name, trend_data in trends.items():
            direction = trend_data['direction']
            if direction == 'improving':
                improving_count += 1
            elif direction == 'declining':
                declining_count += 1
            else:
                stable_count += 1

        total_trends = len(trends)

        if improving_count >= total_trends * 0.6:
            return 'recovering_well'
        elif declining_count >= total_trends * 0.6:
            return 'declining'
        elif stable_count >= total_trends * 0.6:
            return 'stable'
        else:
            return 'mixed_signals'

    def _generate_wellness_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on wellness patterns."""
        recommendations = []

        # Sleep recommendations
        if 'sleep_duration' in patterns:
            avg_sleep = patterns['sleep_duration']['average_hours']
            if avg_sleep < 7:
                recommendations.append(f"Average sleep ({avg_sleep:.1f}h) is below recommended 7-9 hours")
            elif avg_sleep > 9.5:
                recommendations.append(f"Average sleep ({avg_sleep:.1f}h) is unusually high - consider sleep quality assessment")

            if patterns['sleep_duration']['nights_under_7h'] > patterns['sleep_duration'].get('total_nights', 7) * 0.3:
                recommendations.append("Too many nights with insufficient sleep - prioritize sleep schedule")

        if 'sleep_quality' in patterns:
            if patterns['sleep_quality']['poor_nights'] > 0:
                recommendations.append("Consider sleep hygiene improvements for better sleep quality")

        # Stress recommendations
        if 'stress' in patterns:
            if patterns['stress']['average'] > 50:
                recommendations.append("Elevated stress levels - consider stress management techniques")
            if patterns['stress']['high_stress_days'] > patterns['stress'].get('total_days', 14) * 0.4:
                recommendations.append("Frequent high stress days - evaluate training load and life stressors")

        return recommendations

    def get_training_load_modifier(self, date: datetime = None) -> float:
        """Get a training load modifier based on wellness data.

        Args:
            date: Date to assess (defaults to today)

        Returns:
            Multiplier for training load (0.5 = reduce by 50%, 1.0 = normal, 1.2 = can increase)
        """
        readiness = self.get_readiness_score(date)

        if readiness['readiness_score'] is None:
            return 1.0  # No adjustment without data

        score = readiness['readiness_score']

        # Convert readiness score to training load modifier
        if score >= 80:
            return 1.2  # Can handle 20% more load
        elif score >= 65:
            return 1.0  # Normal load
        elif score >= 50:
            return 0.85  # Reduce load by 15%
        elif score >= 35:
            return 0.7   # Reduce load by 30%
        else:
            return 0.5   # Reduce load by 50%


def get_garmin_scores_analyzer(user_id: str = "default") -> GarminScoresAnalyzer:
    """Get Garmin scores analyzer for user."""
    return GarminScoresAnalyzer(user_id)

# Legacy function for backward compatibility
def get_wellness_analyzer(user_id: str = "default") -> GarminScoresAnalyzer:
    """Legacy function - use get_garmin_scores_analyzer instead."""
    return GarminScoresAnalyzer(user_id)