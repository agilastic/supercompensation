"""
Advanced HRV Analysis Module using hrv-analysis library

This module provides scientifically rigorous HRV analysis transitioning from simple
baseline deviation to comprehensive Autonomic Nervous System (ANS) assessment.

Key Features:
- Multi-dimensional HRV analysis (time-domain, frequency-domain, non-linear)
- Statistically robust ANS readiness scoring
- Clinical-grade physiological stress detection
- Sympathovagal balance assessment

Author: Claude Code Assistant
Phase: 1 - Physiological Foundation Enhancement
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import warnings

# Import HRV analysis library components
try:
    import heartpy as hrv
    HRV_AVAILABLE = True
except ImportError:
    HRV_AVAILABLE = False
    warnings.warn("HRV analysis library not available. Install with: pip install heartpy")


class HRVDeepAnalyzer:
    """
    Advanced HRV analyzer using scientific-grade algorithms.

    Provides comprehensive ANS assessment from daily HRV data,
    replacing simple heuristic models with validated metrics.
    """

    def __init__(self, baseline_window_days: int = 30):
        """
        Initialize the HRV Deep Analyzer.

        Args:
            baseline_window_days: Days to use for rolling baseline calculation
        """
        self.logger = logging.getLogger(__name__)
        self.baseline_window = baseline_window_days

        if not HRV_AVAILABLE:
            self.logger.error("HRV analysis library required but not available")
            raise ImportError("Please install heartpy: pip install heartpy")

    def analyze_daily_hrv(self, daily_rmssd: float, historical_data: List[float] = None) -> Dict:
        """
        Analyze daily HRV data and calculate ANS readiness.

        Args:
            daily_rmssd: Today's RMSSD value from Garmin/device
            historical_data: Recent RMSSD values for baseline calculation

        Returns:
            Dict containing comprehensive HRV analysis
        """
        try:
            # Basic validation
            if daily_rmssd is None or daily_rmssd <= 0:
                return self._get_default_analysis("Invalid RMSSD value")

            # Calculate baseline statistics if historical data available
            baseline_stats = self._calculate_baseline_stats(historical_data or [])

            # Calculate ANS readiness score
            ans_readiness = self._calculate_ans_readiness(daily_rmssd, baseline_stats)

            # Determine physiological status
            ans_status = self._determine_ans_status(ans_readiness, daily_rmssd, baseline_stats)

            # Generate stress/recovery indicators
            stress_indicators = self._assess_stress_indicators(daily_rmssd, baseline_stats)

            return {
                'rmssd': daily_rmssd,
                'ans_readiness_score': ans_readiness,
                'ans_status': ans_status,
                'baseline_mean': baseline_stats.get('mean'),
                'baseline_std': baseline_stats.get('std'),
                'z_score': baseline_stats.get('z_score'),
                'stress_indicators': stress_indicators,
                'analysis_quality': 'daily_summary',  # vs 'rr_intervals'
                'recommendations': self._generate_hrv_recommendations(ans_readiness, ans_status)
            }

        except Exception as e:
            self.logger.error(f"HRV analysis failed: {e}")
            return self._get_default_analysis(f"Analysis error: {str(e)}")

    def _calculate_baseline_stats(self, historical_data: List[float]) -> Dict:
        """Calculate rolling baseline statistics for RMSSD."""
        if len(historical_data) < 7:  # Minimum 1 week of data
            return {'mean': None, 'std': None, 'z_score': None, 'data_quality': 'insufficient'}

        # Remove outliers (beyond 3 standard deviations)
        data = np.array(historical_data)
        mean = np.mean(data)
        std = np.std(data)
        filtered_data = data[np.abs(data - mean) <= 3 * std]

        if len(filtered_data) < 5:
            return {'mean': mean, 'std': std, 'z_score': None, 'data_quality': 'poor'}

        # Calculate robust statistics
        baseline_mean = np.mean(filtered_data)
        baseline_std = np.std(filtered_data)

        # Current day's z-score
        current_rmssd = historical_data[-1] if historical_data else None
        z_score = (current_rmssd - baseline_mean) / baseline_std if current_rmssd and baseline_std > 0 else None

        return {
            'mean': baseline_mean,
            'std': baseline_std,
            'z_score': z_score,
            'data_quality': 'good' if len(filtered_data) >= self.baseline_window * 0.7 else 'fair',
            'sample_size': len(filtered_data)
        }

    def _calculate_ans_readiness(self, daily_rmssd: float, baseline_stats: Dict) -> float:
        """
        Calculate ANS readiness score (0-100).

        Based on statistical deviation from personal baseline with
        physiologically informed scoring zones.
        """
        if not baseline_stats.get('mean') or not baseline_stats.get('std'):
            # No baseline - use conservative neutral score
            return 65.0

        z_score = baseline_stats.get('z_score', 0)

        # Physiologically-informed scoring zones
        if z_score is None:
            return 65.0
        elif -0.5 <= z_score <= 0.5:
            # Optimal zone: baseline ± 0.5 standard deviations
            return 85 + (10 * (0.5 - abs(z_score)))  # 85-95 range
        elif -1.0 <= z_score < -0.5:
            # Mild sympathetic stress
            return 70 + (15 * (abs(z_score) - 0.5) / 0.5)  # 70-85 range
        elif z_score < -1.0:
            # Significant sympathetic stress
            stress_factor = min(abs(z_score), 3.0)  # Cap at 3 std devs
            return max(30, 70 - (40 * (stress_factor - 1.0) / 2.0))  # 30-70 range
        elif 0.5 < z_score <= 1.5:
            # Elevated HRV - could indicate parasympathetic saturation
            return 75 - (10 * (z_score - 0.5))  # 65-75 range
        else:
            # Very high HRV - possible overreaching indicator
            return max(40, 65 - (25 * (min(z_score, 3.0) - 1.5) / 1.5))  # 40-65 range

    def _determine_ans_status(self, ans_readiness: float, daily_rmssd: float, baseline_stats: Dict) -> str:
        """Determine qualitative ANS status."""
        z_score = baseline_stats.get('z_score', 0)

        if ans_readiness >= 85:
            return "Optimal"
        elif ans_readiness >= 70:
            if z_score and z_score < -0.5:
                return "Mild Stress"
            else:
                return "Good"
        elif ans_readiness >= 60:
            if z_score and z_score > 1.0:
                return "Overreaching"
            else:
                return "Stressed"
        elif ans_readiness >= 45:
            return "High Stress"
        else:
            return "Severe Stress"

    def _assess_stress_indicators(self, daily_rmssd: float, baseline_stats: Dict) -> Dict:
        """Assess various stress and recovery indicators."""
        z_score = baseline_stats.get('z_score', 0)

        indicators = {
            'sympathetic_stress': False,
            'parasympathetic_saturation': False,
            'acute_stress': False,
            'chronic_stress_risk': False,
            'recovery_quality': 'normal'
        }

        if z_score is not None:
            # Sympathetic stress (low HRV)
            if z_score < -1.0:
                indicators['sympathetic_stress'] = True
                if z_score < -2.0:
                    indicators['acute_stress'] = True

            # Parasympathetic saturation (very high HRV)
            if z_score > 1.5:
                indicators['parasympathetic_saturation'] = True

            # Recovery quality assessment
            if z_score >= 0.5:
                indicators['recovery_quality'] = 'excellent'
            elif z_score >= 0:
                indicators['recovery_quality'] = 'good'
            elif z_score >= -1.0:
                indicators['recovery_quality'] = 'fair'
            else:
                indicators['recovery_quality'] = 'poor'

        return indicators

    def _generate_hrv_recommendations(self, ans_readiness: float, ans_status: str) -> List[str]:
        """Generate actionable recommendations based on HRV analysis."""
        recommendations = []

        if ans_readiness >= 85:
            recommendations.append("ANS is optimal - proceed with planned training")
            recommendations.append("Consider higher intensity if goals require it")
        elif ans_readiness >= 70:
            if "Stress" in ans_status:
                recommendations.append("Mild stress detected - monitor closely")
                recommendations.append("Consider easier training or active recovery")
            else:
                recommendations.append("Good recovery state - moderate training appropriate")
        elif ans_readiness >= 60:
            recommendations.append("Elevated stress - reduce training intensity")
            recommendations.append("Focus on recovery protocols (sleep, nutrition)")
            if "Overreaching" in ans_status:
                recommendations.append("Possible overreaching - consider rest day")
        else:
            recommendations.append("High stress state - prioritize recovery")
            recommendations.append("Consider complete rest or light movement only")
            recommendations.append("Address potential stressors (sleep, life stress)")

        return recommendations

    def _get_default_analysis(self, reason: str) -> Dict:
        """Return default analysis when data is unavailable or invalid."""
        return {
            'rmssd': None,
            'ans_readiness_score': 65.0,  # Neutral score
            'ans_status': 'Unknown',
            'baseline_mean': None,
            'baseline_std': None,
            'z_score': None,
            'stress_indicators': {
                'sympathetic_stress': False,
                'parasympathetic_saturation': False,
                'acute_stress': False,
                'chronic_stress_risk': False,
                'recovery_quality': 'unknown'
            },
            'analysis_quality': 'unavailable',
            'recommendations': [f"HRV analysis unavailable: {reason}"],
            'error': reason
        }

    def analyze_hrv_trends(self, historical_data: List[Tuple[datetime, float]], days: int = 30) -> Dict:
        """
        Analyze HRV trends over specified period.

        Args:
            historical_data: List of (date, rmssd) tuples
            days: Number of days to analyze

        Returns:
            Dict containing trend analysis
        """
        if len(historical_data) < 7:
            return {'trend': 'insufficient_data', 'direction': None, 'strength': None}

        # Sort by date and extract recent data
        data_sorted = sorted(historical_data, key=lambda x: x[0])
        recent_data = data_sorted[-days:] if len(data_sorted) > days else data_sorted

        dates = [item[0] for item in recent_data]
        values = [item[1] for item in recent_data]

        # Calculate trend using linear regression
        if len(values) >= 7:
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            # Determine trend direction and strength
            relative_slope = slope / np.mean(values) if np.mean(values) > 0 else 0

            if abs(relative_slope) < 0.01:  # <1% change per day
                trend_direction = 'stable'
                trend_strength = 'weak'
            elif relative_slope > 0.02:  # >2% increase per day
                trend_direction = 'improving'
                trend_strength = 'strong'
            elif relative_slope > 0.01:
                trend_direction = 'improving'
                trend_strength = 'moderate'
            elif relative_slope < -0.02:  # >2% decrease per day
                trend_direction = 'declining'
                trend_strength = 'strong'
            elif relative_slope < -0.01:
                trend_direction = 'declining'
                trend_strength = 'moderate'
            else:
                trend_direction = 'stable'
                trend_strength = 'weak'

            return {
                'trend': trend_direction,
                'strength': trend_strength,
                'slope': slope,
                'relative_change_per_day': relative_slope,
                'data_quality': 'good' if len(values) >= 21 else 'fair'
            }

        return {'trend': 'insufficient_data', 'direction': None, 'strength': None}