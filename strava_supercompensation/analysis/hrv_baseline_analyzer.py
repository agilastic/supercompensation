"""
HRV Baseline Analysis using German Sports Science

Calculates individual HRV baseline from personal history and analyzes current deviation.
Uses German sports science methodology for autonomic nervous system assessment:
- Personal HRV baseline calculation (30-day rolling average)
- Baseline deviation analysis
- Autonomic nervous system balance assessment
- Training readiness classification

Based on: "Sport: Das Lehrbuch für das Sportstudium" - Güllich & Krüger
Reference: Belastungs-Beanspruchungs-Konzept (Load-Stress Concept)

Author: Sport Medicine Professional & Olympic Trainer
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..db import get_db
from ..db.models import HRVData, SleepData, WellnessData


class ReadinessLevel(Enum):
    """Training readiness classification"""
    EXCELLENT = "excellent"     # 85-100: Ready for high-intensity training
    GOOD = "good"              # 70-84: Ready for moderate-high training
    MODERATE = "moderate"      # 55-69: Light-moderate training only
    POOR = "poor"              # 40-54: Recovery/easy training only
    CRITICAL = "critical"      # 0-39: Rest required


@dataclass
class HRVBaseline:
    """Individual HRV baseline parameters"""
    mean_rmssd: float
    std_deviation: float
    coefficient_variation: float
    lower_threshold: float      # Mean - 1 SD
    upper_threshold: float      # Mean + 1 SD
    training_readiness_zone: Tuple[float, float]  # Optimal training range


class HRVBaselineAnalyzer:
    """
    HRV baseline analysis implementing German sports science methodology.

    Calculates individual HRV baseline and analyzes current deviation for
    autonomic nervous system balance and training readiness assessment.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

        # German sports science weighting (based on research)
        self.component_weights = {
            'hrv_baseline_deviation': 0.30,    # HRV deviation from baseline
            'autonomic_balance': 0.25,         # Sympathetic/parasympathetic balance
            'sleep_architecture': 0.20,        # Sleep stage distribution
            'recovery_debt': 0.15,             # Accumulated recovery debt
            'stress_load': 0.10               # Current stress load
        }

    def calculate_comprehensive_readiness(self, date: datetime = None) -> Dict[str, any]:
        """
        Calculate comprehensive training readiness based on German sports science.

        Args:
            date: Analysis date (defaults to today)

        Returns:
            Comprehensive readiness analysis with German methodology
        """
        if date is None:
            date = datetime.utcnow()

        with self.db.get_session() as session:
            # Get individual HRV baseline (rolling 30-day)
            hrv_baseline = self._calculate_hrv_baseline(session, date, days=30)

            # Get recent wellness data
            wellness_data = self._get_recent_wellness_data(date, days=7)

            # Component analyses
            components = {}

            # 1. HRV Baseline Deviation Analysis
            hrv_analysis = self._analyze_hrv_baseline_deviation(date, hrv_baseline)
            if hrv_analysis:
                components['hrv_baseline_deviation'] = hrv_analysis

            # 2. Autonomic Nervous System Balance
            ans_analysis = self._analyze_autonomic_balance(wellness_data)
            if ans_analysis:
                components['autonomic_balance'] = ans_analysis

            # 3. Sleep Architecture Analysis
            sleep_analysis = self._analyze_sleep_architecture(date)
            if sleep_analysis:
                components['sleep_architecture'] = sleep_analysis

            # 4. Recovery Debt Assessment
            recovery_debt = self._calculate_recovery_debt(date, days=14)
            if recovery_debt:
                components['recovery_debt'] = recovery_debt

            # 5. Current Stress Load
            stress_analysis = self._analyze_current_stress(date)
            if stress_analysis:
                components['stress_load'] = stress_analysis

        # Calculate weighted readiness score
        readiness_score = self._calculate_weighted_readiness(components)

        # Determine readiness level
        readiness_level = self._classify_readiness_level(readiness_score)

        # Generate German methodology recommendations
        recommendations = self._generate_german_recommendations(
            readiness_score, readiness_level, components
        )

        return {
            'readiness_score': readiness_score,
            'readiness_level': readiness_level.value,
            'hrv_baseline': hrv_baseline.__dict__ if hrv_baseline else None,
            'components': components,
            'recommendations': recommendations,
            'methodology': 'German Sports Science (Belastungs-Beanspruchungs-Konzept)',
            'analysis_date': date.isoformat(),
            'data_confidence': self._assess_data_confidence(components)
        }

    def _calculate_hrv_baseline(self, session, date: datetime, days: int = 30) -> Optional[HRVBaseline]:
        """Calculate individual HRV baseline from recent data"""
        end_date = date
        start_date = date - timedelta(days=days)

        hrv_records = session.query(HRVData).filter(
            HRVData.user_id == self.user_id,
            HRVData.date >= start_date,
            HRVData.date <= end_date,
            HRVData.hrv_rmssd.isnot(None)
        ).order_by(HRVData.date).all()

        if len(hrv_records) < 7:  # Need minimum 1 week of data
            return None

        # Extract RMSSD values (ensure we access within session)
        rmssd_values = []
        for record in hrv_records:
            if record.hrv_rmssd is not None:
                rmssd_values.append(float(record.hrv_rmssd))

        if len(rmssd_values) < 7:
            return None

        # Calculate baseline statistics
        mean_rmssd = np.mean(rmssd_values)
        std_deviation = np.std(rmssd_values)
        coefficient_variation = (std_deviation / mean_rmssd) * 100 if mean_rmssd > 0 else 0

        # German sports science thresholds
        lower_threshold = mean_rmssd - std_deviation
        upper_threshold = mean_rmssd + std_deviation

        # Training readiness zone (within 0.5 SD of mean)
        readiness_lower = mean_rmssd - (0.5 * std_deviation)
        readiness_upper = mean_rmssd + (0.5 * std_deviation)

        return HRVBaseline(
            mean_rmssd=mean_rmssd,
            std_deviation=std_deviation,
            coefficient_variation=coefficient_variation,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            training_readiness_zone=(readiness_lower, readiness_upper)
        )

    def _analyze_hrv_baseline_deviation(self, date: datetime,
                                      baseline: HRVBaseline) -> Optional[Dict]:
        """Analyze current HRV deviation from individual baseline"""
        if not baseline:
            return None

        # Get today's HRV
        with self.db.get_session() as session:
            hrv_record = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= date - timedelta(days=1),
                HRVData.date <= date + timedelta(days=1),
                HRVData.hrv_rmssd.isnot(None)
            ).order_by(HRVData.date.desc()).first()

            # Extract data within session context
            if not hrv_record or not hrv_record.hrv_rmssd:
                return None

            current_rmssd = hrv_record.hrv_rmssd

        # Calculate deviation from baseline
        deviation_percent = ((current_rmssd - baseline.mean_rmssd) / baseline.mean_rmssd) * 100

        # Assess deviation significance
        if current_rmssd < baseline.lower_threshold:
            deviation_status = "significantly_below"
            readiness_impact = 30  # Major negative impact
        elif current_rmssd < baseline.training_readiness_zone[0]:
            deviation_status = "below_optimal"
            readiness_impact = 15  # Moderate negative impact
        elif current_rmssd <= baseline.training_readiness_zone[1]:
            deviation_status = "optimal"
            readiness_impact = 0   # No impact
        elif current_rmssd <= baseline.upper_threshold:
            deviation_status = "above_optimal"
            readiness_impact = 5   # Slight positive impact
        else:
            deviation_status = "significantly_above"
            readiness_impact = -10  # May indicate overreaching

        # Calculate component score (0-100)
        base_score = 70  # Neutral baseline
        component_score = max(0, min(100, base_score - readiness_impact + (deviation_percent * 0.5)))

        return {
            'score': component_score,
            'weight': self.component_weights['hrv_baseline_deviation'],
            'current_rmssd': current_rmssd,
            'baseline_mean': baseline.mean_rmssd,
            'deviation_percent': deviation_percent,
            'deviation_status': deviation_status,
            'readiness_zone': baseline.training_readiness_zone,
            'interpretation': self._interpret_hrv_deviation(deviation_status, deviation_percent)
        }

    def _analyze_autonomic_balance(self, wellness_data: List) -> Optional[Dict]:
        """Analyze autonomic nervous system balance indicators"""
        if not wellness_data:
            return None

        # Use HRV status and stress levels as ANS indicators
        recent_data = wellness_data[:3]  # Last 3 days

        ans_indicators = []
        stress_levels = []

        for data in recent_data:
            if data.get('hrv_status'):
                # Convert HRV status to ANS balance score
                status_scores = {
                    'BALANCED': 80,
                    'GOOD': 75,
                    'FAIR': 60,
                    'POOR': 40,
                    'UNBALANCED': 30
                }
                score = status_scores.get(data['hrv_status'].upper(), 60)
                ans_indicators.append(score)

            # Note: stress_avg is not in HRVData, would need separate WellnessData query
            # For now, skip stress levels from this method

        if not ans_indicators and not stress_levels:
            return None

        # Calculate composite ANS balance score
        all_scores = ans_indicators + stress_levels
        ans_balance_score = np.mean(all_scores)

        # Assess balance quality
        if ans_balance_score >= 80:
            balance_status = "excellent_balance"
        elif ans_balance_score >= 70:
            balance_status = "good_balance"
        elif ans_balance_score >= 60:
            balance_status = "moderate_balance"
        elif ans_balance_score >= 50:
            balance_status = "poor_balance"
        else:
            balance_status = "severely_imbalanced"

        return {
            'score': ans_balance_score,
            'weight': self.component_weights['autonomic_balance'],
            'balance_status': balance_status,
            'hrv_status_count': len(ans_indicators),
            'stress_readings': len(stress_levels),
            'trend': self._calculate_trend(all_scores),
            'interpretation': self._interpret_ans_balance(balance_status, ans_balance_score)
        }

    def _analyze_sleep_architecture(self, date: datetime) -> Optional[Dict]:
        """Analyze sleep architecture and quality"""
        with self.db.get_session() as session:
            sleep_record = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= date - timedelta(days=1),
                SleepData.date <= date + timedelta(days=1)
            ).order_by(SleepData.date.desc()).first()

            # Extract data within session context
            if not sleep_record:
                return None

            sleep_data = {
                'total_sleep_time': sleep_record.total_sleep_time,
                'sleep_efficiency': sleep_record.sleep_efficiency,
                'sleep_score': sleep_record.sleep_score,
                'deep_sleep_time': sleep_record.deep_sleep_time,
                'light_sleep_time': sleep_record.light_sleep_time,
                'rem_sleep_time': sleep_record.rem_sleep_time,
                'wake_time': sleep_record.wake_time
            }

        components = {}
        total_score = 0
        weight_sum = 0

        # Sleep duration analysis
        if sleep_data.get('total_sleep_time'):
            duration_hours = sleep_data['total_sleep_time'] / 3600

            # Optimal sleep duration: 7-9 hours for athletes
            if 7.5 <= duration_hours <= 8.5:
                duration_score = 100
            elif 7.0 <= duration_hours <= 9.0:
                duration_score = 85
            elif 6.5 <= duration_hours <= 9.5:
                duration_score = 70
            else:
                duration_score = max(30, 100 - abs(duration_hours - 8) * 15)

            components['duration'] = {
                'score': duration_score,
                'hours': duration_hours,
                'weight': 0.3
            }
            total_score += duration_score * 0.3
            weight_sum += 0.3

        # Sleep score (direct from Garmin)
        if sleep_data.get('sleep_score'):
            components['quality'] = {
                'score': sleep_data['sleep_score'],
                'weight': 0.4
            }
            total_score += sleep_data['sleep_score'] * 0.4
            weight_sum += 0.4

        # Sleep stage distribution (if available)
        if (sleep_data.get('deep_sleep_time') and sleep_data.get('light_sleep_time') and
            sleep_data.get('rem_sleep_time') and sleep_data.get('total_sleep_time')):

            total_sleep = sleep_data['total_sleep_time']
            deep_percent = (sleep_data['deep_sleep_time'] / total_sleep) * 100
            rem_percent = (sleep_data['rem_sleep_time'] / total_sleep) * 100

            # Optimal percentages: Deep 13-23%, REM 20-25%
            deep_score = 100 if 13 <= deep_percent <= 23 else max(50, 100 - abs(deep_percent - 18) * 3)
            rem_score = 100 if 20 <= rem_percent <= 25 else max(50, 100 - abs(rem_percent - 22.5) * 4)

            architecture_score = (deep_score + rem_score) / 2

            components['architecture'] = {
                'score': architecture_score,
                'deep_percent': deep_percent,
                'rem_percent': rem_percent,
                'weight': 0.3
            }
            total_score += architecture_score * 0.3
            weight_sum += 0.3

        if weight_sum == 0:
            return None

        final_score = total_score / weight_sum

        return {
            'score': final_score,
            'weight': self.component_weights['sleep_architecture'],
            'components': components,
            'interpretation': self._interpret_sleep_quality(final_score)
        }

    def _calculate_recovery_debt(self, date: datetime, days: int = 14) -> Optional[Dict]:
        """Calculate accumulated recovery debt based on training load vs recovery"""
        # This would integrate with training load analysis
        # For now, return a simplified assessment based on HRV trends

        end_date = date
        start_date = date - timedelta(days=days)

        with self.db.get_session() as session:
            hrv_records = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= start_date,
                HRVData.date <= end_date,
                HRVData.hrv_rmssd.isnot(None)
            ).order_by(HRVData.date).all()

            # Extract RMSSD values within session context
            if len(hrv_records) < 7:
                return None

            rmssd_values = [r.hrv_rmssd for r in hrv_records]

        # Simple linear trend
        days_array = np.arange(len(rmssd_values))
        trend_slope = np.polyfit(days_array, rmssd_values, 1)[0]

        # Assess recovery debt based on trend
        if trend_slope > 1:
            debt_level = "recovering"
            debt_score = 80
        elif trend_slope > 0:
            debt_level = "stable"
            debt_score = 70
        elif trend_slope > -1:
            debt_level = "accumulating"
            debt_score = 50
        else:
            debt_level = "high_debt"
            debt_score = 30

        return {
            'score': debt_score,
            'weight': self.component_weights['recovery_debt'],
            'debt_level': debt_level,
            'hrv_trend_slope': trend_slope,
            'days_analyzed': len(rmssd_values),
            'interpretation': self._interpret_recovery_debt(debt_level, trend_slope)
        }

    def _analyze_current_stress(self, date: datetime) -> Optional[Dict]:
        """Analyze current stress level indicators"""
        with self.db.get_session() as session:
            wellness_record = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= date - timedelta(days=1),
                WellnessData.date <= date + timedelta(days=1)
            ).order_by(WellnessData.date.desc()).first()

            # Extract data within session context
            if not wellness_record or wellness_record.stress_avg is None:
                return None

            stress_level = wellness_record.stress_avg

        # Convert stress to readiness score (inverted)
        stress_score = max(0, 100 - stress_level)

        # Classify stress level
        if stress_level <= 25:
            stress_category = "low_stress"
        elif stress_level <= 50:
            stress_category = "moderate_stress"
        elif stress_level <= 75:
            stress_category = "high_stress"
        else:
            stress_category = "very_high_stress"

        return {
            'score': stress_score,
            'weight': self.component_weights['stress_load'],
            'stress_level': stress_level,
            'stress_category': stress_category,
            'interpretation': self._interpret_stress_level(stress_category, stress_level)
        }

    def _calculate_weighted_readiness(self, components: Dict) -> float:
        """Calculate weighted readiness score from components"""
        if not components:
            return 50.0  # Neutral score if no data

        total_score = 0
        total_weight = 0

        for component_name, component_data in components.items():
            score = component_data.get('score', 50)
            weight = component_data.get('weight', 0)

            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return total_score / total_weight

    def _classify_readiness_level(self, score: float) -> ReadinessLevel:
        """Classify readiness level based on score"""
        if score >= 85:
            return ReadinessLevel.EXCELLENT
        elif score >= 70:
            return ReadinessLevel.GOOD
        elif score >= 55:
            return ReadinessLevel.MODERATE
        elif score >= 40:
            return ReadinessLevel.POOR
        else:
            return ReadinessLevel.CRITICAL

    def _generate_german_recommendations(self, score: float, level: ReadinessLevel,
                                       components: Dict) -> List[str]:
        """Generate training recommendations based on German sports science"""
        recommendations = []

        # Primary recommendation based on readiness level
        if level == ReadinessLevel.EXCELLENT:
            recommendations.extend([
                "Excellent readiness - suitable for high-intensity training",
                "Consider periodization peak training or competition",
                "Maintain current recovery strategies"
            ])
        elif level == ReadinessLevel.GOOD:
            recommendations.extend([
                "Good readiness for moderate to high intensity training",
                "Monitor for signs of accumulating fatigue",
                "Quality over quantity in training sessions"
            ])
        elif level == ReadinessLevel.MODERATE:
            recommendations.extend([
                "Moderate readiness - focus on aerobic base training",
                "Avoid high-intensity intervals",
                "Enhance recovery strategies (sleep, nutrition, stress management)"
            ])
        elif level == ReadinessLevel.POOR:
            recommendations.extend([
                "Poor readiness - active recovery only",
                "Address underlying stressors",
                "Consider extending recovery period"
            ])
        else:  # CRITICAL
            recommendations.extend([
                "Critical readiness state - complete rest recommended",
                "Investigate potential overreaching/overtraining",
                "Consult sports medicine professional if persistent"
            ])

        # Component-specific recommendations
        for component_name, component_data in components.items():
            if component_name == 'hrv_baseline_deviation':
                if component_data.get('deviation_status') == 'significantly_below':
                    recommendations.append("HRV significantly below baseline - prioritize recovery")
                elif component_data.get('deviation_status') == 'significantly_above':
                    recommendations.append("HRV unusually high - monitor for overreaching signs")

            elif component_name == 'sleep_architecture':
                if component_data.get('score', 0) < 60:
                    recommendations.append("Sleep quality compromised - optimize sleep hygiene")

            elif component_name == 'autonomic_balance':
                if component_data.get('balance_status') in ['poor_balance', 'severely_imbalanced']:
                    recommendations.append("Autonomic imbalance detected - stress management needed")

        return recommendations

    def _get_recent_wellness_data(self, date: datetime, days: int = 7) -> List:
        """Get recent wellness data for analysis"""
        end_date = date
        start_date = date - timedelta(days=days)

        with self.db.get_session() as session:
            hrv_records = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date.desc()).all()

            # Extract data within session context to avoid session binding issues
            wellness_data = []
            for record in hrv_records:
                wellness_data.append({
                    'date': record.date,
                    'hrv_rmssd': record.hrv_rmssd,
                    'hrv_status': record.hrv_status,
                    'hrv_score': record.hrv_score
                })

            return wellness_data

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"

        # Simple trend: compare first half to second half
        mid = len(values) // 2
        first_half_avg = np.mean(values[:mid]) if mid > 0 else values[0]
        second_half_avg = np.mean(values[mid:])

        if second_half_avg > first_half_avg * 1.05:
            return "improving"
        elif second_half_avg < first_half_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _assess_data_confidence(self, components: Dict) -> str:
        """Assess confidence level in the analysis"""
        component_count = len(components)

        if component_count >= 4:
            return "high"
        elif component_count >= 2:
            return "moderate"
        elif component_count >= 1:
            return "low"
        else:
            return "insufficient"

    # Interpretation methods
    def _interpret_hrv_deviation(self, status: str, deviation_percent: float) -> str:
        """Interpret HRV deviation meaning"""
        interpretations = {
            "significantly_below": f"HRV {deviation_percent:.1f}% below baseline - indicates high stress/fatigue",
            "below_optimal": f"HRV {deviation_percent:.1f}% below optimal - slight stress accumulation",
            "optimal": f"HRV within optimal range ({deviation_percent:+.1f}%) - good adaptation state",
            "above_optimal": f"HRV {deviation_percent:.1f}% above optimal - good recovery",
            "significantly_above": f"HRV {deviation_percent:.1f}% significantly elevated - possible overreaching"
        }
        return interpretations.get(status, "Unknown HRV status")

    def _interpret_ans_balance(self, status: str, score: float) -> str:
        """Interpret autonomic nervous system balance"""
        interpretations = {
            "excellent_balance": f"Excellent ANS balance ({score:.0f}/100) - optimal readiness",
            "good_balance": f"Good ANS balance ({score:.0f}/100) - ready for training",
            "moderate_balance": f"Moderate ANS balance ({score:.0f}/100) - light training suitable",
            "poor_balance": f"Poor ANS balance ({score:.0f}/100) - recovery needed",
            "severely_imbalanced": f"Severe ANS imbalance ({score:.0f}/100) - rest required"
        }
        return interpretations.get(status, "Unknown ANS status")

    def _interpret_sleep_quality(self, score: float) -> str:
        """Interpret sleep quality score"""
        if score >= 85:
            return f"Excellent sleep quality ({score:.0f}/100) - optimal recovery"
        elif score >= 70:
            return f"Good sleep quality ({score:.0f}/100) - adequate recovery"
        elif score >= 55:
            return f"Moderate sleep quality ({score:.0f}/100) - suboptimal recovery"
        else:
            return f"Poor sleep quality ({score:.0f}/100) - compromised recovery"

    def _interpret_recovery_debt(self, debt_level: str, slope: float) -> str:
        """Interpret recovery debt status"""
        interpretations = {
            "recovering": f"Recovery trend positive (slope: {slope:.2f}) - debt reducing",
            "stable": f"Recovery stable (slope: {slope:.2f}) - balanced state",
            "accumulating": f"Recovery debt accumulating (slope: {slope:.2f}) - needs attention",
            "high_debt": f"High recovery debt (slope: {slope:.2f}) - immediate action needed"
        }
        return interpretations.get(debt_level, "Unknown recovery status")

    def _interpret_stress_level(self, category: str, level: float) -> str:
        """Interpret current stress level"""
        interpretations = {
            "low_stress": f"Low stress level ({level:.0f}) - excellent for training",
            "moderate_stress": f"Moderate stress level ({level:.0f}) - manageable",
            "high_stress": f"High stress level ({level:.0f}) - impacts training capacity",
            "very_high_stress": f"Very high stress level ({level:.0f}) - significant impact on readiness"
        }
        return interpretations.get(category, "Unknown stress status")


def get_hrv_baseline_analyzer(user_id: str = "default") -> HRVBaselineAnalyzer:
    """Get HRV baseline analyzer instance"""
    return HRVBaselineAnalyzer(user_id)