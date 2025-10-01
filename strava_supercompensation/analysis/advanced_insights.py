"""
Advanced Training Analysis Insights
Detects patterns, imbalances, and risks in training data that standard metrics miss.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import logging

from ..db import get_db
from ..db.models import Activity, Metric, HRVData, SleepData, WellnessData, BodyComposition, BloodPressure
from ..config import config


@dataclass
class HealthDataAnomaly:
    """Represents a detected health data anomaly."""
    date: datetime
    metric: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'minor', 'major', 'critical'
    likely_cause: str


@dataclass
class RiskPattern:
    """Represents a risky training pattern."""
    date: datetime
    risk_type: str
    description: str
    metrics: Dict[str, float]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class TrainingImbalance:
    """Represents detected training imbalances."""
    imbalance_type: str
    current_percentage: float
    recommended_percentage: float
    risk_level: str
    recommendations: List[str]


class AdvancedInsightsAnalyzer:
    """Advanced analysis for detecting training patterns, risks, and imbalances."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.logger = logging.getLogger(__name__)

    def analyze_comprehensive_insights(self, days_back: int = 60) -> Dict[str, Any]:
        """
        Comprehensive analysis addressing the identified issues:
        1. Contradictory readiness/overtraining signals
        2. Health data anomalies
        3. Sleep-training correlations
        4. Training imbalances
        5. Risk patterns
        """
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        results = {
            'analysis_period': f"{start_date} to {end_date}",
            'contradictory_signals': self._analyze_contradictory_signals(start_date, end_date),
            'health_anomalies': self._detect_health_anomalies(start_date, end_date),
            'sleep_training_risks': self._analyze_sleep_training_correlation(start_date, end_date),
            'training_imbalances': self._analyze_training_imbalances(start_date, end_date),
            'injury_risk_patterns': self._detect_injury_risk_patterns(start_date, end_date),
            'recovery_system_analysis': self._analyze_recovery_systems(start_date, end_date),
            'critical_recommendations': []
        }

        # Generate critical recommendations based on findings
        results['critical_recommendations'] = self._generate_critical_recommendations(results)

        return results

    def _analyze_contradictory_signals(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Detect contradictory readiness vs overtraining signals."""
        with self.db.get_session() as session:
            # Get latest metrics
            latest_metrics = session.query(Metric).filter(
                Metric.date >= start_date,
                Metric.date <= end_date
            ).order_by(Metric.date.desc()).limit(7).all()

            if not latest_metrics:
                return {'status': 'no_data'}

            latest = latest_metrics[0]

            # Calculate contradictory signals
            form_signal = "positive" if latest.form > 20 else "neutral" if latest.form > -10 else "negative"
            fitness_level = "high" if latest.fitness > 120 else "moderate" if latest.fitness > 80 else "low"
            fatigue_level = "high" if latest.fatigue > 80 else "moderate" if latest.fatigue > 50 else "low"

            # Detect contradictions
            contradictions = []

            if latest.form > 50 and fatigue_level == "high":
                contradictions.append({
                    'date': latest.date,
                    'type': 'high_form_high_fatigue',
                    'description': f"Form shows peaked state ({latest.form:.1f}) but fatigue is {fatigue_level} ({latest.fatigue:.1f})",
                    'interpretation': "Form may be artificially high due to recent rest, masking underlying fatigue accumulation",
                    'risk': "high"
                })

            if fitness_level == "high" and latest.form < -20:
                contradictions.append({
                    'date': latest.date,
                    'type': 'high_fitness_negative_form',
                    'description': f"High fitness ({latest.fitness:.1f}) but negative form ({latest.form:.1f})",
                    'interpretation': "Chronic overreaching - fitness built through excessive fatigue accumulation",
                    'risk': "critical"
                })

            # Calculate readiness vs metrics alignment
            expected_readiness = self._calculate_expected_readiness(latest.fitness, latest.fatigue, latest.form)

            return {
                'status': 'analyzed',
                'latest_metrics': {
                    'fitness': latest.fitness,
                    'fatigue': latest.fatigue,
                    'form': latest.form,
                    'date': latest.date.strftime('%Y-%m-%d')
                },
                'contradictions': contradictions,
                'signal_alignment': {
                    'form_signal': form_signal,
                    'fitness_level': fitness_level,
                    'fatigue_level': fatigue_level,
                    'expected_readiness': expected_readiness,
                    'alignment_score': len(contradictions) == 0
                }
            }

    def _detect_health_anomalies(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Enhanced anomaly detection for all health metrics with comprehensive validation."""
        anomalies = []

        with self.db.get_session() as session:
            # Get body composition data
            body_comp_data = session.query(BodyComposition).filter(
                BodyComposition.user_id == self.user_id,
                BodyComposition.date >= start_date,
                BodyComposition.date <= end_date
            ).order_by(BodyComposition.date).all()

            # Get HRV data for validation
            hrv_data = session.query(HRVData).filter(
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date).all()

            # Get sleep data for validation
            sleep_data = session.query(SleepData).filter(
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).order_by(SleepData.date).all()

            # Get blood pressure data for validation
            bp_data = session.query(BloodPressure).filter(
                BloodPressure.date >= start_date,
                BloodPressure.date <= end_date
            ).order_by(BloodPressure.date).all()

            # Enhanced body composition validation
            if len(body_comp_data) >= 2:
                self._validate_body_composition(body_comp_data, anomalies)

            # HRV validation
            if len(hrv_data) >= 5:
                self._validate_hrv_metrics(hrv_data, anomalies)

            # Sleep validation
            if len(sleep_data) >= 5:
                self._validate_sleep_metrics(sleep_data, anomalies)

            # Blood pressure validation
            if len(bp_data) >= 3:
                self._validate_blood_pressure(bp_data, anomalies)

        return {
            'status': 'analyzed',
            'anomalies': [
                {
                    'date': a.date.strftime('%Y-%m-%d'),
                    'metric': a.metric,
                    'value': a.value,
                    'expected_range': a.expected_range,
                    'severity': a.severity,
                    'likely_cause': a.likely_cause
                }
                for a in anomalies
            ],
            'total_anomalies': len(anomalies),
            'critical_anomalies': len([a for a in anomalies if a.severity == 'critical'])
        }

    def _validate_body_composition(self, body_comp_data: List, anomalies: List[HealthDataAnomaly]) -> None:
        """Validate body composition data for outliers and impossible changes."""
        # Calculate baselines and detect outliers
        weights = [bc.weight_kg for bc in body_comp_data if bc.weight_kg]
        bf_percentages = [bc.body_fat_percent for bc in body_comp_data if bc.body_fat_percent]
        water_percentages = [bc.water_percent for bc in body_comp_data if bc.water_percent]

        # Weight validation
        if len(weights) > 1:
            weight_mean = statistics.mean(weights)
            weight_std = statistics.stdev(weights)

            for bc in body_comp_data:
                if bc.weight_kg:
                    z_score = abs(bc.weight_kg - weight_mean) / (weight_std + 0.1)
                    if z_score > 2.5:
                        anomalies.append(HealthDataAnomaly(
                            date=bc.date,
                            metric="weight",
                            value=bc.weight_kg,
                            expected_range=(weight_mean - 2*weight_std, weight_mean + 2*weight_std),
                            severity="major" if z_score > 3 else "minor",
                            likely_cause="Data entry error or different person" if z_score > 4 else "Dehydration/fluid retention"
                        ))

        # Body fat percentage validation
        if len(bf_percentages) > 1:
            bf_mean = statistics.mean(bf_percentages)
            bf_std = statistics.stdev(bf_percentages)

            for bc in body_comp_data:
                if bc.body_fat_percent:
                    # Check for physiologically impossible values
                    if bc.body_fat_percent < 3 or bc.body_fat_percent > 50:
                        anomalies.append(HealthDataAnomaly(
                            date=bc.date,
                            metric="body_fat_percent",
                            value=bc.body_fat_percent,
                            expected_range=(3, 50),
                            severity="critical",
                            likely_cause="Scale malfunction or measurement error"
                        ))
                    else:
                        z_score = abs(bc.body_fat_percent - bf_mean) / (bf_std + 0.1)
                        if z_score > 2.5:
                            anomalies.append(HealthDataAnomaly(
                                date=bc.date,
                                metric="body_fat_percent",
                                value=bc.body_fat_percent,
                                expected_range=(bf_mean - 2*bf_std, bf_mean + 2*bf_std),
                                severity="major" if z_score > 3 else "minor",
                                likely_cause="Measurement inconsistency or hydration changes"
                            ))

        # Check for impossible rapid changes
        for i in range(1, len(body_comp_data)):
            prev = body_comp_data[i-1]
            curr = body_comp_data[i]
            days_diff = (curr.date - prev.date).days

            if days_diff > 0 and prev.weight_kg and curr.weight_kg:
                weight_change = abs(curr.weight_kg - prev.weight_kg)
                max_realistic_change = days_diff * 0.2  # 200g per day max realistic

                if weight_change > max_realistic_change and weight_change > 2.0:
                    anomalies.append(HealthDataAnomaly(
                        date=curr.date,
                        metric="weight_change",
                        value=curr.weight_kg - prev.weight_kg,
                        expected_range=(-max_realistic_change, max_realistic_change),
                        severity="critical" if weight_change > 8 else "major",
                        likely_cause="Scale malfunction or different person using scale"
                    ))

    def _validate_hrv_metrics(self, hrv_data: List, anomalies: List[HealthDataAnomaly]) -> None:
        """Validate HRV metrics for physiological plausibility."""
        rmssd_values = [hrv.hrv_rmssd for hrv in hrv_data if hrv.hrv_rmssd and hrv.hrv_rmssd > 0]

        if len(rmssd_values) > 2:
            rmssd_mean = statistics.mean(rmssd_values)
            rmssd_std = statistics.stdev(rmssd_values)

            for hrv in hrv_data:
                if hrv.hrv_rmssd:
                    # Check for physiologically impossible values
                    if hrv.hrv_rmssd < 5 or hrv.hrv_rmssd > 200:
                        anomalies.append(HealthDataAnomaly(
                            date=hrv.date,
                            metric="hrv_rmssd",
                            value=hrv.hrv_rmssd,
                            expected_range=(5, 200),
                            severity="critical",
                            likely_cause="Device malfunction or measurement artifact"
                        ))
                    else:
                        z_score = abs(hrv.hrv_rmssd - rmssd_mean) / (rmssd_std + 0.1)
                        if z_score > 3.0:  # More lenient for HRV due to natural variability
                            anomalies.append(HealthDataAnomaly(
                                date=hrv.date,
                                metric="hrv_rmssd",
                                value=hrv.hrv_rmssd,
                                expected_range=(rmssd_mean - 3*rmssd_std, rmssd_mean + 3*rmssd_std),
                                severity="major" if z_score > 4 else "minor",
                                likely_cause="High stress, illness, or device error" if hrv.hrv_rmssd < rmssd_mean else "Exceptional recovery state"
                            ))

    def _validate_sleep_metrics(self, sleep_data: List, anomalies: List[HealthDataAnomaly]) -> None:
        """Validate sleep metrics for realistic values."""
        sleep_scores = [sleep.sleep_score for sleep in sleep_data if sleep.sleep_score and sleep.sleep_score > 0]

        if len(sleep_scores) > 2:
            sleep_mean = statistics.mean(sleep_scores)
            sleep_std = statistics.stdev(sleep_scores)

            for sleep in sleep_data:
                if sleep.sleep_score:
                    # Check for impossible sleep scores
                    if sleep.sleep_score < 0 or sleep.sleep_score > 100:
                        anomalies.append(HealthDataAnomaly(
                            date=sleep.date,
                            metric="sleep_score",
                            value=sleep.sleep_score,
                            expected_range=(0, 100),
                            severity="critical",
                            likely_cause="Data corruption or app malfunction"
                        ))
                    else:
                        z_score = abs(sleep.sleep_score - sleep_mean) / (sleep_std + 0.1)
                        if z_score > 2.5:
                            anomalies.append(HealthDataAnomaly(
                                date=sleep.date,
                                metric="sleep_score",
                                value=sleep.sleep_score,
                                expected_range=(sleep_mean - 2.5*sleep_std, sleep_mean + 2.5*sleep_std),
                                severity="major" if z_score > 3 else "minor",
                                likely_cause="Sleep disturbance, stress, or tracking error" if sleep.sleep_score < sleep_mean else "Exceptionally good sleep conditions"
                            ))

    def _validate_blood_pressure(self, bp_data: List, anomalies: List[HealthDataAnomaly]) -> None:
        """Validate blood pressure readings for medical plausibility."""
        for bp in bp_data:
            if bp.systolic and bp.diastolic:
                # Check for physiologically impossible values
                if bp.systolic < 70 or bp.systolic > 250:
                    anomalies.append(HealthDataAnomaly(
                        date=bp.date,
                        metric="systolic_bp",
                        value=bp.systolic,
                        expected_range=(70, 250),
                        severity="critical",
                        likely_cause="Device malfunction or measurement error"
                    ))

                if bp.diastolic < 40 or bp.diastolic > 150:
                    anomalies.append(HealthDataAnomaly(
                        date=bp.date,
                        metric="diastolic_bp",
                        value=bp.diastolic,
                        expected_range=(40, 150),
                        severity="critical",
                        likely_cause="Device malfunction or measurement error"
                    ))

                # Check for impossible pulse pressure
                pulse_pressure = bp.systolic - bp.diastolic
                if pulse_pressure < 20 or pulse_pressure > 100:
                    anomalies.append(HealthDataAnomaly(
                        date=bp.date,
                        metric="pulse_pressure",
                        value=pulse_pressure,
                        expected_range=(20, 100),
                        severity="major",
                        likely_cause="Measurement inconsistency or arterial stiffness" if pulse_pressure > 100 else "Device error"
                    ))

            # Validate heart rate if present
            if bp.heart_rate:
                if bp.heart_rate < 30 or bp.heart_rate > 220:
                    anomalies.append(HealthDataAnomaly(
                        date=bp.date,
                        metric="resting_heart_rate",
                        value=bp.heart_rate,
                        expected_range=(30, 220),
                        severity="critical",
                        likely_cause="Device malfunction or movement artifact"
                    ))

    def _analyze_sleep_training_correlation(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Analyze correlation between sleep quality and training load."""
        risks = []

        with self.db.get_session() as session:
            # Get activities and sleep data
            activities = session.query(Activity).filter(
                Activity.start_date >= datetime.combine(start_date, datetime.min.time()),
                Activity.start_date <= datetime.combine(end_date, datetime.max.time())
            ).all()

            sleep_data = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).all()

            # Create date-indexed dictionaries
            daily_loads = {}
            daily_sleep = {}

            for activity in activities:
                activity_date = activity.start_date.date()
                if activity.training_load:
                    daily_loads[activity_date] = daily_loads.get(activity_date, 0) + activity.training_load

            for sleep in sleep_data:
                daily_sleep[sleep.date] = sleep.sleep_score

            # Analyze risky patterns
            for date in daily_loads:
                load = daily_loads[date]
                prev_date = date - timedelta(days=1)
                sleep_score = daily_sleep.get(prev_date)

                if sleep_score and load > 300:  # High training load
                    if sleep_score < 60:  # Poor sleep
                        risks.append(RiskPattern(
                            date=date,
                            risk_type="high_load_poor_sleep",
                            description=f"High training load ({load:.0f} TSS) after poor sleep ({sleep_score}/100)",
                            metrics={'training_load': load, 'sleep_score': sleep_score},
                            severity="critical" if load > 500 and sleep_score < 50 else "high"
                        ))

            # Calculate correlations if enough data
            correlation = None
            if len(daily_loads) > 5 and len(daily_sleep) > 5:
                common_dates = set(daily_loads.keys()) & set(daily_sleep.keys())
                if len(common_dates) > 3:
                    loads = [daily_loads[d] for d in common_dates]
                    sleeps = [daily_sleep[d] for d in common_dates]
                    correlation = np.corrcoef(loads, sleeps)[0, 1] if len(loads) > 1 else None

        return {
            'status': 'analyzed',
            'risk_patterns': [
                {
                    'date': r.date.strftime('%Y-%m-%d'),
                    'type': r.risk_type,
                    'description': r.description,
                    'severity': r.severity,
                    'metrics': r.metrics
                }
                for r in risks
            ],
            'sleep_load_correlation': correlation,
            'high_risk_days': len([r for r in risks if r.severity in ['high', 'critical']]),
            'recommendations': self._generate_sleep_recommendations(risks)
        }

    def _analyze_training_imbalances(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Analyze training imbalances, especially strength training deficits."""
        imbalances = []

        with self.db.get_session() as session:
            activities = session.query(Activity).filter(
                Activity.start_date >= datetime.combine(start_date, datetime.min.time()),
                Activity.start_date <= datetime.combine(end_date, datetime.max.time())
            ).all()

            if not activities:
                return {'status': 'no_data'}

            # Calculate sport distribution
            sport_time = {}
            sport_load = {}
            total_time = 0
            total_load = 0

            for activity in activities:
                sport = activity.type
                time_hours = (activity.moving_time or 0) / 3600
                load = activity.training_load or 0

                sport_time[sport] = sport_time.get(sport, 0) + time_hours
                sport_load[sport] = sport_load.get(sport, 0) + load
                total_time += time_hours
                total_load += load

            # Analyze strength training deficit
            # Case-insensitive matching for strength activities
            strength_keywords = ['weight', 'workout', 'strength']
            strength_time = sum(
                time for sport, time in sport_time.items()
                if any(keyword in sport.lower() for keyword in strength_keywords)
            )
            strength_percentage = (strength_time / total_time * 100) if total_time > 0 else 0

            # Also calculate recent 30-day trend for comparison
            recent_30d_end = end_date
            recent_30d_start = end_date - timedelta(days=30)
            recent_activities = [a for a in activities if recent_30d_start <= a.start_date.date() <= recent_30d_end]

            if recent_activities:
                recent_total_time = sum((a.moving_time or 0) / 3600 for a in recent_activities)
                recent_strength_time = sum(
                    (a.moving_time or 0) / 3600 for a in recent_activities
                    if any(keyword in a.type.lower() for keyword in strength_keywords)
                )
                recent_strength_percentage = (recent_strength_time / recent_total_time * 100) if recent_total_time > 0 else 0
            else:
                recent_strength_percentage = strength_percentage

            if strength_percentage < 15:  # Should be 15-20% for endurance athletes
                imbalances.append(TrainingImbalance(
                    imbalance_type="strength_deficit",
                    current_percentage=strength_percentage,
                    recommended_percentage=18.0,
                    risk_level="critical" if strength_percentage < 5 else "high",
                    recommendations=[
                        "Add 2-3 strength training sessions per week",
                        "Focus on functional movement patterns",
                        "Include core stability work",
                        "Prioritize injury prevention exercises"
                    ]
                ))

            # Analyze sport variety
            primary_sports = ['Run', 'Ride', 'Swim']
            primary_time = sum(sport_time.get(sport, 0) for sport in primary_sports)
            if primary_time > 0:
                sport_ratios = {sport: (sport_time.get(sport, 0) / primary_time) for sport in primary_sports}

                # Check for over-dominance of one sport
                max_sport = max(sport_ratios, key=sport_ratios.get)
                if sport_ratios[max_sport] > 0.7:  # One sport dominates >70%
                    imbalances.append(TrainingImbalance(
                        imbalance_type="sport_overdominance",
                        current_percentage=sport_ratios[max_sport] * 100,
                        recommended_percentage=60.0,
                        risk_level="medium",
                        recommendations=[
                            f"Reduce {max_sport} volume slightly",
                            "Add cross-training variety",
                            "Include complementary movement patterns"
                        ]
                    ))

        return {
            'status': 'analyzed',
            'imbalances': [
                {
                    'type': imb.imbalance_type,
                    'current_pct': imb.current_percentage,
                    'recommended_pct': imb.recommended_percentage,
                    'risk_level': imb.risk_level,
                    'recommendations': imb.recommendations
                }
                for imb in imbalances
            ],
            'recent_strength_pct': recent_strength_percentage,  # Add 30-day trend
            'sport_distribution': {
                sport: {
                    'time_hours': sport_time.get(sport, 0),
                    'time_percentage': (sport_time.get(sport, 0) / total_time * 100) if total_time > 0 else 0,
                    'load_tss': sport_load.get(sport, 0),
                    'load_percentage': (sport_load.get(sport, 0) / total_load * 100) if total_load > 0 else 0
                }
                for sport in sport_time
            }
        }

    def _detect_injury_risk_patterns(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Detect patterns that increase injury risk."""
        risk_patterns = []

        with self.db.get_session() as session:
            activities = session.query(Activity).filter(
                Activity.start_date >= datetime.combine(start_date, datetime.min.time()),
                Activity.start_date <= datetime.combine(end_date, datetime.max.time())
            ).order_by(Activity.start_date).all()

            # Analyze load progressions
            weekly_loads = {}
            for activity in activities:
                week_start = activity.start_date.date() - timedelta(days=activity.start_date.weekday())
                weekly_loads[week_start] = weekly_loads.get(week_start, 0) + (activity.training_load or 0)

            # Check for excessive weekly load increases
            sorted_weeks = sorted(weekly_loads.keys())
            for i in range(1, len(sorted_weeks)):
                prev_week = sorted_weeks[i-1]
                curr_week = sorted_weeks[i]

                prev_load = weekly_loads[prev_week]
                curr_load = weekly_loads[curr_week]

                if prev_load > 0:
                    increase = (curr_load - prev_load) / prev_load * 100
                    if increase > 25:  # >25% weekly increase
                        risk_patterns.append(RiskPattern(
                            date=curr_week,
                            risk_type="excessive_load_increase",
                            description=f"Weekly load increased by {increase:.0f}% ({prev_load:.0f} → {curr_load:.0f} TSS)",
                            metrics={'previous_load': prev_load, 'current_load': curr_load, 'increase_pct': increase},
                            severity="critical" if increase > 50 else "high"
                        ))

        return {
            'status': 'analyzed',
            'risk_patterns': [
                {
                    'date': r.date.strftime('%Y-%m-%d') if hasattr(r.date, 'strftime') else str(r.date),
                    'type': r.risk_type,
                    'description': r.description,
                    'severity': r.severity,
                    'metrics': r.metrics
                }
                for r in risk_patterns
            ],
            'total_risks': len(risk_patterns),
            'high_risks': len([r for r in risk_patterns if r.severity in ['high', 'critical']])
        }

    def _analyze_recovery_systems(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Any]:
        """Analyze recovery across different physiological systems."""
        with self.db.get_session() as session:
            # Get HRV and sleep data for recovery analysis
            hrv_data = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= start_date,
                HRVData.date <= end_date
            ).order_by(HRVData.date).all()

            sleep_data = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= start_date,
                SleepData.date <= end_date
            ).order_by(SleepData.date).all()

            # Calculate recovery trends
            recovery_analysis = {
                'autonomic_recovery': self._analyze_hrv_trends(hrv_data),
                'sleep_recovery': self._analyze_sleep_trends(sleep_data),
                'overall_status': 'analyzing'
            }

            # Determine overall recovery status
            autonomic_score = recovery_analysis['autonomic_recovery'].get('trend_score', 50)
            sleep_score = recovery_analysis['sleep_recovery'].get('trend_score', 50)

            overall_score = (autonomic_score + sleep_score) / 2

            if overall_score < 40:
                recovery_status = "poor"
                concern_level = "critical"
            elif overall_score < 60:
                recovery_status = "declining"
                concern_level = "high"
            elif overall_score < 75:
                recovery_status = "moderate"
                concern_level = "medium"
            else:
                recovery_status = "good"
                concern_level = "low"

            recovery_analysis['overall_status'] = {
                'score': overall_score,
                'status': recovery_status,
                'concern_level': concern_level,
                'interpretation': self._interpret_recovery_status(recovery_status, autonomic_score, sleep_score)
            }

        return recovery_analysis

    def _calculate_expected_readiness(self, fitness: float, fatigue: float, form: float) -> float:
        """Calculate expected readiness based on metrics."""
        # Simplified readiness calculation
        fitness_factor = min(fitness / 100, 1.0) * 0.4
        fatigue_factor = max(0, (100 - fatigue) / 100) * 0.4
        form_factor = max(0, min((form + 50) / 100, 1.0)) * 0.2

        return (fitness_factor + fatigue_factor + form_factor) * 100

    def _generate_sleep_recommendations(self, risks: List[RiskPattern]) -> List[str]:
        """Generate sleep-based recommendations."""
        if not risks:
            return ["Continue maintaining good sleep hygiene"]

        recommendations = [
            "Prioritize 7-9 hours of quality sleep before high-intensity sessions",
            "Consider reducing training load after nights with sleep score <60",
            "Implement consistent sleep schedule and pre-sleep routine",
            "Monitor sleep quality trends and adjust training accordingly"
        ]

        critical_risks = [r for r in risks if r.severity == 'critical']
        if critical_risks:
            recommendations.insert(0, "URGENT: Mandatory rest day after poor sleep before high-load sessions")

        return recommendations

    def _analyze_hrv_trends(self, hrv_data: List) -> Dict[str, Any]:
        """Analyze HRV trends for autonomic recovery."""
        if not hrv_data:
            return {'status': 'no_data', 'trend_score': 50}

        recent_scores = [h.hrv_rmssd for h in hrv_data[-14:] if h.hrv_rmssd]  # Last 2 weeks
        if len(recent_scores) < 3:
            return {'status': 'insufficient_data', 'trend_score': 50}

        avg_recent = statistics.mean(recent_scores)
        baseline_scores = [h.hrv_rmssd for h in hrv_data[:21] if h.hrv_rmssd]  # First 3 weeks as baseline

        if baseline_scores:
            baseline_avg = statistics.mean(baseline_scores)
            trend_pct = ((avg_recent - baseline_avg) / baseline_avg) * 100
        else:
            trend_pct = 0

        # Convert trend to score (higher HRV is better)
        trend_score = max(0, min(100, 50 + trend_pct))

        return {
            'status': 'analyzed',
            'recent_avg': avg_recent,
            'baseline_avg': statistics.mean(baseline_scores) if baseline_scores else None,
            'trend_pct': trend_pct,
            'trend_score': trend_score,
            'interpretation': "improving" if trend_pct > 5 else "declining" if trend_pct < -10 else "stable"
        }

    def _analyze_sleep_trends(self, sleep_data: List) -> Dict[str, Any]:
        """Analyze sleep trends for recovery."""
        if not sleep_data:
            return {'status': 'no_data', 'trend_score': 50}

        recent_scores = [s.sleep_score for s in sleep_data[-14:] if s.sleep_score]
        if len(recent_scores) < 3:
            return {'status': 'insufficient_data', 'trend_score': 50}

        avg_recent = statistics.mean(recent_scores)
        poor_sleep_days = len([s for s in recent_scores if s < 60])

        trend_score = max(0, min(100, avg_recent - (poor_sleep_days * 5)))

        return {
            'status': 'analyzed',
            'recent_avg': avg_recent,
            'poor_sleep_days': poor_sleep_days,
            'trend_score': trend_score,
            'interpretation': "good" if avg_recent > 75 else "concerning" if avg_recent < 60 else "moderate"
        }

    def _interpret_recovery_status(self, status: str, autonomic: float, sleep: float) -> str:
        """Interpret overall recovery status."""
        interpretations = {
            "poor": f"Critical recovery deficit detected. Both autonomic (HRV) and sleep systems showing stress. Immediate intervention required.",
            "declining": f"Recovery systems under stress. Autonomic: {autonomic:.0f}%, Sleep: {sleep:.0f}%. Reduce training load.",
            "moderate": f"Recovery systems managing but not optimal. Monitor trends closely.",
            "good": f"Recovery systems functioning well. Continue current approach."
        }
        return interpretations.get(status, "Unknown recovery status")

    def _generate_critical_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate critical recommendations based on all analysis."""
        recommendations = []

        # Contradictory signals
        contradictions = results.get('contradictory_signals', {}).get('contradictions', [])
        critical_contradictions = [c for c in contradictions if c.get('risk') == 'critical']
        if critical_contradictions:
            recommendations.append("🚨 CRITICAL: Contradictory readiness signals detected - form metrics may be misleading due to chronic overreaching")

        # Health anomalies
        health_anomalies = results.get('health_anomalies', {})
        if health_anomalies.get('critical_anomalies', 0) > 0:
            recommendations.append("⚠️ URGENT: Critical health data anomalies detected - verify scale accuracy and data entry")

        # Sleep-training risks
        sleep_risks = results.get('sleep_training_risks', {})
        if sleep_risks.get('high_risk_days', 0) > 0:
            recommendations.append("💤 HIGH RISK: Multiple high-load training days after poor sleep - mandatory rest protocol needed")

        # Training imbalances
        imbalances = results.get('training_imbalances', {}).get('imbalances', [])
        strength_deficit = next((imb for imb in imbalances if imb['type'] == 'strength_deficit'), None)
        if strength_deficit and strength_deficit['risk_level'] == 'critical':
            # Calculate 30-day trend for comparison
            recent_pct = results.get('training_imbalances', {}).get('recent_strength_pct', strength_deficit['current_pct'])
            if abs(recent_pct - strength_deficit['current_pct']) > 1.0:
                # Show trend if significantly different
                trend = "improving" if recent_pct > strength_deficit['current_pct'] else "declining"
                recommendations.append(f"💪 STRENGTH DEFICIT: {strength_deficit['current_pct']:.1f}% (60d avg) → {recent_pct:.1f}% (30d recent, {trend}) - maintain or increase current level")

        # Recovery systems
        recovery = results.get('recovery_system_analysis', {}).get('overall_status', {})
        if recovery.get('concern_level') == 'critical':
            recommendations.append("🔄 RECOVERY CRISIS: Multiple physiological systems failing to recover - immediate deload required")

        if not recommendations:
            recommendations.append("✅ No critical issues detected in current analysis period")

        return recommendations