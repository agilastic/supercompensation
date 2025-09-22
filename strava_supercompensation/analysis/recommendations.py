"""Training recommendation engine based on supercompensation analysis."""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..config import config
from ..db import get_db
from ..db.models import Metric, Activity
from .multisport_metrics import MultiSportCalculator
from .garmin_scores_analyzer import get_garmin_scores_analyzer
from .hrv_baseline_analyzer import get_hrv_baseline_analyzer
from .multisystem_recovery import MultiSystemRecoveryModel, create_recovery_analysis_report
from .environmental_factors import EnvironmentalAnalyzer, EnvironmentalProfile


class TrainingRecommendation(Enum):
    """Training recommendation types."""

    REST = "REST"  # Complete rest or very light activity
    RECOVERY = "RECOVERY"  # Active recovery, easy pace
    EASY = "EASY"  # Easy aerobic training
    MODERATE = "MODERATE"  # Moderate intensity training
    HARD = "HARD"  # High intensity training
    PEAK = "PEAK"  # Peak performance, race or time trial


class RecommendationEngine:
    """Generate training recommendations based on current state."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.garmin_scores = get_garmin_scores_analyzer(user_id)  # Garmin device scores
        self.hrv_baseline = get_hrv_baseline_analyzer(user_id)  # HRV baseline analysis
        self.multisystem_recovery = MultiSystemRecoveryModel()  # Multi-system recovery
        self.environmental_analyzer = EnvironmentalAnalyzer()

    def get_recommendation(self) -> Dict[str, any]:
        """Get training recommendation for today."""
        with self.db.get_session() as session:
            # Get latest metric
            latest_metric = session.query(Metric).order_by(Metric.date.desc()).first()

            if not latest_metric:
                return self._no_data_recommendation()

            # Analyze current state
            fitness = latest_metric.fitness
            fatigue = latest_metric.fatigue
            form = latest_metric.form

            # Get recent training pattern
            recent_loads = self._get_recent_loads(session, days=7)

            # Get recent activities to check for races or intense efforts
            recent_activities = self._get_recent_activities(session, days=3)

            # Check if training was already completed today
            today_activities = self._get_today_activities(session)

            # Get wellness data for enhanced recommendations
            wellness_readiness = self._get_wellness_readiness()

            # Get HRV baseline analysis (German sports science)
            hrv_baseline_analysis = self.hrv_baseline.calculate_comprehensive_readiness()

            # Get all recent activities for advanced recovery analysis
            all_recent_activities = self._get_recent_activities(session, days=14)

            # Generate advanced recovery analysis
            recovery_analysis = create_recovery_analysis_report(all_recent_activities)

            # Generate recommendation
            recommendation = self._calculate_recommendation(
                fitness, fatigue, form, recent_loads, recent_activities,
                wellness_readiness, recovery_analysis, hrv_baseline_analysis,
                today_activities
            )

            # Update database
            latest_metric.recommendation = recommendation["type"]
            session.commit()

            return recommendation

    def _calculate_recommendation(
        self,
        fitness: float,
        fatigue: float,
        form: float,
        recent_loads: List[float],
        recent_activities: List[Dict] = None,
        wellness_readiness: Dict = None,
        recovery_analysis: Dict = None,
        hrv_baseline_analysis: Dict = None,
        today_activities: List[Dict] = None
    ) -> Dict[str, any]:
        """Calculate training recommendation based on metrics and recent activities."""

        # CHECK TODAY'S ACTIVITIES FIRST
        if today_activities:
            total_today_load = sum(a.get('training_load', 0) for a in today_activities)
            total_today_duration = sum(a.get('moving_time', 0) for a in today_activities) / 60  # minutes

            # If any meaningful training already done today (adjusted thresholds)
            # 30+ TSS or 20+ minutes is enough to count as "already trained"
            if total_today_load >= 30 or total_today_duration >= 20:
                activity_names = ', '.join([a.get('name', 'Activity') for a in today_activities[:2]])
                return {
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        f"Training already completed today ({len(today_activities)} {'activity' if len(today_activities) == 1 else 'activities'})",
                        f"Today's load: {total_today_load:.0f} TSS, {total_today_duration:.0f} minutes",
                        f"Activities: {activity_names}",
                        "Focus on recovery and nutrition"
                    ],
                    "metrics": {
                        "fitness": round(fitness, 1),
                        "fatigue": round(fatigue, 1),
                        "form": round(form, 1),
                        "today_load": round(total_today_load, 0),
                        "today_duration": round(total_today_duration, 0)
                    }
                }

        # Normalize values for decision making
        fatigue_ratio = fatigue / (fitness + 1e-5)  # Prevent division by zero
        avg_recent_load = sum(recent_loads) / len(recent_loads) if recent_loads else 0

        # Initialize recommendation with wellness context
        recommendation = {
            "type": TrainingRecommendation.MODERATE.value,
            "intensity": "moderate",
            "duration_minutes": 60,
            "notes": [],
            "metrics": {
                "fitness": round(fitness, 1),
                "fatigue": round(fatigue, 1),
                "form": round(form, 1),
                "fatigue_ratio": round(fatigue_ratio, 2),
            },
            "wellness": wellness_readiness
        }

        # Apply wellness-based modifications early in decision tree
        wellness_modifier = 1.0
        if wellness_readiness and wellness_readiness.get('readiness_score') is not None:
            readiness_score = wellness_readiness['readiness_score']

            # Get training load modifier from wellness data
            wellness_modifier = self.garmin_scores.get_training_load_modifier()

            # Add wellness info to metrics
            recommendation["metrics"]["wellness_score"] = round(readiness_score, 1)
            recommendation["metrics"]["wellness_modifier"] = round(wellness_modifier, 2)

            # Early intervention for very poor wellness
            if readiness_score < 35:
                return {
                    **recommendation,
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        "Poor wellness readiness detected",
                        f"Readiness score: {readiness_score:.0f}/100",
                        "Prioritize recovery today"
                    ] + wellness_readiness.get('recommendations', [])[:2]
                }
            elif readiness_score < 50:
                # Force recovery/easy training regardless of other metrics
                recommendation.update({
                    "type": TrainingRecommendation.RECOVERY.value,
                    "intensity": "recovery",
                    "duration_minutes": max(20, int(30 * wellness_modifier)),
                })
                recommendation["notes"].append(f"Wellness score ({readiness_score:.0f}/100) suggests easy training")
                if wellness_readiness.get('recommendations'):
                    recommendation["notes"].extend(wellness_readiness['recommendations'][:1])

        # HRV BASELINE ANALYSIS: German sports science methodology
        hrv_baseline_modifier = 1.0
        if hrv_baseline_analysis and hrv_baseline_analysis.get('readiness_score') is not None:
            readiness_score = hrv_baseline_analysis['readiness_score']
            readiness_level = hrv_baseline_analysis.get('readiness_level', 'moderate')

            # Add advanced wellness metrics
            recommendation["metrics"]["german_readiness_score"] = round(readiness_score, 1)
            recommendation["metrics"]["german_readiness_level"] = readiness_level

            # Get HRV baseline information if available
            if hrv_baseline_analysis.get('hrv_baseline'):
                baseline = hrv_baseline_analysis['hrv_baseline']
                recommendation["metrics"]["hrv_baseline_mean"] = round(baseline['mean_rmssd'], 1)
                recommendation["metrics"]["hrv_coefficient_variation"] = round(baseline['coefficient_variation'], 1)

            # Apply readiness-based modifications
            if readiness_level == 'critical':
                return {
                    **recommendation,
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        f"CRITICAL wellness state detected (German methodology)",
                        f"Readiness score: {readiness_score:.0f}/100",
                        "Complete rest required for autonomic recovery"
                    ] + hrv_baseline_analysis.get('recommendations', [])[:2]
                }
            elif readiness_level == 'poor':
                recommendation.update({
                    "type": TrainingRecommendation.RECOVERY.value,
                    "intensity": "recovery",
                    "duration_minutes": max(20, int(30 * 0.6)),
                })
                recommendation["notes"].append(f"German methodology indicates poor readiness ({readiness_score:.0f}/100)")

            # Calculate HRV baseline-based training modifier
            if readiness_score >= 85:
                hrv_baseline_modifier = 1.2   # Enhanced capacity
            elif readiness_score >= 70:
                hrv_baseline_modifier = 1.0   # Normal capacity
            elif readiness_score >= 55:
                hrv_baseline_modifier = 0.8   # Reduced capacity
            elif readiness_score >= 40:
                hrv_baseline_modifier = 0.5   # Severely reduced
            else:
                hrv_baseline_modifier = 0.0   # No training capacity

            recommendation["metrics"]["hrv_baseline_modifier"] = round(hrv_baseline_modifier, 2)

            # Add component-specific insights
            components = hrv_baseline_analysis.get('components', {})
            if 'hrv_baseline_deviation' in components:
                hrv_comp = components['hrv_baseline_deviation']
                deviation_status = hrv_comp.get('deviation_status', 'unknown')
                if deviation_status in ['significantly_below', 'significantly_above']:
                    recommendation["notes"].append(f"HRV {deviation_status.replace('_', ' ')} baseline")

        # ADVANCED RECOVERY ANALYSIS: Multi-system recovery assessment
        recovery_modifier = 1.0
        if recovery_analysis and recovery_analysis.get('recovery_status'):
            recovery_status = recovery_analysis['recovery_status']
            overall_recovery = recovery_status.get('overall', 70)

            # Add recovery metrics to recommendation
            recommendation["metrics"]["recovery_overall"] = round(overall_recovery, 1)
            recommendation["metrics"]["recovery_neural"] = round(recovery_status.get('neural', 70), 1)
            recommendation["metrics"]["recovery_structural"] = round(recovery_status.get('structural', 70), 1)

            # Calculate recovery-based training modifier
            if overall_recovery < 40:
                recovery_modifier = 0.0  # Complete rest
            elif overall_recovery < 60:
                recovery_modifier = 0.4  # Light activity only
            elif overall_recovery < 75:
                recovery_modifier = 0.7  # Moderate training
            else:
                recovery_modifier = 1.0  # Full training ready

            # Apply most restrictive modifier (wellness vs recovery)
            final_modifier = min(wellness_modifier, recovery_modifier)
            recommendation["metrics"]["recovery_modifier"] = round(recovery_modifier, 2)
            recommendation["metrics"]["final_modifier"] = round(final_modifier, 2)

            # Get system-specific insights
            recovery_recommendations = recovery_analysis.get('recommendations', {})
            if recovery_recommendations.get('primary_recommendation'):
                recovery_rec = recovery_recommendations['primary_recommendation']

                # Override recommendation based on recovery analysis
                if recovery_rec == 'REST' and overall_recovery < 50:
                    return {
                        **recommendation,
                        "type": TrainingRecommendation.REST.value,
                        "intensity": "rest",
                        "duration_minutes": 0,
                        "notes": [
                            f"Multi-system recovery analysis: {recovery_recommendations.get('rationale', 'Rest required')}",
                            f"Overall recovery: {overall_recovery:.0f}/100",
                            f"Neural system: {recovery_status.get('neural', 70):.0f}/100",
                            f"Structural system: {recovery_status.get('structural', 70):.0f}/100"
                        ]
                    }
                elif recovery_rec in ['ACTIVE_RECOVERY', 'LIGHT_TRAINING'] and overall_recovery < 65:
                    recommendation.update({
                        "type": TrainingRecommendation.RECOVERY.value,
                        "intensity": "recovery",
                        "duration_minutes": max(20, int(40 * final_modifier)),
                    })
                    recommendation["notes"].append(f"Recovery systems suggest: {recovery_recommendations.get('intensity_guidance', 'Light activity')}")

        # ENVIRONMENTAL ANALYSIS: Adjust for current conditions
        environmental_modifier = 1.0
        environmental_conditions = self._get_environmental_conditions()
        if environmental_conditions:
            env_analysis = self.environmental_analyzer.analyze_environmental_impact(
                environmental_conditions, activity_type="endurance"
            )

            # Add environmental metrics
            recommendation["metrics"]["env_load_multiplier"] = round(env_analysis['load_multiplier'], 2)
            recommendation["metrics"]["env_performance_impact"] = round(env_analysis['performance_impact'], 1)
            recommendation["metrics"]["env_stress_level"] = env_analysis['overall_stress_level']

            # Apply environmental adjustments
            environmental_modifier = env_analysis['load_multiplier']

            # Add environmental recommendations
            if env_analysis['specific_stressors']:
                recommendation["notes"].extend([
                    f"Environmental stress: {', '.join(env_analysis['specific_stressors'][:2])}"
                ])

            # Adjust recommendation based on extreme environmental conditions
            if env_analysis['overall_stress_level'] == 'extreme':
                return {
                    **recommendation,
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        "EXTREME environmental conditions detected",
                        f"Safety assessment: {env_analysis.get('safety_warnings', ['Dangerous conditions'])[0] if env_analysis.get('safety_warnings') else 'Unsafe for training'}",
                        "Training not recommended"
                    ] + env_analysis.get('recommendations', [])[:2]
                }
            elif env_analysis['overall_stress_level'] == 'high_stress':
                # Reduce intensity for high environmental stress
                if recommendation["type"] in ["HARD", "PEAK"]:
                    recommendation.update({
                        "type": TrainingRecommendation.MODERATE.value,
                        "intensity": "moderate",
                        "duration_minutes": max(30, int(recommendation["duration_minutes"] * 0.7))
                    })
                    recommendation["notes"].append("Intensity reduced due to environmental stress")
            else:
                # Apply environmental modifier to duration for all non-extreme conditions
                if environmental_modifier != 1.0 and recommendation["duration_minutes"] > 0:
                    # Adjust duration inversely to load multiplier
                    # Higher environmental stress = shorter duration
                    adjusted_duration = recommendation["duration_minutes"] / environmental_modifier
                    recommendation["duration_minutes"] = max(20, int(adjusted_duration))

                    # Add note about environmental adjustment
                    if environmental_modifier > 1.0:
                        recommendation["notes"].append(
                            f"Duration reduced by {int((1 - 1/environmental_modifier) * 100)}% due to environmental conditions"
                        )
                    elif environmental_modifier < 1.0:
                        recommendation["notes"].append(
                            f"Duration increased by {int((1/environmental_modifier - 1) * 100)}% due to favorable conditions"
                        )

        # PRIORITY 1: Check for recent race or very intense activity (LEGACY - replaced by advanced recovery)
        # if recent_activities:
        #     recovery_analysis = self._analyze_recovery_needs(recent_activities)
        #     if recovery_analysis["needs_recovery"]:
        #         return recovery_analysis["recommendation"]

        # Check for very high recent load (last 2 days)
        if recent_loads and len(recent_loads) >= 2:
            last_two_days_load = sum(recent_loads[:2])
            if last_two_days_load > avg_recent_load * 3:  # Much higher than average
                if recommendation["type"] not in [TrainingRecommendation.REST.value, TrainingRecommendation.RECOVERY.value]:
                    recommendation.update({
                        "type": TrainingRecommendation.RECOVERY.value,
                        "intensity": "recovery",
                        "duration_minutes": 30,
                        "notes": ["Very high recent training load", "Recovery needed"],
                    })

        # PRIORITY 2: High fatigue - recommend rest or recovery
        if fatigue_ratio > config.FATIGUE_HIGH_THRESHOLD:
            if fatigue > fitness * 1.5:
                recommendation.update({
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": ["Very high fatigue detected", "Complete rest recommended"],
                })
            else:
                recommendation.update({
                    "type": TrainingRecommendation.RECOVERY.value,
                    "intensity": "recovery",
                    "duration_minutes": 30,
                    "notes": ["High fatigue", "Active recovery recommended"],
                })

        # Good form - ready for quality training (but only if not recently raced and wellness allows)
        elif form > config.FORM_HIGH_THRESHOLD and recommendation["type"] not in [
            TrainingRecommendation.REST.value,
            TrainingRecommendation.RECOVERY.value,
            TrainingRecommendation.EASY.value
        ]:
            # Check if wellness allows high intensity
            wellness_allows_intensity = wellness_modifier >= 0.85 if wellness_readiness else True

            if form > fitness * 0.1 and wellness_allows_intensity:  # Peak form + good wellness
                duration = int(90 * wellness_modifier)
                notes = ["Excellent form!", "Perfect for race or time trial"]
                if wellness_modifier < 1.0:
                    notes.append(f"Duration adjusted for wellness ({wellness_modifier:.1f}x)")

                recommendation.update({
                    "type": TrainingRecommendation.PEAK.value,
                    "intensity": "peak",
                    "duration_minutes": duration,
                    "notes": notes,
                })
            elif wellness_allows_intensity:
                duration = int(75 * wellness_modifier)
                notes = ["Good form", "Ready for high intensity training"]
                if wellness_modifier < 1.0:
                    notes.append(f"Duration adjusted for wellness ({wellness_modifier:.1f}x)")

                recommendation.update({
                    "type": TrainingRecommendation.HARD.value,
                    "intensity": "hard",
                    "duration_minutes": duration,
                    "notes": notes,
                })
            else:
                # Good form but poor wellness - downgrade intensity
                recommendation.update({
                    "type": TrainingRecommendation.EASY.value,
                    "intensity": "easy",
                    "duration_minutes": int(60 * wellness_modifier),
                    "notes": ["Good form but wellness suggests easy training", f"Wellness score: {wellness_readiness['readiness_score']:.0f}/100"],
                })

        # Poor form - easy training only
        elif form < config.FORM_LOW_THRESHOLD:
            recommendation.update({
                "type": TrainingRecommendation.EASY.value,
                "intensity": "easy",
                "duration_minutes": 45,
                "notes": ["Low form", "Keep intensity easy"],
            })

        # Neutral state - moderate training
        else:
            # Check training monotony
            if len(set(recent_loads)) == 1 and recent_loads[0] > 0:
                recommendation["notes"].append("Consider varying training intensity")

            # Check if building or maintaining
            if fitness < 30:  # Low fitness base
                recommendation.update({
                    "type": TrainingRecommendation.EASY.value,
                    "intensity": "easy",
                    "duration_minutes": 60,
                    "notes": ["Building fitness base", "Focus on consistency"],
                })
            else:
                recommendation["notes"].append("Maintain current training rhythm")

        # Add progressive overload advice
        if avg_recent_load > 0:
            if recommendation["type"] in [TrainingRecommendation.MODERATE.value, TrainingRecommendation.HARD.value]:
                suggested_load = avg_recent_load * 1.1  # 10% progression
                recommendation["suggested_load"] = round(suggested_load, 0)
                recommendation["notes"].append(f"Consider training load around {suggested_load:.0f}")

        return recommendation

    def _get_recent_loads(self, session, days: int = 7) -> List[float]:
        """Get recent daily training loads."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        metrics = session.query(Metric).filter(
            Metric.date >= cutoff_date
        ).order_by(Metric.date.desc()).all()

        return [m.daily_load for m in metrics]

    def _get_recent_activities(self, session, days: int = 3) -> List[Dict]:
        """Get recent activities with details."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        activities = session.query(Activity).filter(
            Activity.start_date >= cutoff_date
        ).order_by(Activity.start_date.desc()).all()

        return [
            {
                "strava_id": a.strava_id,
                "name": a.name,
                "type": a.type,
                "workout_type": a.workout_type,  # 1=Race flag from Strava
                "start_date": a.start_date,
                "distance": a.distance,
                "moving_time": a.moving_time,
                "training_load": a.training_load,
                "average_heartrate": a.average_heartrate,
                "max_heartrate": a.max_heartrate,
                "average_speed": a.average_speed,
            }
            for a in activities
        ]

    def _get_today_activities(self, session) -> List[Dict]:
        """Get activities completed today."""
        # Get activities from today (UTC)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        activities = session.query(Activity).filter(
            Activity.start_date >= today_start,
            Activity.start_date < today_end
        ).order_by(Activity.start_date.desc()).all()

        return [
            {
                "strava_id": a.strava_id,
                "name": a.name,
                "type": a.type,
                "workout_type": a.workout_type,
                "start_date": a.start_date,
                "distance": a.distance,
                "moving_time": a.moving_time,
                "training_load": a.training_load,
                "average_heartrate": a.average_heartrate,
                "max_heartrate": a.max_heartrate,
                "average_speed": a.average_speed,
            }
            for a in activities
        ]

    def _days_since_intense_activity(self, activities: List[Dict]) -> Optional[int]:
        """Calculate days since last race or very intense activity.

        Returns:
            Number of days since intense activity, or None if no intense activity found
        """
        if not activities:
            return None

        today = datetime.utcnow().date()

        for activity in activities:
            # PRIORITY 1: Check Strava's official workout_type flag
            # workout_type: 0=Default, 1=Race, 2=Long Run, 3=Workout
            is_race = activity.get("workout_type") == 1

            # FALLBACK: Check if it's a race by name (if not flagged)
            if not is_race:
                name_lower = (activity.get("name") or "").lower()
                is_race = any(keyword in name_lower for keyword in ["race", "competition", "marathon", "10k", "5k", "triathlon"])

            # Check for very high intensity based on various factors
            is_intense = False

            # Check by distance and time (likely race efforts)
            distance = activity.get("distance", 0)
            time_minutes = (activity.get("moving_time", 0) / 60) if activity.get("moving_time") else 0

            # Half marathon distance (around 21km)
            if 20000 <= distance <= 22000 and 70 <= time_minutes <= 180:
                is_intense = True
            # Marathon distance
            elif 41000 <= distance <= 43000:
                is_intense = True
            # 10K race effort
            elif 9500 <= distance <= 10500 and 28 <= time_minutes <= 70:
                is_intense = True
            # 5K race effort
            elif 4800 <= distance <= 5200 and 13 <= time_minutes <= 35:
                is_intense = True

            # Check by training load (very high load indicates intense effort)
            load = activity.get("training_load", 0)
            if load > 200:  # Very high single-session load
                is_intense = True

            # Check by average heart rate (if near max)
            if activity.get("average_heartrate") and activity.get("max_heartrate"):
                avg_hr = activity["average_heartrate"]
                max_hr = activity["max_heartrate"]
                if avg_hr > max_hr * 0.85:  # Sustained effort above 85% max HR
                    is_intense = True

            # Check running pace for intensity
            if activity.get("type") == "Run" and activity.get("average_speed") and time_minutes > 20:
                speed_ms = activity["average_speed"]
                pace_min_per_km = (1000 / speed_ms) / 60 if speed_ms > 0 else 999

                # Fast pace sustained for significant time
                if pace_min_per_km < 4.5 and time_minutes > 30:  # Sub 4:30/km for 30+ min
                    is_intense = True
                elif pace_min_per_km < 5.0 and time_minutes > 60:  # Sub 5:00/km for 60+ min
                    is_intense = True

            if is_race or is_intense:
                activity_date = activity["start_date"].date() if hasattr(activity["start_date"], 'date') else activity["start_date"]
                days_since = (today - activity_date).days
                return days_since

        return None

    def _analyze_recovery_needs(self, activities: List[Dict]) -> Dict:
        """Analyze recovery needs based on recent activities with sport-specific protocols."""
        multisport_calc = MultiSportCalculator()
        today = datetime.utcnow().date()

        for activity in activities:
            # Check if it's a race
            is_race = activity.get("workout_type") == 1
            if not is_race:
                name_lower = (activity.get("name") or "").lower()
                is_race = any(keyword in name_lower for keyword in ["race", "competition", "marathon", "10k", "5k", "triathlon"])

            # Calculate required recovery time
            recovery_hours = multisport_calc.calculate_recovery_time(activity)

            # Calculate hours since activity
            activity_datetime = activity["start_date"]
            if hasattr(activity_datetime, 'date'):
                activity_date = activity_datetime.date()
                activity_dt = activity_datetime
            else:
                activity_date = activity_datetime
                activity_dt = datetime.combine(activity_date, datetime.min.time())

            now = datetime.utcnow()
            hours_since = (now - activity_dt).total_seconds() / 3600

            # Check if still in recovery period
            if hours_since < recovery_hours:
                days_since = int(hours_since / 24)
                remaining_hours = recovery_hours - hours_since

                # Get sport-specific recommendations
                sport_type = multisport_calc.get_sport_type(activity.get("type", ""))
                cross_training = multisport_calc.get_cross_training_recommendations(sport_type, activity.get("training_load", 0))

                if days_since == 0:
                    # Same day as intense activity
                    return {
                        "needs_recovery": True,
                        "recommendation": {
                            "type": TrainingRecommendation.REST.value,
                            "intensity": "rest",
                            "duration_minutes": 0,
                            "notes": [
                                f"Recovery needed after today's {activity.get('type', 'activity')}",
                                f"Complete rest for {remaining_hours:.0f} more hours",
                                "Focus on hydration, nutrition, and sleep"
                            ],
                            "cross_training": [],
                            "metrics": {}
                        }
                    }
                elif days_since == 1 and remaining_hours > 12:
                    # Day after with significant recovery needed
                    notes = [
                        f"Recovery day after yesterday's {activity.get('type', 'activity')}",
                        f"Still need {remaining_hours:.0f} hours of recovery"
                    ]

                    if is_race:
                        notes.append("Post-race recovery is critical for adaptation")

                    # Add cross-training options for non-impact sports
                    if sport_type.value == "running" and len(cross_training) > 0:
                        notes.append(f"Alternative: {cross_training[0]['activity']} - {cross_training[0]['duration']}")

                    return {
                        "needs_recovery": True,
                        "recommendation": {
                            "type": TrainingRecommendation.REST.value,
                            "intensity": "rest",
                            "duration_minutes": 0,
                            "notes": notes,
                            "cross_training": cross_training[:2],  # Show top 2 options
                            "metrics": {}
                        }
                    }
                elif remaining_hours > 0:
                    # Partial recovery needed
                    duration = min(30, int(remaining_hours / 2))
                    notes = [
                        f"Active recovery phase ({remaining_hours:.0f}h recovery remaining)",
                        f"Keep intensity very low"
                    ]

                    # Suggest cross-training if primary sport was high-impact
                    if sport_type.value == "running" and len(cross_training) > 0:
                        notes.append(f"Consider: {cross_training[0]['activity']} instead")

                    return {
                        "needs_recovery": True,
                        "recommendation": {
                            "type": TrainingRecommendation.RECOVERY.value,
                            "intensity": "recovery",
                            "duration_minutes": duration,
                            "notes": notes,
                            "cross_training": cross_training[:3],
                            "metrics": {}
                        }
                    }

        return {"needs_recovery": False, "recommendation": {}}

    def _get_wellness_readiness(self) -> Dict[str, any]:
        """Get wellness readiness data for today."""
        try:
            return self.garmin_scores.get_readiness_score()
        except Exception:
            # If wellness data isn't available, return None
            return None

    def _no_data_recommendation(self) -> Dict[str, any]:
        """Recommendation when no data is available."""
        return {
            "type": "NO_DATA",
            "intensity": "unknown",
            "duration_minutes": 0,
            "notes": ["No training data available", "Sync activities first"],
            "metrics": {
                "fitness": 0,
                "fatigue": 0,
                "form": 0,
                "fatigue_ratio": 0,
            }
        }

    def get_weekly_plan(self) -> List[Dict[str, any]]:
        """Generate a weekly training plan."""
        recommendations = []
        current_state = self._get_current_state()

        if not current_state:
            return []

        fitness = current_state["fitness"]
        fatigue = current_state["fatigue"]

        # Check if already trained today (for day 0)
        today_activities = None
        if self.db:
            with self.db.get_session() as session:
                today_activities = self._get_today_activities(session)

        # Simulate a week of training
        for day in range(7):
            # Decay fitness and fatigue
            fitness *= 0.98  # Approximate daily decay
            fatigue *= 0.90  # Faster fatigue decay

            form = fitness - fatigue

            # Special handling for today (day 0)
            if day == 0 and today_activities:
                total_today_load = sum(a.get('training_load', 0) for a in today_activities)
                total_today_duration = sum(a.get('moving_time', 0) for a in today_activities) / 60

                # If already trained today (same logic as main recommendation)
                if total_today_load >= 30 or total_today_duration >= 20:
                    rec_type = TrainingRecommendation.REST.value
                    load = total_today_load  # Use actual load from today
                else:
                    # Not enough training yet today
                    rec_type = TrainingRecommendation.RECOVERY.value
                    load = 20
            # Regular pattern for future days
            elif day in [2, 5]:  # Hard days
                if form > 0:
                    rec_type = TrainingRecommendation.HARD.value
                    load = 100
                else:
                    rec_type = TrainingRecommendation.MODERATE.value
                    load = 60
            elif day in [0, 6]:  # Rest/recovery days (day 0 handled above)
                rec_type = TrainingRecommendation.RECOVERY.value
                load = 20
            else:  # Moderate days
                rec_type = TrainingRecommendation.MODERATE.value
                load = 60

            # Update state for next day
            fitness += load * config.FITNESS_MAGNITUDE / config.FITNESS_DECAY_RATE
            fatigue += load * config.FATIGUE_MAGNITUDE / config.FATIGUE_DECAY_RATE

            recommendations.append({
                "day": day + 1,
                "date": (datetime.utcnow() + timedelta(days=day)).date().isoformat(),
                "recommendation": rec_type,
                "suggested_load": load,
                "predicted_form": round(form, 1),
            })

        return recommendations

    def _get_current_state(self) -> Optional[Dict[str, float]]:
        """Get current fitness and fatigue values."""
        with self.db.get_session() as session:
            latest_metric = session.query(Metric).order_by(Metric.date.desc()).first()
            if latest_metric:
                return {
                    "fitness": latest_metric.fitness,
                    "fatigue": latest_metric.fatigue,
                    "form": latest_metric.form,
                }
        return None

    def _activity_to_dict(self, activity) -> Dict:
        """Convert Activity model to dictionary for advanced recovery analysis."""
        return {
            'start_date': activity.start_date,
            'type': activity.type,
            'moving_time': activity.moving_time,
            'distance': activity.distance,
            'average_heartrate': activity.average_heartrate,
            'max_heartrate': activity.max_heartrate,
            'average_speed': activity.average_speed,
            'average_watts': activity.average_watts,
            'training_load': activity.training_load,
            'workout_type': activity.workout_type
        }

    def _get_environmental_conditions(self) -> Optional[EnvironmentalProfile]:
        """
        Get environmental conditions from recent activity data.

        Uses real data from Strava activities:
        - Temperature from recent activities
        - Elevation data from recent activities
        - Estimated humidity/wind based on temperature and season
        """
        try:
            with self.db.get_session() as session:
                # Get recent activity with temperature data (last 7 days)
                recent_activity = session.query(Activity).filter(
                    Activity.start_date >= datetime.now() - timedelta(days=7)
                ).order_by(Activity.start_date.desc()).first()

                # Extract environmental data from activity
                temperature_celsius = None
                altitude_meters = None

                if recent_activity and recent_activity.raw_data:
                    import json
                    raw_data = json.loads(recent_activity.raw_data)
                    temperature_celsius = raw_data.get('average_temp')

                    # Use elevation data if available
                    if recent_activity.total_elevation_gain:
                        # Estimate altitude from elevation gain (rough approximation)
                        altitude_meters = recent_activity.total_elevation_gain
                    else:
                        altitude_meters = raw_data.get('elev_high', 200.0)

                # Fall back to seasonal defaults if no recent activity data
                if temperature_celsius is None:
                    # Estimate based on current month (Northern Hemisphere)
                    current_month = datetime.now().month
                    if current_month in [12, 1, 2]:  # Winter
                        temperature_celsius = 5.0
                    elif current_month in [3, 4, 5]:  # Spring
                        temperature_celsius = 15.0
                    elif current_month in [6, 7, 8]:  # Summer
                        temperature_celsius = 25.0
                    else:  # Fall
                        temperature_celsius = 12.0

                if altitude_meters is None:
                    altitude_meters = 200.0  # Default near sea level

                # Estimate humidity based on temperature
                if temperature_celsius < 10:
                    humidity_percent = 70.0  # Higher humidity in cold
                elif temperature_celsius > 25:
                    humidity_percent = 45.0  # Lower humidity in heat
                else:
                    humidity_percent = 55.0  # Moderate humidity

                # Estimate wind based on season and temperature
                if temperature_celsius < 5 or temperature_celsius > 30:
                    wind_speed_ms = 5.0  # More wind in extreme conditions
                else:
                    wind_speed_ms = 3.0  # Light breeze in moderate conditions

                return EnvironmentalProfile(
                    temperature_celsius=temperature_celsius,
                    humidity_percent=humidity_percent,
                    altitude_meters=altitude_meters,
                    air_quality_index=45,          # Default good air quality
                    wind_speed_ms=wind_speed_ms
                )

        except Exception as e:
            # Fall back to moderate conditions if data fetch fails
            return EnvironmentalProfile(
                temperature_celsius=15.0,
                humidity_percent=55.0,
                altitude_meters=200.0,
                air_quality_index=45,
                wind_speed_ms=3.0
            )