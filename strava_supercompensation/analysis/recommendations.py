"""Training recommendation engine based on supercompensation analysis."""

from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from ..config import config
from ..db import get_db
from ..db.models import Metric, Activity, PeriodizationState
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

        # Initialize sport usage tracking
        self._plan_sport_usage = {}
        self._week_sport_usage = {}
        self._current_plan_days = []

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

        # 🚨 CRITICAL HRV SAFETY CHECK FIRST (Olympic/Sports Medicine Protocol)
        if hrv_baseline_analysis and hrv_baseline_analysis.get('readiness_score') is not None:
            hrv_score = hrv_baseline_analysis['readiness_score']
            if hrv_score < 45:  # Critical threshold for comprehensive readiness
                return {
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "activity": "Complete Rest",
                    "activity_rationale": "Comprehensive readiness critically low - recovery required",
                    "alternative_activities": ["Light stretching", "Meditation", "Sleep"],
                    "notes": [
                        f"🚨 CRITICAL: Comprehensive Readiness {hrv_score:.0f}/100",
                        "German sports science threshold: >60 for training",
                        "Multiple biomarkers indicate overreaching risk",
                        "Monitor daily until readiness >60"
                    ],
                    "metrics": {
                        "comprehensive_readiness": hrv_score,
                        "readiness_status": "CRITICAL",
                        "training_contraindicated": True
                    }
                }

        # CHECK TODAY'S ACTIVITIES FIRST
        if today_activities:
            # Apply sport-specific load multipliers
            total_today_load = sum(
                a.get('training_load', 0) * config.get_sport_load_multiplier(a.get('type', 'Unknown'))
                for a in today_activities
            )
            total_today_duration = sum(a.get('moving_time', 0) for a in today_activities) / 60  # minutes

            # If any meaningful training already done today (adjusted thresholds)
            # 30+ TSS or 20+ minutes is enough to count as "already trained"
            if total_today_load >= 30 or total_today_duration >= 20:
                activity_names = ', '.join([a.get('name', 'Activity') for a in today_activities[:2]])

                # Get sport-specific REST recommendations
                sport_recommendation = self._get_sport_specific_recommendation(
                    TrainingRecommendation.REST.value, form, today_activities, recent_activities
                )

                return {
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "activity": sport_recommendation["activity"],
                    "activity_rationale": sport_recommendation["rationale"],
                    "alternative_activities": sport_recommendation["alternatives"],
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

        # Get sport-specific activity recommendation
        sport_recommendation = self._get_sport_specific_recommendation(
            TrainingRecommendation.MODERATE.value, form, today_activities, recent_activities
        )

        # Initialize recommendation with wellness context
        recommendation = {
            "type": TrainingRecommendation.MODERATE.value,
            "intensity": "moderate",
            "duration_minutes": 60,
            "activity": sport_recommendation["activity"],
            "activity_rationale": sport_recommendation["rationale"],
            "alternative_activities": sport_recommendation["alternatives"],
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

        # Use refined wellness metrics if available
        if wellness_readiness:
            # Prefer refined readiness over device readiness
            readiness_score = wellness_readiness.get('refined_readiness',
                                                    wellness_readiness.get('readiness_score', 70))
            wellness_modifier = wellness_readiness.get('training_modifier', 1.0)

            # Add detailed wellness info to metrics
            recommendation["metrics"]["wellness_score"] = round(readiness_score, 1)
            recommendation["metrics"]["wellness_modifier"] = round(wellness_modifier, 2)

            # Add individual wellness components if available
            if 'hrv_score' in wellness_readiness:
                recommendation["metrics"]["hrv_score"] = wellness_readiness['hrv_score']
                recommendation["metrics"]["hrv_status"] = wellness_readiness.get('hrv_status', 'unknown')
            if 'sleep_score' in wellness_readiness:
                recommendation["metrics"]["sleep_score"] = wellness_readiness['sleep_score']
            if 'stress_avg' in wellness_readiness:
                recommendation["metrics"]["stress_avg"] = wellness_readiness['stress_avg']

            # Early intervention based on refined readiness
            if wellness_modifier == 0.0:  # Complete rest needed
                return {
                    **recommendation,
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        f"🚨 Critical wellness state ({readiness_score:.0f}/100)",
                        "Complete rest required"
                    ] + wellness_readiness.get('recommendations', [])
                }
            elif wellness_modifier <= 0.5:  # Significant reduction needed
                recommendation.update({
                    "type": TrainingRecommendation.RECOVERY.value,
                    "intensity": "recovery",
                    "duration_minutes": max(20, int(30 * wellness_modifier)),
                })
                recommendation["notes"].extend(wellness_readiness.get('recommendations', []))
            elif wellness_modifier < 1.0:  # Some reduction needed
                # Add recommendations but don't override type yet
                recommendation["notes"].append(f"Wellness suggests {int((1-wellness_modifier)*100)}% reduction")
                if wellness_readiness.get('recommendations'):
                    recommendation["notes"].extend(wellness_readiness['recommendations'][:2])

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

            # Apply readiness-based modifications with HRV override
            if readiness_level == 'critical' or readiness_score < 45:
                return {
                    **recommendation,
                    "type": TrainingRecommendation.REST.value,
                    "intensity": "rest",
                    "duration_minutes": 0,
                    "notes": [
                        f"🚨 CRITICAL HRV state detected",
                        f"HRV Readiness: {readiness_score:.0f}/100 (Normal: >70)",
                        "Autonomic nervous system recovery required",
                        "Training contraindicated until HRV >55"
                    ] + hrv_baseline_analysis.get('recommendations', [])[:2]
                }
            elif readiness_level == 'poor' or readiness_score < 60:
                recommendation.update({
                    "type": TrainingRecommendation.RECOVERY.value,
                    "intensity": "recovery",
                    "duration_minutes": max(20, int(40 * 0.7)),
                })
                recommendation["notes"].append(f"⚠️ Poor HRV readiness ({readiness_score:.0f}/100) - Recovery only")

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
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        metrics = session.query(Metric).filter(
            Metric.date >= cutoff_date
        ).order_by(Metric.date.desc()).all()

        return [m.daily_load for m in metrics]

    def _get_recent_activities(self, session, days: int = 3) -> List[Dict]:
        """Get recent activities with details."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
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
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
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

        today = datetime.now(timezone.utc).date()

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
        today = datetime.now(timezone.utc).date()

        for activity in activities:
            # Check if it's a race
            is_race = activity.get("workout_type") == 1
            if not is_race:
                name_lower = (activity.get("name") or "").lower()
                is_race = any(keyword in name_lower for keyword in ["race", "competition", "marathon", "10k", "5k", "triathlon"])

            # Calculate required recovery time (with sport-specific load adjustment)
            # Adjust activity load for sport-specific impact
            adjusted_activity = activity.copy()
            adjusted_activity['training_load'] = activity.get('training_load', 0) * config.get_sport_load_multiplier(activity.get('type', 'Unknown'))
            recovery_hours = multisport_calc.calculate_recovery_time(adjusted_activity)

            # Calculate hours since activity
            activity_datetime = activity["start_date"]
            if hasattr(activity_datetime, 'date'):
                activity_date = activity_datetime.date()
                activity_dt = activity_datetime
            else:
                activity_date = activity_datetime
                activity_dt = datetime.combine(activity_date, datetime.min.time())

            now = datetime.now(timezone.utc)
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
        """Get comprehensive wellness readiness data including HRV, sleep, and stress."""
        from ..db.models import HRVData, SleepData, WellnessData

        wellness_data = {}

        try:
            # First try to get Garmin device readiness score
            device_readiness = self.garmin_scores.get_readiness_score()
            if device_readiness:
                wellness_data.update(device_readiness)
        except Exception:
            pass

        # Now get specific metrics from database for refined analysis
        with self.db.get_session() as session:
            # Get latest HRV data
            latest_hrv = session.query(HRVData).filter_by(
                user_id=self.user_id
            ).order_by(HRVData.date.desc()).first()

            # Get latest sleep data
            latest_sleep = session.query(SleepData).filter_by(
                user_id=self.user_id
            ).order_by(SleepData.date.desc()).first()

            # Get latest wellness/stress data
            latest_wellness = session.query(WellnessData).filter_by(
                user_id=self.user_id
            ).order_by(WellnessData.date.desc()).first()

            # Compile comprehensive wellness assessment
            if latest_hrv:
                wellness_data['hrv_score'] = latest_hrv.hrv_score
                wellness_data['hrv_status'] = latest_hrv.hrv_status
                wellness_data['hrv_date'] = latest_hrv.date

            if latest_sleep:
                wellness_data['sleep_score'] = latest_sleep.sleep_score
                wellness_data['sleep_duration'] = latest_sleep.total_sleep_time / 3600 if latest_sleep.total_sleep_time else None
                wellness_data['sleep_date'] = latest_sleep.date

            if latest_wellness:
                wellness_data['stress_avg'] = latest_wellness.stress_avg
                wellness_data['stress_max'] = latest_wellness.stress_max
                wellness_data['wellness_date'] = latest_wellness.date

        # Calculate refined readiness score based on all metrics
        if wellness_data:
            wellness_data['refined_readiness'] = self._calculate_refined_readiness(wellness_data)
            wellness_data['training_modifier'] = self._calculate_training_modifier(wellness_data)
            wellness_data['recommendations'] = self._generate_wellness_recommendations(wellness_data)

        return wellness_data if wellness_data else None

    def _calculate_refined_readiness(self, wellness_data: Dict) -> float:
        """Calculate readiness score using the same method as garmin_scores_analyzer.py."""
        from ..config import config

        components = {}
        overall_score = None

        # HRV Score (use configured weight from .env)
        if 'hrv_score' in wellness_data and wellness_data['hrv_score'] is not None:
            components['hrv'] = {
                'score': wellness_data['hrv_score'],
                'weight': config.READINESS_WEIGHT_HRV
            }

        # Sleep Score (use configured weight from .env)
        if 'sleep_score' in wellness_data and wellness_data['sleep_score'] is not None:
            components['sleep'] = {
                'score': wellness_data['sleep_score'],
                'weight': config.READINESS_WEIGHT_SLEEP
            }

        # Stress Score (inverted from Garmin, use configured weight from .env)
        if 'stress_avg' in wellness_data and wellness_data['stress_avg'] is not None:
            stress_score = max(0, 100 - wellness_data['stress_avg'])  # Invert stress to readiness
            components['stress'] = {
                'score': stress_score,
                'weight': config.READINESS_WEIGHT_STRESS
            }

        # Calculate weighted overall score if we have any components (same logic as garmin_scores_analyzer)
        if components:
            total_weight = sum(comp['weight'] for comp in components.values())
            if total_weight > 0:
                weighted_sum = sum(comp['score'] * comp['weight'] for comp in components.values())
                overall_score = weighted_sum / total_weight

        return round(overall_score, 1) if overall_score is not None else 70.0

    def _calculate_training_modifier(self, wellness_data: Dict) -> float:
        """Calculate a training intensity/volume modifier based on wellness."""
        refined_readiness = wellness_data.get('refined_readiness', 70)

        # Create training modifier based on readiness thresholds
        if refined_readiness >= 85:
            return 1.1  # Can handle 10% more load
        elif refined_readiness >= 70:
            return 1.0  # Normal training
        elif refined_readiness >= 60:
            return 0.85  # Reduce by 15%
        elif refined_readiness >= 50:
            return 0.7  # Reduce by 30%
        elif refined_readiness >= 40:
            return 0.5  # Half intensity/volume
        else:
            return 0.0  # Rest day recommended

    def _generate_wellness_recommendations(self, wellness_data: Dict) -> List[str]:
        """Generate specific recommendations based on wellness metrics."""
        recommendations = []
        refined_readiness = wellness_data.get('refined_readiness', 70)

        # HRV-based recommendations
        if 'hrv_status' in wellness_data:
            status = wellness_data['hrv_status']
            if status == 'unbalanced':
                recommendations.append("HRV unbalanced - consider easy aerobic work")
            elif status == 'low':
                recommendations.append("Low HRV - prioritize recovery")

        # Sleep-based recommendations
        if 'sleep_score' in wellness_data:
            sleep_score = wellness_data['sleep_score']
            if sleep_score < 60:
                recommendations.append("Poor sleep quality - reduce training intensity")
            if 'sleep_duration' in wellness_data:
                duration = wellness_data['sleep_duration']
                if duration and duration < 7:
                    recommendations.append(f"Only {duration:.1f}h sleep - consider afternoon nap")

        # Stress-based recommendations
        if 'stress_avg' in wellness_data:
            stress = wellness_data['stress_avg']
            if stress > 70:
                recommendations.append("High stress - avoid high-intensity training")
            elif stress > 50:
                recommendations.append("Moderate stress - monitor recovery closely")

        # Overall readiness recommendations
        if refined_readiness < 50:
            recommendations.insert(0, f"⚠️ Low readiness ({refined_readiness:.0f}/100) - recovery priority")
        elif refined_readiness > 80:
            recommendations.insert(0, f"✅ Excellent readiness ({refined_readiness:.0f}/100) - ready for hard training")

        return recommendations

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

    def get_training_plan(self, days: int = 7) -> List[Dict[str, any]]:
        """Generate a training plan for specified number of days (7 or 30)."""
        recommendations = []
        current_state = self._get_current_state()

        if not current_state:
            return []

        fitness = current_state["fitness"]
        fatigue = current_state["fatigue"]

        # Check if already trained today (for day 0) and get recent activities
        today_activities = None
        recent_activities = []
        periodization_state = None

        if self.db:
            with self.db.get_session() as session:
                today_activities = self._get_today_activities(session)
                recent_activities = self._get_recent_activities(session, days=7)

            # Update and get current periodization state
            periodization_state = self.update_periodization_state()

        # Reset sport usage tracking for this plan generation
        self._plan_sport_usage = {}
        self._week_sport_usage = {}  # Track sports per week

        # Generate training plan
        for day in range(days):
            # Apply daily decay first (except for day 0) using proper Banister exponential decay
            if day > 0:
                # Proper exponential decay: new_value = old_value * exp(-1/time_constant)
                import math
                fitness *= math.exp(-1 / config.FITNESS_DECAY_RATE)
                fatigue *= math.exp(-1 / config.FATIGUE_DECAY_RATE)

            form = fitness - fatigue

            # Special handling for today (day 0)
            if day == 0 and today_activities:
                # Apply sport-specific load multipliers
                total_today_load = sum(
                    a.get('training_load', 0) * config.get_sport_load_multiplier(a.get('type', 'Unknown'))
                    for a in today_activities
                )
                total_today_duration = sum(a.get('moving_time', 0) for a in today_activities) / 60

                # If already trained today (same logic as main recommendation)
                if total_today_load >= 30 or total_today_duration >= 20:
                    rec_type = TrainingRecommendation.REST.value
                    load = 0  # REST means no additional load for future calculations
                else:
                    # Not enough training yet today
                    rec_type = TrainingRecommendation.RECOVERY.value
                    load = 20
            else:
                # Determine training type based on periodization state
                if periodization_state:
                    rec_type, load = self._get_periodized_recommendation_with_state(day, form, fatigue, fitness, periodization_state)
                else:
                    # Fallback to old method if no state available
                    rec_type, load = self._get_periodized_recommendation(day, form, fatigue, fitness)

            # Apply training load for this day using proper Banister model
            if load > 0:
                fitness += load * config.FITNESS_MAGNITUDE
                fatigue += load * config.FATIGUE_MAGNITUDE
                form = fitness - fatigue

            # Store updated form and fitness for display (after adding training load)
            current_form = form
            current_fitness = fitness

            # Get sport-specific activity for this day
            if day == 0 and today_activities:
                # For today, use today's activities
                sport_rec = self._get_sport_specific_recommendation(rec_type, form, today_activities, recent_activities)
            else:
                # For future days, simulate recent activities by using current pattern
                sport_rec = self._get_sport_specific_recommendation(rec_type, form, [], recent_activities)

            # 🏆 Calculate consistent date for both display and race logic
            plan_date = datetime.now() + timedelta(days=day)

            plan_entry = {
                "day": day + 1,
                "date": plan_date.date().isoformat(),
                "recommendation": rec_type,
                "activity": sport_rec["activity"],
                "activity_rationale": sport_rec["rationale"],
                "alternative_activities": sport_rec["alternatives"],
                "suggested_load": load,
                "predicted_form": round(current_form, 1),
                "predicted_fitness": round(current_fitness, 1),
            }

            # Add double session fields if present
            if sport_rec.get("second_activity"):
                plan_entry["second_activity"] = sport_rec["second_activity"]
                plan_entry["session_timing"] = sport_rec["session_timing"]
                plan_entry["double_rationale"] = sport_rec["double_rationale"]

            # 🏆 Add race day information (using same plan_date)
            days_to_race = config.days_to_next_race(plan_date)
            is_race_day = False

            # Check if this is a race day (compare only dates, not times)
            race_dates = config.get_race_dates()
            for race_date in race_dates:
                if race_date.date() == plan_date.date():
                    is_race_day = True
                    race_type, race_distance = config.get_race_details(race_date)
                    plan_entry["is_race_day"] = True
                    plan_entry["race_priority"] = config.get_race_priority(race_date)
                    plan_entry["race_type"] = race_type
                    plan_entry["race_distance"] = race_distance

                    # Format race description
                    if race_type == "Run":
                        if race_distance == 21.1:
                            race_name = "Half Marathon"
                        elif race_distance == 42.2:
                            race_name = "Marathon"
                        elif race_distance <= 10:
                            race_name = f"{int(race_distance)}K"
                        else:
                            race_name = f"{race_distance}km Run"
                    elif race_type == "Ride":
                        race_name = f"{int(race_distance)}km Ride"
                    elif race_type == "Hike":
                        race_name = f"{int(race_distance)}km Hike"
                    else:
                        race_name = f"{race_type} {race_distance}km"

                    plan_entry["activity"] = f"🏆 RACE - {race_name}"
                    taper_days = config.get_taper_duration_for_race(race_date)
                    plan_entry["notes"] = plan_entry.get("notes", []) + [
                        f"{plan_entry['race_priority']} Priority - {taper_days} day taper"
                    ]
                    break

            # Add tapering phase indicators (4-day taper)
            if not is_race_day and config.is_in_taper_phase(plan_date) and days_to_race != float('inf'):
                plan_entry["taper_phase"] = True
                plan_entry["days_to_race"] = int(days_to_race)

                if days_to_race <= 1:
                    plan_entry["taper_stage"] = "PEAK"
                    plan_entry["activity"] = f"⭐ PEAK - {plan_entry['activity']}"
                elif days_to_race <= 2:
                    plan_entry["taper_stage"] = "FINAL"
                    plan_entry["activity"] = f"🔥 FINAL - {plan_entry['activity']}"
                elif days_to_race <= 4:
                    plan_entry["taper_stage"] = "TAPER"
                    plan_entry["activity"] = f"📉 TAPER - {plan_entry['activity']}"

            recommendations.append(plan_entry)

            # Note: Training load is already applied above using proper Banister model
            # fitness += load * config.FITNESS_MAGNITUDE (applied above)
            # fatigue += load * config.FATIGUE_MAGNITUDE (applied above)

        return recommendations

    def get_weekly_plan(self) -> List[Dict[str, any]]:
        """Get 7-day training plan (backward compatibility)."""
        return self.get_training_plan(days=7)

    def _get_periodized_recommendation(self, day: int, form: float, fatigue: float, fitness: float) -> Tuple[str, int]:
        """
        Generate periodized training recommendation based on day and current state.
        Uses 3:1 loading pattern with weekly variation and race peaking.
        """
        week_num = day // 7  # Week number (0-based)
        day_of_week = day % 7  # Day within week (0=Sunday)
        current_date = datetime.now() + timedelta(days=day)

        # 🏆 RACE PEAKING SYSTEM (Olympic Competition Periodization)
        race_date, days_to_race, taper_duration = config.get_next_race_with_taper(current_date)
        is_taper_phase = config.is_in_taper_phase(current_date)

        # 🏥 POST-RACE RECOVERY CHECK
        is_recovering, days_since_race, recovery_duration, past_race_date = config.is_in_recovery_phase(current_date)

        # 🏥 POST-RACE RECOVERY OVERRIDE (Prevent overtraining after races)
        if is_recovering:
            # Get recovery intensity limit based on days since race
            intensity_limit = config.get_recovery_intensity_limit(days_since_race)

            if days_since_race <= 1:
                # Complete rest day after race
                return TrainingRecommendation.REST.value, 0
            elif days_since_race <= 3:
                # Very light recovery only
                return TrainingRecommendation.RECOVERY.value, int(30 * intensity_limit)
            elif days_since_race <= 7:
                # Light activity only
                return TrainingRecommendation.EASY.value, int(60 * intensity_limit)
            else:
                # Gradual return to normal training but still limited
                if intensity_limit < 0.75:
                    # Still in recovery period, limit to easy/moderate
                    max_intensity = TrainingRecommendation.MODERATE.value
                    max_duration = int(120 * intensity_limit)

                    # Override any high intensity recommendations
                    if form > 5 and day_of_week in [2, 4]:  # Normal hard days
                        return max_intensity, max_duration
                # Let normal logic continue but with intensity cap

        # Base training pattern (modified by form and fatigue)
        fatigue_ratio = fatigue / (fitness + 1e-5)

        # Weekly volume progression with race peaking override
        if is_taper_phase and days_to_race != float('inf'):
            # 🏆 TAPERING PROTOCOL (Sport-specific taper)
            week_mult = config.get_taper_volume_reduction(days_to_race)

            # If beyond configured days, use gradual reduction
            if days_to_race > 5:
                week_mult = 0.85  # Light reduction for early taper
        else:
            # Standard periodization when not tapering
            week_multipliers = [1.0, 1.1, 1.2, 0.6]  # 3 weeks build, 1 week recovery
            week_mult = week_multipliers[week_num % 4]

        # 🏆 RACE-SPECIFIC INTENSITY MODIFICATIONS (Sport-specific taper)
        if is_taper_phase and days_to_race != float('inf'):
            if race_date:
                race_type, race_distance = config.get_race_details(race_date)

                if days_to_race <= 1:  # Day before race - REST/ACTIVATION
                    return TrainingRecommendation.RECOVERY.value, max(15, int(20 * week_mult))
                elif days_to_race <= 2:  # 2 days out - LIGHT OPENERS
                    return TrainingRecommendation.EASY.value, int(30 * week_mult)
                elif days_to_race <= taper_duration:  # Within taper period
                    # Adjust intensity based on race type
                    if race_type == "Run" and race_distance >= 42.2:  # Marathon
                        # More conservative taper for marathon
                        return TrainingRecommendation.EASY.value, int(40 * week_mult)
                    elif race_type == "Ride" and race_distance >= 200:  # Long ride
                        # Maintain some intensity for long rides
                        return TrainingRecommendation.MODERATE.value, int(50 * week_mult)
                    else:
                        # Standard taper intensity
                        if form > 20:
                            return TrainingRecommendation.MODERATE.value, int(45 * week_mult)
                        else:
                            return TrainingRecommendation.EASY.value, int(40 * week_mult)

        # High fatigue = reduce intensity
        if fatigue_ratio > 0.8:
            if day_of_week in [0, 6]:  # Weekend
                return TrainingRecommendation.REST.value, 0
            else:
                return TrainingRecommendation.RECOVERY.value, int(20 * week_mult)

        # Good form = quality work possible (Olympic thresholds)
        if form > 5:  # Lower threshold for more quality sessions
            # Tuesday/Thursday = Hard days (Olympic intensity)
            if day_of_week in [2, 4]:
                if form > 25:
                    return TrainingRecommendation.PEAK.value, int(120 * week_mult)
                else:
                    return TrainingRecommendation.HARD.value, int(100 * week_mult)
            # Saturday = Long moderate (Olympic endurance foundation)
            elif day_of_week == 6:
                return TrainingRecommendation.MODERATE.value, int(90 * week_mult)
            # Sunday = Active recovery (substantial aerobic base)
            elif day_of_week == 0:
                return TrainingRecommendation.RECOVERY.value, int(40 * week_mult)
            # Monday/Wednesday/Friday = Aerobic base building
            else:
                if day_of_week in [1, 5]:
                    return TrainingRecommendation.EASY.value, int(60 * week_mult)
                else:
                    return TrainingRecommendation.MODERATE.value, int(80 * week_mult)

        # Poor form = focus on recovery
        elif form < -5:
            if day_of_week in [0, 3, 6]:  # More rest days
                return TrainingRecommendation.REST.value, 0
            else:
                return TrainingRecommendation.RECOVERY.value, int(20 * week_mult)

        # Neutral form = balanced approach (Olympic base building)
        else:
            if day_of_week in [2, 4]:  # Moderate intensity days (Olympic load)
                return TrainingRecommendation.MODERATE.value, int(85 * week_mult)
            elif day_of_week in [0, 6]:  # Weekend substantial base
                return TrainingRecommendation.EASY.value, int(65 * week_mult)
            else:
                return TrainingRecommendation.MODERATE.value, int(75 * week_mult)

    def _get_sport_specific_recommendation(
        self,
        intensity: str,
        form: float,
        today_activities: List[Dict] = None,
        recent_activities: List[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate sport-specific activity recommendations based on:
        - Recent sport usage patterns
        - Recovery needs
        - Injury prevention
        - Environmental conditions
        """
        # Get recent sport usage (last 7 days)
        recent_sports = self._analyze_recent_sport_usage(recent_activities or [])

        # Define sport characteristics
        sport_profiles = {
            "Run": {
                "impact": "high",
                "intensity_capacity": ["EASY", "MODERATE", "HARD", "PEAK"],
                "recovery_time": 48,  # hours
                "weather_dependent": True,
                "equipment": "minimal"
            },
            "Ride": {
                "impact": "low",
                "intensity_capacity": ["RECOVERY", "EASY", "MODERATE", "HARD", "PEAK"],
                "recovery_time": 24,
                "weather_dependent": True,
                "equipment": "bike"
            },
            "Hike": {
                "impact": "medium",
                "intensity_capacity": ["RECOVERY", "EASY", "MODERATE"],
                "recovery_time": 24,
                "weather_dependent": True,
                "equipment": "minimal"
            },
            "Workout": {
                "impact": "low",
                "intensity_capacity": ["RECOVERY", "EASY", "MODERATE", "HARD"],
                "recovery_time": 24,
                "weather_dependent": False,
                "equipment": "gym"
            },
            "WeightTraining": {
                "impact": "low",
                "intensity_capacity": ["MODERATE", "HARD"],
                "recovery_time": 48,
                "weather_dependent": False,
                "equipment": "gym"
            },
            "Rowing": {
                "impact": "very_low",
                "intensity_capacity": ["RECOVERY", "EASY", "MODERATE", "HARD"],
                "recovery_time": 24,
                "weather_dependent": False,
                "equipment": "machine"
            }
        }

        # Get primary recommendation
        primary_activity = self._select_primary_activity(
            intensity, form, recent_sports, sport_profiles
        )

        # Get alternatives
        alternatives = self._get_alternative_activities(
            intensity, primary_activity, sport_profiles, recent_sports
        )

        # Generate rationale
        rationale = self._generate_activity_rationale(
            primary_activity, intensity, recent_sports, today_activities
        )

        # Check if double session is recommended
        double_session = self._get_double_session_recommendation(
            intensity, primary_activity, recent_sports, today_activities
        )

        result = {
            "activity": primary_activity,
            "rationale": rationale,
            "alternatives": alternatives
        }

        if double_session:
            result.update(double_session)

        return result

    def _get_double_session_recommendation(
        self,
        intensity: str,
        primary_activity: str,
        recent_sports: Dict,
        today_activities: List[Dict] = None
    ) -> Optional[Dict]:
        """
        Determine if a double session is recommended based on Olympic training principles.

        Returns dict with second_activity, session_timing, and double_rationale if recommended.
        """
        # Skip double sessions for REST or if already trained today
        if intensity == "REST" or (today_activities and len(today_activities) > 0):
            return None

        # Get recent strength training usage
        recent_strength = recent_sports.get("WeightTraining", {}).get("last_used", 999)
        recent_workout = recent_sports.get("Workout", {}).get("last_used", 999)

        # Smart double session combinations based on sports science
        double_session_combinations = {
            # EASY intensity - perfect for combining cardio + strength
            "Zone 2 Endurance Ride": {
                "condition": lambda: intensity == "EASY" and recent_strength > 3,
                "second_activity": "Strength Training (Evening)",
                "timing": "6+ hours apart (AM cardio, PM strength)",
                "rationale": "Low-impact cardio doesn't interfere with strength gains • Optimal hormone response"
            },
            "Conversational Run": {
                "condition": lambda: intensity == "EASY" and recent_strength > 3,
                "second_activity": "Mobility & Core Work",
                "timing": "4+ hours apart (AM run, PM mobility)",
                "rationale": "Easy run + evening mobility prevents injury • Enhances recovery"
            },
            "Conversational Hike": {
                "condition": lambda: intensity == "EASY" and recent_strength > 2,
                "second_activity": "Yoga/Stretching",
                "timing": "Evening (2+ hours after hike)",
                "rationale": "Nature therapy + flexibility work • Mental and physical recovery"
            },

            # MODERATE intensity - carefully selected combinations
            "Tempo Run": {
                "condition": lambda: intensity == "MODERATE" and recent_strength > 4,
                "second_activity": "Recovery Stretching",
                "timing": "Evening (3+ hours after run)",
                "rationale": "Post-tempo recovery work prevents stiffness • Lactate clearance"
            },
            "Strength Training": {
                "condition": lambda: intensity == "MODERATE" and recent_sports.get("Ride", {}).get("last_used", 999) > 1,
                "second_activity": "Easy Recovery Ride",
                "timing": "4+ hours apart (AM strength, PM easy cycling)",
                "rationale": "Active recovery cycling enhances muscle recovery • Different energy systems"
            },

            # HARD intensity - recovery-focused second sessions only
            "Threshold Intervals": {
                "condition": lambda: intensity == "HARD",
                "second_activity": "Foam Rolling & Stretching",
                "timing": "Evening (3+ hours after intervals)",
                "rationale": "Essential recovery work after high intensity • Prevents muscle tightness"
            },
            "FTP Intervals (Bike)": {
                "condition": lambda: intensity == "HARD",
                "second_activity": "Gentle Yoga",
                "timing": "Evening (2+ hours after intervals)",
                "rationale": "Parasympathetic recovery after intense training • Stress hormone regulation"
            },

            # PEAK intensity - minimal second sessions
            "VO2 Max Intervals": {
                "condition": lambda: intensity == "PEAK",
                "second_activity": "Light Mobility Work",
                "timing": "Evening (4+ hours after intervals)",
                "rationale": "Gentle movement aids recovery from peak efforts • Maintain flexibility"
            }
        }

        # Check if primary activity qualifies for double session
        combo = double_session_combinations.get(primary_activity)
        if combo and combo["condition"]():
            return {
                "second_activity": combo["second_activity"],
                "session_timing": combo["timing"],
                "double_rationale": combo["rationale"]
            }

        # General rules for strength training gaps (Olympic training principle)
        if recent_strength > 4:  # Haven't done strength in 4+ days
            if intensity in ["EASY", "MODERATE"]:  # Only on EASY/MODERATE, not RECOVERY
                return {
                    "second_activity": "Strength Training (Evening)",
                    "session_timing": "6+ hours apart",
                    "double_rationale": f"⚠️ Strength training needed ({recent_strength} days) • Essential for balanced development"
                }

        return None

    def _analyze_recent_sport_usage(self, recent_activities: List[Dict]) -> Dict[str, Dict]:
        """Analyze recent sport usage patterns."""
        sport_analysis = {}

        for activity in recent_activities:
            sport_type = activity.get('type', 'Unknown')
            days_ago = (datetime.now(timezone.utc).date() - activity.get('start_date').date()).days

            if sport_type not in sport_analysis:
                sport_analysis[sport_type] = {
                    'last_used': days_ago,
                    'frequency': 0,
                    'total_load': 0,
                    'avg_load': 0,
                    'recent_intensity': 'moderate'
                }

            sport_analysis[sport_type]['frequency'] += 1
            sport_analysis[sport_type]['total_load'] += activity.get('training_load', 0)
            sport_analysis[sport_type]['last_used'] = min(sport_analysis[sport_type]['last_used'], days_ago)

        # Calculate averages
        for sport_data in sport_analysis.values():
            if sport_data['frequency'] > 0:
                sport_data['avg_load'] = sport_data['total_load'] / sport_data['frequency']

        return sport_analysis

    def _get_user_sport_preferences(self) -> Dict[str, float]:
        """Get user's sport preferences from configuration."""
        from ..config import config
        return {sport: config.get_sport_preference(sport)
                for sport in config.get_enabled_sports()}

    def _calculate_sport_rotation_score(self, sport: str, recent_sports: Dict, intensity: str) -> float:
        """Calculate rotation score for smart sport selection (higher = better choice)."""

        user_prefs = self._get_user_sport_preferences()
        base_preference = user_prefs.get(sport, 0.1)  # Default low preference for unknown sports

        # Skip sports with 0.0 preference (disabled sports)
        if base_preference <= 0.0:
            return 0.0

        # Days since last use (higher is better for rotation)
        days_since_use = recent_sports.get(sport, {}).get('last_used', 3)  # Default to 3 if never used
        recency_score = min(days_since_use / 7.0, 1.5)  # Max score at 7+ days (slower rotation)

        # Frequency penalty (recent overuse gets lower score)
        recent_frequency = recent_sports.get(sport, {}).get('frequency', 0)
        frequency_penalty = max(0.2, 1.0 - (recent_frequency * 0.2))  # Penalty for overuse

        # Intensity appropriateness
        intensity_match = self._get_intensity_appropriateness(sport, intensity)

        # Recovery considerations (avoid high-impact after high-impact)
        recovery_bonus = self._get_recovery_appropriateness(sport, recent_sports)

        # Balance preferences with rotation - respect user preferences more
        # Don't flatten preferences too much, user chose them for a reason
        balanced_preference = base_preference

        # Final score with balanced weighting
        total_score = balanced_preference * recency_score * frequency_penalty * intensity_match * recovery_bonus

        return total_score

    def _get_intensity_appropriateness(self, sport: str, intensity: str) -> float:
        """How well a sport matches the required intensity (0.1-1.0)."""

        sport_intensity_map = {
            "WeightTraining": {
                "REST": 0.1, "RECOVERY": 0.4, "EASY": 0.6, "MODERATE": 1.0, "HARD": 0.9, "PEAK": 0.7
            },
            "Ride": {
                "REST": 0.1, "RECOVERY": 1.0, "EASY": 1.0, "MODERATE": 1.0, "HARD": 1.0, "PEAK": 1.0
            },
            "Run": {
                "REST": 0.1, "RECOVERY": 0.6, "EASY": 1.0, "MODERATE": 1.0, "HARD": 1.0, "PEAK": 1.0
            },
            "Hike": {
                "REST": 0.1, "RECOVERY": 0.9, "EASY": 1.0, "MODERATE": 0.8, "HARD": 0.4, "PEAK": 0.2
            },
            "Workout": {
                "REST": 0.1, "RECOVERY": 0.5, "EASY": 0.7, "MODERATE": 1.0, "HARD": 0.9, "PEAK": 0.6
            },
            "Rowing": {
                "REST": 0.1, "RECOVERY": 0.8, "EASY": 0.9, "MODERATE": 1.0, "HARD": 1.0, "PEAK": 0.8
            },
            "Swim": {
                "REST": 0.1, "RECOVERY": 1.0, "EASY": 1.0, "MODERATE": 1.0, "HARD": 1.0, "PEAK": 0.9
            },
            "AlpineSki": {
                "REST": 0.1, "RECOVERY": 0.4, "EASY": 0.7, "MODERATE": 1.0, "HARD": 1.0, "PEAK": 1.0
            },
            "Yoga": {
                "REST": 0.1, "RECOVERY": 1.0, "EASY": 0.9, "MODERATE": 0.7, "HARD": 0.5, "PEAK": 0.3
            }
        }

        return sport_intensity_map.get(sport, {}).get(intensity, 0.5)

    def _get_recovery_appropriateness(self, sport: str, recent_sports: Dict) -> float:
        """Calculate recovery appropriateness bonus (avoid high-impact after high-impact)."""

        # Check recent high-impact activities
        recent_run = recent_sports.get("Run", {}).get("last_used", 7)
        recent_high_impact = min(recent_run, 7)

        sport_impact = {
            "Run": "high",
            "WeightTraining": "medium",
            "Hike": "medium",
            "AlpineSki": "high",
            "Ride": "low",
            "Workout": "low",
            "Rowing": "low",
            "Swim": "low",
            "Yoga": "low"
        }

        current_impact = sport_impact.get(sport, "medium")

        # If recent high-impact activity, prefer low-impact
        if recent_high_impact <= 1:  # Within 1 day
            if current_impact == "low":
                return 1.5  # Bonus for low-impact
            elif current_impact == "medium":
                return 1.0  # Neutral
            else:
                return 0.4  # Penalty for high-impact
        else:
            return 1.0  # No penalty/bonus

    def _get_underused_sport_bonus(self, sport: str, recent_sports: Dict) -> str:
        """Check if sport should get underused bonus recommendation."""

        days_since_use = recent_sports.get(sport, {}).get('last_used', 999)

        # WeightTraining is critical for balanced training
        if sport == "WeightTraining" and days_since_use > 7:
            return "⚠️ Strength training needed - not done in over 7 days"

        # Other sports
        if days_since_use > 10:
            return f"Consider {sport} - last used {days_since_use} days ago"

        return ""

    def _select_primary_activity(
        self,
        intensity: str,
        form: float,
        recent_sports: Dict,
        sport_profiles: Dict
    ) -> str:
        """Select primary activity using smart rotation algorithm."""

        # For REST intensity
        if intensity == "REST":
            return "Complete Rest"

        # Use smart rotation algorithm for all other intensities
        from ..config import config
        available_sports = config.get_enabled_sports()

        # Calculate rotation scores for all enabled sports
        sport_scores = {}
        for sport in available_sports:
            score = self._calculate_sport_rotation_score(sport, recent_sports, intensity)
            sport_scores[sport] = score

        # WEEKLY TARGET ALLOCATION SYSTEM - Permanent Solution
        best_sport = self._select_sport_by_weekly_targets(sport_scores, recent_sports, intensity)

        # Track usage for this plan generation
        self._plan_sport_usage[best_sport] = self._plan_sport_usage.get(best_sport, 0) + 1

        # Track weekly usage
        if not hasattr(self, '_current_plan_days'):
            self._current_plan_days = []
        self._current_plan_days.append(best_sport)

        current_week = (len(self._current_plan_days) - 1) // 7 + 1
        week_key = f"week_{current_week}"
        if week_key not in self._week_sport_usage:
            self._week_sport_usage[week_key] = {}
        self._week_sport_usage[week_key][best_sport] = self._week_sport_usage[week_key].get(best_sport, 0) + 1

        # Generate activity name based on sport and intensity
        activity_name = self._get_activity_name_for_sport_intensity(best_sport, intensity)

        return activity_name

    def _select_sport_by_weekly_targets(self, sport_scores: dict, recent_sports: dict, intensity: str) -> str:
        """
        PERMANENT SOLUTION: Select sport based on weekly preference targets.

        Algorithm:
        1. Calculate current week position and day within week
        2. Determine weekly targets from user preferences
        3. Check current week progress vs targets
        4. Prioritize sports that are behind their weekly targets
        5. Fall back to rotation scores if targets are met
        """
        from ..config import config

        # Initialize tracking if needed
        if not hasattr(self, '_current_plan_days'):
            self._current_plan_days = []
        if not hasattr(self, '_week_sport_usage'):
            self._week_sport_usage = {}

        # Calculate current position in plan
        current_day_in_plan = len(self._current_plan_days)
        current_week = (current_day_in_plan // 7) + 1
        day_in_week = current_day_in_plan % 7  # 0-6
        week_key = f"week_{current_week}"

        # Get current week usage
        current_week_usage = self._week_sport_usage.get(week_key, {})

        # Calculate weekly targets based on user preferences
        weekly_targets = self._calculate_weekly_targets()

        # Determine which sports are behind their weekly targets
        eligible_sports = []
        for sport in sport_scores.keys():
            current_count = current_week_usage.get(sport, 0)
            target_count = weekly_targets.get(sport, 0)

            # Check if sport is behind target (accounting for week progress)
            # If we're on day 3 of 7, we should have ~43% of weekly target completed
            expected_progress = (day_in_week + 1) / 7.0
            expected_count = target_count * expected_progress

            if current_count < expected_count or current_count < target_count:
                # Apply intensity filtering - some sports better for certain intensities
                intensity_match = self._get_intensity_appropriateness(sport, intensity)
                if intensity_match > 0.3:  # Only consider sports suitable for this intensity
                    priority = (expected_count - current_count) * intensity_match
                    eligible_sports.append((sport, priority))

        # Select sport with highest priority (most behind target)
        if eligible_sports:
            eligible_sports.sort(key=lambda x: x[1], reverse=True)
            return eligible_sports[0][0]

        # Fallback: if all targets met, use rotation scores but avoid overuse
        available_sports = []
        for sport, score in sport_scores.items():
            current_count = current_week_usage.get(sport, 0)
            max_per_week = max(2, int(weekly_targets.get(sport, 0) * 1.5))  # Allow slight overuse

            if current_count < max_per_week:
                intensity_match = self._get_intensity_appropriateness(sport, intensity)
                if intensity_match > 0.3:
                    adjusted_score = score * intensity_match
                    available_sports.append((sport, adjusted_score))

        if available_sports:
            available_sports.sort(key=lambda x: x[1], reverse=True)
            return available_sports[0][0]

        # Ultimate fallback: best intensity match
        best_sport = max(sport_scores.keys(),
                        key=lambda s: self._get_intensity_appropriateness(s, intensity))
        return best_sport

    def _calculate_weekly_targets(self) -> dict:
        """Calculate target number of sessions per week for each sport based on preferences."""
        from ..config import config

        # Get user preferences (these are weekly percentages)
        preferences = config.SPORT_PREFERENCES

        # Calculate sessions per week (6 active days, 1 rest day for Olympic training)
        active_days_per_week = 6  # Olympic-level training frequency
        weekly_targets = {}

        for sport, preference in preferences.items():
            if preference > 0:
                target_sessions = preference * active_days_per_week
                weekly_targets[sport] = max(0.5, target_sessions)  # Minimum 0.5 to ensure inclusion

        return weekly_targets

    def _get_activity_name_for_sport_intensity(self, sport: str, intensity: str) -> str:
        """Generate specific activity name based on sport and intensity."""

        activity_map = {
            "Ride": {
                "RECOVERY": "Easy Recovery Ride",
                "EASY": "Zone 2 Endurance Ride",
                "MODERATE": "Tempo Ride",
                "HARD": "FTP Intervals (Bike)",
                "PEAK": "VO2 Max Bike Intervals"
            },
            "Run": {
                "RECOVERY": "Easy Recovery Jog",
                "EASY": "Conversational Run",
                "MODERATE": "Tempo Run",
                "HARD": "Threshold Intervals",
                "PEAK": "VO2 Max Intervals"
            },
            "Hike": {
                "RECOVERY": "Gentle Nature Walk",
                "EASY": "Conversational Hike",
                "MODERATE": "Brisk Hiking",
                "HARD": "Hill Hiking",
                "PEAK": "Mountain Hiking"
            },
            "WeightTraining": {
                "RECOVERY": "Mobility & Light Weights",
                "EASY": "General Strength",
                "MODERATE": "Strength Training",
                "HARD": "Power/Strength Focus",
                "PEAK": "Max Strength Session"
            },
            "Workout": {
                "RECOVERY": "Yoga/Stretching",
                "EASY": "Light Circuit Training",
                "MODERATE": "Circuit Training",
                "HARD": "HIIT Workout",
                "PEAK": "Intense Functional Training"
            },
            "Rowing": {
                "RECOVERY": "Easy Recovery Row",
                "EASY": "Steady State Rowing",
                "MODERATE": "Tempo Rowing",
                "HARD": "Rowing Intervals",
                "PEAK": "Sprint Rowing"
            },
            "Swim": {
                "RECOVERY": "Easy Recovery Swim",
                "EASY": "Zone 2 Swimming",
                "MODERATE": "Tempo Swimming",
                "HARD": "Swimming Intervals",
                "PEAK": "Sprint Swimming"
            },
            "AlpineSki": {
                "RECOVERY": "Gentle Ski Touring",
                "EASY": "Relaxed Skiing",
                "MODERATE": "Alpine Skiing",
                "HARD": "Aggressive Skiing",
                "PEAK": "Race Training"
            },
            "Yoga": {
                "RECOVERY": "Restorative Yoga",
                "EASY": "Gentle Yoga Flow",
                "MODERATE": "Vinyasa Yoga",
                "HARD": "Power Yoga",
                "PEAK": "Advanced Yoga"
            }
        }

        return activity_map.get(sport, {}).get(intensity, f"{intensity.title()} {sport}")

    def _get_alternative_activities(
        self,
        intensity: str,
        primary_activity: str,
        sport_profiles: Dict,
        recent_sports: Dict
    ) -> List[str]:
        """Get 2-3 alternative activities for the same intensity."""
        alternatives = []

        if intensity == "REST":
            return ["Gentle Yoga", "Stretching", "Massage"]

        if intensity == "RECOVERY":
            alternatives = ["Easy Walk", "Gentle Yoga", "Swimming (if available)"]
        elif intensity == "EASY":
            alternatives = ["Zone 2 Bike", "Easy Hike", "Easy Jog"]
        elif intensity == "MODERATE":
            alternatives = ["Tempo Run", "Endurance Ride", "Circuit Training"]
        elif intensity in ["HARD", "PEAK"]:
            alternatives = ["Interval Run", "Bike Intervals", "Rowing Intervals"]

        # Remove the primary activity from alternatives
        alternatives = [alt for alt in alternatives if alt != primary_activity]
        return alternatives[:3]

    def _generate_activity_rationale(
        self,
        activity: str,
        intensity: str,
        recent_sports: Dict,
        today_activities: List[Dict] = None
    ) -> str:
        """Generate explanation for activity choice."""

        if intensity == "REST":
            return "Complete rest recommended after recent training load"

        # Generate smart rotation rationale
        rationale_parts = []

        # Check what was done recently for recovery considerations
        recent_run = recent_sports.get("Run", {}).get("last_used", 999)
        recent_ride = recent_sports.get("Ride", {}).get("last_used", 999)
        recent_weight = recent_sports.get("WeightTraining", {}).get("last_used", 999)

        # Activity-specific rationale
        if "Ride" in activity or "Cycling" in activity:
            if recent_run <= 1:
                rationale_parts.append("Low-impact cycling after recent running")
            elif recent_ride >= 3:
                rationale_parts.append(f"Cycling rotation (last ride {recent_ride} days ago)")
            else:
                rationale_parts.append("Cycling for aerobic development")

        elif "Run" in activity or "Jog" in activity:
            if recent_run >= 2:
                rationale_parts.append(f"Running rotation (adequate recovery: {recent_run} days)")
            elif recent_run <= 1:
                rationale_parts.append("Running despite recent run - high training load needed")
            else:
                rationale_parts.append("Running for sport-specific development")

        elif "WeightTraining" in activity or "Strength" in activity:
            if recent_weight > 7:
                rationale_parts.append("⚠️ Strength training prioritized - not done in over 7 days")
            else:
                rationale_parts.append("Strength training for balanced development")

        elif "Hike" in activity:
            rationale_parts.append("Hiking for low-impact endurance and mental refreshment")

        elif "Rowing" in activity:
            rationale_parts.append("Full-body, low-impact cardiovascular training")

        elif "Workout" in activity or "Circuit" in activity or "HIIT" in activity:
            rationale_parts.append("Functional training for strength-endurance combination")

        # Add rotation insight
        underused_sports = []
        for sport in ["WeightTraining", "Rowing", "Hike"]:
            days_ago = recent_sports.get(sport, {}).get("last_used", 999)
            if days_ago > 7:
                underused_sports.append(f"{sport} ({days_ago}+ days)")

        if underused_sports:
            rationale_parts.append(f"Consider variety: {', '.join(underused_sports)} overdue")

        return " • ".join(rationale_parts) if rationale_parts else "Selected based on smart rotation algorithm"

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

    def get_or_create_periodization_state(self, user_id: str = "default") -> PeriodizationState:
        """Get or create periodization state for user."""
        if not self.db:
            # Initialize a default state if no database
            state = PeriodizationState()
            state.user_id = user_id
            state.cycle_start_date = datetime.now(timezone.utc)
            state.current_week = 1
            state.current_phase = "BUILD"
            return state

        with self.db.get_session() as session:
            state = session.query(PeriodizationState).filter_by(user_id=user_id).first()

            if not state:
                # Create new periodization state
                state = PeriodizationState(
                    user_id=user_id,
                    cycle_start_date=datetime.now(timezone.utc),
                    current_week=1,
                    current_phase="BUILD",
                    cycle_length_weeks=4,
                    build_weeks=3,
                    recovery_weeks=1,
                    baseline_fitness=self._get_current_fitness(),
                    auto_progression=True
                )
                session.add(state)
                session.commit()
                session.refresh(state)
                # Detach the object from session to avoid "not bound to a Session" errors
                session.expunge(state)

            return state

    def update_periodization_state(self, user_id: str = "default") -> PeriodizationState:
        """Update periodization state based on current date and training."""
        if not self.db:
            return self.get_or_create_periodization_state(user_id)

        with self.db.get_session() as session:
            state = session.query(PeriodizationState).filter_by(user_id=user_id).first()

            if not state:
                return self.get_or_create_periodization_state(user_id)

            # Check if we need to advance week or cycle
            if state.should_advance_cycle():
                # Start new cycle
                state.cycle_start_date = datetime.now(timezone.utc)
                state.current_week = 1
                state.current_phase = "BUILD"
                state.baseline_fitness = self._get_current_fitness()
                state.last_phase_change = datetime.now(timezone.utc)

            elif state.should_advance_week():
                # Advance to next week
                state.current_week += 1
                old_phase = state.current_phase
                state.current_phase = state.get_current_phase_type()

                if old_phase != state.current_phase:
                    state.last_phase_change = datetime.now(timezone.utc)

            # Always update the phase to ensure it's correct
            state.current_phase = state.get_current_phase_type()
            state.updated_at = datetime.now(timezone.utc)

            session.commit()
            session.refresh(state)
            # Detach the object from session to avoid "not bound to a Session" errors
            session.expunge(state)
            return state

    def _get_periodized_recommendation_with_state(self, day: int, form: float, fatigue: float, fitness: float,
                                                periodization_state: PeriodizationState) -> Tuple[str, int]:
        """
        Generate periodized training recommendation using actual periodization state.
        This replaces the simple mathematical cycle with real phase tracking.
        """
        # Calculate which day in the actual cycle we're looking at
        actual_cycle_day = periodization_state.get_current_cycle_day() + day
        actual_week = (actual_cycle_day // 7) + 1
        day_of_week = actual_cycle_day % 7

        # 🏥 POST-RACE RECOVERY CHECK (Priority override for race recovery)
        current_date = datetime.now() + timedelta(days=day)
        is_recovering, days_since_race, recovery_duration, past_race_date = config.is_in_recovery_phase(current_date)

        # 🏥 POST-RACE RECOVERY OVERRIDE (Prevent overtraining after races)
        if is_recovering:
            # Get recovery intensity limit based on days since race
            intensity_limit = config.get_recovery_intensity_limit(days_since_race)

            if days_since_race <= 1:
                # Complete rest day after race
                return TrainingRecommendation.REST.value, 0
            elif days_since_race <= 3:
                # Very light recovery only
                return TrainingRecommendation.RECOVERY.value, max(10, int(30 * intensity_limit))
            elif days_since_race <= 7:
                # Light activity only
                return TrainingRecommendation.EASY.value, max(20, int(60 * intensity_limit))
            else:
                # Gradual return to normal training but still limited
                if intensity_limit < 0.75:
                    # Still in recovery period, limit to easy/moderate
                    max_intensity = TrainingRecommendation.MODERATE.value
                    max_duration = max(30, int(120 * intensity_limit))

                    # Override any high intensity recommendations
                    if form > 5 and day_of_week in [2, 4]:  # Normal hard days
                        return max_intensity, max_duration
                # Let normal logic continue but with intensity cap

        # 🏆 RACE PEAKING SYSTEM (Race taper takes priority over periodization)
        race_date, days_to_race, taper_duration = config.get_next_race_with_taper(current_date)
        is_taper_phase = config.is_in_taper_phase(current_date)

        # 🏆 RACE-SPECIFIC TAPER OVERRIDE (Override periodization during taper)
        if is_taper_phase and days_to_race != float('inf') and race_date:
            race_type, race_distance = config.get_race_details(race_date)
            taper_mult = config.get_taper_volume_reduction(days_to_race)

            if days_to_race <= 1:  # Day before race - REST/ACTIVATION
                return TrainingRecommendation.REST.value, 0
            elif days_to_race <= 2:  # 2 days out - LIGHT OPENERS
                return TrainingRecommendation.EASY.value, int(30 * taper_mult)
            elif days_to_race <= taper_duration:  # Within taper period
                # Adjust intensity based on race type and distance
                if race_type == "Run" and race_distance >= 42.2:  # Marathon/Ultra
                    # More conservative taper for long runs
                    return TrainingRecommendation.EASY.value, int(40 * taper_mult)
                elif race_type == "Ride" and race_distance >= 200:  # Long ride
                    # Maintain some intensity for long rides
                    return TrainingRecommendation.MODERATE.value, int(50 * taper_mult)
                else:
                    # Standard taper intensity based on current form
                    if form > 20:
                        return TrainingRecommendation.MODERATE.value, int(45 * taper_mult)
                    else:
                        return TrainingRecommendation.EASY.value, int(40 * taper_mult)

        # Determine phase based on actual state, not mathematical cycle
        if actual_week <= periodization_state.build_weeks:
            current_phase = "BUILD"
            week_mult = 1.0 + (actual_week - 1) * 0.1  # Progressive loading
        elif actual_week == periodization_state.build_weeks + 1:
            current_phase = "PEAK"
            week_mult = 1.2  # Peak intensity
        else:
            current_phase = "RECOVERY"
            week_mult = 0.6  # Recovery week

        # Base training pattern (modified by form and fatigue)
        fatigue_ratio = fatigue / (fitness + 1e-5)

        # High fatigue = reduce intensity regardless of planned phase
        if fatigue_ratio > 0.8:
            if day_of_week in [0, 6]:  # Weekend
                return TrainingRecommendation.REST.value, 0
            else:
                return TrainingRecommendation.RECOVERY.value, int(20 * week_mult)

        # Phase-specific recommendations
        if current_phase == "BUILD":
            # Build phase: progressive loading
            if day_of_week in [0, 6]:  # Weekend rest or easy
                if form > 5:
                    return TrainingRecommendation.EASY.value, int(50 * week_mult)
                else:
                    return TrainingRecommendation.RECOVERY.value, int(30 * week_mult)
            elif day_of_week in [1, 4]:  # Quality days
                if form > 0:
                    return TrainingRecommendation.MODERATE.value, int(60 * week_mult)
                else:
                    return TrainingRecommendation.EASY.value, int(40 * week_mult)
            elif day_of_week in [2, 5]:  # Hard days
                if form > 5:
                    return TrainingRecommendation.PEAK.value, int(120 * week_mult)
                else:
                    return TrainingRecommendation.MODERATE.value, int(60 * week_mult)
            else:  # Recovery days
                return TrainingRecommendation.EASY.value, int(40 * week_mult)

        elif current_phase == "PEAK":
            # Peak phase: high intensity, moderate volume
            if day_of_week in [0, 6]:  # Weekend
                return TrainingRecommendation.MODERATE.value, int(80 * week_mult)
            elif day_of_week in [1, 3, 5]:  # Quality days
                if form > 0:
                    return TrainingRecommendation.PEAK.value, int(120 * week_mult)
                else:
                    return TrainingRecommendation.MODERATE.value, int(70 * week_mult)
            else:  # Recovery days
                return TrainingRecommendation.EASY.value, int(50 * week_mult)

        else:  # RECOVERY phase
            # Recovery phase: low intensity, allow supercompensation
            if day_of_week in [0, 6]:  # Weekend
                return TrainingRecommendation.RECOVERY.value, int(30 * week_mult)
            elif day_of_week in [1, 3, 5]:  # Light training
                return TrainingRecommendation.EASY.value, int(40 * week_mult)
            else:  # Very easy days
                return TrainingRecommendation.RECOVERY.value, int(25 * week_mult)

    def _get_current_fitness(self) -> float:
        """Get current fitness level for periodization tracking."""
        if not self.db:
            return 70.0  # Default fitness level

        try:
            with self.db.get_session() as session:
                latest_metric = session.query(Metric).order_by(Metric.date.desc()).first()
                return latest_metric.fitness if latest_metric else 70.0
        except Exception:
            return 70.0