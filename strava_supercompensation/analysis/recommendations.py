"""Training recommendation engine based on supercompensation analysis."""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..config import config
from ..db import get_db
from ..db.models import Metric, Activity
from .multisport_metrics import MultiSportCalculator
from .wellness_analyzer import get_wellness_analyzer


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
        self.wellness_analyzer = get_wellness_analyzer(user_id)

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

            # Get wellness data for enhanced recommendations
            wellness_readiness = self._get_wellness_readiness()

            # Generate recommendation
            recommendation = self._calculate_recommendation(
                fitness, fatigue, form, recent_loads, recent_activities, wellness_readiness
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
        wellness_readiness: Dict = None
    ) -> Dict[str, any]:
        """Calculate training recommendation based on metrics and recent activities."""

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
            wellness_modifier = self.wellness_analyzer.get_training_load_modifier()

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

        # PRIORITY 1: Check for recent race or very intense activity
        if recent_activities:
            recovery_analysis = self._analyze_recovery_needs(recent_activities)

            if recovery_analysis["needs_recovery"]:
                return recovery_analysis["recommendation"]

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
            return self.wellness_analyzer.get_readiness_score()
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

        # Simulate a week of training
        for day in range(7):
            # Decay fitness and fatigue
            fitness *= 0.98  # Approximate daily decay
            fatigue *= 0.90  # Faster fatigue decay

            form = fitness - fatigue

            # Simple weekly pattern
            if day in [2, 5]:  # Hard days
                if form > 0:
                    rec_type = TrainingRecommendation.HARD.value
                    load = 100
                else:
                    rec_type = TrainingRecommendation.MODERATE.value
                    load = 60
            elif day in [0, 6]:  # Rest/recovery days
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