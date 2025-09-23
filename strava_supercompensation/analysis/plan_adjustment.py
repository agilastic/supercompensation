"""Plan adjustment system for adaptive training plan modifications.

This module provides:
1. Real-time plan adjustments based on actual vs planned performance
2. Adaptive load modifications based on recovery status
3. Sport-specific adjustments for different activities
4. Weather and life constraint adaptations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .advanced_planning import WorkoutPlan, WorkoutType, AdvancedPlanGenerator
from .advanced_model import EnhancedFitnessFatigueModel, PerPotModel
from .multisystem_recovery import MultiSystemRecoveryModel
from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric


class AdjustmentReason(Enum):
    """Reasons for plan adjustments."""
    OVERTRAINING_RISK = "overtraining_risk"
    POOR_RECOVERY = "poor_recovery"
    ILLNESS = "illness"
    SCHEDULE_CONFLICT = "schedule_conflict"
    WEATHER = "weather"
    EQUIPMENT_ISSUE = "equipment_issue"
    PERFORMANCE_DELTA = "performance_delta"
    FATIGUE_ACCUMULATION = "fatigue_accumulation"
    TRAVEL = "travel"
    LIFE_STRESS = "life_stress"


class AdjustmentType(Enum):
    """Types of adjustments that can be made."""
    SKIP_WORKOUT = "skip_workout"
    REDUCE_LOAD = "reduce_load"
    CHANGE_SPORT = "change_sport"
    MOVE_WORKOUT = "move_workout"
    REPLACE_WITH_RECOVERY = "replace_with_recovery"
    EXTEND_RECOVERY = "extend_recovery"
    MODIFY_INTENSITY = "modify_intensity"
    SPLIT_WORKOUT = "split_workout"


@dataclass
class Adjustment:
    """Represents a single plan adjustment."""
    date: datetime
    original_workout: WorkoutPlan
    adjusted_workout: Optional[WorkoutPlan]
    reason: AdjustmentReason
    adjustment_type: AdjustmentType
    confidence: float  # 0-1, confidence in the adjustment
    automatic: bool    # Whether adjustment was made automatically
    user_approved: bool = False
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


class PlanAdjustmentEngine:
    """Engine for making intelligent plan adjustments."""

    def __init__(self, user_id: str = "default"):
        """Initialize adjustment engine."""
        self.user_id = user_id
        self.db = get_db()

        # Initialize models
        self.ff_model = EnhancedFitnessFatigueModel(user_id=user_id)
        self.perpot_model = PerPotModel(user_id=user_id)
        self.recovery_model = MultiSystemRecoveryModel()
        self.plan_generator = AdvancedPlanGenerator(user_id=user_id)

        # Adjustment thresholds
        self.overtraining_threshold = 0.7
        self.poor_recovery_threshold = 60.0
        self.performance_delta_threshold = 0.3
        self.fatigue_ratio_threshold = 1.5

    def evaluate_plan_adjustments(
        self,
        current_plan: List[WorkoutPlan],
        actual_performance: Dict[str, Any] = None,
        wellness_data: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None
    ) -> List[Adjustment]:
        """Evaluate current plan and suggest adjustments.

        Args:
            current_plan: List of planned workouts
            actual_performance: Recent actual performance data
            wellness_data: Current wellness metrics
            constraints: External constraints (weather, schedule, etc.)

        Returns:
            List of suggested adjustments
        """
        adjustments = []

        # Get current state
        current_state = self._get_current_training_state()

        # Check for various adjustment triggers
        adjustments.extend(self._check_overtraining_risk(current_plan, current_state))
        adjustments.extend(self._check_recovery_status(current_plan, wellness_data))
        adjustments.extend(self._check_performance_variance(current_plan, actual_performance))
        adjustments.extend(self._check_fatigue_accumulation(current_plan, current_state))

        if constraints:
            adjustments.extend(self._apply_external_constraints(current_plan, constraints))

        # Sort by importance and confidence
        adjustments.sort(key=lambda a: (a.confidence, a.reason.value), reverse=True)

        return adjustments

    def apply_adjustments(
        self,
        current_plan: List[WorkoutPlan],
        adjustments: List[Adjustment],
        auto_apply_threshold: float = 0.8
    ) -> Tuple[List[WorkoutPlan], List[Adjustment]]:
        """Apply adjustments to the training plan.

        Args:
            current_plan: Original plan
            adjustments: List of adjustments to apply
            auto_apply_threshold: Automatically apply adjustments above this confidence

        Returns:
            Tuple of (adjusted_plan, applied_adjustments)
        """
        adjusted_plan = current_plan.copy()
        applied_adjustments = []

        for adjustment in adjustments:
            # Auto-apply high-confidence adjustments
            if adjustment.confidence >= auto_apply_threshold or adjustment.automatic:
                adjusted_plan = self._apply_single_adjustment(adjusted_plan, adjustment)
                adjustment.user_approved = True
                applied_adjustments.append(adjustment)

        # Recalculate physiological response after adjustments
        adjusted_plan = self._recalculate_plan_metrics(adjusted_plan)

        return adjusted_plan, applied_adjustments

    def _check_overtraining_risk(
        self,
        plan: List[WorkoutPlan],
        current_state: Dict[str, float]
    ) -> List[Adjustment]:
        """Check for overtraining risk and suggest adjustments."""
        adjustments = []

        # Simulate plan to detect overtraining
        loads = np.array([w.planned_load for w in plan])
        t = np.arange(len(loads))
        perpot_results = self.perpot_model.simulate(loads, t)

        # Find overtraining risk periods
        risk_days = np.where(perpot_results['overtraining_risk'])[0]

        for day_idx in risk_days:
            if day_idx < len(plan):
                workout = plan[day_idx]

                # Suggest recovery instead of planned workout
                adjustment = Adjustment(
                    date=workout.date,
                    original_workout=workout,
                    adjusted_workout=self._create_recovery_workout(workout),
                    reason=AdjustmentReason.OVERTRAINING_RISK,
                    adjustment_type=AdjustmentType.REPLACE_WITH_RECOVERY,
                    confidence=0.9,
                    automatic=True,
                    notes=f"PerPot model indicates overtraining risk. "
                           f"Strain potential > 1.0 on day {day_idx + 1}"
                )
                adjustments.append(adjustment)

        return adjustments

    def _check_recovery_status(
        self,
        plan: List[WorkoutPlan],
        wellness_data: Dict[str, Any]
    ) -> List[Adjustment]:
        """Check recovery status and suggest adjustments."""
        adjustments = []

        if not wellness_data:
            return adjustments

        # Check various recovery indicators
        recovery_score = wellness_data.get('recovery_score', 100)
        hrv_status = wellness_data.get('hrv_status', 'normal')
        sleep_quality = wellness_data.get('sleep_quality', 100)
        stress_level = wellness_data.get('stress_level', 0)

        # Calculate composite recovery
        composite_recovery = (
            recovery_score * 0.4 +
            (100 - stress_level) * 0.3 +
            sleep_quality * 0.3
        )

        if composite_recovery < self.poor_recovery_threshold:
            # Find next high-intensity workouts to modify
            for i, workout in enumerate(plan[:7]):  # Next week
                if workout.intensity_level in ['hard', 'peak']:
                    adjustment = Adjustment(
                        date=workout.date,
                        original_workout=workout,
                        adjusted_workout=self._reduce_workout_intensity(workout),
                        reason=AdjustmentReason.POOR_RECOVERY,
                        adjustment_type=AdjustmentType.REDUCE_LOAD,
                        confidence=0.8,
                        automatic=False,
                        notes=f"Poor recovery status: {composite_recovery:.0f}/100. "
                               f"Reducing intensity for better adaptation."
                    )
                    adjustments.append(adjustment)

        return adjustments

    def _check_performance_variance(
        self,
        plan: List[WorkoutPlan],
        actual_performance: Dict[str, Any]
    ) -> List[Adjustment]:
        """Check for significant performance variance and adjust accordingly."""
        adjustments = []

        if not actual_performance:
            return adjustments

        # Analyze recent performance trends
        recent_activities = actual_performance.get('recent_activities', [])
        if len(recent_activities) < 3:
            return adjustments

        # Calculate performance variance
        performance_scores = []
        for activity in recent_activities:
            expected_load = activity.get('expected_load', 100)
            actual_load = activity.get('actual_load', 100)
            rpe = activity.get('rpe', 5)  # 1-10 scale

            # Performance score based on load completion and RPE
            load_ratio = actual_load / expected_load if expected_load > 0 else 1.0
            rpe_score = (10 - rpe) / 10  # Convert to 0-1 scale

            performance_score = (load_ratio + rpe_score) / 2
            performance_scores.append(performance_score)

        avg_performance = np.mean(performance_scores)
        performance_trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]

        # If performance is declining or consistently poor
        if avg_performance < 0.6 or performance_trend < -0.1:
            # Suggest easier workouts for the next few days
            for i, workout in enumerate(plan[:5]):
                if workout.intensity_level in ['hard', 'peak']:
                    adjustment = Adjustment(
                        date=workout.date,
                        original_workout=workout,
                        adjusted_workout=self._reduce_workout_intensity(workout),
                        reason=AdjustmentReason.PERFORMANCE_DELTA,
                        adjustment_type=AdjustmentType.MODIFY_INTENSITY,
                        confidence=0.7,
                        automatic=False,
                        notes=f"Performance declining: avg={avg_performance:.2f}, "
                               f"trend={performance_trend:.3f}. Reducing load temporarily."
                    )
                    adjustments.append(adjustment)

        return adjustments

    def _check_fatigue_accumulation(
        self,
        plan: List[WorkoutPlan],
        current_state: Dict[str, float]
    ) -> List[Adjustment]:
        """Check for excessive fatigue accumulation."""
        adjustments = []

        current_fitness = current_state.get('fitness', 50)
        current_fatigue = current_state.get('fatigue', 30)
        fatigue_ratio = current_fatigue / (current_fitness + 1e-5)

        if fatigue_ratio > self.fatigue_ratio_threshold:
            # Insert recovery days
            for i, workout in enumerate(plan[:7]):
                if i % 2 == 0 and workout.workout_type not in [WorkoutType.REST, WorkoutType.RECOVERY]:
                    adjustment = Adjustment(
                        date=workout.date,
                        original_workout=workout,
                        adjusted_workout=self._create_recovery_workout(workout),
                        reason=AdjustmentReason.FATIGUE_ACCUMULATION,
                        adjustment_type=AdjustmentType.REPLACE_WITH_RECOVERY,
                        confidence=0.8,
                        automatic=True,
                        notes=f"High fatigue ratio: {fatigue_ratio:.2f}. "
                               f"Adding recovery to prevent overreaching."
                    )
                    adjustments.append(adjustment)

        return adjustments

    def _apply_external_constraints(
        self,
        plan: List[WorkoutPlan],
        constraints: Dict[str, Any]
    ) -> List[Adjustment]:
        """Apply external constraints like weather, schedule conflicts."""
        adjustments = []

        # Weather constraints
        weather = constraints.get('weather', {})
        for date, forecast in weather.items():
            workout_date = datetime.fromisoformat(date) if isinstance(date, str) else date
            workout_date_obj = workout_date.date() if hasattr(workout_date, 'date') else workout_date
            workout = next((w for w in plan if (w.date.date() if hasattr(w.date, 'date') else w.date) == workout_date_obj), None)

            if workout and self._is_weather_unsuitable(workout, forecast):
                adjustment = Adjustment(
                    date=workout.date,
                    original_workout=workout,
                    adjusted_workout=self._adapt_for_weather(workout, forecast),
                    reason=AdjustmentReason.WEATHER,
                    adjustment_type=AdjustmentType.CHANGE_SPORT,
                    confidence=0.9,
                    automatic=True,
                    notes=f"Weather adjustment: {forecast.get('condition', 'poor')} conditions"
                )
                adjustments.append(adjustment)

        # Schedule constraints
        schedule_conflicts = constraints.get('schedule_conflicts', [])
        for conflict in schedule_conflicts:
            conflict_date = conflict.get('date')
            conflict_date_obj = conflict_date.date() if hasattr(conflict_date, 'date') else conflict_date
            workout = next((w for w in plan if (w.date.date() if hasattr(w.date, 'date') else w.date) == conflict_date_obj), None)

            if workout:
                adjustment = Adjustment(
                    date=workout.date,
                    original_workout=workout,
                    adjusted_workout=None,  # Skip or move
                    reason=AdjustmentReason.SCHEDULE_CONFLICT,
                    adjustment_type=AdjustmentType.SKIP_WORKOUT,
                    confidence=1.0,
                    automatic=False,
                    notes=f"Schedule conflict: {conflict.get('reason', 'unavailable')}"
                )
                adjustments.append(adjustment)

        return adjustments

    def _create_recovery_workout(self, original: WorkoutPlan) -> WorkoutPlan:
        """Create a recovery workout based on original workout."""
        recovery = WorkoutPlan(
            date=original.date,
            day_number=original.day_number,
            week_number=original.week_number,
            mesocycle=original.mesocycle,
            workout_type=WorkoutType.RECOVERY,
            primary_sport=self._get_recovery_sport(original.primary_sport),
            alternative_sports=["Walk", "Yoga", "Swim"],
            planned_load=min(30, original.planned_load * 0.3),
            intensity_level="easy",
            hr_zones={'z1': 90, 'z2': 10, 'z3': 0, 'z4': 0, 'z5': 0},
            total_duration_min=max(20, original.total_duration_min // 2),
            warmup_min=5,
            main_set_min=max(10, original.total_duration_min // 2 - 10),
            cooldown_min=5,
            title=f"Recovery {self._get_recovery_sport(original.primary_sport)}",
            description=f"Easy recovery session to promote adaptation. "
                       f"Keep effort very low (conversational pace).",
            workout_structure={'warmup': 5, 'main': 15, 'cooldown': 5, 'sets': []},
            primary_system="recovery",
            secondary_systems=[],
            expected_adaptations=["recovery", "parasympathetic_activation"],
            recovery_hours_needed=12,
            next_day_recommendation="Can continue normal training",
            can_skip=True,
            can_move=True,
            priority=3
        )
        return recovery

    def _reduce_workout_intensity(self, original: WorkoutPlan) -> WorkoutPlan:
        """Reduce workout intensity while maintaining structure."""
        reduced = WorkoutPlan(
            date=original.date,
            day_number=original.day_number,
            week_number=original.week_number,
            mesocycle=original.mesocycle,
            workout_type=original.workout_type,
            primary_sport=original.primary_sport,
            alternative_sports=original.alternative_sports,
            planned_load=original.planned_load * 0.7,  # 30% reduction
            intensity_level=self._reduce_intensity_level(original.intensity_level),
            hr_zones=self._reduce_hr_zones(original.hr_zones),
            total_duration_min=original.total_duration_min,
            warmup_min=original.warmup_min,
            main_set_min=original.main_set_min,
            cooldown_min=original.cooldown_min,
            title=f"Reduced {original.title}",
            description=f"Modified intensity: {original.description} "
                       f"(Reduced load for recovery)",
            workout_structure=original.workout_structure,
            primary_system=original.primary_system,
            secondary_systems=original.secondary_systems,
            expected_adaptations=original.expected_adaptations,
            recovery_hours_needed=int(original.recovery_hours_needed * 0.8),
            next_day_recommendation=original.next_day_recommendation,
            can_skip=original.can_skip,
            can_move=original.can_move,
            priority=original.priority
        )
        return reduced

    def _is_weather_unsuitable(
        self,
        workout: WorkoutPlan,
        forecast: Dict[str, Any]
    ) -> bool:
        """Check if weather is unsuitable for planned workout."""
        condition = forecast.get('condition', '').lower()
        temp = forecast.get('temperature', 20)
        wind = forecast.get('wind_speed', 0)
        rain = forecast.get('precipitation', 0)

        # Outdoor sports that are weather-sensitive
        outdoor_sports = ['run', 'ride', 'bike', 'cycling']

        if workout.primary_sport.lower() in outdoor_sports:
            # Check various weather conditions
            if 'storm' in condition or 'severe' in condition:
                return True
            if rain > 80:  # Heavy rain
                return True
            if temp < -10 or temp > 40:  # Extreme temperatures
                return True
            if wind > 50:  # Strong winds for cycling
                return True

        return False

    def _adapt_for_weather(
        self,
        workout: WorkoutPlan,
        forecast: Dict[str, Any]
    ) -> WorkoutPlan:
        """Adapt workout for weather conditions."""
        # Map outdoor activities to indoor alternatives
        indoor_alternatives = {
            'run': 'Treadmill',
            'ride': 'VirtualRide',
            'bike': 'VirtualRide',
            'cycling': 'VirtualRide'
        }

        new_sport = indoor_alternatives.get(
            workout.primary_sport.lower(),
            workout.primary_sport
        )

        adapted = WorkoutPlan(
            date=workout.date,
            day_number=workout.day_number,
            week_number=workout.week_number,
            mesocycle=workout.mesocycle,
            workout_type=workout.workout_type,
            primary_sport=new_sport,
            alternative_sports=["Gym", "Indoor", "Swim"],
            planned_load=workout.planned_load,
            intensity_level=workout.intensity_level,
            hr_zones=workout.hr_zones,
            total_duration_min=workout.total_duration_min,
            warmup_min=workout.warmup_min,
            main_set_min=workout.main_set_min,
            cooldown_min=workout.cooldown_min,
            title=f"Indoor {workout.title}",
            description=f"Weather adapted: {workout.description}",
            workout_structure=workout.workout_structure,
            primary_system=workout.primary_system,
            secondary_systems=workout.secondary_systems,
            expected_adaptations=workout.expected_adaptations,
            recovery_hours_needed=workout.recovery_hours_needed,
            next_day_recommendation=workout.next_day_recommendation,
            can_skip=workout.can_skip,
            can_move=workout.can_move,
            priority=workout.priority
        )
        return adapted

    def _apply_single_adjustment(
        self,
        plan: List[WorkoutPlan],
        adjustment: Adjustment
    ) -> List[WorkoutPlan]:
        """Apply a single adjustment to the plan."""
        adjusted_plan = plan.copy()

        # Find the workout to adjust
        adj_date_obj = adjustment.date.date() if hasattr(adjustment.date, 'date') else adjustment.date
        workout_idx = next(
            (i for i, w in enumerate(adjusted_plan) if (w.date.date() if hasattr(w.date, 'date') else w.date) == adj_date_obj),
            None
        )

        if workout_idx is None:
            return adjusted_plan

        if adjustment.adjustment_type == AdjustmentType.SKIP_WORKOUT:
            # Remove the workout
            adjusted_plan.pop(workout_idx)

        elif adjustment.adjustment_type in [
            AdjustmentType.REDUCE_LOAD,
            AdjustmentType.REPLACE_WITH_RECOVERY,
            AdjustmentType.CHANGE_SPORT,
            AdjustmentType.MODIFY_INTENSITY
        ]:
            # Replace with adjusted workout
            if adjustment.adjusted_workout:
                adjusted_plan[workout_idx] = adjustment.adjusted_workout

        elif adjustment.adjustment_type == AdjustmentType.MOVE_WORKOUT:
            # Move workout to different day (implementation depends on target date)
            # For now, just skip - more complex logic needed for rescheduling
            adjusted_plan.pop(workout_idx)

        return adjusted_plan

    def _recalculate_plan_metrics(
        self,
        plan: List[WorkoutPlan]
    ) -> List[WorkoutPlan]:
        """Recalculate physiological metrics after plan adjustments."""
        loads = np.array([w.planned_load for w in plan])
        t = np.arange(len(loads))

        # Recalculate fitness-fatigue
        fitness, fatigue, performance = self.ff_model.impulse_response(loads, t)

        # Update workout predictions
        for i, workout in enumerate(plan):
            if i < len(fitness):
                workout.predicted_fitness = fitness[i]
                workout.predicted_fatigue = fatigue[i]
                workout.predicted_form = fitness[i] - fatigue[i]

        return plan

    def _get_current_training_state(self) -> Dict[str, float]:
        """Get current training state from database."""
        with self.db.get_session() as session:
            latest = session.query(Metric).order_by(Metric.date.desc()).first()

            if latest:
                return {
                    'fitness': latest.fitness,
                    'fatigue': latest.fatigue,
                    'form': latest.form
                }
            else:
                return {
                    'fitness': 50.0,
                    'fatigue': 30.0,
                    'form': 20.0
                }

    def _get_recovery_sport(self, original_sport: str) -> str:
        """Get appropriate recovery sport."""
        recovery_map = {
            'Run': 'Walk',
            'Ride': 'EasyRide',
            'Swim': 'EasySwim',
            'VirtualRide': 'Walk'
        }
        return recovery_map.get(original_sport, 'Walk')

    def _reduce_intensity_level(self, original: str) -> str:
        """Reduce intensity level by one step."""
        intensity_levels = ['rest', 'easy', 'moderate', 'hard', 'peak']
        try:
            current_idx = intensity_levels.index(original)
            return intensity_levels[max(0, current_idx - 1)]
        except ValueError:
            return 'easy'

    def _reduce_hr_zones(self, original_zones: Dict[str, float]) -> Dict[str, float]:
        """Shift heart rate zones to lower intensities."""
        return {
            'z1': min(100, original_zones.get('z1', 20) + 20),
            'z2': max(0, original_zones.get('z2', 50) + 10),
            'z3': max(0, original_zones.get('z3', 20) - 10),
            'z4': max(0, original_zones.get('z4', 10) - 15),
            'z5': max(0, original_zones.get('z5', 0) - 5)
        }


def create_adjustment_engine(user_id: str = "default") -> PlanAdjustmentEngine:
    """Factory function to create adjustment engine."""
    return PlanAdjustmentEngine(user_id=user_id)