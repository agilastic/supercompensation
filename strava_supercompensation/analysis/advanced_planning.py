"""Advanced 30-day training plan generation with visualization.

This module provides:
1. Mesocycle-based periodization (4-week blocks)
2. Adaptive load progression based on current state
3. Sport-specific planning with cross-training
4. Interactive visualization capabilities
5. Plan adjustment based on actual vs planned performance
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from .advanced_model import (
    FitnessFatigueModel,
    PerPotModel,
    OptimalControlProblem,
    TrainingGoal
)
from .multisystem_recovery import MultiSystemRecoveryModel, RecoverySystem
from .data_validation import DataValidator, ValidationResult
from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric, PeriodizationState


class MesocycleType(Enum):
    """Types of 4-week training mesocycles."""
    ADAPTATION = "adaptation"      # Base building, gradual increase
    ACCUMULATION = "accumulation"  # High volume, moderate intensity
    INTENSIFICATION = "intensification"  # Higher intensity, lower volume
    REALIZATION = "realization"    # Taper and peak
    RECOVERY = "recovery"          # Active recovery and regeneration
    MAINTENANCE = "maintenance"    # Maintain fitness with minimal stress


class WorkoutType(Enum):
    """Specific workout types with physiological targets."""
    REST = "rest"
    RECOVERY = "recovery"           # <60% max HR, regeneration
    AEROBIC = "aerobic"            # 60-75% max HR, base building
    TEMPO = "tempo"                # 75-85% max HR, lactate threshold
    THRESHOLD = "threshold"        # 85-90% max HR, anaerobic threshold
    VO2MAX = "vo2max"              # 90-95% max HR, maximal oxygen uptake
    NEUROMUSCULAR = "neuromuscular"  # 95-100% max HR, power/speed
    LONG = "long"                  # Extended duration, aerobic
    FARTLEK = "fartlek"           # Variable intensity
    INTERVALS = "intervals"        # Structured high intensity


@dataclass
class WorkoutPlan:
    """Detailed workout plan for a single day."""
    date: datetime
    day_number: int
    week_number: int
    mesocycle: MesocycleType
    workout_type: WorkoutType
    primary_sport: str
    alternative_sports: List[str]

    # Load and intensity
    planned_load: float
    intensity_level: str  # easy, moderate, hard, peak
    hr_zones: Dict[str, float]  # time in each zone

    # Duration and structure
    total_duration_min: int
    warmup_min: int
    main_set_min: int
    cooldown_min: int

    # Detailed description
    title: str
    description: str
    workout_structure: Dict[str, Any]

    # Physiological targets
    primary_system: str  # aerobic, anaerobic, neuromuscular
    secondary_systems: List[str]
    expected_adaptations: List[str]

    # Recovery needs
    recovery_hours_needed: int
    next_day_recommendation: str

    # Metrics
    predicted_fitness: float = 0
    predicted_fatigue: float = 0
    predicted_form: float = 0
    overtraining_risk: float = 0

    # Adjustability
    can_skip: bool = False
    can_move: bool = True
    priority: int = 5  # 1-10, higher is more important

    # Two-a-day session support
    second_activity_title: Optional[str] = None
    second_activity_duration: Optional[int] = None  # minutes
    second_activity_rationale: Optional[str] = None


@dataclass
class Mesocycle:
    """4-week training block with specific focus."""
    cycle_type: MesocycleType
    start_date: datetime
    end_date: datetime
    weeks: List[List[WorkoutPlan]]

    # Load progression
    weekly_loads: List[float]
    load_pattern: str  # "3:1", "2:1", "progressive", "stable"

    # Intensity distribution
    easy_percentage: float
    moderate_percentage: float
    hard_percentage: float

    # Goals and adaptations
    primary_goal: str
    expected_fitness_gain: float
    expected_fatigue_peak: float

    # Key workouts
    key_workouts: List[WorkoutPlan]
    test_workout: Optional[WorkoutPlan] = None


class TrainingPlanGenerator:
    """Generate 30-day training plans using optimization algorithms."""

    def __init__(self, user_id: str = "default"):
        """Initialize plan generator with user context."""
        self.user_id = user_id
        self.db = get_db()
        self.logger = logging.getLogger(__name__)

        # Load configuration from .env
        self.config = config

        # Initialize models with configuration
        self.ff_model = FitnessFatigueModel(user_id=user_id)
        self.perpot_model = PerPotModel(user_id=user_id)
        self.recovery_model = MultiSystemRecoveryModel()
        self.data_validator = DataValidator()

        # Load user preferences and history
        self._load_user_context()

    def _load_user_context(self):
        """Load user preferences, history, and constraints."""
        with self.db.get_session() as session:
            # Get recent activities for pattern analysis
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            activities = session.query(Activity).filter(
                Activity.start_date >= cutoff
            ).all()

            # Analyze patterns from history
            historical_sports = self._analyze_sport_preferences(activities)
            self.typical_duration = self._analyze_duration_patterns(activities)
            self.weekly_pattern = self._analyze_weekly_patterns(activities)

            # Load sport preferences from configuration
            self.enabled_sports = self._get_enabled_sports_from_config()
            self.sport_preferences = self._get_sport_preferences_from_config()
            self.recovery_times = self._get_recovery_times_from_config()
            self.load_multipliers = self._get_load_multipliers_from_config()

            # Debug configuration loading

            # Combine historical data with configuration
            self.preferred_sports = self._combine_sport_preferences(historical_sports)

            # Get current state
            latest_metric = session.query(Metric).order_by(
                Metric.date.desc()
            ).first()

            if latest_metric:
                # Use data validator for comprehensive metric validation
                validation_results = {}
                validation_results['fitness'] = self.data_validator.validate_metric(
                    'fitness', latest_metric.fitness
                )
                validation_results['fatigue'] = self.data_validator.validate_metric(
                    'fatigue', latest_metric.fatigue
                )
                validation_results['form'] = self.data_validator.validate_metric(
                    'form', latest_metric.form
                )

                # Check if all metrics are valid
                all_valid = all(result.is_valid for result in validation_results.values())

                if all_valid:
                    self.current_fitness = latest_metric.fitness
                    self.current_fatigue = latest_metric.fatigue
                    self.current_form = latest_metric.form
                else:
                    # Log validation issues and use suggested values or defaults
                    invalid_metrics = [name for name, result in validation_results.items() if not result.is_valid]
                    self.logger.warning(f"Invalid metrics detected: {invalid_metrics}. Applying corrections.")

                    # Use validated/corrected values
                    self.current_fitness = (validation_results['fitness'].suggested_value
                                           if not validation_results['fitness'].is_valid and validation_results['fitness'].suggested_value
                                           else latest_metric.fitness if validation_results['fitness'].is_valid
                                           else 85.0)  # Elite athlete baseline

                    self.current_fatigue = (validation_results['fatigue'].suggested_value
                                           if not validation_results['fatigue'].is_valid and validation_results['fatigue'].suggested_value
                                           else latest_metric.fatigue if validation_results['fatigue'].is_valid
                                           else 45.0)  # Elite athlete baseline

                    self.current_form = (validation_results['form'].suggested_value
                                        if not validation_results['form'].is_valid and validation_results['form'].suggested_value
                                        else latest_metric.form if validation_results['form'].is_valid
                                        else self.current_fitness - self.current_fatigue)  # Calculate from other metrics
            else:
                # No metrics exist - use beginner-friendly defaults
                self.current_fitness = 75.0   # Moderate fitness base
                self.current_fatigue = 35.0   # Moderate fatigue
                self.current_form = 40.0      # Positive form

    def generate_training_plan(
        self,
        duration_days: int = 30,
        goal: str = "fitness",  # fitness, performance, recovery, maintenance
        target_event_date: Optional[datetime] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a complete training plan for specified duration.

        Args:
            duration_days: Length of plan in days (30, 60, or 90)
            goal: Primary training goal
            target_event_date: Optional target event/race date
            constraints: User constraints (rest days, max hours, etc.)

        Returns:
            Complete training plan with visualizations
        """
        constraints = constraints or {}

        # Enhanced mesocycle planning for extended durations
        mesocycles = self._plan_mesocycle_sequence(duration_days, goal, target_event_date)
        if len(mesocycles) > 1:
            self.logger.info(f"Extended plan: {len(mesocycles)} mesocycles over {duration_days} days")

        # Use first mesocycle for initial settings (backward compatibility)
        mesocycle = mesocycles[0]
        if mesocycle.cycle_type == MesocycleType.RECOVERY:
            self.logger.info("FORCED RECOVERY: Overriding goal due to overtraining risk")

        # FIXED: Optimize load distribution across ALL mesocycles
        optimized_loads = self._optimize_load_distribution(
            mesocycles, goal, constraints, duration_days
        )

        # Calculate strength days using first mesocycle start date
        strength_days_config = config.TRAINING_STRENGTH_DAYS
        configured_strength_days = []
        if strength_days_config:
            try:
                configured_strength_days = [int(x.strip()) for x in strength_days_config.split(',')]
            except (ValueError, AttributeError):
                configured_strength_days = []

        strength_days = []
        start_date = mesocycles[0].start_date
        for day_offset in range(duration_days):
            plan_date = start_date + timedelta(days=day_offset)
            actual_weekday = plan_date.weekday()  # 0=Monday, 6=Sunday
            if actual_weekday in configured_strength_days:
                strength_days.append(day_offset)

        # Create detailed daily workouts
        daily_workouts = self._create_daily_workouts(
            mesocycle, optimized_loads, duration_days, strength_days
        )

        # Simulate physiological response
        simulated_response = self._simulate_training_response(
            daily_workouts
        )

        # Add recovery and adaptation periods
        final_plan = self._add_recovery_periods(
            daily_workouts, simulated_response
        )

        # Generate visualizations
        visualizations = self._create_visualizations(
            final_plan, simulated_response, duration_days
        )

        return {
            'mesocycle': mesocycle,
            'daily_workouts': final_plan,
            'simulated_response': simulated_response,
            'visualizations': visualizations,
            'summary': self._create_plan_summary(final_plan, simulated_response, duration_days)
        }

    def _determine_mesocycle_type(
        self,
        goal: str,
        target_date: Optional[datetime]
    ) -> MesocycleType:
        """Determine appropriate mesocycle based on goal and timing."""
        if target_date:
            days_to_event = (target_date - datetime.now(timezone.utc)).days

            if days_to_event <= 14:
                return MesocycleType.REALIZATION  # Taper
            elif days_to_event <= 30:
                return MesocycleType.INTENSIFICATION  # Peak intensity
            elif days_to_event <= 60:
                return MesocycleType.ACCUMULATION  # Build volume
            else:
                return MesocycleType.ADAPTATION  # Base building

        # No target event, base on goal
        goal_map = {
            'fitness': MesocycleType.ACCUMULATION,
            'performance': MesocycleType.INTENSIFICATION,
            'recovery': MesocycleType.RECOVERY,
            'maintenance': MesocycleType.MAINTENANCE
        }

        return goal_map.get(goal, MesocycleType.ADAPTATION)

    def _intelligent_mesocycle_selection(
        self,
        goal: str,
        target_date: Optional[datetime]
    ) -> MesocycleType:
        """Intelligent mesocycle selection based on current athlete state.

        REQUIREMENT 1: Implements state-aware coaching intelligence.
        Checks overtraining risk and readiness before selecting mesocycle type.
        Forces RECOVERY mesocycle when athlete needs it, regardless of goal.
        """
        try:
            # Get current athlete state using IntegratedTrainingAnalyzer
            from .model_integration import IntegratedTrainingAnalyzer

            analyzer = IntegratedTrainingAnalyzer()

            # Get current state through daily recommendation (which includes all metrics we need)
            recommendation = analyzer.get_daily_recommendation()

            readiness_score = recommendation.get('readiness_score', 50.0)

            # Check for overtraining risk in the current state
            current_state = analyzer._get_current_state()
            overtraining_risk = False
            if current_state and current_state.get('form', 0) < -30:
                overtraining_risk = True

            # Critical thresholds for forcing recovery
            CRITICAL_READINESS_THRESHOLD = 40.0  # Below this = mandatory recovery

            # INTELLIGENT OVERRIDE: Force recovery if athlete needs it
            if overtraining_risk or readiness_score < CRITICAL_READINESS_THRESHOLD:
                print(f"🚨 INTELLIGENT OVERRIDE: Forcing RECOVERY mesocycle")
                print(f"   Readiness: {readiness_score:.1f}% (threshold: {CRITICAL_READINESS_THRESHOLD}%)")
                print(f"   Overtraining risk: {overtraining_risk}")
                print(f"   Original goal '{goal}' overridden for athlete safety")

                return MesocycleType.RECOVERY

            # If athlete is in good condition, proceed with normal goal-based selection
            print(f"✅ Athlete state is good: Readiness {readiness_score:.1f}%, No overtraining")
            return self._determine_mesocycle_type(goal, target_date)

        except Exception as e:
            # Fallback to standard selection if state analysis fails
            print(f"⚠️ State analysis failed ({e}), using standard mesocycle selection")
            return self._determine_mesocycle_type(goal, target_date)

    def _create_mesocycle(
        self,
        cycle_type: MesocycleType,
        start_date: datetime,
        constraints: Dict
    ) -> Mesocycle:
        """Create a 4-week mesocycle structure."""

        # Define load patterns for each mesocycle type (implementing 3:1 and 4:1 periodization)
        patterns = {
            MesocycleType.ADAPTATION: {
                'loads': [60, 70, 80, 45],  # Classic 3:1 - build, build, build, recover
                'pattern': '3:1',
                'easy': 0.75, 'moderate': 0.20, 'hard': 0.05  # Predominantly aerobic
            },
            MesocycleType.ACCUMULATION: {
                'loads': [75, 85, 95, 50],  # 3:1 high volume progression
                'pattern': '3:1',
                'easy': 0.70, 'moderate': 0.25, 'hard': 0.05  # Volume focus
            },
            MesocycleType.INTENSIFICATION: {
                'loads': [70, 75, 80, 85, 45],  # 4:1 pattern for intensity
                'pattern': '4:1',
                'easy': 0.60, 'moderate': 0.25, 'hard': 0.15  # Higher intensity ratio
            },
            MesocycleType.REALIZATION: {
                'loads': [70, 55, 40, 25],  # Progressive taper
                'pattern': 'taper',
                'easy': 0.70, 'moderate': 0.20, 'hard': 0.10  # Maintain sharpness
            },
            MesocycleType.RECOVERY: {
                'loads': [30, 45, 60, 50],  # Structured recovery: Deep -> Active -> Base Building -> Test
                'pattern': 'structured_recovery',
                'easy': 0.85, 'moderate': 0.15, 'hard': 0.0  # Recovery focused
            },
            MesocycleType.MAINTENANCE: {
                'loads': [65, 65, 65, 50],  # Stable with recovery
                'pattern': '3:1',
                'easy': 0.75, 'moderate': 0.20, 'hard': 0.05  # Maintenance focus
            }
        }

        pattern = patterns[cycle_type]

        return Mesocycle(
            cycle_type=cycle_type,
            start_date=start_date,
            end_date=start_date + timedelta(days=28),
            weeks=[],  # Will be populated with workouts
            weekly_loads=pattern['loads'],
            load_pattern=pattern['pattern'],
            easy_percentage=pattern['easy'],
            moderate_percentage=pattern['moderate'],
            hard_percentage=pattern['hard'],
            primary_goal=self._get_mesocycle_goal(cycle_type),
            expected_fitness_gain=sum(pattern['loads']) * 0.1,
            expected_fatigue_peak=max(pattern['loads']),
            key_workouts=[]
        )

    def _optimize_load_distribution(
        self,
        mesocycles: List[Mesocycle],
        goal: str,
        constraints: Dict[str, Any] = None,
        duration_days: int = 30
    ) -> np.ndarray:
        """Optimize daily load distribution using physiological principles.

        Implements progressive overload with proper intensity variation
        across multiple mesocycles for extended duration plans.

        CRITICAL: Enforces physiological constraints to prevent impossible plans.
        """
        constraints = constraints or {}

        # PHYSIOLOGICAL CONSTRAINTS - Critical for safe training progression
        MAX_DAILY_TSS = 400          # Maximum daily TSS for elite athletes
        MAX_WEEKLY_TSS_INCREASE = 50 # Maximum weekly TSS increase (7 CTL points)
        MAX_CTL_RAMP_RATE = 7        # Maximum CTL increase per week
        MIN_RECOVERY_RATIO = 0.3     # Minimum ratio of easy days to total training days

        # Get rest days from constraints (user configuration)
        configured_rest_days = constraints.get('rest_days', [0])  # Default Monday rest

        # Get strength days from configuration
        strength_days_config = config.TRAINING_STRENGTH_DAYS
        configured_strength_days = []
        if strength_days_config:
            try:
                configured_strength_days = [int(x.strip()) for x in strength_days_config.split(',')]
            except (ValueError, AttributeError):
                configured_strength_days = []

        # FIXED: Use first mesocycle's start_date for all date calculations
        rest_days = []
        strength_days = []
        start_date = mesocycles[0].start_date  # Use first mesocycle's start date consistently

        for day_offset in range(duration_days):
            plan_date = start_date + timedelta(days=day_offset)
            actual_weekday = plan_date.weekday()  # 0=Monday, 6=Sunday
            if actual_weekday in configured_rest_days:
                rest_days.append(day_offset)
            if actual_weekday in configured_strength_days:
                strength_days.append(day_offset)

        # Get max weekly hours constraint
        max_weekly_hours = constraints.get('max_weekly_hours', 14)

        # FIXED: Create load distribution across all mesocycles
        daily_loads = np.zeros(duration_days)

        # PHYSIOLOGICAL BASE: Start from current fitness level (CTL)
        # Elite athletes: use current CTL * 7 as sustainable weekly TSS baseline
        current_ctl = self.current_fitness
        base_weekly_tss = min(current_ctl * 7, max_weekly_hours * 60)  # Respect time budget constraints

        self.logger.info(f"Base weekly TSS: {base_weekly_tss:.0f} (from CTL: {current_ctl:.1f}, max hours: {max_weekly_hours})")

        # Process all mesocycles sequentially
        day_offset = 0

        for mesocycle_idx, mesocycle in enumerate(mesocycles):
            # Use mesocycle weekly_loads as scaling factors for progressive overload
            weekly_load_factors = mesocycle.weekly_loads  # [80, 90, 100, 60] for accumulation

            # Calculate actual weekly TSS targets with PHYSIOLOGICAL CONSTRAINTS
            weekly_targets = []
            previous_week_tss = base_weekly_tss if mesocycle_idx == 0 else weekly_targets[-1] if weekly_targets else base_weekly_tss

            for week_idx, load_factor in enumerate(weekly_load_factors):
                # Scale base TSS by the load factor (treating 100 as 100% of base)
                target_weekly_tss = base_weekly_tss * (load_factor / 100.0)

                # CONSTRAINT: Limit weekly TSS increases to prevent overreaching
                if week_idx > 0 or mesocycle_idx > 0:
                    max_allowed_increase = previous_week_tss + MAX_WEEKLY_TSS_INCREASE
                    target_weekly_tss = min(target_weekly_tss, max_allowed_increase)

                # CONSTRAINT: Cap absolute maximum weekly TSS
                max_weekly_tss = MAX_DAILY_TSS * 7 * 0.6  # 60% loading across 7 days
                target_weekly_tss = min(target_weekly_tss, max_weekly_tss)

                weekly_targets.append(target_weekly_tss)
                previous_week_tss = target_weekly_tss

            # Distribute loads across each week in this mesocycle
            for week_num in range(len(weekly_load_factors)):  # Process all weeks in mesocycle
                week_start = day_offset + week_num * 7
                week_end = min(week_start + 7, duration_days)

                if week_end <= week_start or week_start >= duration_days:
                    break

                weekly_target_load = weekly_targets[week_num]

                # Create intensity distribution for the week
                week_days = week_end - week_start
                week_loads = self._create_weekly_load_pattern(
                    weekly_target_load,
                    week_days,
                    mesocycle,
                    week_num,
                    [day - week_start for day in rest_days if week_start <= day < week_end]
                )

                # Assign to daily_loads with PHYSIOLOGICAL CONSTRAINTS
                for i, load in enumerate(week_loads):
                    if week_start + i < duration_days:
                        # CONSTRAINT: Cap daily TSS at physiological maximum
                        constrained_load = min(load, MAX_DAILY_TSS)
                        daily_loads[week_start + i] = constrained_load

                        # Log if capping occurred
                        if constrained_load < load:
                            self.logger.warning(f"Day {week_start + i}: Capped TSS from {load:.0f} to {constrained_load:.0f}")

            # Move to next mesocycle (typically 4 weeks = 28 days)
            day_offset += len(weekly_load_factors) * 7

        return daily_loads

    def _create_weekly_load_pattern(
        self,
        weekly_target_load: float,
        week_days: int,
        mesocycle: Mesocycle,
        week_num: int,
        week_rest_days: List[int]
    ) -> np.ndarray:
        """Create intelligent weekly load pattern with POLARIZED intensity distribution.

        Implements true 80/20 polarized model:
        - 80% of training time at easy intensity (Zone 1-2)
        - ~15% at moderate intensity (Zone 3)
        - ~5% at hard intensity (Zone 4-5)

        CRITICAL: Follows the coach's prescription for elite endurance training.
        """
        loads = np.zeros(week_days)

        # Apply rest days first
        for rest_day in week_rest_days:
            if 0 <= rest_day < week_days:
                loads[rest_day] = 0

        # Calculate available training days
        available_days = [i for i in range(week_days) if i not in week_rest_days]

        if not available_days:
            return loads

        # POLARIZED INTENSITY DISTRIBUTION based on sports science
        if mesocycle.cycle_type == MesocycleType.RECOVERY:
            # Recovery week: all easy - maintain polarized approach even in recovery
            for day in available_days:
                loads[day] = weekly_target_load / len(available_days)
        else:
            # POLARIZED MODEL: Schedule intensity days strategically
            # Key principle: Only 1-2 hard days per week maximum

            # Classify days by intensity zone
            hard_days = []      # Zone 4-5 (threshold, VO2max, neuromuscular)
            moderate_days = []  # Zone 3 (tempo, sweet spot)
            easy_days = []      # Zone 1-2 (aerobic, recovery)

            for day in available_days:
                day_of_week = day % 7  # 0=Monday, 1=Tuesday, etc.

                # Hard sessions: Tuesday and Friday (never consecutive)
                if day_of_week in [1, 4] and len(hard_days) < 2:
                    hard_days.append(day)
                # Moderate sessions: Wednesday or Saturday (1 per week max)
                elif day_of_week in [2, 5] and len(moderate_days) == 0 and day not in hard_days:
                    moderate_days.append(day)
                else:
                    easy_days.append(day)

            # Ensure we don't exceed polarized ratios
            total_training_days = len(available_days)
            max_hard_days = max(1, int(total_training_days * 0.2))  # Max 20% hard
            max_moderate_days = max(1, int(total_training_days * 0.15))  # Max 15% moderate

            # Adjust if we have too many intensity days
            while len(hard_days) > max_hard_days:
                moved_day = hard_days.pop()
                easy_days.append(moved_day)

            while len(moderate_days) > max_moderate_days:
                moved_day = moderate_days.pop()
                easy_days.append(moved_day)

            # LOAD DISTRIBUTION following polarized principles
            # Hard days: High quality, lower volume per session
            if hard_days:
                hard_load_per_day = weekly_target_load * 0.25 / len(hard_days)  # 25% on hard days
                for day in hard_days:
                    loads[day] = hard_load_per_day

            # Moderate days: Tempo/threshold work
            if moderate_days:
                moderate_load_per_day = weekly_target_load * 0.20 / len(moderate_days)  # 20% on moderate
                for day in moderate_days:
                    loads[day] = moderate_load_per_day

            # Easy days: Base building volume (majority of training)
            if easy_days:
                easy_load_per_day = weekly_target_load * 0.55 / len(easy_days)  # 55% on easy days
                for day in easy_days:
                    loads[day] = easy_load_per_day

            # Ensure at least one long day on weekend if available
            weekend_days = [d for d in available_days if d % 7 in [5, 6]]  # Sat/Sun
            if weekend_days and week_num < 3:  # Not in recovery week
                # Make one weekend day a long session
                long_day = weekend_days[0]
                if long_day not in hard_days:
                    loads[long_day] = min(loads[long_day] * 1.3, weekly_target_load * 0.3)

        return loads

    def _apply_dynamic_load_scaling(self, base_loads: np.ndarray, strength_days: List[int] = None) -> np.ndarray:
        """Apply dynamic load scaling based on predicted daily readiness.

        REQUIREMENT 2: Implements intelligent daily load adjustments.
        Uses predicted Form (TSB) to modulate daily training loads.
        Higher form = slightly higher loads, lower form = reduced loads.
        """
        # First, predict the form progression with base loads
        t = np.arange(len(base_loads))
        predicted_fitness, predicted_fatigue, predicted_performance = self.ff_model.calculate_fitness_fatigue(
            base_loads, t,
            initial_fitness=self.current_fitness,
            initial_fatigue=self.current_fatigue
        )

        # Calculate form (TSB) for each day
        predicted_form = predicted_fitness - predicted_fatigue

        # Apply readiness modifiers based on predicted form
        adjusted_loads = np.zeros_like(base_loads)

        for day in range(len(base_loads)):
            # Calculate readiness modifier based on predicted form
            # Formula: modifier = 1.0 + (predicted_form / 100)
            # This gives a range roughly 0.5 to 1.5 for typical form values (-50 to +50)
            form_value = predicted_form[day]
            readiness_modifier = 1.0 + (form_value / 100.0)

            # Clamp modifier to reasonable bounds (0.5 to 1.2)
            readiness_modifier = max(0.5, min(1.2, readiness_modifier))

            # Apply modifier to base load
            adjusted_loads[day] = base_loads[day] * readiness_modifier

            # Special handling for rest days and mandatory strength days
            if base_loads[day] == 0:  # Rest day
                adjusted_loads[day] = 0
            elif self._is_strength_day(day, strength_days or []):  # Tuesday strength training
                # Strength training gets fixed load regardless of form
                adjusted_loads[day] = max(40, adjusted_loads[day])  # Minimum effective strength load

        return adjusted_loads

    def _is_strength_day(self, day: int, strength_days: List[int]) -> bool:
        """Check if this day is a configured mandatory strength training day."""
        if not strength_days:
            return False
        return day in strength_days

    def _create_daily_workouts(
        self,
        mesocycle: Mesocycle,
        loads: np.ndarray,
        duration_days: int = 30,
        strength_days: List[int] = None
    ) -> List[WorkoutPlan]:
        """Create detailed daily workout plans with two-a-day support.

        REQUIREMENT 2: Implements dynamic load scaling based on daily readiness.
        """
        workouts = []
        weekly_time_budget_min = config.TRAINING_MAX_WEEKLY_HOURS * 60
        strength_last_day = -10  # Track when strength training was last done

        # REQUIREMENT 2: Dynamic load scaling based on predicted daily readiness
        adjusted_loads = self._apply_dynamic_load_scaling(loads, strength_days)

        for day in range(duration_days):
            date = mesocycle.start_date + timedelta(days=day)
            week_num = day // 7 + 1

            # Determine workout type based on adjusted load and day
            workout_type = self._determine_workout_type(
                adjusted_loads[day], day % 7, week_num
            )

            # Create workout structure with dynamically adjusted load
            workout = self._create_workout_structure(
                date, day, week_num, mesocycle.cycle_type,
                workout_type, adjusted_loads[day], strength_days
            )

            # Add second session logic
            self._add_second_session_if_appropriate(
                workout, day, strength_last_day
            )

            # Track strength training
            if (workout.second_activity_title and
                "Strength" in workout.second_activity_title):
                strength_last_day = day

            workouts.append(workout)

        # Apply weekly time budget constraints
        self._apply_weekly_time_budget(workouts, weekly_time_budget_min)

        return workouts

    def _create_workout_structure(
        self,
        date: datetime,
        day_num: int,
        week_num: int,
        mesocycle_type: MesocycleType,
        workout_type: WorkoutType,
        load: float,
        strength_days: List[int] = None
    ) -> WorkoutPlan:
        """Create detailed workout structure."""

        # Determine sport based on patterns and variety
        primary_sport = self._select_sport(day_num, workout_type, strength_days)
        alternatives = self._get_alternative_sports(primary_sport, workout_type)

        # Apply sport-specific load multipliers
        sport_multiplier = self.load_multipliers.get(primary_sport, 1.0)
        adjusted_load = load * sport_multiplier

        # Calculate duration based on load and workout type
        duration = self._calculate_duration(adjusted_load, workout_type)

        # Create workout structure
        structure = self._build_workout_structure(workout_type, duration, load)

        # Calculate physiological targets
        hr_zones = self._calculate_hr_zones(workout_type)

        # Determine recovery needs
        recovery_hours = self._calculate_recovery_needs(load, workout_type)

        return WorkoutPlan(
            date=date,
            day_number=day_num,
            week_number=week_num,
            mesocycle=mesocycle_type,
            workout_type=workout_type,
            primary_sport=primary_sport,
            alternative_sports=alternatives,
            planned_load=adjusted_load,
            intensity_level=self._get_intensity_level(workout_type),
            hr_zones=hr_zones,
            total_duration_min=duration,
            warmup_min=structure['warmup'],
            main_set_min=structure['main'],
            cooldown_min=structure['cooldown'],
            title=self._generate_workout_title(workout_type, primary_sport),
            description=self._generate_workout_description(
                workout_type, duration, adjusted_load
            ),
            workout_structure=structure,
            primary_system=self._get_primary_system(workout_type),
            secondary_systems=self._get_secondary_systems(workout_type),
            expected_adaptations=self._get_expected_adaptations(workout_type),
            recovery_hours_needed=recovery_hours,
            next_day_recommendation=self._get_next_day_recommendation(
                adjusted_load, workout_type
            ),
            can_skip=(workout_type == WorkoutType.RECOVERY),
            can_move=(workout_type != WorkoutType.LONG),
            priority=self._get_workout_priority(workout_type, week_num)
        )

    def _simulate_training_response(
        self,
        workouts: List[WorkoutPlan]
    ) -> Dict[str, np.ndarray]:
        """Simulate physiological response to training plan."""

        # Extract loads
        loads = np.array([w.planned_load for w in workouts])
        t = np.arange(len(loads))

        # FIXED: Fitness-Fatigue simulation with continuous state
        fitness, fatigue, performance = self.ff_model.calculate_fitness_fatigue(
            loads,
            t,
            initial_fitness=self.current_fitness,
            initial_fatigue=self.current_fatigue
        )

        # PerPot simulation for overtraining risk
        perpot_results = self.perpot_model.simulate(loads, t)

        # Multi-system recovery simulation
        recovery_status = np.zeros((len(loads), 3))  # metabolic, neural, structural

        for i in range(1, len(loads)):
            if loads[i-1] > 0:
                # Calculate recovery for each system
                hours_since = 24
                recovery_status[i, 0] = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.METABOLIC, hours_since, loads[i-1]
                )
                recovery_status[i, 1] = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.NEURAL, hours_since, loads[i-1]
                )
                recovery_status[i, 2] = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.STRUCTURAL, hours_since, loads[i-1]
                )

        # Update workout predictions
        for i, workout in enumerate(workouts):
            workout.predicted_fitness = fitness[i]
            workout.predicted_fatigue = fatigue[i]
            workout.predicted_form = fitness[i] - fatigue[i]
            workout.overtraining_risk = float(perpot_results['overtraining_risk'][i])

        return {
            'fitness': fitness,
            'fatigue': fatigue,
            'performance': performance,
            'form': fitness - fatigue,
            'perpot_performance': perpot_results['performance_potential'],
            'overtraining_risk': perpot_results['overtraining_risk'],
            'recovery_status': recovery_status,
            'overall_recovery': np.min(recovery_status, axis=1) * 100
        }

    def _add_recovery_periods(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray]
    ) -> List[WorkoutPlan]:
        """Add strategic recovery periods based on simulated response."""

        # Identify high-risk periods
        overtraining_days = np.where(response['overtraining_risk'])[0]
        low_recovery_days = np.where(response['overall_recovery'] < 60)[0]

        # Check for sustained overtraining requiring structured recovery mesocycle
        if len(overtraining_days) > 3:
            return self._create_recovery_mesocycle(workouts, overtraining_days)

        # Standard recovery modifications for isolated overtraining days
        for day in overtraining_days:
            if day < len(workouts):
                # Convert to recovery workout
                workouts[day].workout_type = WorkoutType.RECOVERY
                workouts[day].planned_load *= 0.5
                workouts[day].description = "Modified to recovery due to overtraining risk"

        for day in low_recovery_days:
            if day < len(workouts) and workouts[day].workout_type not in [
                WorkoutType.REST, WorkoutType.RECOVERY
            ]:
                # Reduce load
                workouts[day].planned_load *= 0.7
                workouts[day].description += " (reduced load for recovery)"

        return workouts

    def _create_recovery_mesocycle(
        self,
        workouts: List[WorkoutPlan],
        overtraining_days: np.ndarray
    ) -> List[WorkoutPlan]:
        """Create STRUCTURED recovery mesocycle following sports science principles.

        Implements the coach's prescription:
        Week 1: Deep recovery (TSS 20-35) - Complete rest and very light active recovery
        Week 2: Active recovery (TSS 35-50) - Structured active recovery with mobility
        Week 3: Base building (TSS 50-70) - Return to aerobic base with strength
        Week 4: Assessment (TSS 60-80) - Test readiness with moderate sessions
        """

        # Week 1: Deep Recovery - Minimum viable movement
        for i in range(min(7, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                day_in_week = i % 7
                if day_in_week in [0, 3, 6]:  # Monday, Thursday, Sunday rest
                    workouts[i].workout_type = WorkoutType.REST
                    workouts[i].planned_load = 0
                    workouts[i].total_duration_min = 0
                    workouts[i].title = "Complete Rest"
                    workouts[i].description = "Deep recovery - complete rest"
                else:
                    workouts[i].workout_type = WorkoutType.RECOVERY
                    workouts[i].planned_load = 20 + i * 2  # 20-32 TSS
                    workouts[i].total_duration_min = 30 + i * 5  # 30-60 min
                    workouts[i].intensity_level = 'easy'
                    workouts[i].description = f"Deep Recovery Day {i+1} - Light movement only"
                    selected_sport = "Walk" if i % 2 == 0 else self._select_sport(i, WorkoutType.RECOVERY)
                    workouts[i].title = self._generate_workout_title(WorkoutType.RECOVERY, selected_sport)

        # Week 2: Active Recovery - Structured movement with purpose
        for i in range(7, min(14, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                day_in_week = i % 7
                if day_in_week == 0:  # Monday rest
                    workouts[i].workout_type = WorkoutType.REST
                    workouts[i].planned_load = 0
                    workouts[i].title = "Rest Day"
                elif day_in_week in [2, 5]:  # Tuesday, Friday - structured sessions
                    workouts[i].workout_type = WorkoutType.AEROBIC
                    workouts[i].planned_load = 45 + (i-7) * 1  # 45-52 TSS
                    workouts[i].total_duration_min = 60 + (i-7) * 5  # 60-95 min
                    workouts[i].intensity_level = 'easy'
                    workouts[i].description = f"Active Recovery Week - Zone 2 aerobic"
                    selected_sport = self._select_sport(i, WorkoutType.AEROBIC)
                    workouts[i].title = self._generate_workout_title(WorkoutType.AEROBIC, selected_sport)
                else:
                    workouts[i].workout_type = WorkoutType.RECOVERY
                    workouts[i].planned_load = 30 + (i-7) * 1  # 30-37 TSS
                    workouts[i].total_duration_min = 40 + (i-7) * 3
                    workouts[i].description = "Recovery with mobility focus"
                    selected_sport = "Walk" if i % 3 == 0 else self._select_sport(i, WorkoutType.RECOVERY)
                    workouts[i].title = self._generate_workout_title(WorkoutType.RECOVERY, selected_sport)

        # Week 3: Base Building - Return to structure with strength
        for i in range(14, min(21, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                day_in_week = i % 7
                if day_in_week == 0:  # Monday rest
                    workouts[i].workout_type = WorkoutType.REST
                elif day_in_week in [1, 3, 5]:  # Tue, Thu, Sat - key sessions
                    if day_in_week == 1:  # Tuesday - strength focus
                        workouts[i].workout_type = WorkoutType.AEROBIC
                        workouts[i].planned_load = 55 + (i-14) * 2
                        workouts[i].primary_sport = "WeightTraining"
                        workouts[i].description = "Base building with strength focus"
                        workouts[i].title = "Aerobic WeightTraining"
                        # Add second session
                        workouts[i].second_activity_title = "Easy Aerobic"
                        workouts[i].second_activity_duration = 30
                    else:
                        workouts[i].workout_type = WorkoutType.AEROBIC
                        workouts[i].planned_load = 60 + (i-14) * 2  # 60-72 TSS
                        workouts[i].total_duration_min = 75 + (i-14) * 5
                        workouts[i].description = f"Base Building Week - Aerobic foundation"
                        selected_sport = self._select_sport(i, WorkoutType.AEROBIC)
                        workouts[i].title = self._generate_workout_title(WorkoutType.AEROBIC, selected_sport)
                else:
                    workouts[i].workout_type = WorkoutType.RECOVERY
                    workouts[i].planned_load = 40 + (i-14) * 1
                    workouts[i].description = "Active recovery between sessions"

        # Week 4: Assessment - Test readiness for return to training
        for i in range(21, min(28, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                day_in_week = i % 7
                if day_in_week == 0:  # Monday rest
                    workouts[i].workout_type = WorkoutType.REST
                elif day_in_week in [2, 4]:  # Wed, Fri - assessment sessions
                    workouts[i].workout_type = WorkoutType.TEMPO
                    workouts[i].planned_load = 70 + (i-21) * 2  # 70-84 TSS
                    workouts[i].total_duration_min = 80 + (i-21) * 3
                    workouts[i].intensity_level = 'moderate'
                    workouts[i].description = f"Assessment Week - Testing readiness"
                    selected_sport = self._select_sport(i, WorkoutType.TEMPO)
                    workouts[i].title = self._generate_workout_title(WorkoutType.TEMPO, selected_sport)
                else:
                    workouts[i].workout_type = WorkoutType.AEROBIC
                    workouts[i].planned_load = 50 + (i-21) * 2
                    workouts[i].description = "Aerobic maintenance between tests"

        return workouts

    def _create_visualizations(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray],
        duration_days: int = 30
    ) -> Dict[str, Any]:
        """Create visualization data for the training plan."""

        # Prepare daily summary data
        daily_data = []
        for i, workout in enumerate(workouts):
            daily_data.append({
                'date': workout.date.isoformat(),
                'day': i + 1,
                'week': workout.week_number,
                'workout_type': workout.workout_type.value,
                'sport': workout.primary_sport,
                'load': workout.planned_load,
                'duration': workout.total_duration_min,
                'intensity': workout.intensity_level,
                'fitness': response['fitness'][i],
                'fatigue': response['fatigue'][i],
                'form': response['form'][i],
                'recovery': response['overall_recovery'][i] if i < len(response['overall_recovery']) else 100,
                'overtraining_risk': workout.overtraining_risk,
                'title': workout.title,
                'description': workout.description
            })

        # Create weekly summary
        weekly_summary = self._create_weekly_summary(workouts, response, duration_days)

        # Create load distribution chart data
        load_distribution = self._create_load_distribution(workouts)

        # Create intensity distribution
        intensity_distribution = self._create_intensity_distribution(workouts)

        return {
            'daily_data': daily_data,
            'weekly_summary': weekly_summary,
            'load_distribution': load_distribution,
            'intensity_distribution': intensity_distribution,
            'fitness_progression': {
                'dates': [w.date.isoformat() for w in workouts],
                'fitness': response['fitness'].tolist(),
                'fatigue': response['fatigue'].tolist(),
                'form': response['form'].tolist()
            },
            'recovery_status': {
                'dates': [w.date.isoformat() for w in workouts],
                'overall': response['overall_recovery'].tolist(),
                'metabolic': response['recovery_status'][:, 0].tolist(),
                'neural': response['recovery_status'][:, 1].tolist(),
                'structural': response['recovery_status'][:, 2].tolist()
            }
        }

    def generate_30_day_plan(
        self,
        goal: str = "fitness",
        target_event_date: Optional[datetime] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a 30-day training plan (backward compatibility)."""
        return self.generate_training_plan(30, goal, target_event_date, constraints)

    def generate_60_day_plan(
        self,
        goal: str = "fitness",
        target_event_date: Optional[datetime] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a 60-day training plan with extended mesocycle progression."""
        return self.generate_training_plan(60, goal, target_event_date, constraints)

    def _plan_mesocycle_sequence(
        self,
        duration_days: int,
        goal: str,
        target_event_date: Optional[datetime] = None
    ) -> List[object]:
        """Plan sequence of mesocycles for extended training periods.

        For 30-day plans: Single 4-week mesocycle
        For 60-day plans: Two 4-week mesocycles with progression
        For 90-day plans: Three 4-week mesocycles with full periodization
        """
        mesocycles = []
        weeks_needed = (duration_days + 6) // 7  # Round up to nearest week

        # Plan start date
        from datetime import timezone
        plan_start_date = datetime.now(timezone.utc)

        current_date = plan_start_date
        remaining_weeks = weeks_needed

        cycle_count = 0
        while remaining_weeks > 0:
            cycle_count += 1

            # Determine mesocycle type based on position in sequence
            if cycle_count == 1:
                # First cycle: Use intelligent selection
                mesocycle_type = self._intelligent_mesocycle_selection(goal, target_event_date)
            elif remaining_weeks <= 4:
                # Final cycle: Peak or Recovery based on goal
                if goal in ['performance', 'peak']:
                    mesocycle_type = MesocycleType.REALIZATION  # Peak performance
                else:
                    mesocycle_type = MesocycleType.RECOVERY
            else:
                # Middle cycles: Progressive build
                mesocycle_type = MesocycleType.ACCUMULATION  # Build phase

            # Create mesocycle (4 weeks max per cycle)
            cycle_weeks = min(4, remaining_weeks)
            mesocycle = self._create_mesocycle(
                mesocycle_type,
                current_date,
                {'duration_weeks': cycle_weeks}
            )

            mesocycles.append(mesocycle)

            # Move to next cycle
            current_date = current_date + timedelta(weeks=cycle_weeks)
            remaining_weeks -= cycle_weeks

        return mesocycles

    def _create_plan_summary(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray],
        duration_days: int = 30
    ) -> Dict[str, Any]:
        """Create summary statistics for the plan."""

        total_load = sum(w.planned_load for w in workouts)
        total_duration = sum(w.total_duration_min for w in workouts)

        # Count workout types
        workout_counts = {}
        for w in workouts:
            workout_counts[w.workout_type.value] = workout_counts.get(
                w.workout_type.value, 0
            ) + 1

        # Calculate improvements
        fitness_gain = response['fitness'][-1] - response['fitness'][0]
        final_form = response['form'][-1]

        # Identify key workouts
        key_workouts = [
            w for w in workouts
            if w.workout_type in [WorkoutType.THRESHOLD, WorkoutType.VO2MAX, WorkoutType.LONG]
        ]

        return {
            'total_load': total_load,
            'total_duration_hours': total_duration / 60,
            'average_daily_load': total_load / duration_days,
            'average_daily_duration': total_duration / duration_days,
            'workout_distribution': workout_counts,
            'fitness_gain': fitness_gain,
            'final_form': final_form,
            'peak_fatigue': np.max(response['fatigue']),
            'overtraining_days': np.sum(response['overtraining_risk']),
            'key_workouts_count': len(key_workouts),
            'rest_days': workout_counts.get('rest', 0),
            'recovery_days': workout_counts.get('recovery', 0),
            'hard_days': sum(1 for w in workouts if w.intensity_level == 'hard'),
            'easy_days': sum(1 for w in workouts if w.intensity_level == 'easy')
        }

    # Helper methods
    def _analyze_sport_preferences(self, activities: List[Activity]) -> List[str]:
        """Analyze preferred sports from activity history."""
        sport_counts = {}
        for activity in activities:
            sport = activity.type
            sport_counts[sport] = sport_counts.get(sport, 0) + 1

        # Sort by frequency
        sorted_sports = sorted(sport_counts.items(), key=lambda x: x[1], reverse=True)
        return [sport for sport, _ in sorted_sports[:5]]  # Top 5 sports

    def _analyze_duration_patterns(self, activities: List[Activity]) -> float:
        """Analyze typical workout duration."""
        durations = [
            a.moving_time / 60 for a in activities
            if a.moving_time is not None
        ]
        return np.median(durations) if durations else 60.0

    def _analyze_weekly_patterns(self, activities: List[Activity]) -> Dict[int, float]:
        """Analyze weekly training patterns."""
        weekly_loads = {}
        for activity in activities:
            week_day = activity.start_date.weekday()
            load = activity.training_load or 0
            if week_day not in weekly_loads:
                weekly_loads[week_day] = []
            weekly_loads[week_day].append(load)

        # Calculate average load per weekday
        return {
            day: np.mean(loads) if loads else 0
            for day, loads in weekly_loads.items()
        }

    def _get_enabled_sports_from_config(self) -> List[str]:
        """Get enabled sports from configuration."""
        enabled_sports = []

        # Config stores enabled sports as a dictionary
        if hasattr(self.config, 'ENABLED_SPORTS'):
            for sport_name, is_enabled in self.config.ENABLED_SPORTS.items():
                if is_enabled:
                    enabled_sports.append(sport_name)

        return enabled_sports

    def _get_sport_preferences_from_config(self) -> Dict[str, float]:
        """Get sport preferences from configuration."""
        if hasattr(self.config, 'SPORT_PREFERENCES'):
            return dict(self.config.SPORT_PREFERENCES)
        return {}

    def _get_recovery_times_from_config(self) -> Dict[str, float]:
        """Get sport-specific recovery times from configuration."""
        if hasattr(self.config, 'SPORT_RECOVERY_TIMES'):
            return dict(self.config.SPORT_RECOVERY_TIMES)
        return {}

    def _get_load_multipliers_from_config(self) -> Dict[str, float]:
        """Get sport-specific load multipliers from configuration."""
        if hasattr(self.config, 'SPORT_LOAD_MULTIPLIERS'):
            return dict(self.config.SPORT_LOAD_MULTIPLIERS)
        return {}

    def _combine_sport_preferences(self, historical_sports: List[str]) -> List[str]:
        """Combine historical data with configuration preferences."""
        # Start with enabled sports from config
        preferred_sports = []

        # Add sports that are both enabled and have preferences > 0
        for sport in self.enabled_sports:
            if self.sport_preferences.get(sport, 0) > 0:
                preferred_sports.append(sport)

        # If no configured preferences, fall back to historical data
        if not preferred_sports:
            preferred_sports = [s for s in historical_sports if s in self.enabled_sports]

        # If still empty, use all enabled sports
        if not preferred_sports:
            preferred_sports = self.enabled_sports

        return preferred_sports

    def _get_mesocycle_goal(self, cycle_type: MesocycleType) -> str:
        """Get primary goal for mesocycle type."""
        goals = {
            MesocycleType.ADAPTATION: "Build aerobic base and adaptation",
            MesocycleType.ACCUMULATION: "Increase training volume and load",
            MesocycleType.INTENSIFICATION: "Improve power and speed",
            MesocycleType.REALIZATION: "Peak performance and freshness",
            MesocycleType.RECOVERY: "Regeneration and recovery",
            MesocycleType.MAINTENANCE: "Maintain fitness with minimal stress"
        }
        return goals.get(cycle_type, "General fitness improvement")


    def _determine_workout_type(
        self,
        load: float,
        day_of_week: int,
        week_num: int
    ) -> WorkoutType:
        """Determine workout type based on load and schedule with proper intensity variation."""
        if load == 0:
            return WorkoutType.REST
        elif load < 25:
            return WorkoutType.RECOVERY
        elif load < 60:
            return WorkoutType.AEROBIC
        elif load < 85:
            # Medium loads: choose between TEMPO and THRESHOLD for intensity variety
            if day_of_week in [1, 4]:  # Tuesday/Friday - hard days
                return WorkoutType.THRESHOLD
            else:
                return WorkoutType.TEMPO
        elif load < 110:
            # High loads: choose based on day and training focus
            if day_of_week == 5:  # Saturday - long sessions
                return WorkoutType.LONG
            elif day_of_week in [1, 4]:  # Tuesday/Friday - intense intervals
                return WorkoutType.VO2MAX
            else:
                return WorkoutType.THRESHOLD
        else:
            # Very high loads: peak intensity workouts
            if day_of_week == 5:  # Saturday
                return WorkoutType.LONG
            else:
                return WorkoutType.INTERVALS

    def _select_sport(self, day_num: int, workout_type: WorkoutType, strength_days: List[int] = None) -> str:
        """Select appropriate sport for the workout using configuration preferences."""

        # REQUIREMENT 3: Force WeightTraining on configured strength days (e.g., Tuesday)
        if self._is_strength_day(day_num, strength_days or []) and workout_type != WorkoutType.REST:
            return "WeightTraining"

        # Get primary sports from configuration for each workout type
        primary_endurance = next((sport for sport in self.enabled_sports if sport in ["Run", "Ride", "Hike"]), "Run")
        primary_strength = next((sport for sport in self.enabled_sports if sport in ["WeightTraining", "Workout"]), primary_endurance)

        # Default sports for workout types - use configured enabled sports
        default_sports = {
            WorkoutType.REST: "Rest",
            WorkoutType.RECOVERY: "Walk",
            WorkoutType.AEROBIC: primary_endurance,
            WorkoutType.TEMPO: primary_endurance,
            WorkoutType.THRESHOLD: primary_endurance,
            WorkoutType.VO2MAX: primary_endurance,
            WorkoutType.NEUROMUSCULAR: primary_strength,
            WorkoutType.LONG: primary_endurance,
            WorkoutType.INTERVALS: primary_endurance,
            WorkoutType.FARTLEK: primary_endurance
        }

        # Use sport preferences from configuration
        if hasattr(self, 'preferred_sports') and self.preferred_sports and workout_type != WorkoutType.REST:
            # Weighted selection based on preferences
            import random
            random.seed(day_num)  # Deterministic for testing
            weights = [self.sport_preferences.get(sport, 0.1) for sport in self.preferred_sports]
            if sum(weights) > 0:
                selected_sport = random.choices(self.preferred_sports, weights=weights)[0]
                return selected_sport
            else:
                # Rotation if no weights
                sport_index = day_num % len(self.preferred_sports)
                return self.preferred_sports[sport_index]

        # Fallback to default - use first enabled sport instead of hardcoded "Run"
        primary_sport = next(iter(self.enabled_sports), "Run") if self.enabled_sports else "Run"
        default_sport = default_sports.get(workout_type, primary_sport)
        return default_sport

    def _get_alternative_sports(
        self,
        primary: str,
        workout_type: WorkoutType
    ) -> List[str]:
        """Get alternative sports for cross-training based on enabled sports from config."""
        # Get enabled sports from configuration excluding the primary sport
        alternative_options = [sport for sport in self.enabled_sports if sport != primary]

        # Return up to 2 alternatives, prioritizing complementary sports
        if len(alternative_options) >= 2:
            return alternative_options[:2]
        elif len(alternative_options) == 1:
            return alternative_options
        else:
            # Fallback if no alternatives configured
            return []

    def _calculate_duration(self, load: float, workout_type: WorkoutType) -> int:
        """Calculate realistic workout duration from load and type."""
        if load == 0:
            return 0

        # FIXED: More realistic TSS to duration conversion
        # Base conversion: 1 TSS ≈ 1 minute for moderate intensity
        base_duration = load * 1.0

        # FIXED: Use configured duration multipliers from .env
        duration_multipliers = {
            WorkoutType.REST: config.DURATION_MULTIPLIERS["REST"],
            WorkoutType.RECOVERY: config.DURATION_MULTIPLIERS["RECOVERY"],
            WorkoutType.AEROBIC: config.DURATION_MULTIPLIERS["AEROBIC"],
            WorkoutType.TEMPO: config.DURATION_MULTIPLIERS["TEMPO"],
            WorkoutType.THRESHOLD: config.DURATION_MULTIPLIERS["THRESHOLD"],
            WorkoutType.VO2MAX: config.DURATION_MULTIPLIERS["VO2MAX"],
            WorkoutType.NEUROMUSCULAR: config.DURATION_MULTIPLIERS["NEUROMUSCULAR"],
            WorkoutType.LONG: config.DURATION_MULTIPLIERS["LONG"],
            WorkoutType.INTERVALS: config.DURATION_MULTIPLIERS["INTERVALS"],
            WorkoutType.FARTLEK: config.DURATION_MULTIPLIERS["FARTLEK"]
        }

        multiplier = duration_multipliers.get(workout_type, 1.0)
        raw_duration = base_duration * multiplier

        # Round to nearest 5 minutes as requested
        rounded_duration = round(raw_duration / 5) * 5

        # Ensure minimum duration for non-rest workouts
        if workout_type != WorkoutType.REST and rounded_duration < 20:
            rounded_duration = 20

        return int(rounded_duration)

    def _build_workout_structure(
        self,
        workout_type: WorkoutType,
        duration: int,
        load: float
    ) -> Dict[str, Any]:
        """Build detailed workout structure."""
        structures = {
            WorkoutType.INTERVALS: {
                'warmup': 15,
                'main': duration - 25,
                'cooldown': 10,
                'sets': [
                    {'duration': 4, 'intensity': 'hard', 'recovery': 2}
                    for _ in range(int((duration - 25) / 6))
                ]
            },
            WorkoutType.TEMPO: {
                'warmup': 10,
                'main': duration - 20,
                'cooldown': 10,
                'sets': [{'duration': duration - 20, 'intensity': 'moderate'}]
            },
            WorkoutType.LONG: {
                'warmup': 10,
                'main': duration - 15,
                'cooldown': 5,
                'sets': [{'duration': duration - 15, 'intensity': 'easy'}]
            }
        }

        # Default structure
        default = {
            'warmup': min(10, duration // 5),
            'main': duration - min(15, duration // 3),
            'cooldown': min(5, duration // 6),
            'sets': []
        }

        return structures.get(workout_type, default)

    def _calculate_hr_zones(self, workout_type: WorkoutType) -> Dict[str, float]:
        """Calculate time in heart rate zones."""
        zone_distributions = {
            WorkoutType.REST: {'z1': 100, 'z2': 0, 'z3': 0, 'z4': 0, 'z5': 0},
            WorkoutType.RECOVERY: {'z1': 100, 'z2': 0, 'z3': 0, 'z4': 0, 'z5': 0},
            WorkoutType.AEROBIC: {'z1': 20, 'z2': 70, 'z3': 10, 'z4': 0, 'z5': 0},
            WorkoutType.TEMPO: {'z1': 10, 'z2': 30, 'z3': 50, 'z4': 10, 'z5': 0},
            WorkoutType.THRESHOLD: {'z1': 10, 'z2': 20, 'z3': 40, 'z4': 30, 'z5': 0},
            WorkoutType.VO2MAX: {'z1': 10, 'z2': 10, 'z3': 20, 'z4': 40, 'z5': 20},
            WorkoutType.INTERVALS: {'z1': 15, 'z2': 15, 'z3': 20, 'z4': 30, 'z5': 20},
            WorkoutType.LONG: {'z1': 10, 'z2': 60, 'z3': 25, 'z4': 5, 'z5': 0}
        }

        return zone_distributions.get(
            workout_type,
            {'z1': 20, 'z2': 50, 'z3': 20, 'z4': 10, 'z5': 0}
        )

    def _calculate_recovery_needs(
        self,
        load: float,
        workout_type: WorkoutType
    ) -> int:
        """Calculate recovery hours needed after workout."""
        base_recovery = load * 0.3  # Base on load

        # FIXED: Use configured recovery multipliers from .env
        recovery_multipliers = {
            WorkoutType.REST: config.RECOVERY_MULTIPLIERS["REST"],
            WorkoutType.RECOVERY: config.RECOVERY_MULTIPLIERS["RECOVERY"],
            WorkoutType.AEROBIC: config.RECOVERY_MULTIPLIERS["AEROBIC"],
            WorkoutType.TEMPO: config.RECOVERY_MULTIPLIERS["TEMPO"],
            WorkoutType.THRESHOLD: config.RECOVERY_MULTIPLIERS["THRESHOLD"],
            WorkoutType.VO2MAX: config.RECOVERY_MULTIPLIERS["VO2MAX"],
            WorkoutType.NEUROMUSCULAR: config.RECOVERY_MULTIPLIERS["NEUROMUSCULAR"],
            WorkoutType.LONG: config.RECOVERY_MULTIPLIERS["LONG"],
            WorkoutType.INTERVALS: config.RECOVERY_MULTIPLIERS["INTERVALS"]
        }

        multiplier = recovery_multipliers.get(workout_type, 1.0)
        return int(base_recovery * multiplier)

    def _get_intensity_level(self, workout_type: WorkoutType) -> str:
        """Get intensity level description."""
        intensity_map = {
            WorkoutType.REST: "rest",
            WorkoutType.RECOVERY: "easy",
            WorkoutType.AEROBIC: "easy",
            WorkoutType.TEMPO: "moderate",
            WorkoutType.THRESHOLD: "hard",
            WorkoutType.VO2MAX: "peak",
            WorkoutType.NEUROMUSCULAR: "peak",
            WorkoutType.LONG: "moderate",
            WorkoutType.INTERVALS: "hard",
            WorkoutType.FARTLEK: "variable"
        }
        return intensity_map.get(workout_type, "moderate")

    def _generate_workout_title(
        self,
        workout_type: WorkoutType,
        sport: str
    ) -> str:
        """Generate descriptive workout title."""
        titles = {
            WorkoutType.REST: "Rest Day",
            WorkoutType.RECOVERY: f"Recovery {sport}",
            WorkoutType.AEROBIC: f"Aerobic {sport}",
            WorkoutType.TEMPO: f"Tempo {sport}",
            WorkoutType.THRESHOLD: f"Threshold {sport}",
            WorkoutType.VO2MAX: f"VO2max {sport}",
            WorkoutType.LONG: f"Long {sport}",
            WorkoutType.INTERVALS: f"{sport} Intervals",
            WorkoutType.FARTLEK: f"Fartlek {sport}"
        }
        return titles.get(workout_type, f"{sport} Workout")

    def _generate_workout_description(
        self,
        workout_type: WorkoutType,
        duration: int,
        load: float
    ) -> str:
        """Generate detailed workout description."""
        descriptions = {
            WorkoutType.INTERVALS: f"High-intensity intervals for {duration} min. "
                                  f"Focus on form and power. TSS: {load:.0f}",
            WorkoutType.TEMPO: f"Steady tempo effort for {duration} min. "
                              f"Stay at threshold pace. TSS: {load:.0f}",
            WorkoutType.LONG: f"Long endurance session for {duration} min. "
                             f"Build aerobic base. TSS: {load:.0f}",
            WorkoutType.RECOVERY: f"Easy recovery for {duration} min. "
                                 f"Keep heart rate low. TSS: {load:.0f}"
        }

        default = f"{workout_type.value.capitalize()} workout for {duration} min. TSS: {load:.0f}"
        return descriptions.get(workout_type, default)

    def _get_primary_system(self, workout_type: WorkoutType) -> str:
        """Get primary energy system targeted."""
        systems = {
            WorkoutType.REST: "recovery",
            WorkoutType.RECOVERY: "recovery",
            WorkoutType.AEROBIC: "aerobic",
            WorkoutType.TEMPO: "aerobic",
            WorkoutType.THRESHOLD: "anaerobic",
            WorkoutType.VO2MAX: "anaerobic",
            WorkoutType.NEUROMUSCULAR: "neuromuscular",
            WorkoutType.LONG: "aerobic",
            WorkoutType.INTERVALS: "anaerobic"
        }
        return systems.get(workout_type, "aerobic")

    def _get_secondary_systems(self, workout_type: WorkoutType) -> List[str]:
        """Get secondary systems engaged."""
        secondary = {
            WorkoutType.TEMPO: ["anaerobic"],
            WorkoutType.THRESHOLD: ["aerobic", "neuromuscular"],
            WorkoutType.VO2MAX: ["aerobic", "neuromuscular"],
            WorkoutType.INTERVALS: ["aerobic", "neuromuscular"],
            WorkoutType.LONG: ["fat_metabolism"]
        }
        return secondary.get(workout_type, [])

    def _get_expected_adaptations(self, workout_type: WorkoutType) -> List[str]:
        """Get expected physiological adaptations."""
        adaptations = {
            WorkoutType.AEROBIC: ["mitochondrial_density", "capillarization"],
            WorkoutType.TEMPO: ["lactate_threshold", "glycogen_storage"],
            WorkoutType.THRESHOLD: ["lactate_buffering", "vo2max"],
            WorkoutType.VO2MAX: ["vo2max", "cardiac_output"],
            WorkoutType.LONG: ["fat_oxidation", "glycogen_storage"],
            WorkoutType.INTERVALS: ["neuromuscular_power", "lactate_tolerance"]
        }
        return adaptations.get(workout_type, ["general_fitness"])

    def _get_next_day_recommendation(
        self,
        load: float,
        workout_type: WorkoutType
    ) -> str:
        """Get recommendation for next day's training."""
        if workout_type in [WorkoutType.VO2MAX, WorkoutType.INTERVALS]:
            return "Easy recovery or rest recommended"
        elif workout_type == WorkoutType.LONG:
            return "Recovery or easy aerobic recommended"
        elif load > 100:
            return "Moderate or easy day recommended"
        else:
            return "Normal training can continue"

    def _get_workout_priority(
        self,
        workout_type: WorkoutType,
        week_num: int
    ) -> int:
        """Get workout priority (1-10)."""
        priorities = {
            WorkoutType.REST: 3,
            WorkoutType.RECOVERY: 4,
            WorkoutType.AEROBIC: 5,
            WorkoutType.TEMPO: 7,
            WorkoutType.THRESHOLD: 8,
            WorkoutType.VO2MAX: 9,
            WorkoutType.LONG: 8,
            WorkoutType.INTERVALS: 9
        }

        # Increase priority in peak weeks
        base_priority = priorities.get(workout_type, 5)
        if week_num == 3:  # Peak week
            base_priority += 1

        return min(10, base_priority)

    def _create_weekly_summary(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray],
        duration_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Create weekly summary statistics."""
        weekly_summaries = []

        for week in range((duration_days + 6) // 7):  # Calculate weeks needed to cover all days
            week_start = week * 7
            week_end = min(week_start + 7, len(workouts))
            week_workouts = workouts[week_start:week_end]

            if week_workouts:
                weekly_summaries.append({
                    'week': week + 1,
                    'total_load': sum(w.planned_load for w in week_workouts),
                    'total_duration': sum(w.total_duration_min for w in week_workouts),
                    'hard_days': sum(1 for w in week_workouts if w.intensity_level == 'hard'),
                    'easy_days': sum(1 for w in week_workouts if w.intensity_level == 'easy'),
                    'avg_fitness': np.mean(response['fitness'][week_start:week_end]),
                    'avg_fatigue': np.mean(response['fatigue'][week_start:week_end]),
                    'avg_form': np.mean(response['form'][week_start:week_end])
                })

        return weekly_summaries

    def _create_load_distribution(
        self,
        workouts: List[WorkoutPlan]
    ) -> Dict[str, List[float]]:
        """Create load distribution data."""
        return {
            'dates': [w.date.isoformat() for w in workouts],
            'loads': [w.planned_load for w in workouts]
        }

    def _create_intensity_distribution(
        self,
        workouts: List[WorkoutPlan]
    ) -> Dict[str, int]:
        """Create intensity distribution summary."""
        distribution = {'easy': 0, 'moderate': 0, 'hard': 0, 'peak': 0, 'rest': 0}

        for workout in workouts:
            distribution[workout.intensity_level] = distribution.get(
                workout.intensity_level, 0
            ) + 1

        return distribution

    def _add_second_session_if_appropriate(
        self,
        workout: WorkoutPlan,
        day: int,
        strength_last_day: int
    ):
        """Add second session based on sports science principles."""
        # No second session on REST or RECOVERY days
        if workout.workout_type in [WorkoutType.REST, WorkoutType.RECOVERY]:
            return

        # Check if strength training is needed (gap of 4+ days)
        strength_gap = day - strength_last_day
        needs_strength = strength_gap >= 4

        # Science-based intensity pairing logic
        if workout.intensity_level in ["easy", "moderate"]:
            # On EASY/MODERATE days, add strength training if needed
            if needs_strength:
                workout.second_activity_title = "Strength Training"
                workout.second_activity_duration = 45  # Already a multiple of 5
                workout.second_activity_rationale = f"Strength training needed ({strength_gap} days ago)"
            elif day % 7 == 0:  # Weekly mobility session (less frequent)
                workout.second_activity_title = "Mobility/Flexibility"
                workout.second_activity_duration = 25  # Already a multiple of 5
                workout.second_activity_rationale = "Weekly maintenance mobility session"

        elif workout.intensity_level in ["hard", "peak"]:
            # On HARD/PEAK days, only add recovery if truly high intensity
            if workout.workout_type in [WorkoutType.THRESHOLD, WorkoutType.VO2MAX, WorkoutType.INTERVALS]:
                workout.second_activity_title = "Recovery/Stretching"
                workout.second_activity_duration = 20  # Already a multiple of 5
                workout.second_activity_rationale = "Active recovery after high intensity"
            # TEMPO and LONG workouts don't need mandatory recovery sessions

    def _apply_weekly_time_budget(
        self,
        workouts: List[WorkoutPlan],
        weekly_budget_min: float
    ):
        """Apply weekly time budget constraints and scale workouts if needed."""
        weeks = {}

        # Group workouts by week
        for workout in workouts:
            week_num = workout.week_number
            if week_num not in weeks:
                weeks[week_num] = []
            weeks[week_num].append(workout)

        # Check and adjust each week
        for week_num, week_workouts in weeks.items():
            # Calculate total weekly time
            total_time = sum(
                w.total_duration_min + (w.second_activity_duration or 0)
                for w in week_workouts
            )

            # Scale down if over budget
            if total_time > weekly_budget_min:
                scale_factor = weekly_budget_min / total_time

                for workout in week_workouts:
                    # Scale primary workout duration
                    workout.total_duration_min = int(workout.total_duration_min * scale_factor)
                    workout.main_set_min = int(workout.main_set_min * scale_factor)
                    workout.warmup_min = int(workout.warmup_min * scale_factor)
                    workout.cooldown_min = int(workout.cooldown_min * scale_factor)

                    # Scale second session duration
                    if workout.second_activity_duration:
                        workout.second_activity_duration = int(
                            workout.second_activity_duration * scale_factor
                        )

                    # Update rationale to mention scaling
                    if workout.second_activity_rationale:
                        workout.second_activity_rationale += f" (scaled {scale_factor:.1%})"