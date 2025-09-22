"""Training periodization and plan generation module."""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric
from .sports_metrics import SportsMetricsCalculator, RecoveryMetrics


class PeriodizationType(Enum):
    """Types of periodization models."""

    LINEAR = "linear"  # Traditional linear progression
    UNDULATING = "undulating"  # Daily variation in intensity
    BLOCK = "block"  # Focused blocks of specific training
    POLARIZED = "polarized"  # 80% easy, 20% hard
    PYRAMIDAL = "pyramidal"  # High volume base, moderate threshold, low high-intensity


class TrainingPhase(Enum):
    """Training phases in periodized plan."""

    BASE = "base"  # Aerobic base building
    BUILD = "build"  # Increasing intensity
    PEAK = "peak"  # Race preparation
    TAPER = "taper"  # Pre-competition taper
    RECOVERY = "recovery"  # Post-competition recovery
    MAINTENANCE = "maintenance"  # Off-season maintenance


@dataclass
class TrainingWeek:
    """Represents a week of training."""

    week_number: int
    phase: TrainingPhase
    total_load: float
    intensity_distribution: Dict[str, float]  # easy, moderate, hard percentages
    key_workouts: List[Dict]
    notes: str


@dataclass
class TrainingDay:
    """Represents a single training day."""

    date: datetime
    workout_type: str  # rest, easy, tempo, intervals, long, race
    planned_load: float
    duration_minutes: int
    intensity: str  # recovery, aerobic, threshold, vo2max
    description: str
    warmup: Optional[str] = None
    main_set: Optional[str] = None
    cooldown: Optional[str] = None


class PeriodizationPlanner:
    """Generate periodized training plans based on athlete goals and current fitness."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.metrics_calc = SportsMetricsCalculator()
        self.recovery = RecoveryMetrics()

    def create_training_plan(
        self,
        goal_date: datetime,
        goal_type: str,  # "5k", "10k", "half_marathon", "marathon", "century", "sprint_tri", etc.
        current_fitness: float,
        periodization_type: PeriodizationType = PeriodizationType.LINEAR,
        weeks_available: Optional[int] = None
    ) -> List[TrainingWeek]:
        """Create a periodized training plan.

        Args:
            goal_date: Target race/event date
            goal_type: Type of goal event
            current_fitness: Current CTL/fitness level
            periodization_type: Type of periodization to use
            weeks_available: Number of weeks until goal (calculated if not provided)

        Returns:
            List of TrainingWeek objects
        """
        if weeks_available is None:
            weeks_available = (goal_date - datetime.now()).days // 7

        # Get goal-specific parameters
        goal_params = self._get_goal_parameters(goal_type)

        # Calculate training phases
        phases = self._calculate_phases(weeks_available, goal_params)

        # Generate weekly plan
        training_plan = []
        current_week_fitness = current_fitness

        for week_num in range(weeks_available):
            phase = self._get_phase_for_week(week_num, phases)

            # Calculate weekly parameters based on periodization type
            if periodization_type == PeriodizationType.LINEAR:
                week_plan = self._create_linear_week(
                    week_num, phase, current_week_fitness, goal_params
                )
            elif periodization_type == PeriodizationType.UNDULATING:
                week_plan = self._create_undulating_week(
                    week_num, phase, current_week_fitness, goal_params
                )
            elif periodization_type == PeriodizationType.BLOCK:
                week_plan = self._create_block_week(
                    week_num, phase, current_week_fitness, goal_params
                )
            elif periodization_type == PeriodizationType.POLARIZED:
                week_plan = self._create_polarized_week(
                    week_num, phase, current_week_fitness, goal_params
                )
            else:  # PYRAMIDAL
                week_plan = self._create_pyramidal_week(
                    week_num, phase, current_week_fitness, goal_params
                )

            training_plan.append(week_plan)

            # Update fitness estimate
            current_week_fitness = self._estimate_fitness_progression(
                current_week_fitness, week_plan.total_load
            )

        return training_plan

    def _get_goal_parameters(self, goal_type: str) -> Dict:
        """Get training parameters based on goal type."""
        # Define typical training parameters for different goals
        goal_configs = {
            "5k": {
                "peak_weekly_load": 400,
                "peak_long_run_pct": 0.25,
                "speed_work_pct": 0.20,
                "tempo_pct": 0.15,
                "base_phase_weeks": 4,
                "build_phase_weeks": 6,
                "peak_phase_weeks": 2,
                "taper_weeks": 1,
            },
            "10k": {
                "peak_weekly_load": 500,
                "peak_long_run_pct": 0.30,
                "speed_work_pct": 0.15,
                "tempo_pct": 0.20,
                "base_phase_weeks": 4,
                "build_phase_weeks": 8,
                "peak_phase_weeks": 3,
                "taper_weeks": 1,
            },
            "half_marathon": {
                "peak_weekly_load": 600,
                "peak_long_run_pct": 0.35,
                "speed_work_pct": 0.10,
                "tempo_pct": 0.25,
                "base_phase_weeks": 6,
                "build_phase_weeks": 8,
                "peak_phase_weeks": 3,
                "taper_weeks": 2,
            },
            "marathon": {
                "peak_weekly_load": 800,
                "peak_long_run_pct": 0.35,
                "speed_work_pct": 0.05,
                "tempo_pct": 0.20,
                "base_phase_weeks": 8,
                "build_phase_weeks": 10,
                "peak_phase_weeks": 4,
                "taper_weeks": 3,
            },
            "century": {  # 100-mile bike ride
                "peak_weekly_load": 1000,
                "peak_long_ride_pct": 0.40,
                "speed_work_pct": 0.10,
                "tempo_pct": 0.20,
                "base_phase_weeks": 8,
                "build_phase_weeks": 10,
                "peak_phase_weeks": 3,
                "taper_weeks": 2,
            },
            "sprint_tri": {
                "peak_weekly_load": 500,
                "peak_long_workout_pct": 0.25,
                "speed_work_pct": 0.20,
                "tempo_pct": 0.20,
                "base_phase_weeks": 4,
                "build_phase_weeks": 6,
                "peak_phase_weeks": 2,
                "taper_weeks": 1,
            },
            "olympic_tri": {
                "peak_weekly_load": 700,
                "peak_long_workout_pct": 0.30,
                "speed_work_pct": 0.15,
                "tempo_pct": 0.25,
                "base_phase_weeks": 6,
                "build_phase_weeks": 8,
                "peak_phase_weeks": 3,
                "taper_weeks": 2,
            },
        }

        return goal_configs.get(goal_type, goal_configs["10k"])  # Default to 10k

    def _calculate_phases(self, total_weeks: int, goal_params: Dict) -> Dict[TrainingPhase, Tuple[int, int]]:
        """Calculate training phase distribution."""
        phases = {}
        current_week = 0

        # Calculate phase durations based on available time
        if total_weeks >= 16:
            # Full plan
            base_weeks = min(goal_params["base_phase_weeks"], total_weeks // 4)
            build_weeks = min(goal_params["build_phase_weeks"], total_weeks // 3)
            peak_weeks = min(goal_params["peak_phase_weeks"], total_weeks // 6)
            taper_weeks = min(goal_params["taper_weeks"], 2)
        else:
            # Shortened plan
            base_weeks = max(2, total_weeks // 4)
            build_weeks = max(3, total_weeks // 2)
            peak_weeks = max(1, total_weeks // 8)
            taper_weeks = 1

        phases[TrainingPhase.BASE] = (current_week, current_week + base_weeks)
        current_week += base_weeks

        phases[TrainingPhase.BUILD] = (current_week, current_week + build_weeks)
        current_week += build_weeks

        phases[TrainingPhase.PEAK] = (current_week, current_week + peak_weeks)
        current_week += peak_weeks

        phases[TrainingPhase.TAPER] = (current_week, current_week + taper_weeks)

        return phases

    def _get_phase_for_week(self, week_num: int, phases: Dict) -> TrainingPhase:
        """Determine which phase a given week belongs to."""
        for phase, (start, end) in phases.items():
            if start <= week_num < end:
                return phase
        return TrainingPhase.MAINTENANCE

    def _create_linear_week(
        self, week_num: int, phase: TrainingPhase, fitness: float, params: Dict
    ) -> TrainingWeek:
        """Create a week following linear periodization."""
        # Base load calculation
        if phase == TrainingPhase.BASE:
            load_multiplier = 0.6 + (week_num * 0.02)
            intensity_dist = {"easy": 80, "moderate": 15, "hard": 5}
        elif phase == TrainingPhase.BUILD:
            load_multiplier = 0.8 + (week_num * 0.015)
            intensity_dist = {"easy": 70, "moderate": 20, "hard": 10}
        elif phase == TrainingPhase.PEAK:
            load_multiplier = 1.0
            intensity_dist = {"easy": 60, "moderate": 25, "hard": 15}
        elif phase == TrainingPhase.TAPER:
            load_multiplier = 0.6 - (week_num * 0.05)
            intensity_dist = {"easy": 70, "moderate": 20, "hard": 10}
        else:
            load_multiplier = 0.5
            intensity_dist = {"easy": 85, "moderate": 10, "hard": 5}

        total_load = params["peak_weekly_load"] * load_multiplier

        # Create key workouts based on phase
        key_workouts = self._generate_key_workouts(phase, params)

        return TrainingWeek(
            week_number=week_num + 1,
            phase=phase,
            total_load=total_load,
            intensity_distribution=intensity_dist,
            key_workouts=key_workouts,
            notes=f"{phase.value.capitalize()} phase - Week {week_num + 1}"
        )

    def _create_undulating_week(
        self, week_num: int, phase: TrainingPhase, fitness: float, params: Dict
    ) -> TrainingWeek:
        """Create a week following undulating periodization (daily variation)."""
        # Base load varies within the week
        base_load = params["peak_weekly_load"] * 0.7

        # Create high/low/medium pattern
        if week_num % 3 == 0:
            total_load = base_load * 1.2  # High week
            intensity_dist = {"easy": 60, "moderate": 25, "hard": 15}
        elif week_num % 3 == 1:
            total_load = base_load * 0.8  # Low week
            intensity_dist = {"easy": 85, "moderate": 10, "hard": 5}
        else:
            total_load = base_load  # Medium week
            intensity_dist = {"easy": 70, "moderate": 20, "hard": 10}

        key_workouts = self._generate_key_workouts(phase, params)

        return TrainingWeek(
            week_number=week_num + 1,
            phase=phase,
            total_load=total_load,
            intensity_distribution=intensity_dist,
            key_workouts=key_workouts,
            notes=f"Undulating pattern - {phase.value}"
        )

    def _create_block_week(
        self, week_num: int, phase: TrainingPhase, fitness: float, params: Dict
    ) -> TrainingWeek:
        """Create a week following block periodization."""
        # Focused blocks with specific adaptations
        block_type = (week_num // 3) % 3  # 3-week blocks

        if block_type == 0:  # Endurance block
            total_load = params["peak_weekly_load"] * 0.8
            intensity_dist = {"easy": 90, "moderate": 8, "hard": 2}
            focus = "Endurance block"
        elif block_type == 1:  # Threshold block
            total_load = params["peak_weekly_load"] * 0.7
            intensity_dist = {"easy": 60, "moderate": 35, "hard": 5}
            focus = "Threshold block"
        else:  # VO2max/Speed block
            total_load = params["peak_weekly_load"] * 0.6
            intensity_dist = {"easy": 65, "moderate": 15, "hard": 20}
            focus = "Speed block"

        key_workouts = self._generate_key_workouts(phase, params)

        return TrainingWeek(
            week_number=week_num + 1,
            phase=phase,
            total_load=total_load,
            intensity_distribution=intensity_dist,
            key_workouts=key_workouts,
            notes=focus
        )

    def _create_polarized_week(
        self, week_num: int, phase: TrainingPhase, fitness: float, params: Dict
    ) -> TrainingWeek:
        """Create a week following polarized training (80/20 rule)."""
        # Strict 80% easy, 20% hard distribution
        total_load = params["peak_weekly_load"] * (0.6 + week_num * 0.02)

        if phase in [TrainingPhase.BASE, TrainingPhase.RECOVERY]:
            intensity_dist = {"easy": 90, "moderate": 5, "hard": 5}
        elif phase == TrainingPhase.TAPER:
            intensity_dist = {"easy": 75, "moderate": 10, "hard": 15}
            total_load *= 0.6
        else:
            intensity_dist = {"easy": 80, "moderate": 5, "hard": 15}

        key_workouts = self._generate_key_workouts(phase, params)

        return TrainingWeek(
            week_number=week_num + 1,
            phase=phase,
            total_load=total_load,
            intensity_distribution=intensity_dist,
            key_workouts=key_workouts,
            notes="Polarized training approach"
        )

    def _create_pyramidal_week(
        self, week_num: int, phase: TrainingPhase, fitness: float, params: Dict
    ) -> TrainingWeek:
        """Create a week following pyramidal distribution."""
        # Pyramidal: ~70% easy, ~20% threshold, ~10% VO2max
        total_load = params["peak_weekly_load"] * (0.5 + week_num * 0.025)

        if phase == TrainingPhase.BASE:
            intensity_dist = {"easy": 80, "moderate": 15, "hard": 5}
        elif phase == TrainingPhase.BUILD:
            intensity_dist = {"easy": 70, "moderate": 20, "hard": 10}
        elif phase == TrainingPhase.PEAK:
            intensity_dist = {"easy": 65, "moderate": 23, "hard": 12}
        elif phase == TrainingPhase.TAPER:
            intensity_dist = {"easy": 75, "moderate": 15, "hard": 10}
            total_load *= 0.5
        else:
            intensity_dist = {"easy": 85, "moderate": 10, "hard": 5}

        key_workouts = self._generate_key_workouts(phase, params)

        return TrainingWeek(
            week_number=week_num + 1,
            phase=phase,
            total_load=total_load,
            intensity_distribution=intensity_dist,
            key_workouts=key_workouts,
            notes="Pyramidal intensity distribution"
        )

    def _generate_key_workouts(self, phase: TrainingPhase, params: Dict) -> List[Dict]:
        """Generate key workouts for the week based on phase."""
        workouts = []

        if phase == TrainingPhase.BASE:
            workouts.append({
                "type": "long_run",
                "description": "Long easy run/ride",
                "intensity": "aerobic",
                "duration_pct": params.get("peak_long_run_pct", 0.30)
            })
            workouts.append({
                "type": "tempo",
                "description": "Steady state tempo",
                "intensity": "threshold",
                "duration_pct": 0.15
            })

        elif phase == TrainingPhase.BUILD:
            workouts.append({
                "type": "long_run",
                "description": "Progressive long run",
                "intensity": "aerobic_to_threshold",
                "duration_pct": params.get("peak_long_run_pct", 0.30)
            })
            workouts.append({
                "type": "intervals",
                "description": "Threshold intervals",
                "intensity": "threshold",
                "duration_pct": 0.20
            })
            workouts.append({
                "type": "speed",
                "description": "Speed development",
                "intensity": "vo2max",
                "duration_pct": 0.10
            })

        elif phase == TrainingPhase.PEAK:
            workouts.append({
                "type": "race_pace",
                "description": "Race pace simulation",
                "intensity": "race_pace",
                "duration_pct": 0.25
            })
            workouts.append({
                "type": "intervals",
                "description": "VO2max intervals",
                "intensity": "vo2max",
                "duration_pct": 0.15
            })
            workouts.append({
                "type": "tempo",
                "description": "Tempo maintenance",
                "intensity": "threshold",
                "duration_pct": 0.15
            })

        elif phase == TrainingPhase.TAPER:
            workouts.append({
                "type": "race_pace",
                "description": "Race pace touch",
                "intensity": "race_pace",
                "duration_pct": 0.10
            })
            workouts.append({
                "type": "sharpener",
                "description": "Pre-race sharpener",
                "intensity": "mixed",
                "duration_pct": 0.10
            })

        else:  # RECOVERY or MAINTENANCE
            workouts.append({
                "type": "easy",
                "description": "Recovery run/ride",
                "intensity": "recovery",
                "duration_pct": 0.20
            })

        return workouts

    def _estimate_fitness_progression(self, current_fitness: float, weekly_load: float) -> float:
        """Estimate fitness change based on weekly training load."""
        # Simple model: fitness increases with consistent load
        # But also accounts for recovery needs
        fitness_gain = (weekly_load * 0.01) - (current_fitness * 0.02)  # Natural decay
        return max(0, current_fitness + fitness_gain)

    def generate_daily_plan(self, training_week: TrainingWeek, start_date: datetime) -> List[TrainingDay]:
        """Generate daily workouts for a training week.

        Args:
            training_week: TrainingWeek object with weekly parameters
            start_date: Starting date for the week

        Returns:
            List of 7 TrainingDay objects
        """
        daily_plan = []
        daily_loads = self._distribute_weekly_load(training_week)

        for day_num in range(7):
            day_date = start_date + timedelta(days=day_num)

            # Determine workout type based on day and key workouts
            if day_num == 0:  # Monday - Easy recovery
                workout = self._create_easy_day(daily_loads[day_num])
            elif day_num == 1:  # Tuesday - Quality
                workout = self._create_quality_day(daily_loads[day_num], training_week)
            elif day_num == 2:  # Wednesday - Easy/moderate
                workout = self._create_moderate_day(daily_loads[day_num])
            elif day_num == 3:  # Thursday - Quality or rest
                if training_week.phase == TrainingPhase.TAPER:
                    workout = self._create_rest_day()
                else:
                    workout = self._create_quality_day(daily_loads[day_num], training_week)
            elif day_num == 4:  # Friday - Easy or rest
                workout = self._create_easy_day(daily_loads[day_num])
            elif day_num == 5:  # Saturday - Long workout
                workout = self._create_long_day(daily_loads[day_num], training_week)
            else:  # Sunday - Recovery or rest
                if training_week.total_load > 500:
                    workout = self._create_recovery_day(daily_loads[day_num])
                else:
                    workout = self._create_rest_day()

            workout.date = day_date
            daily_plan.append(workout)

        return daily_plan

    def _distribute_weekly_load(self, training_week: TrainingWeek) -> List[float]:
        """Distribute weekly load across 7 days."""
        total_load = training_week.total_load

        # Standard distribution pattern (can be customized)
        if training_week.phase == TrainingPhase.TAPER:
            # Taper: Front-load the week
            distribution = [0.20, 0.25, 0.15, 0.10, 0.10, 0.15, 0.05]
        elif training_week.phase == TrainingPhase.PEAK:
            # Peak: Balanced with key sessions
            distribution = [0.10, 0.20, 0.10, 0.20, 0.05, 0.30, 0.05]
        else:
            # Standard distribution
            distribution = [0.10, 0.15, 0.10, 0.15, 0.10, 0.35, 0.05]

        return [total_load * pct for pct in distribution]

    def _create_easy_day(self, load: float) -> TrainingDay:
        """Create an easy training day."""
        duration = int(load * 0.6)  # Rough conversion from load to minutes
        return TrainingDay(
            date=datetime.now(),  # Will be overwritten
            workout_type="easy",
            planned_load=load,
            duration_minutes=duration,
            intensity="aerobic",
            description="Easy aerobic training",
            warmup="10 min easy warm-up",
            main_set=f"{duration-20} min at easy aerobic pace",
            cooldown="10 min cool-down"
        )

    def _create_quality_day(self, load: float, week: TrainingWeek) -> TrainingDay:
        """Create a quality/interval training day."""
        duration = int(load * 0.5)

        # Select workout based on phase
        if week.phase == TrainingPhase.BUILD:
            return TrainingDay(
                date=datetime.now(),
                workout_type="intervals",
                planned_load=load,
                duration_minutes=duration,
                intensity="threshold",
                description="Threshold intervals",
                warmup="15 min progressive warm-up",
                main_set="4x8 min at threshold pace, 2 min recovery",
                cooldown="10 min easy cool-down"
            )
        elif week.phase == TrainingPhase.PEAK:
            return TrainingDay(
                date=datetime.now(),
                workout_type="intervals",
                planned_load=load,
                duration_minutes=duration,
                intensity="vo2max",
                description="VO2max intervals",
                warmup="20 min warm-up with strides",
                main_set="5x3 min at VO2max pace, 3 min recovery",
                cooldown="15 min easy cool-down"
            )
        else:
            return TrainingDay(
                date=datetime.now(),
                workout_type="tempo",
                planned_load=load,
                duration_minutes=duration,
                intensity="threshold",
                description="Tempo run",
                warmup="15 min warm-up",
                main_set="20-30 min at tempo pace",
                cooldown="10 min cool-down"
            )

    def _create_moderate_day(self, load: float) -> TrainingDay:
        """Create a moderate training day."""
        duration = int(load * 0.55)
        return TrainingDay(
            date=datetime.now(),
            workout_type="moderate",
            planned_load=load,
            duration_minutes=duration,
            intensity="aerobic",
            description="Moderate aerobic training",
            warmup="10 min easy warm-up",
            main_set=f"{duration-20} min at moderate aerobic pace",
            cooldown="10 min cool-down"
        )

    def _create_long_day(self, load: float, week: TrainingWeek) -> TrainingDay:
        """Create a long training day."""
        duration = int(load * 0.7)

        if week.phase == TrainingPhase.BUILD:
            description = "Progressive long run"
            main_set = f"{duration*0.6:.0f} min easy, {duration*0.3:.0f} min moderate, {duration*0.1:.0f} min at tempo"
        elif week.phase == TrainingPhase.PEAK:
            description = "Long run with race pace segments"
            main_set = f"{duration-30} min with 3x10 min at race pace"
        else:
            description = "Long easy run"
            main_set = f"{duration-20} min at easy aerobic pace"

        return TrainingDay(
            date=datetime.now(),
            workout_type="long",
            planned_load=load,
            duration_minutes=duration,
            intensity="aerobic",
            description=description,
            warmup="10 min easy warm-up",
            main_set=main_set,
            cooldown="10 min easy cool-down"
        )

    def _create_recovery_day(self, load: float) -> TrainingDay:
        """Create a recovery day."""
        duration = int(load * 0.4)
        return TrainingDay(
            date=datetime.now(),
            workout_type="recovery",
            planned_load=load,
            duration_minutes=duration,
            intensity="recovery",
            description="Active recovery",
            main_set=f"{duration} min very easy recovery pace"
        )

    def _create_rest_day(self) -> TrainingDay:
        """Create a rest day."""
        return TrainingDay(
            date=datetime.now(),
            workout_type="rest",
            planned_load=0,
            duration_minutes=0,
            intensity="rest",
            description="Complete rest day",
            main_set="Rest and recovery"
        )