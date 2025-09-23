"""Advanced 30-day training plan generation with visualization.

This module provides:
1. Mesocycle-based periodization (4-week blocks)
2. Adaptive load progression based on current state
3. Sport-specific planning with cross-training
4. Interactive visualization capabilities
5. Plan adjustment based on actual vs planned performance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from .advanced_model import (
    EnhancedFitnessFatigueModel,
    PerPotModel,
    OptimalControlProblem,
    TrainingGoal
)
from .multisystem_recovery import MultiSystemRecoveryModel, RecoverySystem
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


class AdvancedPlanGenerator:
    """Generate sophisticated 30-day training plans."""

    def __init__(self, user_id: str = "default"):
        """Initialize plan generator with user context."""
        self.user_id = user_id
        self.db = get_db()

        # Initialize models
        self.ff_model = EnhancedFitnessFatigueModel(user_id=user_id)
        self.perpot_model = PerPotModel(user_id=user_id)
        self.recovery_model = MultiSystemRecoveryModel()

        # Load user preferences and history
        self._load_user_context()

    def _load_user_context(self):
        """Load user preferences, history, and constraints."""
        with self.db.get_session() as session:
            # Get recent activities for pattern analysis
            cutoff = datetime.utcnow() - timedelta(days=90)
            activities = session.query(Activity).filter(
                Activity.start_date >= cutoff
            ).all()

            # Analyze patterns
            self.preferred_sports = self._analyze_sport_preferences(activities)
            self.typical_duration = self._analyze_duration_patterns(activities)
            self.weekly_pattern = self._analyze_weekly_patterns(activities)

            # Get current state
            latest_metric = session.query(Metric).order_by(
                Metric.date.desc()
            ).first()

            if latest_metric:
                self.current_fitness = latest_metric.fitness
                self.current_fatigue = latest_metric.fatigue
                self.current_form = latest_metric.form
            else:
                self.current_fitness = 30.0
                self.current_fatigue = 20.0
                self.current_form = 10.0

    def generate_30_day_plan(
        self,
        goal: str = "fitness",  # fitness, performance, recovery, maintenance
        target_event_date: Optional[datetime] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a complete 30-day training plan.

        Args:
            goal: Primary training goal
            target_event_date: Optional target event/race date
            constraints: User constraints (rest days, max hours, etc.)

        Returns:
            Complete training plan with visualizations
        """
        constraints = constraints or {}

        # Determine mesocycle type based on goal and timing
        mesocycle_type = self._determine_mesocycle_type(
            goal, target_event_date
        )

        # Generate base plan structure
        mesocycle = self._create_mesocycle(
            mesocycle_type,
            datetime.utcnow().date(),
            constraints
        )

        # Optimize load distribution
        optimized_loads = self._optimize_load_distribution(
            mesocycle, goal
        )

        # Create detailed daily workouts
        daily_workouts = self._create_daily_workouts(
            mesocycle, optimized_loads
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
            final_plan, simulated_response
        )

        return {
            'mesocycle': mesocycle,
            'daily_workouts': final_plan,
            'simulated_response': simulated_response,
            'visualizations': visualizations,
            'summary': self._create_plan_summary(final_plan, simulated_response)
        }

    def _determine_mesocycle_type(
        self,
        goal: str,
        target_date: Optional[datetime]
    ) -> MesocycleType:
        """Determine appropriate mesocycle based on goal and timing."""
        if target_date:
            days_to_event = (target_date - datetime.utcnow()).days

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

    def _create_mesocycle(
        self,
        cycle_type: MesocycleType,
        start_date: datetime,
        constraints: Dict
    ) -> Mesocycle:
        """Create a 4-week mesocycle structure."""

        # Define load patterns for each mesocycle type
        patterns = {
            MesocycleType.ADAPTATION: {
                'loads': [60, 70, 80, 50],  # Progressive with recovery
                'pattern': '3:1',
                'easy': 0.7, 'moderate': 0.2, 'hard': 0.1
            },
            MesocycleType.ACCUMULATION: {
                'loads': [80, 90, 100, 60],  # High volume
                'pattern': '3:1',
                'easy': 0.6, 'moderate': 0.3, 'hard': 0.1
            },
            MesocycleType.INTENSIFICATION: {
                'loads': [70, 80, 85, 50],  # Higher intensity
                'pattern': '3:1',
                'easy': 0.5, 'moderate': 0.25, 'hard': 0.25
            },
            MesocycleType.REALIZATION: {
                'loads': [60, 50, 40, 30],  # Taper
                'pattern': 'taper',
                'easy': 0.7, 'moderate': 0.2, 'hard': 0.1
            },
            MesocycleType.RECOVERY: {
                'loads': [40, 35, 30, 25],  # Recovery
                'pattern': 'recovery',
                'easy': 0.9, 'moderate': 0.1, 'hard': 0.0
            },
            MesocycleType.MAINTENANCE: {
                'loads': [60, 60, 60, 50],  # Stable
                'pattern': 'stable',
                'easy': 0.7, 'moderate': 0.25, 'hard': 0.05
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
        mesocycle: Mesocycle,
        goal: str
    ) -> np.ndarray:
        """Optimize daily load distribution using advanced models."""

        # Map goal to optimization objective
        goal_map = {
            'fitness': TrainingGoal.MAXIMIZE_FTI,
            'performance': TrainingGoal.BALANCED,
            'recovery': TrainingGoal.MINIMIZE_FATIGUE,
            'maintenance': TrainingGoal.BALANCED
        }

        training_goal = goal_map.get(goal, TrainingGoal.BALANCED)

        # Create optimization problem
        problem = OptimalControlProblem(
            goal=training_goal,
            duration_days=30,
            min_load_threshold=20.0,
            max_daily_load=150.0,
            rest_days=[6, 13, 20, 27],  # Weekly rest days
            constraints={}
        )

        # Solve optimization
        result = self.ff_model.optimize_training_plan(
            problem,
            initial_fitness=self.current_fitness,
            initial_fatigue=self.current_fatigue
        )

        if result['success']:
            return result['loads']
        else:
            # Fallback to pattern-based distribution
            return self._create_fallback_loads(mesocycle)

    def _create_daily_workouts(
        self,
        mesocycle: Mesocycle,
        loads: np.ndarray
    ) -> List[WorkoutPlan]:
        """Create detailed daily workout plans."""
        workouts = []

        for day in range(30):
            date = mesocycle.start_date + timedelta(days=day)
            week_num = day // 7 + 1

            # Determine workout type based on load and day
            workout_type = self._determine_workout_type(
                loads[day], day % 7, week_num
            )

            # Create workout structure
            workout = self._create_workout_structure(
                date, day, week_num, mesocycle.cycle_type,
                workout_type, loads[day]
            )

            workouts.append(workout)

        return workouts

    def _create_workout_structure(
        self,
        date: datetime,
        day_num: int,
        week_num: int,
        mesocycle_type: MesocycleType,
        workout_type: WorkoutType,
        load: float
    ) -> WorkoutPlan:
        """Create detailed workout structure."""

        # Determine sport based on patterns and variety
        primary_sport = self._select_sport(day_num, workout_type)
        alternatives = self._get_alternative_sports(primary_sport, workout_type)

        # Calculate duration based on load and workout type
        duration = self._calculate_duration(load, workout_type)

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
            planned_load=load,
            intensity_level=self._get_intensity_level(workout_type),
            hr_zones=hr_zones,
            total_duration_min=duration,
            warmup_min=structure['warmup'],
            main_set_min=structure['main'],
            cooldown_min=structure['cooldown'],
            title=self._generate_workout_title(workout_type, primary_sport),
            description=self._generate_workout_description(
                workout_type, duration, load
            ),
            workout_structure=structure,
            primary_system=self._get_primary_system(workout_type),
            secondary_systems=self._get_secondary_systems(workout_type),
            expected_adaptations=self._get_expected_adaptations(workout_type),
            recovery_hours_needed=recovery_hours,
            next_day_recommendation=self._get_next_day_recommendation(
                load, workout_type
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

        # Fitness-Fatigue simulation
        fitness, fatigue, performance = self.ff_model.impulse_response(loads, t)

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
        """Create structured recovery mesocycle for sustained overtraining."""

        # Week 1-2: Deep recovery (TSS 20-40)
        for i in range(min(14, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                workouts[i].workout_type = WorkoutType.RECOVERY
                workouts[i].planned_load = min(25 + i * 1, 40)  # Gradual increase
                workouts[i].duration_minutes = 30 + i * 5  # 30-100 minutes
                workouts[i].intensity_level = 'very_easy'
                workouts[i].description = f"Recovery Week {'1' if i < 7 else '2'} - Active recovery"
                workouts[i].title = f"Easy Recovery - Day {i + 1}"

        # Week 3-4: Gradual return to base training (TSS 40-80)
        for i in range(14, min(28, len(workouts))):
            if workouts[i].workout_type != WorkoutType.REST:
                week_day = i - 14
                if week_day % 7 in [0, 3]:  # Two key sessions per week
                    workouts[i].workout_type = WorkoutType.AEROBIC
                    workouts[i].planned_load = 60 + week_day * 2  # Progressive load
                    workouts[i].intensity_level = 'easy'
                    workouts[i].description = f"Base Building Week {'3' if i < 21 else '4'}"
                else:
                    workouts[i].workout_type = WorkoutType.RECOVERY
                    workouts[i].planned_load = 35 + week_day * 1
                    workouts[i].intensity_level = 'very_easy'
                    workouts[i].description = "Recovery between base sessions"

        return workouts

    def _create_visualizations(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray]
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
        weekly_summary = self._create_weekly_summary(workouts, response)

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

    def _create_plan_summary(
        self,
        workouts: List[WorkoutPlan],
        response: Dict[str, np.ndarray]
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
            'average_daily_load': total_load / 30,
            'average_daily_duration': total_duration / 30,
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

    def _create_fallback_loads(self, mesocycle: Mesocycle) -> np.ndarray:
        """Create fallback load distribution if optimization fails."""
        loads = []
        for week in mesocycle.weekly_loads:
            # Create weekly pattern
            week_pattern = [
                week * 0.8,   # Monday
                week * 1.0,   # Tuesday
                week * 0.9,   # Wednesday
                week * 1.1,   # Thursday
                week * 0.7,   # Friday
                week * 1.2,   # Saturday
                0,            # Sunday rest
            ]
            loads.extend(week_pattern[:7])  # Ensure 7 days

        # Extend to 30 days
        while len(loads) < 30:
            loads.append(loads[-7] if len(loads) >= 7 else 50)

        return np.array(loads[:30])

    def _determine_workout_type(
        self,
        load: float,
        day_of_week: int,
        week_num: int
    ) -> WorkoutType:
        """Determine workout type based on load and schedule."""
        if load == 0:
            return WorkoutType.REST
        elif load < 30:
            return WorkoutType.RECOVERY
        elif load < 50:
            return WorkoutType.AEROBIC
        elif load < 70:
            return WorkoutType.TEMPO
        elif load < 90:
            return WorkoutType.THRESHOLD
        elif load < 110:
            return WorkoutType.VO2MAX
        else:
            # High load days alternate between long and intervals
            if day_of_week == 5:  # Saturday
                return WorkoutType.LONG
            else:
                return WorkoutType.INTERVALS

    def _select_sport(self, day_num: int, workout_type: WorkoutType) -> str:
        """Select appropriate sport for the workout."""
        # Default sports for workout types
        default_sports = {
            WorkoutType.REST: "Rest",
            WorkoutType.RECOVERY: "Walk",
            WorkoutType.AEROBIC: "Run",
            WorkoutType.TEMPO: "Ride",
            WorkoutType.THRESHOLD: "Run",
            WorkoutType.VO2MAX: "Run",
            WorkoutType.NEUROMUSCULAR: "Run",
            WorkoutType.LONG: "Ride",
            WorkoutType.INTERVALS: "Run",
            WorkoutType.FARTLEK: "Run"
        }

        # Use preferred sports if available
        if self.preferred_sports:
            # Rotate through preferred sports for variety
            sport_index = day_num % len(self.preferred_sports)
            return self.preferred_sports[sport_index]

        return default_sports.get(workout_type, "Run")

    def _get_alternative_sports(
        self,
        primary: str,
        workout_type: WorkoutType
    ) -> List[str]:
        """Get alternative sports for cross-training."""
        alternatives = {
            "Run": ["Ride", "Swim", "Row"],
            "Ride": ["Run", "Swim", "VirtualRide"],
            "Swim": ["Run", "Ride", "Row"],
            "Walk": ["Yoga", "Swim", "EasyRide"],
            "Rest": ["Yoga", "Stretching", "Meditation"]
        }

        return alternatives.get(primary, ["Run", "Ride", "Swim"])[:2]

    def _calculate_duration(self, load: float, workout_type: WorkoutType) -> int:
        """Calculate workout duration from load and type."""
        # Base duration on load
        base_duration = load * 0.6  # Rough conversion

        # Adjust for workout type
        duration_multipliers = {
            WorkoutType.REST: 0,
            WorkoutType.RECOVERY: 0.5,
            WorkoutType.AEROBIC: 1.0,
            WorkoutType.TEMPO: 0.8,
            WorkoutType.THRESHOLD: 0.7,
            WorkoutType.VO2MAX: 0.6,
            WorkoutType.NEUROMUSCULAR: 0.5,
            WorkoutType.LONG: 1.5,
            WorkoutType.INTERVALS: 0.7,
            WorkoutType.FARTLEK: 0.9
        }

        multiplier = duration_multipliers.get(workout_type, 1.0)
        return int(base_duration * multiplier)

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

        # Adjust for workout type
        recovery_multipliers = {
            WorkoutType.REST: 0,
            WorkoutType.RECOVERY: 0.5,
            WorkoutType.AEROBIC: 1.0,
            WorkoutType.TEMPO: 1.2,
            WorkoutType.THRESHOLD: 1.5,
            WorkoutType.VO2MAX: 2.0,
            WorkoutType.NEUROMUSCULAR: 2.5,
            WorkoutType.LONG: 1.3,
            WorkoutType.INTERVALS: 1.8
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
        response: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Create weekly summary statistics."""
        weekly_summaries = []

        for week in range(5):  # 5 weeks to cover 30 days
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