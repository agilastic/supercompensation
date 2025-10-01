"""
Interactive Training Plan Optimization

Genetic algorithm-based optimization for multi-week training plans with:
- Multi-objective optimization (fitness, recovery, injury prevention)
- Weather-aware training suggestions
- Event-based plan adaptation
- Constraint-based scheduling
- "What-if" scenario planning

Author: AI Training System
Date: 2025-10-01
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class WorkoutType(Enum):
    """Types of workouts in the plan."""
    REST = "rest"
    RECOVERY = "recovery"
    EASY = "easy"
    TEMPO = "tempo"
    THRESHOLD = "threshold"
    VO2MAX = "vo2max"
    INTERVALS = "intervals"
    LONG = "long"
    RACE = "race"
    STRENGTH = "strength"


class TrainingPhase(Enum):
    """Training phases."""
    BASE = "base"
    BUILD = "build"
    PEAK = "peak"
    TAPER = "taper"
    RECOVERY = "recovery"
    TRANSITION = "transition"


@dataclass
class Workout:
    """Individual workout in the training plan."""
    date: datetime
    workout_type: WorkoutType
    duration_minutes: int
    intensity: float  # 0.0-1.0 (percentage of max)
    tss: float  # Training Stress Score
    sport: str  # run, ride, swim, etc.
    description: str = ""
    weather_dependent: bool = False
    can_move: bool = True  # Can this workout be rescheduled?
    priority: int = 5  # 1-10, higher = more important

    def calculate_load(self) -> float:
        """Calculate training load for this workout."""
        return self.tss


@dataclass
class TrainingWeek:
    """Week of training in the plan."""
    week_number: int
    start_date: datetime
    workouts: List[Workout]
    phase: TrainingPhase
    target_tss: float
    target_hours: float

    def get_actual_tss(self) -> float:
        """Calculate actual TSS for the week."""
        return sum(w.tss for w in self.workouts)

    def get_actual_hours(self) -> float:
        """Calculate actual hours for the week."""
        return sum(w.duration_minutes for w in self.workouts) / 60

    def get_hard_days(self) -> int:
        """Count hard training days."""
        hard_types = {WorkoutType.THRESHOLD, WorkoutType.VO2MAX, WorkoutType.INTERVALS, WorkoutType.RACE}
        return sum(1 for w in self.workouts if w.workout_type in hard_types)

    def get_rest_days(self) -> int:
        """Count rest days."""
        return sum(1 for w in self.workouts if w.workout_type == WorkoutType.REST)


@dataclass
class TrainingPlan:
    """Complete multi-week training plan."""
    weeks: List[TrainingWeek]
    goal_event: Optional[datetime] = None
    goal_event_type: str = "race"
    athlete_constraints: Dict[str, Any] = field(default_factory=dict)

    def get_total_tss(self) -> float:
        """Total TSS across all weeks."""
        return sum(week.get_actual_tss() for week in self.weeks)

    def get_total_hours(self) -> float:
        """Total hours across all weeks."""
        return sum(week.get_actual_hours() for week in self.weeks)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert plan to DataFrame for analysis."""
        data = []
        for week in self.weeks:
            for workout in week.workouts:
                data.append({
                    'date': workout.date,
                    'week': week.week_number,
                    'phase': week.phase.value,
                    'type': workout.workout_type.value,
                    'sport': workout.sport,
                    'duration_min': workout.duration_minutes,
                    'intensity': workout.intensity,
                    'tss': workout.tss,
                    'description': workout.description
                })
        return pd.DataFrame(data)


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization goals."""
    maximize_fitness: float = 1.0  # Weight for fitness improvement
    minimize_fatigue: float = 0.8  # Weight for fatigue management
    minimize_injury_risk: float = 1.0  # Weight for injury prevention
    maximize_adherence: float = 0.6  # Weight for plan adherence likelihood
    respect_constraints: float = 1.0  # Weight for constraint satisfaction

    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = (self.maximize_fitness + self.minimize_fatigue +
                self.minimize_injury_risk + self.maximize_adherence +
                self.respect_constraints)
        self.maximize_fitness /= total
        self.minimize_fatigue /= total
        self.minimize_injury_risk /= total
        self.maximize_adherence /= total
        self.respect_constraints /= total


@dataclass
class PlanConstraints:
    """Constraints for training plan optimization."""
    max_weekly_tss: float = 600
    min_weekly_tss: float = 200
    max_daily_tss: float = 200
    max_weekly_hours: float = 15
    min_rest_days_per_week: int = 1
    max_consecutive_hard_days: int = 3
    available_days: List[int] = field(default_factory=lambda: list(range(7)))  # 0=Monday
    restricted_times: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    must_include_workouts: List[WorkoutType] = field(default_factory=list)


class GeneticPlanOptimizer:
    """
    Genetic algorithm optimizer for training plans.

    Uses evolutionary algorithms to find optimal training schedules that:
    - Maximize fitness gains
    - Minimize injury risk
    - Respect athlete constraints
    - Adapt to events and weather
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 elitism_rate: float = 0.1):
        """
        Initialize genetic optimizer.

        Args:
            population_size: Number of plans in each generation
            generations: Number of evolutionary iterations
            mutation_rate: Probability of random mutation
            crossover_rate: Probability of crossover between plans
            elitism_rate: Percentage of best plans to keep unchanged
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.logger = logging.getLogger(__name__)

        # Banister model parameters for fitness simulation
        self.tau_fitness = 42  # Fitness time constant (days)
        self.tau_fatigue = 7   # Fatigue time constant (days)
        self.k_fitness = 1.0   # Fitness gain coefficient
        self.k_fatigue = 2.0   # Fatigue coefficient

    def optimize_plan(self,
                     current_ctl: float,
                     current_atl: float,
                     weeks: int,
                     goal_event: Optional[datetime] = None,
                     constraints: Optional[PlanConstraints] = None,
                     objectives: Optional[OptimizationObjectives] = None) -> TrainingPlan:
        """
        Optimize a training plan using genetic algorithm.

        Args:
            current_ctl: Current chronic training load (fitness)
            current_atl: Current acute training load (fatigue)
            weeks: Number of weeks to plan
            goal_event: Target race/event date
            constraints: Training constraints
            objectives: Optimization objectives

        Returns:
            Optimized training plan
        """
        if constraints is None:
            constraints = PlanConstraints()

        if objectives is None:
            objectives = OptimizationObjectives()
            objectives.normalize_weights()

        # Initialize population
        self.logger.info(f"Initializing population of {self.population_size} training plans")
        population = self._initialize_population(
            current_ctl, current_atl, weeks, goal_event, constraints
        )

        # Evolution loop
        best_fitness_history = []
        best_plan = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_fitness(plan, current_ctl, current_atl, objectives, constraints)
                for plan in population
            ]

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            best_fitness_history.append(gen_best_fitness)

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_plan = deepcopy(population[gen_best_idx])

            if generation % 20 == 0:
                self.logger.info(f"Generation {generation}: Best fitness = {gen_best_fitness:.4f}")

            # Selection
            selected = self._selection(population, fitness_scores)

            # Crossover
            offspring = self._crossover(selected)

            # Mutation
            offspring = self._mutation(offspring, constraints)

            # Elitism: keep best plans
            n_elite = int(self.population_size * self.elitism_rate)
            elite_indices = np.argsort(fitness_scores)[-n_elite:]
            elite = [population[i] for i in elite_indices]

            # New population
            population = elite + offspring[:self.population_size - n_elite]

        self.logger.info(f"Optimization complete. Final fitness: {best_fitness:.4f}")

        return best_plan

    def _initialize_population(self,
                              current_ctl: float,
                              current_atl: float,
                              weeks: int,
                              goal_event: Optional[datetime],
                              constraints: PlanConstraints) -> List[TrainingPlan]:
        """Generate initial population of random training plans."""
        population = []

        for _ in range(self.population_size):
            plan = self._generate_random_plan(
                current_ctl, current_atl, weeks, goal_event, constraints
            )
            population.append(plan)

        return population

    def _generate_random_plan(self,
                             current_ctl: float,
                             current_atl: float,
                             weeks: int,
                             goal_event: Optional[datetime],
                             constraints: PlanConstraints) -> TrainingPlan:
        """Generate a single random (but valid) training plan."""
        start_date = datetime.now()
        training_weeks = []

        for week_num in range(weeks):
            week_start = start_date + timedelta(weeks=week_num)

            # Determine phase
            if goal_event:
                days_to_event = (goal_event - week_start).days
                if days_to_event <= 14:
                    phase = TrainingPhase.TAPER
                elif days_to_event <= 28:
                    phase = TrainingPhase.PEAK
                elif days_to_event <= 56:
                    phase = TrainingPhase.BUILD
                else:
                    phase = TrainingPhase.BASE
            else:
                # Default progression
                if week_num < weeks * 0.4:
                    phase = TrainingPhase.BASE
                elif week_num < weeks * 0.7:
                    phase = TrainingPhase.BUILD
                else:
                    phase = TrainingPhase.PEAK

            # Generate week
            week = self._generate_random_week(
                week_num, week_start, phase, constraints
            )
            training_weeks.append(week)

        return TrainingPlan(
            weeks=training_weeks,
            goal_event=goal_event,
            athlete_constraints={'ctl': current_ctl, 'atl': current_atl}
        )

    def _generate_random_week(self,
                             week_num: int,
                             week_start: datetime,
                             phase: TrainingPhase,
                             constraints: PlanConstraints) -> TrainingWeek:
        """Generate a random training week."""
        # Phase-specific parameters
        phase_params = {
            TrainingPhase.BASE: {'intensity': 0.6, 'volume': 0.8, 'hard_ratio': 0.2},
            TrainingPhase.BUILD: {'intensity': 0.75, 'volume': 1.0, 'hard_ratio': 0.3},
            TrainingPhase.PEAK: {'intensity': 0.85, 'volume': 0.9, 'hard_ratio': 0.35},
            TrainingPhase.TAPER: {'intensity': 0.7, 'volume': 0.5, 'hard_ratio': 0.2},
            TrainingPhase.RECOVERY: {'intensity': 0.5, 'volume': 0.6, 'hard_ratio': 0.1},
            TrainingPhase.TRANSITION: {'intensity': 0.5, 'volume': 0.4, 'hard_ratio': 0.0}
        }

        params = phase_params.get(phase, phase_params[TrainingPhase.BUILD])

        # Target weekly TSS
        base_tss = random.uniform(constraints.min_weekly_tss, constraints.max_weekly_tss)
        target_tss = base_tss * params['volume']
        target_hours = target_tss / 70  # Approximate: 70 TSS/hour average

        # Generate workouts for 7 days
        workouts = []
        for day in range(7):
            day_date = week_start + timedelta(days=day)

            # Determine workout type
            if day in [6]:  # Sunday often rest or long
                if random.random() < 0.5:
                    workout_type = WorkoutType.LONG
                    duration = random.randint(120, 240)
                    intensity = 0.6
                    tss = duration / 60 * 70 * intensity
                else:
                    workout_type = WorkoutType.REST
                    duration = 0
                    intensity = 0
                    tss = 0
            elif random.random() < params['hard_ratio']:
                # Hard workout
                workout_type = random.choice([WorkoutType.THRESHOLD, WorkoutType.VO2MAX, WorkoutType.INTERVALS])
                duration = random.randint(45, 90)
                intensity = random.uniform(0.8, 0.95)
                tss = duration / 60 * 100 * intensity
            elif random.random() < 0.15:
                # Rest day
                workout_type = WorkoutType.REST
                duration = 0
                intensity = 0
                tss = 0
            else:
                # Easy workout
                workout_type = random.choice([WorkoutType.EASY, WorkoutType.RECOVERY])
                duration = random.randint(30, 75)
                intensity = random.uniform(0.5, 0.7)
                tss = duration / 60 * 60 * intensity

            sport = random.choice(['run', 'ride']) if workout_type != WorkoutType.REST else 'rest'

            workout = Workout(
                date=day_date,
                workout_type=workout_type,
                duration_minutes=duration,
                intensity=intensity,
                tss=tss,
                sport=sport,
                description=f"{workout_type.value.title()} {sport}",
                can_move=workout_type != WorkoutType.RACE
            )
            workouts.append(workout)

        return TrainingWeek(
            week_number=week_num,
            start_date=week_start,
            workouts=workouts,
            phase=phase,
            target_tss=target_tss,
            target_hours=target_hours
        )

    def _evaluate_fitness(self,
                         plan: TrainingPlan,
                         initial_ctl: float,
                         initial_atl: float,
                         objectives: OptimizationObjectives,
                         constraints: PlanConstraints) -> float:
        """
        Evaluate fitness of a training plan.

        Multi-objective function combining:
        1. Fitness gain (CTL improvement)
        2. Fatigue management (avoid excessive ATL)
        3. Injury risk (ramp rate, consecutive hard days)
        4. Constraint satisfaction
        5. Adherence likelihood
        """
        # Simulate CTL/ATL progression
        ctl = initial_ctl
        atl = initial_atl

        max_ctl = ctl
        total_injury_risk = 0
        constraint_violations = 0

        for week in plan.weeks:
            weekly_load = []
            hard_days_consecutive = 0
            max_consecutive = 0

            for workout in week.workouts:
                daily_tss = workout.tss
                weekly_load.append(daily_tss)

                # Update CTL/ATL (simplified Banister)
                ctl = ctl + (daily_tss - ctl) * (1 - np.exp(-1/self.tau_fitness))
                atl = atl + (daily_tss - atl) * (1 - np.exp(-1/self.tau_fatigue))

                max_ctl = max(max_ctl, ctl)

                # Track consecutive hard days
                if workout.workout_type in {WorkoutType.THRESHOLD, WorkoutType.VO2MAX, WorkoutType.INTERVALS}:
                    hard_days_consecutive += 1
                    max_consecutive = max(max_consecutive, hard_days_consecutive)
                else:
                    hard_days_consecutive = 0

                # Check daily TSS constraint
                if daily_tss > constraints.max_daily_tss:
                    constraint_violations += 1

            # Weekly ramp rate (injury risk)
            week_tss = sum(weekly_load)
            if week.week_number > 0:
                prev_week_tss = sum(w.tss for w in plan.weeks[week.week_number - 1].workouts)
                ramp_rate = (week_tss - prev_week_tss) / prev_week_tss if prev_week_tss > 0 else 0
                if abs(ramp_rate) > 0.15:  # >15% change
                    total_injury_risk += abs(ramp_rate) - 0.15

            # Check constraints
            if week_tss > constraints.max_weekly_tss:
                constraint_violations += 1
            if week_tss < constraints.min_weekly_tss:
                constraint_violations += 0.5
            if week.get_rest_days() < constraints.min_rest_days_per_week:
                constraint_violations += 1
            if max_consecutive > constraints.max_consecutive_hard_days:
                constraint_violations += 1

        # Calculate objectives
        fitness_gain = (max_ctl - initial_ctl) / initial_ctl  # Normalized fitness improvement
        final_tsb = ctl - atl
        fatigue_score = 1.0 / (1.0 + abs(final_tsb + 5))  # Penalize extreme TSB
        injury_risk_score = 1.0 / (1.0 + total_injury_risk * 10)
        constraint_score = 1.0 / (1.0 + constraint_violations)

        # Adherence likelihood (simpler plans = higher adherence)
        workout_variety = len(set(w.workout_type for week in plan.weeks for w in week.workouts))
        adherence_score = 1.0 - (workout_variety / 10.0)  # Penalize excessive variety

        # Combined fitness (weighted sum)
        combined_fitness = (
            objectives.maximize_fitness * fitness_gain +
            objectives.minimize_fatigue * fatigue_score +
            objectives.minimize_injury_risk * injury_risk_score +
            objectives.respect_constraints * constraint_score +
            objectives.maximize_adherence * adherence_score
        )

        return combined_fitness

    def _selection(self,
                   population: List[TrainingPlan],
                   fitness_scores: List[float]) -> List[TrainingPlan]:
        """Tournament selection for next generation."""
        selected = []
        tournament_size = 5

        for _ in range(len(population)):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(deepcopy(population[winner_idx]))

        return selected

    def _crossover(self, population: List[TrainingPlan]) -> List[TrainingPlan]:
        """Crossover operation: combine weeks from two parent plans."""
        offspring = []

        for i in range(0, len(population) - 1, 2):
            if random.random() < self.crossover_rate:
                parent1 = population[i]
                parent2 = population[i + 1]

                # Single-point crossover at week level
                crossover_point = random.randint(1, len(parent1.weeks) - 1)

                child1_weeks = parent1.weeks[:crossover_point] + parent2.weeks[crossover_point:]
                child2_weeks = parent2.weeks[:crossover_point] + parent1.weeks[crossover_point:]

                child1 = TrainingPlan(weeks=child1_weeks, goal_event=parent1.goal_event)
                child2 = TrainingPlan(weeks=child2_weeks, goal_event=parent2.goal_event)

                offspring.extend([child1, child2])
            else:
                offspring.extend([population[i], population[i + 1]])

        return offspring

    def _mutation(self,
                  population: List[TrainingPlan],
                  constraints: PlanConstraints) -> List[TrainingPlan]:
        """Mutation operation: randomly modify workouts."""
        for plan in population:
            if random.random() < self.mutation_rate:
                # Select random week and workout
                week = random.choice(plan.weeks)
                workout_idx = random.randint(0, len(week.workouts) - 1)
                workout = week.workouts[workout_idx]

                if workout.can_move:
                    # Mutation types
                    mutation_type = random.choice(['intensity', 'duration', 'type', 'swap'])

                    if mutation_type == 'intensity':
                        workout.intensity = np.clip(workout.intensity + random.uniform(-0.1, 0.1), 0.3, 1.0)
                        workout.tss = workout.duration_minutes / 60 * 100 * workout.intensity

                    elif mutation_type == 'duration':
                        workout.duration_minutes = int(np.clip(
                            workout.duration_minutes + random.randint(-15, 15),
                            30, 240
                        ))
                        workout.tss = workout.duration_minutes / 60 * 100 * workout.intensity

                    elif mutation_type == 'type':
                        new_type = random.choice(list(WorkoutType))
                        if new_type != WorkoutType.RACE:
                            workout.workout_type = new_type

                    elif mutation_type == 'swap':
                        # Swap with another workout in the week
                        other_idx = random.randint(0, len(week.workouts) - 1)
                        if week.workouts[other_idx].can_move:
                            week.workouts[workout_idx], week.workouts[other_idx] = \
                                week.workouts[other_idx], week.workouts[workout_idx]

        return population


class WeatherAdaptiveScheduler:
    """
    Weather-aware training schedule adjustments.

    Adapts training plans based on weather forecasts.
    """

    def __init__(self):
        """Initialize weather scheduler."""
        self.logger = logging.getLogger(__name__)

    def suggest_adjustments(self,
                           plan: TrainingPlan,
                           weather_forecast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest plan adjustments based on weather.

        Args:
            plan: Current training plan
            weather_forecast: Dict with keys: date, temp, conditions, wind

        Returns:
            List of suggested adjustments
        """
        suggestions = []

        for week in plan.weeks:
            for workout in week.workouts:
                date_str = workout.date.strftime('%Y-%m-%d')
                if date_str in weather_forecast:
                    weather = weather_forecast[date_str]

                    # Check conditions
                    if weather.get('temp_c', 20) > 32:
                        # Hot weather
                        if workout.workout_type in {WorkoutType.THRESHOLD, WorkoutType.VO2MAX}:
                            suggestions.append({
                                'date': workout.date,
                                'type': 'reduce_intensity',
                                'reason': f"High temperature ({weather['temp_c']}°C)",
                                'recommendation': "Reduce intensity by 10-15% or move to cooler time"
                            })

                    elif weather.get('conditions') == 'rain' and workout.sport == 'ride':
                        suggestions.append({
                            'date': workout.date,
                            'type': 'swap_sport',
                            'reason': "Rain forecast",
                            'recommendation': "Consider indoor trainer or swap to run"
                        })

                    elif weather.get('wind_kph', 0) > 40 and workout.sport == 'ride':
                        suggestions.append({
                            'date': workout.date,
                            'type': 'modify_route',
                            'reason': f"High wind ({weather['wind_kph']} km/h)",
                            'recommendation': "Choose sheltered route or indoor option"
                        })

        return suggestions


class WhatIfAnalyzer:
    """
    What-if scenario planning tool.

    Simulates alternative training scenarios and outcomes.
    """

    def __init__(self):
        """Initialize what-if analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_scenario(self,
                        base_plan: TrainingPlan,
                        scenario: str,
                        modifications: Dict[str, Any],
                        current_ctl: float,
                        current_atl: float) -> Dict[str, Any]:
        """
        Analyze a what-if scenario.

        Args:
            base_plan: Original training plan
            scenario: Scenario name/description
            modifications: Changes to apply (e.g., {'week_3_skip': True})
            current_ctl: Current fitness
            current_atl: Current fatigue

        Returns:
            Analysis results with predicted outcomes
        """
        # Apply modifications
        modified_plan = self._apply_modifications(base_plan, modifications)

        # Simulate both plans
        base_outcome = self._simulate_plan_outcome(base_plan, current_ctl, current_atl)
        modified_outcome = self._simulate_plan_outcome(modified_plan, current_ctl, current_atl)

        return {
            'scenario': scenario,
            'base_outcome': base_outcome,
            'modified_outcome': modified_outcome,
            'differences': {
                'ctl_diff': modified_outcome['final_ctl'] - base_outcome['final_ctl'],
                'tsb_diff': modified_outcome['final_tsb'] - base_outcome['final_tsb'],
                'injury_risk_diff': modified_outcome['injury_risk'] - base_outcome['injury_risk']
            },
            'recommendation': self._generate_recommendation(base_outcome, modified_outcome)
        }

    def _apply_modifications(self,
                            plan: TrainingPlan,
                            modifications: Dict[str, Any]) -> TrainingPlan:
        """Apply modifications to create scenario plan."""
        modified = deepcopy(plan)

        # Example modifications
        if modifications.get('reduce_volume'):
            factor = modifications['reduce_volume']
            for week in modified.weeks:
                for workout in week.workouts:
                    workout.duration_minutes = int(workout.duration_minutes * factor)
                    workout.tss *= factor

        if modifications.get('skip_week'):
            week_num = modifications['skip_week']
            if 0 <= week_num < len(modified.weeks):
                for workout in modified.weeks[week_num].workouts:
                    workout.workout_type = WorkoutType.REST
                    workout.duration_minutes = 0
                    workout.tss = 0

        return modified

    def _simulate_plan_outcome(self,
                               plan: TrainingPlan,
                               initial_ctl: float,
                               initial_atl: float) -> Dict[str, Any]:
        """Simulate plan execution and predict outcomes."""
        ctl = initial_ctl
        atl = initial_atl
        injury_risk = 0

        for week in plan.weeks:
            week_tss = week.get_actual_tss()

            for workout in week.workouts:
                # Update CTL/ATL
                ctl += (workout.tss - ctl) * (1 - np.exp(-1/42))
                atl += (workout.tss - atl) * (1 - np.exp(-1/7))

            # Injury risk from ramp rate
            if week.week_number > 0:
                prev_tss = plan.weeks[week.week_number - 1].get_actual_tss()
                ramp = (week_tss - prev_tss) / prev_tss if prev_tss > 0 else 0
                if abs(ramp) > 0.15:
                    injury_risk += abs(ramp)

        return {
            'final_ctl': ctl,
            'final_atl': atl,
            'final_tsb': ctl - atl,
            'injury_risk': injury_risk,
            'total_tss': plan.get_total_tss(),
            'total_hours': plan.get_total_hours()
        }

    def _generate_recommendation(self,
                                base: Dict[str, Any],
                                modified: Dict[str, Any]) -> str:
        """Generate recommendation based on scenario comparison."""
        ctl_diff = modified['final_ctl'] - base['final_ctl']
        risk_diff = modified['injury_risk'] - base['injury_risk']

        if ctl_diff < -3 and risk_diff < -0.1:
            return "Modified plan reduces injury risk but at cost of fitness. Consider if recovery is priority."
        elif ctl_diff > 3 and risk_diff > 0.1:
            return "Modified plan improves fitness but increases injury risk. Ensure adequate recovery."
        elif risk_diff < -0.1:
            return "Modified plan reduces injury risk with minimal fitness impact. Recommended."
        elif ctl_diff > 2:
            return "Modified plan improves fitness with manageable risk. Consider adopting."
        else:
            return "Modified plan shows similar outcomes to base plan."
