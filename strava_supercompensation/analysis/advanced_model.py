"""Training models based on Schäfer et al. 2015 and Herold & Sommer 2020.

This module implements:
1. Fitness-Fatigue model with proper impulse-response dynamics
2. PerPot model for overtraining protection
3. Optimal control problem formulation for training plan generation
"""

import numpy as np
from scipy import optimize
from scipy.integrate import odeint
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric, AdaptiveModelParameters


class TrainingGoal(Enum):
    """Training goals for optimization."""
    MAXIMIZE_FTI = "maximize_fti"  # Force-Time Integral
    MAXIMIZE_FTI_WITH_THRESHOLD = "maximize_fti_threshold"
    MAXIMIZE_FATIGUE = "maximize_fatigue"
    MINIMIZE_FATIGUE = "minimize_fatigue"
    MAXIMIZE_TUT = "maximize_tut"  # Time-Under-Tension
    BALANCED = "balanced"


@dataclass
class OptimalControlProblem:
    """Defines an optimal control problem for training optimization."""
    goal: TrainingGoal
    duration_days: int
    min_load_threshold: float = 0.0
    max_daily_load: float = 200.0
    rest_days: List[int] = None
    constraints: Dict[str, float] = None

    def __post_init__(self):
        if self.rest_days is None:
            self.rest_days = []
        if self.constraints is None:
            self.constraints = {}


class FitnessFatigueModel:
    """Fitness-Fatigue model with proper impulse-response dynamics.

    Based on Schäfer et al. 2015 and Herold & Sommer 2020.
    """

    def __init__(
        self,
        # --- START FIX ---
        # The parameters are now sourced directly from the centralized config
        # This prevents scaling errors and ensures consistency with the basic model.
        k1: float = None,
        k2: float = None,
        tau1: float = None,
        tau2: float = None,
        # --- END FIX ---
        p_star: float = 0.2,  # Baseline performance
        user_id: str = "default"
    ):
        """Initialize the enhanced model with research-based parameters."""
        # Use config values if not explicitly provided
        self.k1 = k1 if k1 is not None else config.FITNESS_MAGNITUDE
        self.k2 = k2 if k2 is not None else config.FATIGUE_MAGNITUDE
        self.tau1 = tau1 if tau1 is not None else config.FITNESS_DECAY_RATE
        self.tau2 = tau2 if tau2 is not None else config.FATIGUE_DECAY_RATE
        self.p_star = p_star
        self.user_id = user_id

    def calculate_fitness_fatigue(
        self,
        training_loads: np.ndarray,
        t: np.ndarray,
        initial_fitness: float = 0.0,
        initial_fatigue: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate fitness, fatigue, and performance using Banister model.

        FIXED: Now accepts initial fitness and fatigue state for continuous simulation.

        Implements the impulse-response model from the papers:
        g(n) = Σ w(t) * exp(-(n-t)/τ1)  for fitness
        h(n) = Σ w(t) * exp(-(n-t)/τ2)  for fatigue
        p(n) = k1*g(n) - k2*h(n)         for form (TSB)

        Args:
            training_loads: Array of daily training loads (TSS)
            t: Time array (days)
            initial_fitness: Starting fitness level (CTL)
            initial_fatigue: Starting fatigue level (ATL)
        """
        n_days = len(t)
        fitness = np.zeros(n_days)
        fatigue = np.zeros(n_days)

        # FIXED: Initialize with current state, then apply decay and add first day's load
        if n_days > 0:
            # Start day 0 with decayed initial state plus first day's training
            fitness[0] = initial_fitness * np.exp(-1 / self.tau1)
            fatigue[0] = initial_fatigue * np.exp(-1 / self.tau2)

            # Add first day's training impulse
            if len(training_loads) > 0 and training_loads[0] > 0:
                fitness[0] += training_loads[0] * self.k1
                fatigue[0] += training_loads[0] * self.k2

        # Calculate cumulative impulse responses for subsequent days
        for i in range(1, n_days):
            # Decay from previous day
            fitness[i] = fitness[i-1] * np.exp(-1 / self.tau1)
            fatigue[i] = fatigue[i-1] * np.exp(-1 / self.tau2)

            # Add today's training impulse
            if i < len(training_loads) and training_loads[i] > 0:
                fitness[i] += training_loads[i] * self.k1
                fatigue[i] += training_loads[i] * self.k2

        # Calculate form (Training Stress Balance)
        performance = fitness - fatigue

        return fitness, fatigue, performance

    def generate_training_plan(
        self,
        problem: OptimalControlProblem,
        initial_fitness: float = 0.0,
        initial_fatigue: float = 0.0
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Generate training plan using differential evolution.

        Based on the optimization approach from Schäfer et al. 2015.
        Uses initial fitness and fatigue states to start from current condition.
        """
        n_days = problem.duration_days

        def objective(loads):
            """Objective function to optimize based on goal."""
            # Apply rest day constraints
            for day in problem.rest_days:
                if day < n_days:
                    loads[day] = 0

            # Start from initial state
            t = np.arange(n_days)
            fitness_base, fatigue_base, _ = self.calculate_fitness_fatigue(loads, t)
            # Add initial states
            fitness = fitness_base + initial_fitness * np.exp(-t / self.tau1)
            fatigue = fatigue_base + initial_fatigue * np.exp(-t / self.tau2)
            performance = fitness - fatigue

            if problem.goal == TrainingGoal.MAXIMIZE_FTI:
                # Force-Time Integral
                return -np.sum(loads)  # Negative for minimization

            elif problem.goal == TrainingGoal.MAXIMIZE_FTI_WITH_THRESHOLD:
                # FTI above threshold
                above_threshold = np.maximum(0, loads - problem.min_load_threshold)
                return -np.sum(above_threshold)

            elif problem.goal == TrainingGoal.MAXIMIZE_FATIGUE:
                return -np.max(fatigue)

            elif problem.goal == TrainingGoal.MINIMIZE_FATIGUE:
                # Minimize fatigue while achieving target FTI
                target_fti = problem.constraints.get('target_fti', 1000)
                fti_penalty = abs(np.sum(loads) - target_fti) * 10
                return np.max(fatigue) + fti_penalty

            elif problem.goal == TrainingGoal.MAXIMIZE_TUT:
                # Time under tension (number of training days)
                return -np.sum(loads > problem.min_load_threshold)

            else:  # BALANCED
                # Balance performance and recovery
                final_performance = performance[-1]
                avg_fatigue = np.mean(fatigue)
                return -(final_performance - 0.1 * avg_fatigue)

        # Set up bounds
        bounds = [(0, problem.max_daily_load) for _ in range(n_days)]

        # Apply rest day constraints in bounds
        for day in problem.rest_days:
            if day < n_days:
                bounds[day] = (0, 0)

        # Use differential evolution for global optimization
        result = optimize.differential_evolution(
            objective,
            bounds,
            maxiter=300,      # Increased from 100 to 300
            popsize=20,       # Increased from 15 to 20
            atol=1e-6,        # Absolute tolerance
            tol=0.01,         # Relative tolerance
            seed=42,
            workers=1         # Sequential for debugging
        )

        # Calculate final metrics
        optimal_loads = result.x
        t = np.arange(n_days)
        fitness, fatigue, performance = self.calculate_fitness_fatigue(optimal_loads, t)

        return {
            'loads': optimal_loads,
            'fitness': fitness,
            'fatigue': fatigue,
            'performance': performance,
            'objective_value': -result.fun,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else 'No message available'
        }


class PerPotModel:
    """Performance Potential model with overtraining protection.

    Based on Perl's PerPot metamodel from Schäfer et al. 2015.
    """

    def __init__(
        self,
        ds: float = 7.0,   # Delay of Strain Rate
        dr: float = 6.0,   # Delay of Response Rate
        dso: float = 1.5,  # Delay of Strain Overflow Rate
        pp0: float = 0.2,  # Initial Performance Potential
        max_daily_load: float = 200.0,  # User's maximum daily training load (TSS)
        user_id: str = "default"
    ):
        """Initialize PerPot model with parameters."""
        self.ds = ds
        self.dr = dr
        self.dso = dso
        self.pp0 = pp0
        self.max_daily_load = max_daily_load
        self.user_id = user_id

    def simulate(
        self,
        training_loads: np.ndarray,
        t: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Simulate PerPot dynamics with overtraining protection.

        SP: Strain Potential (negative effect)
        RP: Response Potential (positive effect)
        PP: Performance Potential (actual performance)
        """
        n_days = len(t)

        # Initialize potentials
        sp = np.zeros(n_days)  # Strain Potential
        rp = np.zeros(n_days)  # Response Potential
        pp = np.zeros(n_days)  # Performance Potential
        pp[0] = self.pp0

        for i in range(1, n_days):
            # Load rate for the current day i (fixed indexing)
            # Normalize by user's max daily load capacity instead of hardcoded 100
            lr = training_loads[i] / self.max_daily_load if i < len(training_loads) else 0.0

            # Update potentials with load
            sp[i] = sp[i-1] + lr
            rp[i] = rp[i-1] + lr

            # Calculate flow rates
            sr = min(min(1, sp[i]), max(0, pp[i-1])) / self.ds  # Strain Rate
            rr = min(min(1, rp[i]), min(1, 1 - pp[i-1])) / self.dr  # Response Rate

            # Overtraining protection - strain overflow
            or_rate = max(0, sp[i] - 1) / self.dso  # Overflow Rate

            # Update potentials with flows
            sp[i] = sp[i] - sr - or_rate
            rp[i] = rp[i] - rr
            pp[i] = pp[i-1] + rr - sr - or_rate

            # Keep PP in valid range [0, 1]
            pp[i] = np.clip(pp[i], 0, 1)

        return {
            'strain_potential': sp,
            'response_potential': rp,
            'performance_potential': pp,
            'overtraining_risk': sp > config.OVERTRAINING_THRESHOLD  # Configurable threshold
        }


class OptimalControlSolver:
    """Solves optimal control problems for training plan generation.

    Implements the multi-stage optimization approach from Herold & Sommer 2020.
    """

    def __init__(self, model: FitnessFatigueModel):
        """Initialize solver with a model."""
        self.model = model

    def solve_periodization_plan(
        self,
        total_duration: int,
        build_weeks: int,
        recovery_weeks: int,
        peak_weeks: int,
        stage_durations: List[int] = None,
        min_load: float = 0.0,
        max_load: float = 100.0,
        objective: str = 'maximize_fti'
    ) -> Dict[str, np.ndarray]:
        """Create periodized training plan with flexible stage durations.

        Based on the formulation in Herold & Sommer 2020.
        Allows custom stage durations instead of fixed equal stages.
        """
        # Use provided stage durations or calculate based on periodization
        if stage_durations is None:
            # Default periodization: build->recovery->peak cycles
            stage_durations = []
            remaining = total_duration
            cycle_length = build_weeks + recovery_weeks + peak_weeks

            while remaining > 0:
                if remaining >= cycle_length:
                    stage_durations.extend([build_weeks, recovery_weeks, peak_weeks])
                    remaining -= cycle_length
                else:
                    # Distribute remaining days
                    if remaining >= build_weeks:
                        stage_durations.append(build_weeks)
                        remaining -= build_weeks
                    if remaining > 0:
                        stage_durations.append(remaining)
                        remaining = 0

        n_stages = len(stage_durations)
        rest_stages = list(range(1, n_stages, 3))  # Every 3rd stage is recovery

        def stage_objective(x):
            """Objective for multi-stage problem."""
            # x contains loads and stage durations
            loads = x[:n_stages]
            durations = x[n_stages:]

            # Build full load profile
            full_loads = []
            for i in range(n_stages):
                if i in rest_stages:
                    stage_load = 0
                else:
                    stage_load = loads[i]
                full_loads.extend([stage_load] * int(durations[i]))

            full_loads = np.array(full_loads[:total_duration])
            t = np.arange(len(full_loads))

            fitness, fatigue, performance = self.model.calculate_fitness_fatigue(full_loads, t)

            if objective == 'maximize_fti':
                fti = np.sum(full_loads)
                return -fti
            elif objective == 'maximize_fti_above_threshold':
                fti_above = np.sum(np.maximum(0, full_loads - min_load))
                return -fti_above
            elif objective == 'minimize_fatigue_for_target_fti':
                target_fti = 1000
                fti = np.sum(full_loads)
                max_fatigue = np.max(fatigue)
                return max_fatigue + 10 * abs(fti - target_fti)
            else:
                return -performance[-1]

        # Set up bounds
        load_bounds = [(0, max_load) for _ in range(n_stages)]
        duration_bounds = [(1, total_duration // 2) for _ in range(n_stages)]

        # Apply rest stage constraints
        for stage in rest_stages:
            load_bounds[stage] = (0, 0)

        bounds = load_bounds + duration_bounds

        # Initial guess
        x0 = np.zeros(2 * n_stages)
        x0[:n_stages] = max_load / 2  # Medium loads
        x0[n_stages:] = stage_duration  # Equal durations

        # Add constraint that durations sum to total
        def duration_constraint(x):
            return np.sum(x[n_stages:]) - total_duration

        constraints = {'type': 'eq', 'fun': duration_constraint}

        # Optimize
        result = optimize.minimize(
            stage_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Extract solution
        optimal_loads = result.x[:n_stages]
        optimal_durations = result.x[n_stages:]

        # Build full solution
        full_loads = []
        for i in range(n_stages):
            if i in rest_stages:
                stage_load = 0
            else:
                stage_load = optimal_loads[i]
            full_loads.extend([stage_load] * int(optimal_durations[i]))

        full_loads = np.array(full_loads[:total_duration])
        t = np.arange(len(full_loads))
        fitness, fatigue, performance = self.model.calculate_fitness_fatigue(full_loads, t)

        return {
            'stage_loads': optimal_loads,
            'stage_durations': optimal_durations,
            'full_loads': full_loads,
            'fitness': fitness,
            'fatigue': fatigue,
            'performance': performance,
            'success': result.success
        }


class AdaptiveParameterLearning:
    """Adaptive parameter learning using evolutionary algorithms.

    Based on the adaptive approach from Schäfer et al. 2015.
    """

    def __init__(self, user_id: str = "default"):
        """Initialize adaptive learning system."""
        self.user_id = user_id
        self.db = get_db()

    def adapt_with_differential_evolution(
        self,
        training_history: pd.DataFrame,
        performance_data: pd.DataFrame,
        model_class: type
    ) -> Dict[str, float]:
        """Adapt model parameters using differential evolution.

        Args:
            training_history: DataFrame with columns ['date', 'load']
            performance_data: DataFrame with columns ['date', 'performance']
            model_class: Model class to optimize parameters for

        Returns:
            Optimized parameters dictionary
        """

        def objective(params):
            """Objective function - minimize prediction error."""
            if model_class == FitnessFatigueModel:
                k1, k2, tau1, tau2, p_star = params
                model = FitnessFatigueModel(k1, k2, tau1, tau2, p_star)
            else:
                return np.inf

            # Simulate with parameters
            loads = training_history['load'].values
            t = np.arange(len(loads))
            _, _, predicted_performance = model.calculate_fitness_fatigue(loads, t)

            # Calculate error on performance data dates
            error = 0
            for _, row in performance_data.iterrows():
                day_idx = (row['date'] - training_history['date'].min()).days
                if 0 <= day_idx < len(predicted_performance):
                    error += (predicted_performance[day_idx] - row['performance']) ** 2

            return error

        # Set parameter bounds based on physiological constraints
        bounds = [
            (0.01, 0.5),   # k1: fitness magnitude
            (0.01, 0.5),   # k2: fatigue magnitude
            (20, 60),      # tau1: fitness time constant
            (3, 15),       # tau2: fatigue time constant
            (0.1, 0.5)     # p_star: baseline performance
        ]

        # Optimize
        result = optimize.differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )

        # Save to database
        with self.db.get_session() as session:
            params_record = session.query(AdaptiveModelParameters).filter_by(
                user_id=self.user_id
            ).first()

            if not params_record:
                params_record = AdaptiveModelParameters(user_id=self.user_id)
                session.add(params_record)

            # Update with optimized values
            params_record.fitness_magnitude = result.x[0]
            params_record.fatigue_magnitude = result.x[1]
            params_record.fitness_decay = result.x[2]
            params_record.fatigue_decay = result.x[3]
            params_record.last_adaptation = datetime.now(timezone.utc)
            params_record.total_adaptations += 1

            session.commit()

        return {
            'k1': result.x[0],
            'k2': result.x[1],
            'tau1': result.x[2],
            'tau2': result.x[3],
            'p_star': result.x[4],
            'optimization_success': result.success,
            'final_error': result.fun
        }


def generate_optimal_weekly_plan(
    current_fitness: float,
    current_fatigue: float,
    goal: TrainingGoal,
    rest_days: List[int] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """Generate an optimal weekly training plan based on current state and goal.

    Args:
        current_fitness: Current fitness level (0-100)
        current_fatigue: Current fatigue level (0-100)
        goal: Training goal to optimize for
        rest_days: List of weekday indices (0=Monday) to rest

    Returns:
        Dictionary with daily loads and predicted outcomes
    """
    # Initialize model with personalized parameters if available
    model = FitnessFatigueModel()

    # Define optimization problem
    problem = OptimalControlProblem(
        goal=goal,
        duration_days=7,
        min_load_threshold=30.0,
        max_daily_load=150.0,
        rest_days=rest_days or [6],  # Default rest on Sunday
        constraints={
            'target_fti': 500 if goal == TrainingGoal.MINIMIZE_FATIGUE else None
        }
    )

    # Solve optimization problem
    result = model.generate_training_plan(
        problem,
        initial_fitness=current_fitness,
        initial_fatigue=current_fatigue
    )

    # Add practical recommendations
    result['recommendations'] = []
    for i, load in enumerate(result['loads']):
        if load == 0:
            result['recommendations'].append('REST')
        elif load < 40:
            result['recommendations'].append('RECOVERY')
        elif load < 70:
            result['recommendations'].append('EASY')
        elif load < 100:
            result['recommendations'].append('MODERATE')
        elif load < 130:
            result['recommendations'].append('HARD')
        else:
            result['recommendations'].append('PEAK')

    return result