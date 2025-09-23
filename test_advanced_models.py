#!/usr/bin/env python3
"""Test script for advanced training models implementation.

Tests the integration of:
1. Enhanced Fitness-Fatigue model
2. PerPot model with overtraining protection
3. Optimal control problem solver
4. Multi-system recovery integration
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Add project to path
sys.path.append('/Users/alex/workspace/supercompensation')

from strava_supercompensation.analysis.advanced_model import (
    EnhancedFitnessFatigueModel,
    PerPotModel,
    OptimalControlProblem,
    TrainingGoal,
    generate_optimal_weekly_plan
)
from strava_supercompensation.analysis.model_integration import (
    IntegratedTrainingAnalyzer,
    get_integrated_analyzer
)


def test_enhanced_ff_model():
    """Test Enhanced Fitness-Fatigue model."""
    print("\n" + "="*60)
    print("Testing Enhanced Fitness-Fatigue Model")
    print("="*60)

    # Initialize model with research-based parameters
    model = EnhancedFitnessFatigueModel(
        k1=0.15, k2=0.13, tau1=42, tau2=7, p_star=0.2
    )

    # Test with sample training loads
    training_loads = np.array([
        100, 0, 80, 0, 120, 0, 0,  # Week 1: alternating with rest
        90, 0, 100, 0, 110, 0, 0,   # Week 2: progressive
        60, 40, 20, 0, 0, 0, 0,     # Week 3: taper
    ])
    t = np.arange(len(training_loads))

    # Calculate impulse response
    fitness, fatigue, performance = model.impulse_response(training_loads, t)

    print(f"Initial performance: {performance[0]:.2f}")
    print(f"Peak fitness: {np.max(fitness):.2f} on day {np.argmax(fitness)}")
    print(f"Peak fatigue: {np.max(fatigue):.2f} on day {np.argmax(fatigue)}")
    print(f"Final performance: {performance[-1]:.2f}")
    print(f"Peak performance: {np.max(performance):.2f} on day {np.argmax(performance)}")

    # Verify supercompensation effect
    rest_days = np.where(training_loads == 0)[0]
    if len(rest_days) > 0:
        performance_after_rest = performance[rest_days[rest_days > 0]]
        if len(performance_after_rest) > 0:
            print(f"Average performance after rest: {np.mean(performance_after_rest):.2f}")

    return model


def test_perpot_model():
    """Test PerPot model with overtraining detection."""
    print("\n" + "="*60)
    print("Testing PerPot Model")
    print("="*60)

    model = PerPotModel(ds=7.0, dr=6.0, dso=1.5, pp0=0.2)

    # Test with progressively increasing loads
    training_loads = np.array([
        50, 60, 70, 80, 90, 100, 110,  # Increasing load
        120, 130, 140, 150, 160, 170, 180,  # Pushing toward overtraining
    ])
    t = np.arange(len(training_loads))

    results = model.simulate(training_loads, t)

    print(f"Initial performance potential: {results['performance_potential'][0]:.2f}")
    print(f"Final performance potential: {results['performance_potential'][-1]:.2f}")
    print(f"Peak performance: {np.max(results['performance_potential']):.2f}")

    # Check for overtraining
    overtraining_days = np.where(results['overtraining_risk'])[0]
    if len(overtraining_days) > 0:
        print(f"⚠️ Overtraining detected on days: {overtraining_days}")
        print(f"First overtraining on day {overtraining_days[0]} with load {training_loads[overtraining_days[0]]}")
    else:
        print("✓ No overtraining detected")

    return model


def test_optimal_control():
    """Test optimal control problem solver."""
    print("\n" + "="*60)
    print("Testing Optimal Control Problem Solver")
    print("="*60)

    model = EnhancedFitnessFatigueModel()

    # Test different training goals
    goals = [
        TrainingGoal.MAXIMIZE_FTI,
        TrainingGoal.MAXIMIZE_FATIGUE,
        TrainingGoal.MINIMIZE_FATIGUE,
        TrainingGoal.BALANCED
    ]

    for goal in goals:
        print(f"\nOptimizing for: {goal.value}")

        problem = OptimalControlProblem(
            goal=goal,
            duration_days=7,
            min_load_threshold=30.0,
            max_daily_load=150.0,
            rest_days=[6],  # Rest on Sunday
            constraints={'target_fti': 500} if goal == TrainingGoal.MINIMIZE_FATIGUE else {}
        )

        result = model.optimize_training_plan(
            problem,
            initial_fitness=30.0,
            initial_fatigue=20.0
        )

        if result['success']:
            print(f"  Optimal loads: {result['loads'].round(1)}")
            print(f"  Total FTI: {np.sum(result['loads']):.1f}")
            print(f"  Max fatigue: {np.max(result['fatigue']):.1f}")
            print(f"  Final performance: {result['performance'][-1]:.2f}")
        else:
            print("  Optimization failed!")

    return model


def test_integrated_analyzer():
    """Test integrated training analyzer."""
    print("\n" + "="*60)
    print("Testing Integrated Training Analyzer")
    print("="*60)

    analyzer = get_integrated_analyzer("test_user")

    # Generate optimal weekly plan
    print("\nGenerating optimal weekly plan:")
    plan = analyzer.generate_optimal_plan(
        goal='balanced',
        duration_days=7,
        rest_days=[6]  # Sunday rest
    )

    if plan['success']:
        print("\nDaily Training Plan:")
        print("-" * 40)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, detail in enumerate(plan['daily_details']):
            print(f"{days[i]}: {detail['recommendation']:8s} - "
                  f"Load: {detail['load']:5.1f}, "
                  f"Form: {detail['predicted_form']:5.1f}")

        print(f"\nTotal weekly load: {np.sum(plan['loads']):.1f}")
        print(f"Expected fitness gain: {plan['fitness'][-1] - plan['fitness'][0]:.1f}")
    else:
        print("Failed to generate plan")

    return analyzer


def test_weekly_plan_generation():
    """Test weekly plan generation with different goals."""
    print("\n" + "="*60)
    print("Testing Weekly Plan Generation")
    print("="*60)

    # Test with different starting conditions
    scenarios = [
        {"fitness": 60, "fatigue": 20, "goal": TrainingGoal.MAXIMIZE_FTI, "name": "Build Phase"},
        {"fitness": 80, "fatigue": 60, "goal": TrainingGoal.MINIMIZE_FATIGUE, "name": "Recovery Phase"},
        {"fitness": 70, "fatigue": 30, "goal": TrainingGoal.BALANCED, "name": "Maintenance Phase"},
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Starting: Fitness={scenario['fitness']}, Fatigue={scenario['fatigue']}")

        plan = generate_optimal_weekly_plan(
            current_fitness=scenario['fitness'],
            current_fatigue=scenario['fatigue'],
            goal=scenario['goal'],
            rest_days=[6]  # Sunday rest
        )

        if plan['success']:
            print(f"  Recommendations: {', '.join(plan['recommendations'])}")
            print(f"  Total load: {np.sum(plan['loads']):.1f}")
            print(f"  Final form: {plan['performance'][-1]:.2f}")
        else:
            print("  Plan generation failed")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ADVANCED TRAINING MODELS TEST SUITE")
    print("#"*60)

    try:
        # Test individual components
        ff_model = test_enhanced_ff_model()
        perpot_model = test_perpot_model()
        oc_model = test_optimal_control()

        # Test integrated system
        analyzer = test_integrated_analyzer()

        # Test weekly plan generation
        test_weekly_plan_generation()

        print("\n" + "#"*60)
        print("# ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("#"*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())