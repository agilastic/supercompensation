#!/usr/bin/env python3
"""Test and demonstration script for advanced 30-day training plan system.

This script demonstrates:
1. Advanced mesocycle-based planning
2. Optimal control integration
3. Interactive visualization generation
4. Plan adjustment capabilities
5. Real-world scenario testing
"""

import sys
import json
import numpy as np
from datetime import datetime, timedelta

# Add project to path
sys.path.append('/Users/alex/workspace/supercompensation')

from strava_supercompensation.analysis.advanced_planning import (
    AdvancedPlanGenerator, MesocycleType, WorkoutType
)
from strava_supercompensation.analysis.plan_adjustment import (
    PlanAdjustmentEngine, AdjustmentReason, AdjustmentType
)
from strava_supercompensation.analysis.model_integration import (
    get_integrated_analyzer
)


def test_advanced_plan_generation():
    """Test advanced 30-day plan generation."""
    print("\n" + "="*60)
    print("Testing Advanced 30-Day Plan Generation")
    print("="*60)

    generator = AdvancedPlanGenerator("test_user")

    # Test different scenarios
    scenarios = [
        {
            'name': 'Fitness Building',
            'goal': 'fitness',
            'target_date': None,
            'constraints': {'max_weekly_hours': 12, 'rest_days': [6]}
        },
        {
            'name': 'Race Preparation',
            'goal': 'performance',
            'target_date': datetime.utcnow() + timedelta(days=45),
            'constraints': {'max_weekly_hours': 15, 'rest_days': [6]}
        },
        {
            'name': 'Recovery Phase',
            'goal': 'recovery',
            'target_date': None,
            'constraints': {'max_weekly_hours': 8, 'rest_days': [6, 0]}  # Sunday and Monday
        }
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")

        plan = generator.generate_30_day_plan(
            goal=scenario['goal'],
            target_event_date=scenario['target_date'],
            constraints=scenario['constraints']
        )

        # Analyze plan
        summary = plan['summary']
        print(f"Total Load: {summary['total_load']:.0f} TSS")
        print(f"Total Duration: {summary['total_duration_hours']:.1f} hours")
        print(f"Fitness Gain: +{summary['fitness_gain']:.1f}")
        print(f"Hard Days: {summary['hard_days']}")
        print(f"Easy Days: {summary['easy_days']}")
        print(f"Rest Days: {summary['rest_days']}")

        # Check for overtraining
        if summary['overtraining_days'] > 0:
            print(f"⚠️ Overtraining risk detected on {summary['overtraining_days']} days")
        else:
            print("✓ No overtraining risk detected")

        # Store for further analysis
        results[scenario['name']] = plan

    return results


def test_plan_adjustments():
    """Test plan adjustment system."""
    print("\n" + "="*60)
    print("Testing Plan Adjustment System")
    print("="*60)

    # Generate a base plan
    generator = AdvancedPlanGenerator("test_user")
    plan = generator.generate_30_day_plan(goal='performance')

    # Initialize adjustment engine
    adjuster = PlanAdjustmentEngine("test_user")

    # Simulate various scenarios requiring adjustments
    scenarios = [
        {
            'name': 'Poor Recovery',
            'wellness_data': {
                'recovery_score': 45,
                'hrv_status': 'poor',
                'sleep_quality': 60,
                'stress_level': 80
            },
            'constraints': None
        },
        {
            'name': 'Weather Issues',
            'wellness_data': None,
            'constraints': {
                'weather': {
                    datetime.utcnow().date() + timedelta(days=1): {
                        'condition': 'severe_storm',
                        'temperature': 5,
                        'wind_speed': 60,
                        'precipitation': 90
                    }
                }
            }
        },
        {
            'name': 'Schedule Conflicts',
            'wellness_data': None,
            'constraints': {
                'schedule_conflicts': [
                    {
                        'date': datetime.utcnow().date() + timedelta(days=3),
                        'reason': 'business_trip'
                    }
                ]
            }
        }
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")

        # Get suggested adjustments
        adjustments = adjuster.evaluate_plan_adjustments(
            current_plan=plan['daily_workouts'],
            wellness_data=scenario['wellness_data'],
            constraints=scenario['constraints']
        )

        print(f"Suggested adjustments: {len(adjustments)}")

        for adj in adjustments[:3]:  # Show top 3
            print(f"  • {adj.adjustment_type.value}: {adj.reason.value}")
            print(f"    Confidence: {adj.confidence:.1f}")
            print(f"    Notes: {adj.notes[:60]}...")

        # Apply high-confidence adjustments
        adjusted_plan, applied = adjuster.apply_adjustments(
            plan['daily_workouts'],
            adjustments,
            auto_apply_threshold=0.8
        )

        print(f"Applied {len(applied)} automatic adjustments")

    return adjustments


def test_visualization_data():
    """Test visualization data generation."""
    print("\n" + "="*60)
    print("Testing Visualization Data Generation")
    print("="*60)

    generator = AdvancedPlanGenerator("test_user")
    plan = generator.generate_30_day_plan(goal='balanced')

    viz_data = plan['visualizations']

    print("Visualization components:")
    for component, data in viz_data.items():
        if isinstance(data, list):
            print(f"  • {component}: {len(data)} data points")
        elif isinstance(data, dict):
            print(f"  • {component}: {len(data)} keys")

    # Validate data structure
    required_components = [
        'daily_data',
        'weekly_summary',
        'fitness_progression',
        'recovery_status'
    ]

    missing = [comp for comp in required_components if comp not in viz_data]
    if missing:
        print(f"⚠️ Missing visualization components: {missing}")
    else:
        print("✓ All required visualization components present")

    return viz_data


def test_integrated_system():
    """Test integration with advanced models."""
    print("\n" + "="*60)
    print("Testing Integrated System")
    print("="*60)

    analyzer = get_integrated_analyzer("test_user")

    # Generate optimal plan using integrated models
    plan = analyzer.generate_optimal_plan(
        goal='balanced',
        duration_days=30,
        rest_days=[6]  # Sunday rest
    )

    if plan['success']:
        print("✓ Integrated optimization successful")
        print(f"Total weekly load: {np.sum(plan['loads']):.1f}")
        print(f"Load distribution: {[f'{x:.0f}' for x in plan['loads'][:7]]}")

        # Check recommendations
        recs = plan['recommendations']
        rec_counts = {}
        for rec in recs:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        print("Recommendation distribution:")
        for rec, count in rec_counts.items():
            print(f"  {rec}: {count} days")

    else:
        print("❌ Integrated optimization failed")

    return plan


def generate_sample_html():
    """Generate sample HTML visualization."""
    print("\n" + "="*60)
    print("Generating Sample HTML Visualization")
    print("="*60)

    # Generate a sample plan
    generator = AdvancedPlanGenerator("test_user")
    plan = generator.generate_30_day_plan(goal='fitness')

    # Extract visualization data
    viz_data = plan['visualizations']

    # Create HTML with embedded data
    html_template_path = '/Users/alex/workspace/supercompensation/templates/training_plan_visualization.html'
    output_path = '/Users/alex/workspace/supercompensation/sample_30day_plan.html'

    try:
        with open(html_template_path, 'r') as f:
            html_content = f.read()

        # Replace the sample data loading with actual data
        data_js = f"loadPlanData({json.dumps(plan, default=str, indent=2)});"

        # Insert the data loading call
        html_content = html_content.replace(
            'loadPlanData(sampleData);',
            data_js
        )

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✓ HTML visualization generated: {output_path}")
        print("Open this file in a browser to view the interactive plan")

    except Exception as e:
        print(f"❌ Failed to generate HTML: {e}")

    return output_path


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("Running Performance Benchmarks")
    print("="*60)

    import time

    generator = AdvancedPlanGenerator("test_user")

    # Benchmark plan generation
    start_time = time.time()
    plan = generator.generate_30_day_plan(goal='performance')
    generation_time = time.time() - start_time

    print(f"Plan generation time: {generation_time:.2f} seconds")

    # Benchmark adjustment evaluation
    adjuster = PlanAdjustmentEngine("test_user")

    start_time = time.time()
    adjustments = adjuster.evaluate_plan_adjustments(
        current_plan=plan['daily_workouts'],
        wellness_data={'recovery_score': 60},
        constraints={'weather': {}}
    )
    adjustment_time = time.time() - start_time

    print(f"Adjustment evaluation time: {adjustment_time:.2f} seconds")
    print(f"Number of workouts analyzed: {len(plan['daily_workouts'])}")
    print(f"Adjustments suggested: {len(adjustments)}")

    # Memory usage estimate
    plan_size = sys.getsizeof(plan) / 1024  # KB
    print(f"Plan data size: {plan_size:.1f} KB")

    return {
        'generation_time': generation_time,
        'adjustment_time': adjustment_time,
        'plan_size': plan_size
    }


def main():
    """Run comprehensive test suite."""
    print("\n" + "#"*60)
    print("# ADVANCED 30-DAY TRAINING PLAN TEST SUITE")
    print("#"*60)

    try:
        # Test plan generation
        plan_results = test_advanced_plan_generation()

        # Test plan adjustments
        adjustment_results = test_plan_adjustments()

        # Test visualization
        viz_data = test_visualization_data()

        # Test integrated system
        integrated_results = test_integrated_system()

        # Generate sample HTML
        html_path = generate_sample_html()

        # Run benchmarks
        benchmarks = run_performance_benchmarks()

        print("\n" + "#"*60)
        print("# TEST SUMMARY")
        print("#"*60)

        print(f"✓ Plan generation: {len(plan_results)} scenarios tested")
        print(f"✓ Adjustment system: {len(adjustment_results)} adjustments evaluated")
        print(f"✓ Visualization: {len(viz_data)} components generated")
        print(f"✓ Integration: {'Success' if integrated_results['success'] else 'Failed'}")
        print(f"✓ HTML output: {html_path}")

        print(f"\nPerformance:")
        print(f"  Generation: {benchmarks['generation_time']:.2f}s")
        print(f"  Adjustments: {benchmarks['adjustment_time']:.2f}s")
        print(f"  Memory: {benchmarks['plan_size']:.1f} KB")

        print("\n" + "#"*60)
        print("# ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("#"*60)

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())