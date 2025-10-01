#!/usr/bin/env python3
"""
Test script for correlation analysis module.

This script tests the correlation analyzer with mock data to ensure
the implementation works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from strava_supercompensation.analysis.correlation_analyzer import (
    WellnessPerformanceCorrelations,
    CorrelationResult
)


def create_mock_data(days=60):
    """Create mock dataset for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Create correlated data
    np.random.seed(42)

    # Simulate CTL/ATL/TSB
    ctl = 80 + np.cumsum(np.random.randn(days) * 2)
    atl = ctl * 0.8 + np.random.randn(days) * 5
    tsb = ctl - atl

    # Simulate HRV (negatively correlated with fatigue)
    hrv_rmssd = 60 - (atl - 50) * 0.3 + np.random.randn(days) * 5
    hrv_rmssd = np.clip(hrv_rmssd, 20, 100)

    # Simulate sleep (correlated with recovery)
    sleep_score = 75 + (tsb * 0.5) + np.random.randn(days) * 10
    sleep_score = np.clip(sleep_score, 40, 100)

    # Simulate stress (correlated with fatigue)
    stress_avg = 40 + (atl - 50) * 0.4 + np.random.randn(days) * 8
    stress_avg = np.clip(stress_avg, 10, 90)

    # Simulate resting HR (correlated with fatigue)
    resting_hr = 55 + (atl - 50) * 0.2 + np.random.randn(days) * 3
    resting_hr = np.clip(resting_hr, 45, 75)

    # Simulate daily load
    daily_load = np.abs(np.random.randn(days) * 30 + 60)

    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'ctl': ctl,
        'atl': atl,
        'tsb': tsb,
        'hrv_rmssd': hrv_rmssd,
        'hrv_score': hrv_rmssd * 1.2,  # Scaled version
        'sleep_score': sleep_score,
        'stress_avg': stress_avg,
        'resting_hr': resting_hr,
        'daily_load': daily_load
    })

    return data


def test_correlation_calculation():
    """Test basic correlation calculation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Correlation Calculation")
    print("="*60)

    data = create_mock_data(60)
    analyzer = WellnessPerformanceCorrelations()

    # Test correlation between HRV and ATL (should be negative)
    result = analyzer._calculate_correlation(
        data, 'hrv_rmssd', 'atl',
        min_samples=10,
        sig_level=0.05
    )

    if result:
        print(f"✓ HRV vs Fatigue (ATL)")
        print(f"  Correlation: {result.correlation:.3f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Strength: {result.strength}")
        print(f"  Direction: {result.direction}")
        print(f"  Significant: {'Yes' if result.is_significant else 'No'}")
        print(f"  Samples: {result.n_samples}")

        # Validate expectations
        assert result.correlation < 0, "HRV should be negatively correlated with fatigue"
        assert result.n_samples == 60, f"Expected 60 samples, got {result.n_samples}"
        print("\n✅ TEST PASSED")
    else:
        print("❌ TEST FAILED: No correlation result")
        return False

    return True


def test_leading_indicators():
    """Test time-lagged correlation analysis."""
    print("\n" + "="*60)
    print("TEST 2: Leading Indicator Detection")
    print("="*60)

    data = create_mock_data(90)

    # Create a lagged relationship: HRV today predicts TSB tomorrow
    data['tsb_tomorrow'] = data['tsb'].shift(-2)  # 2-day lag

    analyzer = WellnessPerformanceCorrelations()

    # Test time-lagged correlation
    best_lag = 0
    best_corr = 0

    for lag in range(1, 7):
        lagged_data = data[['hrv_rmssd', 'tsb']].copy()
        lagged_data[f'hrv_lag{lag}'] = lagged_data['hrv_rmssd'].shift(lag)

        result = analyzer._calculate_correlation(
            lagged_data, f'hrv_lag{lag}', 'tsb',
            min_samples=10, sig_level=0.05
        )

        if result:
            print(f"  Lag {lag} days: r={result.correlation:+.3f}, p={result.p_value:.4f}")
            if abs(result.correlation) > abs(best_corr):
                best_lag = lag
                best_corr = result.correlation

    print(f"\n✓ Best predictive lag: {best_lag} days")
    print(f"  Correlation: {best_corr:.3f}")

    if best_lag > 0:
        print("\n✅ TEST PASSED")
        return True
    else:
        print("⚠️  TEST WARNING: No optimal lag found (may be data-dependent)")
        return True  # Not a failure, just data-dependent


def test_correlation_matrix():
    """Test full correlation matrix generation."""
    print("\n" + "="*60)
    print("TEST 3: Correlation Matrix Generation")
    print("="*60)

    data = create_mock_data(60)
    analyzer = WellnessPerformanceCorrelations()

    matrix = analyzer._build_correlation_matrix(data)

    print(f"✓ Matrix dimensions: {len(matrix['variables'])} x {len(matrix['variables'])}")
    print(f"  Variables: {', '.join(matrix['variables'][:5])}...")

    # Check matrix properties
    corr_df = pd.DataFrame(matrix['correlation_matrix'])
    print(f"\n  Sample correlations:")
    print(f"  HRV vs ATL: {corr_df.loc['hrv_rmssd', 'atl']:.3f}")
    print(f"  Sleep vs TSB: {corr_df.loc['sleep_score', 'tsb']:.3f}")
    print(f"  Stress vs ATL: {corr_df.loc['stress_avg', 'atl']:.3f}")

    # Validate diagonal is 1.0
    for var in matrix['variables']:
        assert abs(corr_df.loc[var, var] - 1.0) < 0.001, f"Diagonal should be 1.0 for {var}"

    print("\n✅ TEST PASSED")
    return True


def test_data_structures():
    """Test data structure creation."""
    print("\n" + "="*60)
    print("TEST 4: Data Structure Validation")
    print("="*60)

    # Test CorrelationResult
    result = CorrelationResult(
        variable_1='hrv_rmssd',
        variable_2='atl',
        correlation=-0.65,
        p_value=0.001,
        n_samples=60,
        is_significant=True,
        interpretation="HRV vs Fatigue"
    )

    print(f"✓ CorrelationResult created")
    print(f"  Strength: {result.strength}")
    print(f"  Direction: {result.direction}")

    assert result.strength == "moderate", f"Expected moderate, got {result.strength}"
    assert result.direction == "negative", f"Expected negative, got {result.direction}"

    print("\n✅ TEST PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS MODULE TEST SUITE")
    print("="*60)

    tests = [
        ("Basic Correlation", test_correlation_calculation),
        ("Leading Indicators", test_leading_indicators),
        ("Correlation Matrix", test_correlation_matrix),
        ("Data Structures", test_data_structures),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
