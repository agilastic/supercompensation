"""Tests for sports metrics calculations."""

import pytest
import numpy as np
from strava_supercompensation.analysis.sports_metrics import (
    SportsMetricsCalculator,
    TrainingZone,
    RecoveryMetrics
)


class TestSportsMetricsCalculator:
    """Test sports metrics calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SportsMetricsCalculator(athlete_max_hr=180, athlete_ftp=250)

    def test_estimate_max_hr(self):
        """Test max HR estimation."""
        # Test Tanaka formula
        assert self.calculator.estimate_max_hr(30) == 187
        assert self.calculator.estimate_max_hr(40) == 180
        assert self.calculator.estimate_max_hr(50) == 173

    def test_calculate_trimp(self):
        """Test TRIMP calculation."""
        # Test basic TRIMP calculation
        trimp = self.calculator.calculate_trimp(
            duration_minutes=60,
            avg_hr=150,
            max_hr=180,
            gender="male"
        )
        assert trimp > 0
        assert trimp < 200  # Reasonable range for 60 min moderate intensity

        # Test gender difference
        trimp_female = self.calculator.calculate_trimp(
            duration_minutes=60,
            avg_hr=150,
            max_hr=180,
            gender="female"
        )
        assert trimp_female != trimp

    def test_calculate_tss(self):
        """Test TSS calculation."""
        # Test with FTP
        tss = self.calculator.calculate_tss(
            duration_seconds=3600,  # 1 hour
            normalized_power=250,    # At FTP
            intensity_factor=1.0
        )
        assert tss == 100  # 1 hour at FTP = 100 TSS

        # Test harder effort
        tss_hard = self.calculator.calculate_tss(
            duration_seconds=1800,  # 30 min
            normalized_power=275,    # 110% FTP
            intensity_factor=1.1
        )
        assert tss_hard > 50
        assert tss_hard < 70

    def test_calculate_hrss(self):
        """Test HRSS calculation."""
        hrss = self.calculator.calculate_hrss(
            duration_minutes=60,
            avg_hr=153,  # ~85% of 180 max HR
            threshold_hr=153
        )
        assert hrss == 100  # 1 hour at threshold = 100 HRSS

    def test_get_hr_zone(self):
        """Test heart rate zone determination."""
        # Test various HR zones
        assert self.calculator.get_hr_zone(90, 180) == TrainingZone.RECOVERY
        assert self.calculator.get_hr_zone(115, 180) == TrainingZone.AEROBIC
        assert self.calculator.get_hr_zone(135, 180) == TrainingZone.TEMPO
        assert self.calculator.get_hr_zone(155, 180) == TrainingZone.THRESHOLD
        assert self.calculator.get_hr_zone(165, 180) == TrainingZone.VO2MAX

    def test_calculate_zone_distribution(self):
        """Test zone distribution calculation."""
        # Create sample HR data
        hr_data = [
            100, 100, 100,  # Recovery
            120, 120,        # Aerobic
            140,             # Tempo
            155,             # Threshold
            165              # VO2max
        ]

        distribution = self.calculator.calculate_zone_distribution(hr_data, 180)

        assert TrainingZone.RECOVERY in distribution
        assert distribution[TrainingZone.RECOVERY] == pytest.approx(37.5, rel=0.1)
        assert distribution[TrainingZone.AEROBIC] == pytest.approx(25.0, rel=0.1)

    def test_calculate_acwr(self):
        """Test Acute:Chronic Workload Ratio calculation."""
        # Test normal ratio
        acute = [100, 110, 120, 100, 90, 100, 110]  # Last 7 days
        chronic = acute * 4  # Last 28 days (repeated for simplicity)

        acwr = self.calculator.calculate_acute_chronic_workload_ratio(acute, chronic)
        assert acwr == pytest.approx(1.0, rel=0.1)

        # Test high ratio (injury risk)
        acute_high = [150, 160, 170, 150, 140, 150, 160]
        acwr_high = self.calculator.calculate_acute_chronic_workload_ratio(acute_high, chronic)
        assert acwr_high > 1.3  # Danger zone

    def test_calculate_training_monotony(self):
        """Test training monotony calculation."""
        # Test varied training
        varied_loads = [50, 100, 30, 120, 40, 150, 20]
        monotony_varied = self.calculator.calculate_training_monotony(varied_loads)
        assert monotony_varied < 2.0  # Good variation

        # Test monotonous training
        monotonous_loads = [100, 105, 98, 102, 99, 103, 101]
        monotony_high = self.calculator.calculate_training_monotony(monotonous_loads)
        assert monotony_high > 5.0  # High monotony

    def test_calculate_training_strain(self):
        """Test training strain calculation."""
        weekly_load = 700
        monotony = 2.5

        strain = self.calculator.calculate_training_strain(weekly_load, monotony)
        assert strain == 1750

    def test_estimate_vo2max(self):
        """Test VO2max estimation."""
        # Test 5k pace (4:00/km = 4.17 m/s)
        vo2max = self.calculator.estimate_vo2max(
            pace_ms=4.17,
            duration_minutes=20  # ~5k time
        )
        assert vo2max > 45
        assert vo2max < 65

    def test_calculate_efficiency_factor(self):
        """Test Efficiency Factor calculation."""
        ef = self.calculator.calculate_efficiency_factor(
            normalized_power=200,
            avg_hr=150
        )
        assert ef == pytest.approx(1.33, rel=0.01)

    def test_calculate_cardiac_drift(self):
        """Test cardiac drift calculation."""
        # Test positive drift (dehydration/fatigue)
        drift = self.calculator.calculate_cardiac_drift(
            first_half_hr=140,
            second_half_hr=150,
            first_half_pace=4.0,
            second_half_pace=3.9  # Slight slowdown
        )
        assert drift > 5  # Significant drift

        # Test minimal drift (good fitness)
        drift_low = self.calculator.calculate_cardiac_drift(
            first_half_hr=140,
            second_half_hr=142,
            first_half_pace=4.0,
            second_half_pace=4.0  # Consistent pace
        )
        assert drift_low < 3  # Minimal drift


class TestRecoveryMetrics:
    """Test recovery metrics calculations."""

    def test_calculate_hrv_stress(self):
        """Test HRV stress assessment."""
        # Test low stress (good recovery)
        stress = RecoveryMetrics.calculate_hrv_stress(
            hrv_today=55,
            hrv_baseline=50,
            hrv_std=5
        )
        assert stress == "low"

        # Test high stress (poor recovery)
        stress_high = RecoveryMetrics.calculate_hrv_stress(
            hrv_today=40,
            hrv_baseline=50,
            hrv_std=5
        )
        assert stress_high == "high"

        # Test moderate stress
        stress_mod = RecoveryMetrics.calculate_hrv_stress(
            hrv_today=48,
            hrv_baseline=50,
            hrv_std=5
        )
        assert stress_mod == "moderate"

    def test_calculate_recovery_score(self):
        """Test overall recovery score calculation."""
        # Test good recovery
        score = RecoveryMetrics.calculate_recovery_score(
            sleep_hours=8,
            hrv_cv=0.08,
            resting_hr=45,
            baseline_rhr=50,
            subjective_score=8
        )
        assert score > 70  # Good recovery

        # Test poor recovery
        score_poor = RecoveryMetrics.calculate_recovery_score(
            sleep_hours=5,
            hrv_cv=0.03,
            resting_hr=60,
            baseline_rhr=50,
            subjective_score=4
        )
        assert score_poor < 50  # Poor recovery

        # Test with missing data
        score_partial = RecoveryMetrics.calculate_recovery_score(
            sleep_hours=7,
            hrv_cv=0,  # Missing HRV
            resting_hr=50,
            baseline_rhr=50,
            subjective_score=None  # Missing subjective
        )
        assert score_partial >= 0
        assert score_partial <= 100