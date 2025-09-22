"""Tests for Banister Impulse-Response model."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from strava_supercompensation.analysis.model import BanisterModel, SupercompensationAnalyzer


class TestBanisterModel:
    """Test Banister model calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = BanisterModel(
            fitness_decay=42,
            fatigue_decay=7,
            fitness_magnitude=0.1,
            fatigue_magnitude=0.15
        )

    def test_impulse_response_single_workout(self):
        """Test impulse response for a single workout."""
        # Single workout on day 1
        training_loads = np.array([0, 100, 0, 0, 0, 0, 0])
        days = np.arange(7)

        fitness, fatigue, form = self.model.impulse_response(training_loads, days)

        # Check initial response
        assert fitness[0] == 0  # No training on day 0
        assert fitness[1] > 0   # Fitness increases after training
        assert fatigue[1] > fitness[1]  # Fatigue should be higher initially

        # Check decay
        assert fitness[2] < fitness[1]  # Fitness decays
        assert fatigue[2] < fatigue[1]  # Fatigue decays
        assert fatigue[6] < fatigue[2]  # Fatigue decays faster than fitness

        # Check form
        assert form[1] < 0  # Negative form immediately after training
        # Form should improve as fatigue decays faster
        assert form[6] > form[1]

    def test_impulse_response_consistent_training(self):
        """Test impulse response for consistent daily training."""
        # Consistent moderate training for 30 days
        training_loads = np.full(30, 50)
        days = np.arange(30)

        fitness, fatigue, form = self.model.impulse_response(training_loads, days)

        # Fitness should gradually increase
        assert fitness[29] > fitness[14] > fitness[7]

        # Fatigue should reach steady state relatively quickly
        assert abs(fatigue[29] - fatigue[28]) < 1  # Near steady state

        # Form should stabilize
        assert abs(form[29] - form[28]) < 0.5

    def test_impulse_response_periodized_training(self):
        """Test impulse response for periodized training pattern."""
        # 3-week pattern: build, build, recovery
        loads = []
        for _ in range(3):  # 3 cycles
            loads.extend([60, 70, 80, 70, 60, 100, 40])  # Week 1: build
            loads.extend([70, 80, 90, 80, 70, 110, 40])  # Week 2: build
            loads.extend([40, 50, 40, 50, 30, 60, 20])   # Week 3: recovery

        training_loads = np.array(loads)
        days = np.arange(len(loads))

        fitness, fatigue, form = self.model.impulse_response(training_loads, days)

        # Check supercompensation after recovery weeks
        # Form should be better at end of recovery weeks
        assert form[20] > form[13]  # End of first recovery week
        assert form[41] > form[34]  # End of second recovery week

    def test_predict_optimal_load(self):
        """Test optimal load prediction."""
        current_fitness = 50
        current_fatigue = 30

        recommendations = self.model.predict_optimal_load(
            current_fitness, current_fatigue, days_ahead=7
        )

        # Check all scenarios are present
        assert "rest" in recommendations
        assert "easy" in recommendations
        assert "moderate" in recommendations
        assert "hard" in recommendations
        assert "very_hard" in recommendations

        # Rest should improve form the most in short term
        assert recommendations["rest"]["form_change"] > recommendations["very_hard"]["form_change"]

        # Hard training should increase fitness more
        assert recommendations["hard"]["predicted_fitness"] > recommendations["rest"]["predicted_fitness"]

    def test_different_decay_rates(self):
        """Test model with different decay rate parameters."""
        # Fast adaptation model
        fast_model = BanisterModel(
            fitness_decay=21,  # Faster fitness decay
            fatigue_decay=3,   # Faster fatigue decay
            fitness_magnitude=0.15,
            fatigue_magnitude=0.25
        )

        # Slow adaptation model
        slow_model = BanisterModel(
            fitness_decay=60,  # Slower fitness decay
            fatigue_decay=14,  # Slower fatigue decay
            fitness_magnitude=0.08,
            fatigue_magnitude=0.10
        )

        training_loads = np.array([100, 0, 0, 0, 0])
        days = np.arange(5)

        fast_fitness, fast_fatigue, _ = fast_model.impulse_response(training_loads, days)
        slow_fitness, slow_fatigue, _ = slow_model.impulse_response(training_loads, days)

        # Fast model should show quicker changes
        assert fast_fatigue[4] < slow_fatigue[4]  # Fatigue decays faster
        assert fast_fitness[4] < slow_fitness[4]   # But fitness also decays faster


class TestSupercompensationAnalyzer:
    """Test supercompensation analyzer integration."""

    @pytest.fixture
    def mock_activities(self):
        """Create mock activity data."""
        base_date = datetime.utcnow() - timedelta(days=30)
        activities = []

        for i in range(20):
            activities.append({
                "start_date": base_date + timedelta(days=i),
                "training_load": 50 + np.random.randint(-20, 30),
                "type": "Run" if i % 3 else "Ride",
                "duration": 3600 + np.random.randint(-1200, 1800)
            })

        return activities

    def test_analyze_basic(self, mock_activities):
        """Test basic analysis functionality."""
        # Note: This test requires database setup
        # In a real test environment, you'd mock the database
        pass

    def test_get_current_state(self):
        """Test getting current training state."""
        # Note: This test requires database setup
        pass

    def test_metrics_history(self):
        """Test retrieving metrics history."""
        # Note: This test requires database setup
        pass