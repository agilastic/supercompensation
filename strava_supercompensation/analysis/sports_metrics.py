"""Advanced sports science metrics and calculations."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TrainingZone(Enum):
    """Heart rate training zones based on percentage of max HR."""

    RECOVERY = (0.50, 0.60)  # 50-60% of max HR
    AEROBIC = (0.60, 0.70)   # 60-70% of max HR
    TEMPO = (0.70, 0.80)     # 70-80% of max HR
    THRESHOLD = (0.80, 0.90) # 80-90% of max HR
    VO2MAX = (0.90, 0.95)    # 90-95% of max HR
    NEUROMUSCULAR = (0.95, 1.05)  # 95-105% of max HR


class SportsMetricsCalculator:
    """Calculate advanced sports science metrics."""

    def __init__(self, athlete_max_hr: Optional[int] = None, athlete_ftp: Optional[float] = None):
        """Initialize with athlete-specific parameters.

        Args:
            athlete_max_hr: Maximum heart rate (if None, estimated from age)
            athlete_ftp: Functional Threshold Power for cycling (watts)
        """
        self.max_hr = athlete_max_hr
        self.ftp = athlete_ftp

    def estimate_max_hr(self, age: int) -> int:
        """Estimate max HR using Tanaka formula: 208 - 0.7 * age."""
        return int(208 - 0.7 * age)

    def calculate_trimp(self, duration_minutes: float, avg_hr: float, max_hr: Optional[float] = None,
                       gender: str = "male") -> float:
        """Calculate Training Impulse (TRIMP) using Bannister's method.

        TRIMP = duration * HR_ratio * weight_factor
        where HR_ratio = (HR_avg - HR_rest) / (HR_max - HR_rest)

        Args:
            duration_minutes: Exercise duration in minutes
            avg_hr: Average heart rate during exercise
            max_hr: Maximum heart rate (uses stored value if not provided)
            gender: "male" or "female" for different weighting factors

        Returns:
            TRIMP value
        """
        if max_hr is None:
            max_hr = self.max_hr or 180  # Default fallback

        # Assume resting HR of 60 if not known
        rest_hr = 60

        # Calculate heart rate ratio
        hr_ratio = (avg_hr - rest_hr) / (max_hr - rest_hr)
        hr_ratio = max(0, min(1, hr_ratio))  # Clamp between 0 and 1

        # Gender-specific weighting factor
        if gender.lower() == "female":
            weight = 1.67
        else:
            weight = 1.92

        # TRIMP calculation using exponential weighting
        trimp = duration_minutes * hr_ratio * (0.64 * np.exp(weight * hr_ratio))

        return trimp

    def calculate_tss(self, duration_seconds: float, normalized_power: float,
                     intensity_factor: Optional[float] = None) -> float:
        """Calculate Training Stress Score for cycling.

        TSS = (duration_seconds * NP * IF) / (FTP * 3600) * 100

        Args:
            duration_seconds: Duration in seconds
            normalized_power: Normalized Power in watts
            intensity_factor: IF = NP / FTP (calculated if not provided)

        Returns:
            TSS value
        """
        if self.ftp is None:
            # Estimate TSS using time and estimated intensity
            return self._estimate_tss_from_duration(duration_seconds, intensity_factor)

        if intensity_factor is None:
            intensity_factor = normalized_power / self.ftp

        tss = (duration_seconds * normalized_power * intensity_factor) / (self.ftp * 3600) * 100

        return tss

    def _estimate_tss_from_duration(self, duration_seconds: float,
                                   intensity_factor: Optional[float] = None) -> float:
        """Estimate TSS when power data is not available."""
        duration_hours = duration_seconds / 3600

        if intensity_factor is None:
            intensity_factor = 0.75  # Assume moderate intensity

        # Rough TSS estimation: 100 TSS = 1 hour at FTP
        tss = (intensity_factor ** 2) * duration_hours * 100

        return tss

    def calculate_hrss(self, duration_minutes: float, avg_hr: float,
                      threshold_hr: Optional[float] = None) -> float:
        """Calculate Heart Rate Stress Score (HRSS).

        Similar to TSS but based on heart rate instead of power.

        Args:
            duration_minutes: Duration in minutes
            avg_hr: Average heart rate
            threshold_hr: Lactate threshold heart rate (estimated if not provided)

        Returns:
            HRSS value
        """
        if threshold_hr is None:
            # Estimate as 85% of max HR
            threshold_hr = (self.max_hr or 180) * 0.85

        # Calculate intensity relative to threshold
        intensity = avg_hr / threshold_hr

        # HRSS calculation similar to TSS
        hrss = (duration_minutes / 60) * (intensity ** 2) * 100

        return hrss

    def get_hr_zone(self, hr: float, max_hr: Optional[float] = None) -> TrainingZone:
        """Determine training zone based on heart rate.

        Args:
            hr: Current heart rate
            max_hr: Maximum heart rate

        Returns:
            Training zone
        """
        if max_hr is None:
            max_hr = self.max_hr or 180

        hr_percentage = hr / max_hr

        for zone in TrainingZone:
            low, high = zone.value
            if low <= hr_percentage < high:
                return zone

        return TrainingZone.RECOVERY

    def calculate_zone_distribution(self, hr_data: List[float],
                                   max_hr: Optional[float] = None) -> Dict[TrainingZone, float]:
        """Calculate time spent in each training zone.

        Args:
            hr_data: List of heart rate measurements
            max_hr: Maximum heart rate

        Returns:
            Dictionary of zone -> percentage of time
        """
        if not hr_data:
            return {}

        zone_counts = {zone: 0 for zone in TrainingZone}

        for hr in hr_data:
            zone = self.get_hr_zone(hr, max_hr)
            zone_counts[zone] += 1

        total = len(hr_data)
        zone_distribution = {
            zone: (count / total) * 100
            for zone, count in zone_counts.items()
        }

        return zone_distribution

    def calculate_acute_chronic_workload_ratio(self,
                                              acute_loads: List[float],
                                              chronic_loads: List[float]) -> float:
        """Calculate Acute:Chronic Workload Ratio (ACWR).

        ACWR is used to assess injury risk:
        - < 0.8: Undertrained (higher injury risk)
        - 0.8-1.3: Optimal range
        - > 1.5: Danger zone (high injury risk)

        Args:
            acute_loads: Last 7 days of training loads
            chronic_loads: Last 28 days of training loads

        Returns:
            ACWR value
        """
        if not chronic_loads:
            return 1.0

        acute_avg = np.mean(acute_loads[-7:]) if acute_loads else 0
        chronic_avg = np.mean(chronic_loads)

        if chronic_avg == 0:
            return 0 if acute_avg == 0 else 2.0  # Cap at 2.0 for safety

        acwr = acute_avg / chronic_avg

        return acwr

    def calculate_training_monotony(self, daily_loads: List[float]) -> float:
        """Calculate training monotony (lack of variation).

        High monotony (>2.0) combined with high load increases injury risk.

        Args:
            daily_loads: Daily training loads for a week

        Returns:
            Monotony value
        """
        if len(daily_loads) < 2:
            return 0

        mean_load = np.mean(daily_loads)
        std_load = np.std(daily_loads)

        if std_load == 0:
            return 10.0  # Very high monotony if no variation

        monotony = mean_load / std_load

        return monotony

    def calculate_training_strain(self, weekly_load: float, monotony: float) -> float:
        """Calculate training strain.

        Strain = Weekly Load × Monotony
        High strain (>6000) indicates increased injury/illness risk.

        Args:
            weekly_load: Total weekly training load
            monotony: Training monotony value

        Returns:
            Training strain
        """
        return weekly_load * monotony

    def estimate_vo2max(self, pace_ms: float, duration_minutes: float) -> float:
        """Estimate VO2max using Jack Daniels' formula.

        Args:
            pace_ms: Running pace in meters per second
            duration_minutes: Duration of maximal effort

        Returns:
            Estimated VO2max (ml/kg/min)
        """
        # Convert pace to minutes per kilometer
        pace_min_km = (1000 / pace_ms) / 60

        # Estimate based on race performance (simplified)
        # This is a rough approximation
        if duration_minutes <= 11:  # ~3km effort
            vo2max = 133.61 - 4.85 * pace_min_km
        elif duration_minutes <= 30:  # ~10km effort
            vo2max = 120.62 - 3.60 * pace_min_km
        else:  # Marathon pace
            vo2max = 112.28 - 2.95 * pace_min_km

        return max(20, min(80, vo2max))  # Reasonable bounds

    def calculate_efficiency_factor(self, normalized_power: float, avg_hr: float) -> float:
        """Calculate Efficiency Factor (EF) for cycling.

        EF = Normalized Power / Average Heart Rate
        Higher values indicate better aerobic fitness.

        Args:
            normalized_power: NP in watts
            avg_hr: Average heart rate

        Returns:
            Efficiency factor
        """
        if avg_hr == 0:
            return 0

        return normalized_power / avg_hr

    def calculate_cardiac_drift(self, first_half_hr: float, second_half_hr: float,
                              first_half_pace: float, second_half_pace: float) -> float:
        """Calculate cardiac drift percentage.

        Cardiac drift indicates aerobic fitness and heat adaptation.
        Lower values (<5%) indicate good fitness.

        Args:
            first_half_hr: Average HR in first half
            second_half_hr: Average HR in second half
            first_half_pace: Average pace in first half (m/s)
            second_half_pace: Average pace in second half (m/s)

        Returns:
            Cardiac drift percentage
        """
        if first_half_hr == 0 or first_half_pace == 0:
            return 0

        hr_increase = (second_half_hr - first_half_hr) / first_half_hr
        pace_decrease = (first_half_pace - second_half_pace) / first_half_pace

        # Adjust for any pace changes
        cardiac_drift = (hr_increase - pace_decrease) * 100

        return cardiac_drift


class RecoveryMetrics:
    """Calculate recovery-related metrics."""

    @staticmethod
    def calculate_hrv_stress(hrv_today: float, hrv_baseline: float,
                            hrv_std: float) -> str:
        """Assess stress level based on HRV.

        Args:
            hrv_today: Today's HRV measurement
            hrv_baseline: Personal baseline HRV (e.g., 30-day average)
            hrv_std: Standard deviation of baseline

        Returns:
            Stress level: "low", "moderate", "high"
        """
        if hrv_baseline == 0 or hrv_std == 0:
            return "unknown"

        # Calculate z-score
        z_score = (hrv_today - hrv_baseline) / hrv_std

        if z_score > 0.5:
            return "low"  # Good recovery
        elif z_score > -1:
            return "moderate"  # Normal
        else:
            return "high"  # Poor recovery

    @staticmethod
    def calculate_recovery_score(sleep_hours: float, hrv_cv: float,
                                resting_hr: float, baseline_rhr: float,
                                subjective_score: Optional[float] = None) -> float:
        """Calculate overall recovery score (0-100).

        Args:
            sleep_hours: Hours of sleep
            hrv_cv: HRV coefficient of variation
            resting_hr: Current resting heart rate
            baseline_rhr: Baseline resting heart rate
            subjective_score: Subjective recovery rating (1-10)

        Returns:
            Recovery score (0-100)
        """
        score = 0
        weights_used = 0

        # Sleep component (weight: 30%)
        if sleep_hours > 0:
            sleep_score = min(100, (sleep_hours / 8) * 100)
            score += sleep_score * 0.3
            weights_used += 0.3

        # HRV component (weight: 30%)
        if hrv_cv > 0:
            # Higher HRV CV is better (more variability)
            hrv_score = min(100, hrv_cv * 1000)  # Rough scaling
            score += hrv_score * 0.3
            weights_used += 0.3

        # Resting HR component (weight: 25%)
        if resting_hr > 0 and baseline_rhr > 0:
            # Lower RHR relative to baseline is better
            rhr_diff = baseline_rhr - resting_hr
            rhr_score = 50 + (rhr_diff * 5)  # ±10 bpm = ±50 points
            rhr_score = max(0, min(100, rhr_score))
            score += rhr_score * 0.25
            weights_used += 0.25

        # Subjective component (weight: 15%)
        if subjective_score is not None:
            subj_score = (subjective_score / 10) * 100
            score += subj_score * 0.15
            weights_used += 0.15

        # Normalize if not all components available
        if weights_used > 0:
            score = score / weights_used * 100

        return min(100, max(0, score))