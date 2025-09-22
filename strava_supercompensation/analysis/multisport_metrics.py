"""Multi-sport specific training load calculations and recovery protocols."""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class SportType(Enum):
    """Sport-specific categories for training load calculations."""

    RUNNING = "running"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    ROWING = "rowing"
    SKATING = "skating"  # Inline skating
    HIKING = "hiking"
    OTHER = "other"


class RecoveryDemand(Enum):
    """Recovery demand levels for different sports."""

    LOW = 1      # Walking, yoga, easy hiking
    MODERATE = 2 # Swimming, easy cycling
    HIGH = 3     # Running, intense cycling, rowing
    VERY_HIGH = 4 # Racing, ultra-endurance


class MultiSportCalculator:
    """Calculate sport-specific training loads and recovery requirements."""

    def __init__(self):
        """Initialize with sport-specific parameters."""
        # Complete mapping of official Strava activity types
        self.sport_mapping = {
            # Running activities
            "run": SportType.RUNNING,
            "trailrun": SportType.RUNNING,
            "virtualrun": SportType.RUNNING,

            # Cycling activities
            "ride": SportType.CYCLING,
            "virtualride": SportType.CYCLING,
            "mountainbikeride": SportType.CYCLING,
            "gravelride": SportType.CYCLING,
            "ebikeride": SportType.CYCLING,
            "emountainbikeride": SportType.CYCLING,
            "handcycle": SportType.CYCLING,
            "velomobile": SportType.CYCLING,

            # Swimming activities
            "swim": SportType.SWIMMING,
            "openswim": SportType.SWIMMING,
            "kiteswim": SportType.SWIMMING,

            # Rowing activities
            "rowing": SportType.ROWING,
            "canoeing": SportType.ROWING,
            "kayaking": SportType.ROWING,
            "standuppaddling": SportType.ROWING,

            # Skating activities
            "inlineskate": SportType.SKATING,
            "iceskate": SportType.SKATING,
            "backcountryski": SportType.SKATING,
            "alpineski": SportType.SKATING,
            "nordicski": SportType.SKATING,
            "skateski": SportType.SKATING,
            "snowboard": SportType.SKATING,
            "snowshoe": SportType.SKATING,

            # Hiking/Walking activities
            "hike": SportType.HIKING,
            "walk": SportType.HIKING,
            "rockclimbing": SportType.HIKING,
            "climbing": SportType.HIKING,

            # Other activities
            "elliptical": SportType.OTHER,
            "stairstepper": SportType.OTHER,
            "crosstraining": SportType.OTHER,
            "workout": SportType.OTHER,
            "weighttraining": SportType.OTHER,
            "strengthtraining": SportType.OTHER,
            "hiit": SportType.OTHER,
            "pilates": SportType.OTHER,
            "yoga": SportType.OTHER,
            "barre": SportType.OTHER,
            "golf": SportType.OTHER,
            "hanggliding": SportType.OTHER,
            "kitesurf": SportType.OTHER,
            "paragliding": SportType.OTHER,
            "sailing": SportType.OTHER,
            "skateboard": SportType.OTHER,
            "soccer": SportType.OTHER,
            "surfing": SportType.OTHER,
            "tennis": SportType.OTHER,
            "volleyball": SportType.OTHER,
            "windsurf": SportType.OTHER,
            "badminton": SportType.OTHER,
            "cricket": SportType.OTHER,
            "tabletennis": SportType.OTHER,
            "handball": SportType.OTHER,
            "hockey": SportType.OTHER,
            "floorball": SportType.OTHER,
            "lacrosse": SportType.OTHER,
            "rugby": SportType.OTHER,
            "squash": SportType.OTHER,
            "basketball": SportType.OTHER,
            "americanfootball": SportType.OTHER,
            "baseball": SportType.OTHER,
            "bowling": SportType.OTHER,
            "boxing": SportType.OTHER,
            "martialarts": SportType.OTHER,
            "meditation": SportType.OTHER,
            "dancing": SportType.OTHER,
            "esports": SportType.OTHER,
            "wheelchair": SportType.OTHER,
        }

        # Sport-specific base metabolic cost (TSS per hour at moderate intensity)
        self.base_tss_per_hour = {
            SportType.RUNNING: 100,      # High impact, glycolytic
            SportType.CYCLING: 80,       # Lower impact, aerobic bias
            SportType.SWIMMING: 120,     # Full body, technique dependent
            SportType.ROWING: 110,       # Full body, high power (includes kayaking, SUP)
            SportType.SKATING: 85,       # Lower impact than running (includes skiing)
            SportType.HIKING: 40,        # Low intensity, long duration (includes climbing)
            SportType.OTHER: 70,         # General activities, strength training, sports
        }

        # Activity-specific adjustments within sport types
        self.activity_adjustments = {
            # Running variations
            "trailrun": 1.15,           # 15% higher due to terrain

            # Cycling variations
            "mountainbikeride": 1.2,    # 20% higher due to terrain/technique
            "gravelride": 1.1,          # 10% higher than road cycling
            "ebikeride": 0.7,           # 30% lower due to assistance
            "emountainbikeride": 0.8,   # 20% lower than regular MTB

            # Swimming variations
            "openswim": 1.1,            # 10% higher due to conditions
            "kiteswim": 1.3,            # 30% higher due to technique demands

            # Winter sports
            "backcountryski": 1.3,      # High intensity, full body
            "alpineski": 0.9,           # Lower than XC skiing
            "nordicski": 1.1,           # Classic endurance sport
            "snowboard": 0.95,          # Similar to alpine skiing
            "snowshoe": 1.2,            # Higher than hiking due to snow

            # Water sports
            "standuppaddling": 0.9,     # Lower than kayaking
            "kitesurf": 1.4,            # Very high intensity
            "windsurf": 1.2,            # High technique + endurance
            "sailing": 0.6,             # Lower intensity, more tactical

            # Strength/conditioning
            "weighttraining": 0.8,      # Anaerobic, lower TSS
            "strengthtraining": 0.8,    # Similar to weight training
            "hiit": 1.5,                # Very high intensity
            "crosstraining": 1.1,       # Mixed modalities

            # Low intensity
            "yoga": 0.4,                # Very low intensity
            "pilates": 0.5,             # Low-moderate intensity
            "meditation": 0.1,          # Minimal physical load
            "walk": 0.6,                # Lower than hiking

            # Sports (highly variable, using moderate estimates)
            "soccer": 1.3,              # High intermittent intensity
            "basketball": 1.4,          # Very high intermittent
            "tennis": 1.2,              # High intermittent
            "hockey": 1.5,              # Very high intensity
            "rugby": 1.4,               # High contact sport intensity
            "boxing": 1.6,              # Extremely high intensity
            "martialarts": 1.3,         # High intensity, technique
        }

        # Recovery demand by sport type
        self.recovery_demand = {
            SportType.RUNNING: RecoveryDemand.HIGH,      # High impact stress
            SportType.CYCLING: RecoveryDemand.MODERATE,  # Lower impact
            SportType.SWIMMING: RecoveryDemand.MODERATE, # Non-impact
            SportType.ROWING: RecoveryDemand.HIGH,       # High power output
            SportType.SKATING: RecoveryDemand.MODERATE,  # Medium impact
            SportType.HIKING: RecoveryDemand.LOW,        # Low intensity
            SportType.OTHER: RecoveryDemand.MODERATE,
        }

        # Sport-specific intensity zones (based on physiological responses)
        self.intensity_zones = {
            SportType.RUNNING: {
                "recovery": (0.50, 0.65),    # 50-65% HRmax
                "aerobic": (0.65, 0.75),     # 65-75% HRmax
                "tempo": (0.75, 0.85),       # 75-85% HRmax
                "threshold": (0.85, 0.92),   # 85-92% HRmax
                "vo2max": (0.92, 1.00),      # 92-100% HRmax
            },
            SportType.CYCLING: {
                "recovery": (0.55, 0.68),    # Slightly higher due to position
                "aerobic": (0.68, 0.78),
                "tempo": (0.78, 0.87),
                "threshold": (0.87, 0.94),
                "vo2max": (0.94, 1.00),
            },
            SportType.SWIMMING: {
                "recovery": (0.60, 0.70),    # Higher due to technique demands
                "aerobic": (0.70, 0.80),
                "tempo": (0.80, 0.88),
                "threshold": (0.88, 0.95),
                "vo2max": (0.95, 1.00),
            },
            SportType.ROWING: {
                "recovery": (0.55, 0.68),
                "aerobic": (0.68, 0.78),
                "tempo": (0.78, 0.87),
                "threshold": (0.87, 0.94),
                "vo2max": (0.94, 1.00),
            },
            SportType.SKATING: {
                "recovery": (0.55, 0.70),
                "aerobic": (0.70, 0.80),
                "tempo": (0.80, 0.88),
                "threshold": (0.88, 0.95),
                "vo2max": (0.95, 1.00),
            },
        }

    def get_sport_type(self, strava_activity_type: str) -> SportType:
        """Map Strava activity type to internal sport type."""
        return self.sport_mapping.get(strava_activity_type.lower(), SportType.OTHER)

    def calculate_sport_specific_load(self, activity_data: Dict) -> float:
        """Calculate training load with sport-specific adjustments."""
        activity_type = activity_data.get("type", "").lower()
        sport_type = self.get_sport_type(activity_type)

        # Use Strava's suffer_score if available (already sport-normalized)
        if activity_data.get("suffer_score"):
            return float(activity_data["suffer_score"])

        duration_hours = (activity_data.get("moving_time", 0) / 3600.0)
        base_tss = self.base_tss_per_hour[sport_type]

        # Activity-specific adjustment (e.g., trail running vs road running)
        activity_adjustment = self.activity_adjustments.get(activity_type, 1.0)

        # Heart rate intensity factor
        hr_factor = self._calculate_hr_intensity_factor(activity_data, sport_type)

        # Sport-specific power/pace adjustments
        pace_factor = self._calculate_pace_intensity_factor(activity_data, sport_type)

        # Environmental factors
        env_factor = self._calculate_environmental_factor(activity_data, sport_type)

        # Final calculation
        training_load = duration_hours * base_tss * activity_adjustment * hr_factor * pace_factor * env_factor

        return training_load

    def _calculate_hr_intensity_factor(self, activity_data: Dict, sport_type: SportType) -> float:
        """Calculate intensity factor based on heart rate zones."""
        avg_hr = activity_data.get("average_heartrate")
        max_hr = activity_data.get("max_heartrate", 185)  # Default estimate

        if not avg_hr:
            return 1.0  # Default moderate intensity

        hr_ratio = avg_hr / max_hr
        zones = self.intensity_zones.get(sport_type, self.intensity_zones[SportType.RUNNING])

        # Determine zone and apply multiplier
        if hr_ratio < zones["recovery"][1]:
            return 0.6  # Recovery
        elif hr_ratio < zones["aerobic"][1]:
            return 1.0  # Aerobic base
        elif hr_ratio < zones["tempo"][1]:
            return 1.3  # Tempo
        elif hr_ratio < zones["threshold"][1]:
            return 1.6  # Threshold
        else:
            return 2.0  # VO2max+

    def _calculate_pace_intensity_factor(self, activity_data: Dict, sport_type: SportType) -> float:
        """Calculate intensity factor based on pace/speed."""
        speed = activity_data.get("average_speed")
        if not speed:
            return 1.0

        # Sport-specific pace analysis
        if sport_type == SportType.RUNNING:
            pace_min_km = (1000 / speed) / 60 if speed > 0 else 10
            if pace_min_km < 3.5:      # Elite pace
                return 1.8
            elif pace_min_km < 4.0:    # Sub-4:00/km
                return 1.5
            elif pace_min_km < 5.0:    # Moderate
                return 1.0
            elif pace_min_km < 6.0:    # Easy
                return 0.8
            else:                      # Very easy
                return 0.6

        elif sport_type == SportType.CYCLING:
            speed_kmh = speed * 3.6
            if speed_kmh > 40:         # Fast
                return 1.5
            elif speed_kmh > 30:       # Moderate
                return 1.0
            elif speed_kmh > 20:       # Easy
                return 0.8
            else:                      # Very easy
                return 0.6

        elif sport_type == SportType.SKATING:
            speed_kmh = speed * 3.6
            activity_type = activity_data.get("type", "").lower()

            if activity_type == "inlineskate":
                # Inline skating speed zones
                if speed_kmh > 30:         # Very fast inline skating
                    return 1.6
                elif speed_kmh > 25:       # Fast
                    return 1.3
                elif speed_kmh > 20:       # Moderate
                    return 1.0
                elif speed_kmh > 15:       # Easy
                    return 0.8
                else:                      # Very easy
                    return 0.6
            else:
                # Skiing/other skating activities
                if speed_kmh > 25:         # Fast
                    return 1.4
                elif speed_kmh > 18:       # Moderate
                    return 1.0
                else:                      # Easy
                    return 0.7

        return 1.0

    def _calculate_environmental_factor(self, activity_data: Dict, sport_type: SportType) -> float:
        """Calculate environmental stress factor."""
        elevation_gain = activity_data.get("total_elevation_gain", 0)
        duration_hours = (activity_data.get("moving_time", 0) / 3600.0)

        if duration_hours == 0:
            return 1.0

        # Elevation stress (meters per hour)
        elevation_per_hour = elevation_gain / duration_hours

        # Sport-specific elevation impact
        if sport_type in [SportType.RUNNING, SportType.HIKING]:
            if elevation_per_hour > 500:      # Very hilly
                return 1.3
            elif elevation_per_hour > 200:    # Hilly
                return 1.15
            elif elevation_per_hour > 50:     # Rolling
                return 1.05
        elif sport_type == SportType.CYCLING:
            if elevation_per_hour > 800:      # Very hilly cycling
                return 1.25
            elif elevation_per_hour > 300:    # Hilly
                return 1.1
            elif elevation_per_hour > 100:    # Rolling
                return 1.05

        return 1.0

    def calculate_recovery_time(self, activity_data: Dict) -> int:
        """Calculate required recovery time in hours based on activity."""
        activity_type = activity_data.get("type", "").lower()
        sport_type = self.get_sport_type(activity_type)

        duration_hours = (activity_data.get("moving_time", 0) / 3600.0)
        training_load = activity_data.get("training_load", 0)

        # Base recovery demand by sport
        recovery_demand = self.recovery_demand[sport_type]

        # Calculate base recovery hours
        if recovery_demand == RecoveryDemand.LOW:
            base_recovery = duration_hours * 0.5
        elif recovery_demand == RecoveryDemand.MODERATE:
            base_recovery = duration_hours * 1.0
        elif recovery_demand == RecoveryDemand.HIGH:
            base_recovery = duration_hours * 1.5
        else:  # VERY_HIGH
            base_recovery = duration_hours * 2.0

        # Adjust for intensity (training load)
        if training_load > 200:           # Very high load
            intensity_multiplier = 2.0
        elif training_load > 150:         # High load
            intensity_multiplier = 1.5
        elif training_load > 100:         # Moderate load
            intensity_multiplier = 1.0
        elif training_load > 50:          # Low load
            intensity_multiplier = 0.7
        else:                             # Very low load
            intensity_multiplier = 0.3

        # Check if it's a race
        is_race = activity_data.get("workout_type") == 1
        if is_race:
            # Races require much more recovery - minimum 48h for any race
            race_multiplier = 3.0
            # Longer races need even more recovery
            if duration_hours > 3:        # Marathon+
                race_multiplier = 5.0
            elif duration_hours > 1.5:    # Half marathon+
                race_multiplier = 4.0
            else:                         # 5K, 10K races
                race_multiplier = 3.0
        else:
            race_multiplier = 1.0

        total_recovery_hours = base_recovery * intensity_multiplier * race_multiplier

        # Ensure minimum recovery for races
        if is_race:
            total_recovery_hours = max(total_recovery_hours, 48)  # Minimum 48h for any race
            if duration_hours > 1.5:  # Half marathon+
                total_recovery_hours = max(total_recovery_hours, 60)  # Minimum 60h

        # Cap at reasonable maximums
        return min(int(total_recovery_hours), 96)  # Max 4 days

    def get_cross_training_recommendations(self, primary_sport: SportType,
                                         current_load: float) -> List[Dict]:
        """Recommend complementary activities for recovery and cross-training."""
        recommendations = []

        if primary_sport == SportType.RUNNING:
            recommendations = [
                {
                    "activity": "Swimming",
                    "benefit": "Non-impact cardiovascular maintenance",
                    "intensity": "Easy-moderate",
                    "duration": "30-45 minutes"
                },
                {
                    "activity": "Cycling",
                    "benefit": "Active recovery, leg circulation",
                    "intensity": "Easy",
                    "duration": "45-60 minutes"
                },
                {
                    "activity": "Aqua jogging",
                    "benefit": "Running-specific movement without impact",
                    "intensity": "Moderate",
                    "duration": "20-30 minutes"
                }
            ]
        elif primary_sport == SportType.CYCLING:
            recommendations = [
                {
                    "activity": "Running",
                    "benefit": "Impact training for bone density",
                    "intensity": "Easy",
                    "duration": "20-30 minutes"
                },
                {
                    "activity": "Swimming",
                    "benefit": "Upper body strength, different movement pattern",
                    "intensity": "Moderate",
                    "duration": "30-40 minutes"
                },
                {
                    "activity": "Hiking",
                    "benefit": "Active recovery with nature benefits",
                    "intensity": "Easy",
                    "duration": "60-90 minutes"
                }
            ]
        elif primary_sport == SportType.SWIMMING:
            recommendations = [
                {
                    "activity": "Running",
                    "benefit": "Weight-bearing exercise, different energy system",
                    "intensity": "Easy-moderate",
                    "duration": "30-45 minutes"
                },
                {
                    "activity": "Cycling",
                    "benefit": "Leg strength, aerobic maintenance",
                    "intensity": "Moderate",
                    "duration": "45-60 minutes"
                },
                {
                    "activity": "Yoga",
                    "benefit": "Flexibility, shoulder mobility",
                    "intensity": "Light",
                    "duration": "45-60 minutes"
                }
            ]

        return recommendations

    def analyze_training_distribution(self, activities: List[Dict],
                                    days: int = 7) -> Dict:
        """Analyze training distribution across sports."""
        sport_loads = {}
        total_load = 0
        sport_hours = {}
        total_hours = 0

        for activity in activities:
            sport_type = self.get_sport_type(activity.get("type", ""))
            load = activity.get("training_load", 0)
            duration = (activity.get("moving_time", 0) / 3600.0)

            if sport_type not in sport_loads:
                sport_loads[sport_type] = 0
                sport_hours[sport_type] = 0

            sport_loads[sport_type] += load
            sport_hours[sport_type] += duration
            total_load += load
            total_hours += duration

        # Calculate percentages
        load_distribution = {}
        time_distribution = {}

        for sport_type in sport_loads:
            load_pct = (sport_loads[sport_type] / total_load * 100) if total_load > 0 else 0
            time_pct = (sport_hours[sport_type] / total_hours * 100) if total_hours > 0 else 0

            load_distribution[sport_type.value] = round(load_pct, 1)
            time_distribution[sport_type.value] = round(time_pct, 1)

        return {
            "load_distribution": load_distribution,
            "time_distribution": time_distribution,
            "total_load": total_load,
            "total_hours": round(total_hours, 1),
            "sport_loads": {k.value: v for k, v in sport_loads.items()},
            "sport_hours": {k.value: round(v, 1) for k, v in sport_hours.items()}
        }