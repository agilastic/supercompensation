"""
Multi-System Recovery Model Implementation
Based on German Sports Science: Belastungs-Beanspruchungs Model (Load-Stress Model)

Multi-system recovery with different timeframes according to physiological systems:
- Neural transmission: Seconds to minutes
- Creatine phosphate stores: Minutes to hours
- Glycogen replenishment: Hours to days
- Protein structure repair: Days to weeks
- Cellular adaptations: Weeks to months

Author: Sport Medicine Professional & Olympic Trainer
Reference: "Sport: Das Lehrbuch für das Sportstudium" - Güllich & Krüger
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class RecoverySystem(Enum):
    """Physiological systems with different recovery timeframes"""
    NEURAL = "neural"              # Neural transmission recovery
    ENERGETIC = "energetic"        # Creatine phosphate & immediate energy
    METABOLIC = "metabolic"        # Glycogen replenishment & lactate clearance
    STRUCTURAL = "structural"      # Protein synthesis & tissue repair
    ADAPTIVE = "adaptive"          # Cellular adaptations & enzyme synthesis


@dataclass
class SystemRecoveryProfile:
    """Recovery characteristics for each physiological system"""
    name: str
    recovery_half_life: float      # Hours for 50% recovery
    max_recovery_time: float       # Hours for 95% recovery
    load_sensitivity: float        # How sensitive to training load (0-1)
    adaptation_coefficient: float  # Adaptation potential (0-1)


class MultiSystemRecoveryModel:
    """
    Multi-system recovery model based on German sports science methodology.

    Implements the Belastungs-Beanspruchungs-Konzept with system-specific
    recovery kinetics for optimal training load distribution.
    """

    def __init__(self):
        self.recovery_systems = {
            RecoverySystem.NEURAL: SystemRecoveryProfile(
                name="Neural System",
                recovery_half_life=0.25,       # 15 minutes
                max_recovery_time=2.0,         # 2 hours
                load_sensitivity=0.9,          # High sensitivity to load
                adaptation_coefficient=0.15    # Fast adaptation
            ),
            RecoverySystem.ENERGETIC: SystemRecoveryProfile(
                name="Energy Systems",
                recovery_half_life=2.0,        # 2 hours
                max_recovery_time=12.0,        # 12 hours
                load_sensitivity=0.8,          # High sensitivity
                adaptation_coefficient=0.12    # Moderate-fast adaptation
            ),
            RecoverySystem.METABOLIC: SystemRecoveryProfile(
                name="Metabolic Systems",
                recovery_half_life=12.0,       # 12 hours
                max_recovery_time=48.0,        # 48 hours
                load_sensitivity=0.7,          # Moderate sensitivity
                adaptation_coefficient=0.08    # Moderate adaptation
            ),
            RecoverySystem.STRUCTURAL: SystemRecoveryProfile(
                name="Structural Systems",
                recovery_half_life=48.0,       # 48 hours
                max_recovery_time=168.0,       # 7 days
                load_sensitivity=0.6,          # Lower sensitivity
                adaptation_coefficient=0.06    # Slow adaptation
            ),
            RecoverySystem.ADAPTIVE: SystemRecoveryProfile(
                name="Adaptive Systems",
                recovery_half_life=168.0,      # 7 days
                max_recovery_time=720.0,       # 30 days
                load_sensitivity=0.4,          # Low sensitivity
                adaptation_coefficient=0.04    # Very slow adaptation
            )
        }

    def calculate_system_load(self, activity_data: Dict, system: RecoverySystem) -> float:
        """
        Calculate system-specific training load based on activity characteristics.

        Args:
            activity_data: Activity metrics (duration, intensity, type, etc.)
            system: Target physiological system

        Returns:
            System-specific load value (0-100)
        """
        if not activity_data:
            return 0.0

        duration_hours = activity_data.get('moving_time', 0) / 3600.0
        intensity = self._calculate_relative_intensity(activity_data)
        activity_type = activity_data.get('type', 'Unknown')

        # Base load calculation
        base_load = duration_hours * intensity

        # System-specific adjustments
        system_multiplier = self._get_system_multiplier(system, activity_type, intensity)

        # Apply system load sensitivity
        profile = self.recovery_systems[system]
        system_load = base_load * system_multiplier * profile.load_sensitivity

        # Cap at reasonable maximum (equivalent to 4h at threshold)
        return min(system_load, 100.0)

    def _calculate_relative_intensity(self, activity_data: Dict) -> float:
        """Calculate relative exercise intensity (0-1 scale)"""
        # Priority: Heart rate > Power > Pace > RPE
        if activity_data.get('average_heartrate'):
            # Assume 190 bpm max HR for normalization (should be individualized)
            max_hr = 190
            avg_hr = activity_data['average_heartrate']
            return min(avg_hr / max_hr, 1.0)

        elif activity_data.get('average_watts'):
            # Estimate relative intensity from power (requires FTP)
            # For now, use normalized scaling
            watts = activity_data['average_watts']
            estimated_ftp = 250  # Should be individualized
            return min(watts / estimated_ftp, 1.5) / 1.5

        elif activity_data.get('average_speed') and activity_data.get('distance'):
            # Estimate from pace relative to activity type
            activity_type = activity_data.get('type', 'Unknown')
            if activity_type == 'Run':
                # Estimate relative to 5:00/km pace
                avg_pace_per_km = (activity_data['moving_time'] / activity_data['distance']) * 1000
                reference_pace = 300  # 5:00/km in seconds
                return min(reference_pace / avg_pace_per_km, 1.2) / 1.2
            elif activity_type == 'Ride':
                # Estimate relative to 35 km/h
                avg_speed_kmh = activity_data['average_speed'] * 3.6
                return min(avg_speed_kmh / 35.0, 1.0)

        # Default to moderate intensity if no intensity metrics available
        return 0.6

    def _get_system_multiplier(self, system: RecoverySystem, activity_type: str, intensity: float) -> float:
        """Get activity-specific multiplier for each system"""

        # Base multipliers by activity type
        type_multipliers = {
            'Run': {
                RecoverySystem.NEURAL: 1.2,     # High impact stress
                RecoverySystem.ENERGETIC: 1.0,  # Standard energy demand
                RecoverySystem.METABOLIC: 1.1,  # High metabolic stress
                RecoverySystem.STRUCTURAL: 1.3, # High structural stress (impact)
                RecoverySystem.ADAPTIVE: 1.0
            },
            'Ride': {
                RecoverySystem.NEURAL: 0.8,     # Lower neural stress
                RecoverySystem.ENERGETIC: 1.2,  # High energy demand
                RecoverySystem.METABOLIC: 1.0,  # Standard metabolic
                RecoverySystem.STRUCTURAL: 0.7, # Lower structural stress
                RecoverySystem.ADAPTIVE: 1.1
            },
            'Hike': {
                RecoverySystem.NEURAL: 0.6,     # Low neural stress
                RecoverySystem.ENERGETIC: 0.8,  # Moderate energy
                RecoverySystem.METABOLIC: 0.9,  # Moderate metabolic
                RecoverySystem.STRUCTURAL: 0.8, # Moderate structural
                RecoverySystem.ADAPTIVE: 0.7
            },
            'WeightTraining': {
                RecoverySystem.NEURAL: 1.5,     # Very high neural stress
                RecoverySystem.ENERGETIC: 0.6,  # Low energy demand
                RecoverySystem.METABOLIC: 0.5,  # Low metabolic stress
                RecoverySystem.STRUCTURAL: 1.8, # Very high structural stress
                RecoverySystem.ADAPTIVE: 1.2
            }
        }

        base_multiplier = type_multipliers.get(activity_type, {}).get(system, 1.0)

        # Intensity adjustments (higher intensity affects neural/energetic more)
        if system in [RecoverySystem.NEURAL, RecoverySystem.ENERGETIC]:
            intensity_multiplier = 0.5 + (intensity * 1.5)  # 0.5-2.0 range
        else:
            intensity_multiplier = 0.7 + (intensity * 0.8)  # 0.7-1.5 range

        return base_multiplier * intensity_multiplier

    def calculate_system_recovery(self, system: RecoverySystem, hours_since_load: float,
                                 initial_load: float) -> float:
        """
        Calculate current recovery state for a specific system.

        Args:
            system: Physiological system
            hours_since_load: Hours elapsed since training load
            initial_load: Initial system load value

        Returns:
            Recovery percentage (0-100, where 100 = fully recovered)
        """
        if hours_since_load <= 0 or initial_load <= 0:
            return 100.0

        profile = self.recovery_systems[system]

        # Exponential recovery model: R(t) = 100 * (1 - e^(-k*t))
        # where k = ln(2) / half_life
        k = math.log(2) / profile.recovery_half_life

        # Recovery percentage
        recovery_percentage = 100 * (1 - math.exp(-k * hours_since_load))

        # Account for initial load magnitude (higher loads take longer to recover)
        load_factor = min(initial_load / 50.0, 2.0)  # Scale factor based on load
        adjusted_recovery = recovery_percentage / load_factor

        return min(adjusted_recovery, 100.0)

    def calculate_overall_recovery_status(self, activities: List[Dict],
                                        current_time: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive recovery status across all systems.

        Args:
            activities: List of recent activities with timestamps
            current_time: Current timestamp for calculations

        Returns:
            Dictionary with recovery status for each system and overall score
        """
        system_recoveries = {}

        for system in RecoverySystem:
            total_residual_load = 0.0

            # Sum residual loads from all recent activities
            for activity in activities:
                activity_time = activity.get('start_date')
                if not activity_time:
                    continue

                if isinstance(activity_time, str):
                    activity_time = datetime.fromisoformat(activity_time.replace('Z', '+00:00'))

                hours_elapsed = (current_time - activity_time).total_seconds() / 3600.0

                # Only consider activities within the max recovery time for this system
                profile = self.recovery_systems[system]
                if hours_elapsed <= profile.max_recovery_time:
                    initial_load = self.calculate_system_load(activity, system)
                    recovery_pct = self.calculate_system_recovery(system, hours_elapsed, initial_load)
                    residual_load = initial_load * (100 - recovery_pct) / 100
                    total_residual_load += residual_load

            # Convert residual load to recovery percentage
            # Higher residual load = lower recovery status
            recovery_status = max(0, 100 - total_residual_load)
            system_recoveries[system.value] = recovery_status

        # Calculate weighted overall recovery
        # Weight systems by their adaptation importance for training readiness
        weights = {
            RecoverySystem.NEURAL: 0.25,      # Critical for skill/coordination
            RecoverySystem.ENERGETIC: 0.20,   # Important for intensity
            RecoverySystem.METABOLIC: 0.25,   # Critical for endurance
            RecoverySystem.STRUCTURAL: 0.20,  # Important for health
            RecoverySystem.ADAPTIVE: 0.10     # Long-term consideration
        }

        overall_recovery = sum(
            system_recoveries[system.value] * weights[system]
            for system in RecoverySystem
        )

        system_recoveries['overall'] = overall_recovery
        return system_recoveries

    def get_training_recommendations(self, recovery_status: Dict[str, float]) -> Dict[str, str]:
        """
        Generate training recommendations based on system recovery status.

        Args:
            recovery_status: Dictionary of system recovery percentages

        Returns:
            Training recommendations and rationale
        """
        overall = recovery_status.get('overall', 50)
        neural = recovery_status.get('neural', 50)
        energetic = recovery_status.get('energetic', 50)
        metabolic = recovery_status.get('metabolic', 50)
        structural = recovery_status.get('structural', 50)

        recommendations = {
            'primary_recommendation': '',
            'intensity_guidance': '',
            'duration_guidance': '',
            'type_guidance': '',
            'rationale': ''
        }

        # Primary recommendation logic
        if overall >= 85:
            recommendations['primary_recommendation'] = 'HIGH_INTENSITY'
            recommendations['intensity_guidance'] = 'All intensity zones available'
            recommendations['duration_guidance'] = 'Normal to extended duration'
            recommendations['type_guidance'] = 'Any activity type suitable'
            recommendations['rationale'] = 'Excellent recovery across all systems'

        elif overall >= 70:
            if neural >= 80 and energetic >= 80:
                recommendations['primary_recommendation'] = 'MODERATE_HIGH'
                recommendations['intensity_guidance'] = 'Zone 2-4 recommended, brief Zone 5'
                recommendations['duration_guidance'] = 'Normal duration'
                recommendations['type_guidance'] = 'Technical work and moderate intensity'
                recommendations['rationale'] = 'Good neural and energetic recovery'
            else:
                recommendations['primary_recommendation'] = 'MODERATE'
                recommendations['intensity_guidance'] = 'Zone 1-3 focus'
                recommendations['duration_guidance'] = 'Reduced duration'
                recommendations['type_guidance'] = 'Endurance base training'
                recommendations['rationale'] = 'Moderate overall recovery'

        elif overall >= 50:
            if structural < 60:
                recommendations['primary_recommendation'] = 'ACTIVE_RECOVERY'
                recommendations['intensity_guidance'] = 'Zone 1 only'
                recommendations['duration_guidance'] = 'Short to moderate'
                recommendations['type_guidance'] = 'Low-impact activities (swimming, cycling)'
                recommendations['rationale'] = 'Structural recovery needed - avoid high impact'
            else:
                recommendations['primary_recommendation'] = 'LIGHT_TRAINING'
                recommendations['intensity_guidance'] = 'Zone 1-2 only'
                recommendations['duration_guidance'] = 'Reduced duration'
                recommendations['type_guidance'] = 'Low intensity endurance'
                recommendations['rationale'] = 'Partial recovery - maintain aerobic base'

        else:  # overall < 50
            recommendations['primary_recommendation'] = 'REST'
            recommendations['intensity_guidance'] = 'Complete rest or very light movement'
            recommendations['duration_guidance'] = 'Minimal activity'
            recommendations['type_guidance'] = 'Gentle mobility, walking'
            recommendations['rationale'] = 'Poor recovery status - rest required'

        return recommendations


def create_recovery_analysis_report(activities: List[Dict],
                                  current_time: datetime = None) -> Dict:
    """
    Generate comprehensive recovery analysis report.

    Args:
        activities: List of activity data
        current_time: Analysis timestamp (defaults to now)

    Returns:
        Comprehensive recovery analysis with recommendations
    """
    if current_time is None:
        current_time = datetime.now()

    model = MultiSystemRecoveryModel()

    # Calculate recovery status
    recovery_status = model.calculate_overall_recovery_status(activities, current_time)

    # Generate recommendations
    recommendations = model.get_training_recommendations(recovery_status)

    # Calculate system-specific insights
    system_insights = {}
    for system in RecoverySystem:
        status = recovery_status.get(system.value, 50)
        profile = model.recovery_systems[system]

        if status >= 85:
            insight = f"{profile.name}: Excellent recovery - ready for high load"
        elif status >= 70:
            insight = f"{profile.name}: Good recovery - moderate loads suitable"
        elif status >= 50:
            insight = f"{profile.name}: Partial recovery - light loads only"
        else:
            insight = f"{profile.name}: Poor recovery - rest required"

        system_insights[system.value] = insight

    return {
        'timestamp': current_time.isoformat(),
        'recovery_status': recovery_status,
        'recommendations': recommendations,
        'system_insights': system_insights,
        'activities_analyzed': len(activities)
    }