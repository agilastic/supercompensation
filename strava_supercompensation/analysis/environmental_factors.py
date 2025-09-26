"""
Environmental Factor Analysis for Training Load Adjustment

Based on German Sports Science: Environmental modulators affecting performance and recovery
"Sport: Das Lehrbuch für das Sportstudium" - Güllich & Krüger

Environmental factors that affect training load and recovery:
- Temperature stress (heat/cold)
- Altitude adaptation phases
- Humidity impact
- Air quality considerations

Author: Sport Medicine Professional & Olympic Trainer
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import config

class EnvironmentalCondition(Enum):
    """Environmental condition categories"""
    OPTIMAL = "optimal"           # Ideal training conditions
    MILD_STRESS = "mild_stress"   # Slight environmental challenge
    MODERATE_STRESS = "moderate_stress"  # Noticeable environmental stress
    HIGH_STRESS = "high_stress"   # Significant environmental challenge
    EXTREME = "extreme"           # Dangerous/inadvisable conditions


@dataclass
class EnvironmentalProfile:
    """Environmental conditions for a training session"""
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[float] = None
    altitude_meters: Optional[float] = None
    air_quality_index: Optional[int] = None  # AQI 0-500 scale
    wind_speed_ms: Optional[float] = None
    heat_index: Optional[float] = None       # Calculated from temp + humidity
    wind_chill: Optional[float] = None       # Calculated from temp + wind


class EnvironmentalAnalyzer:
    """
    Analyze environmental conditions and their impact on training load and recovery.

    Based on physiological research on environmental stress responses and
    German sports science methodology for training load adjustment.
    """

    def __init__(self):
        # Optimal training ranges (configurable via .env)
        self.optimal_ranges = {
            'temperature': (config.OPTIMAL_TEMP_MIN, config.OPTIMAL_TEMP_MAX),
            'humidity': (config.OPTIMAL_HUMIDITY_MIN, config.OPTIMAL_HUMIDITY_MAX),
            'altitude': (0.0, config.ALTITUDE_THRESHOLD),
            'air_quality': (0, 50),  # AQI standard (Good category)
            'wind_speed': (0.0, config.OPTIMAL_WIND_MAX / 3.6)  # Convert km/h to m/s
        }

    def analyze_environmental_impact(self, conditions: EnvironmentalProfile,
                                   activity_type: str = "endurance") -> Dict[str, any]:
        """
        Analyze environmental conditions and calculate training load adjustments.

        Args:
            conditions: Environmental conditions during training
            activity_type: Type of activity (endurance, power, technical)

        Returns:
            Environmental analysis with load adjustments and recommendations
        """
        analysis = {
            'overall_stress_level': EnvironmentalCondition.OPTIMAL.value,
            'load_multiplier': 1.0,        # Training load adjustment factor
            'recovery_modifier': 1.0,      # Recovery time adjustment
            'performance_impact': 0.0,     # Expected performance degradation (%)
            'specific_stressors': [],
            'recommendations': [],
            'safety_warnings': []
        }

        # Temperature stress analysis
        temp_impact = self._analyze_temperature_stress(conditions, activity_type)
        analysis.update(temp_impact)

        # Humidity impact
        humidity_impact = self._analyze_humidity_stress(conditions)
        self._combine_impacts(analysis, humidity_impact)

        # Altitude impact
        altitude_impact = self._analyze_altitude_stress(conditions)
        self._combine_impacts(analysis, altitude_impact)

        # Air quality impact
        air_quality_impact = self._analyze_air_quality_stress(conditions)
        self._combine_impacts(analysis, air_quality_impact)

        # Calculate combined heat stress (temperature + humidity)
        heat_stress_impact = self._analyze_heat_stress(conditions)
        self._combine_impacts(analysis, heat_stress_impact)

        # Determine overall environmental stress level
        analysis['overall_stress_level'] = self._determine_overall_stress_level(analysis)

        return analysis

    def _analyze_temperature_stress(self, conditions: EnvironmentalProfile,
                                  activity_type: str) -> Dict[str, any]:
        """Analyze temperature-specific stress factors"""
        if conditions.temperature_celsius is None:
            return {'load_multiplier': 1.0, 'recovery_modifier': 1.0, 'performance_impact': 0.0}

        temp = conditions.temperature_celsius
        optimal_min, optimal_max = self.optimal_ranges['temperature']

        impact = {
            'load_multiplier': 1.0,
            'recovery_modifier': 1.0,
            'performance_impact': 0.0,
            'specific_stressors': [],
            'recommendations': []
        }

        # Cold stress analysis (< 10°C)
        if temp < optimal_min:
            cold_severity = (optimal_min - temp) / 10.0  # Scale factor

            if temp < -10:  # Extreme cold
                impact.update({
                    'load_multiplier': 1.4,      # Higher physiological cost
                    'recovery_modifier': 1.3,     # Longer recovery needed
                    'performance_impact': 15.0 + cold_severity * 5
                })
                impact['specific_stressors'].append(f"Extreme cold stress ({temp:.1f}°C)")
                impact['recommendations'].extend([
                    "Extended warm-up required (15-20 minutes)",
                    "Layer clothing system essential",
                    "Monitor for hypothermia signs"
                ])

            elif temp < 0:  # Severe cold
                impact.update({
                    'load_multiplier': 1.25,
                    'recovery_modifier': 1.2,
                    'performance_impact': 8.0 + cold_severity * 3
                })
                impact['specific_stressors'].append(f"Severe cold stress ({temp:.1f}°C)")
                impact['recommendations'].extend([
                    "Extended warm-up needed (10-15 minutes)",
                    "Protect extremities from frostbite"
                ])

            else:  # Moderate cold (0-10°C)
                impact.update({
                    'load_multiplier': 1.1,
                    'recovery_modifier': 1.05,
                    'performance_impact': 2.0 + cold_severity * 2
                })
                impact['specific_stressors'].append(f"Cold stress ({temp:.1f}°C)")
                impact['recommendations'].append("Extra warm-up recommended")

        # Heat stress analysis (> 20°C)
        elif temp > optimal_max:
            heat_severity = (temp - optimal_max) / 10.0

            if temp > 35:  # Extreme heat
                impact.update({
                    'load_multiplier': 1.6,      # Much higher physiological cost
                    'recovery_modifier': 1.5,     # Extended recovery
                    'performance_impact': 20.0 + heat_severity * 8
                })
                impact['specific_stressors'].append(f"Extreme heat stress ({temp:.1f}°C)")
                impact['recommendations'].extend([
                    "Consider postponing training",
                    "Aggressive pre-cooling strategies",
                    "Frequent hydration breaks"
                ])

            elif temp > 30:  # Severe heat
                impact.update({
                    'load_multiplier': 1.4,
                    'recovery_modifier': 1.3,
                    'performance_impact': 12.0 + heat_severity * 5
                })
                impact['specific_stressors'].append(f"Severe heat stress ({temp:.1f}°C)")
                impact['recommendations'].extend([
                    "Reduce training intensity",
                    "Increase hydration frequency",
                    "Monitor for heat illness"
                ])

            elif temp > 25:  # Moderate heat
                impact.update({
                    'load_multiplier': 1.2,
                    'recovery_modifier': 1.15,
                    'performance_impact': 5.0 + heat_severity * 3
                })
                impact['specific_stressors'].append(f"Heat stress ({temp:.1f}°C)")
                impact['recommendations'].extend([
                    "Extra hydration needed",
                    "Consider earlier training times"
                ])

            else:  # Mild heat (20-25°C)
                impact.update({
                    'load_multiplier': 1.05,
                    'recovery_modifier': 1.02,
                    'performance_impact': 1.0 + heat_severity
                })

        # Activity-specific adjustments
        if activity_type == "power":
            # Power activities less affected by temperature
            impact['load_multiplier'] *= 0.8
            impact['performance_impact'] *= 0.6
        elif activity_type == "endurance":
            # Endurance more affected by temperature
            impact['load_multiplier'] *= 1.2
            impact['performance_impact'] *= 1.3

        return impact

    def _analyze_humidity_stress(self, conditions: EnvironmentalProfile) -> Dict[str, any]:
        """Analyze humidity-specific stress factors"""
        if conditions.humidity_percent is None:
            return {'load_multiplier': 1.0, 'recovery_modifier': 1.0, 'performance_impact': 0.0}

        humidity = conditions.humidity_percent
        optimal_min, optimal_max = self.optimal_ranges['humidity']

        impact = {
            'load_multiplier': 1.0,
            'recovery_modifier': 1.0,
            'performance_impact': 0.0,
            'specific_stressors': [],
            'recommendations': []
        }

        # High humidity stress (> 60%)
        if humidity > optimal_max:
            humidity_excess = (humidity - optimal_max) / 40.0  # Scale to 0-1

            if humidity > 90:  # Extreme humidity
                impact.update({
                    'load_multiplier': 1.3,
                    'recovery_modifier': 1.25,
                    'performance_impact': 10.0 + humidity_excess * 5
                })
                impact['specific_stressors'].append(f"Extreme humidity ({humidity:.0f}%)")
                impact['recommendations'].extend([
                    "Impaired sweat evaporation - reduce intensity",
                    "Increased dehydration risk"
                ])

            elif humidity > 75:  # High humidity
                impact.update({
                    'load_multiplier': 1.15,
                    'recovery_modifier': 1.1,
                    'performance_impact': 5.0 + humidity_excess * 3
                })
                impact['specific_stressors'].append(f"High humidity ({humidity:.0f}%)")
                impact['recommendations'].append("Enhanced cooling strategies needed")

        # Low humidity stress (< 40%)
        elif humidity < optimal_min:
            dryness = (optimal_min - humidity) / 40.0

            if humidity < 20:  # Very dry
                impact.update({
                    'load_multiplier': 1.1,
                    'recovery_modifier': 1.05,
                    'performance_impact': 3.0 + dryness * 2
                })
                impact['specific_stressors'].append(f"Very low humidity ({humidity:.0f}%)")
                impact['recommendations'].extend([
                    "Increased fluid loss through respiration",
                    "Extra hydration recommended"
                ])

        return impact

    def _analyze_altitude_stress(self, conditions: EnvironmentalProfile) -> Dict[str, any]:
        """Analyze altitude-specific stress factors with adaptation phases"""
        if conditions.altitude_meters is None:
            return {'load_multiplier': 1.0, 'recovery_modifier': 1.0, 'performance_impact': 0.0}

        altitude = conditions.altitude_meters
        optimal_max = self.optimal_ranges['altitude'][1]

        impact = {
            'load_multiplier': 1.0,
            'recovery_modifier': 1.0,
            'performance_impact': 0.0,
            'specific_stressors': [],
            'recommendations': []
        }

        if altitude > optimal_max:
            # Altitude impact calculation (based on oxygen partial pressure)
            # At sea level: 760 mmHg, at 2000m: ~600 mmHg (~21% reduction)
            oxygen_reduction = 1 - math.exp(-altitude / 8848.0)  # Exponential model

            # Performance impact increases with altitude
            performance_loss = oxygen_reduction * 100

            if altitude > 3500:  # High altitude
                impact.update({
                    'load_multiplier': 1.5 + oxygen_reduction,
                    'recovery_modifier': 1.4 + oxygen_reduction * 0.5,
                    'performance_impact': performance_loss
                })
                impact['specific_stressors'].append(f"High altitude stress ({altitude:.0f}m)")
                impact['recommendations'].extend([
                    "Gradual acclimatization required (7-14 days)",
                    "Monitor for altitude sickness symptoms",
                    "Reduce training intensity initially"
                ])

            elif altitude > 2000:  # Moderate altitude
                impact.update({
                    'load_multiplier': 1.3 + oxygen_reduction * 0.5,
                    'recovery_modifier': 1.2 + oxygen_reduction * 0.3,
                    'performance_impact': performance_loss * 0.7
                })
                impact['specific_stressors'].append(f"Moderate altitude stress ({altitude:.0f}m)")
                impact['recommendations'].extend([
                    "Allow 3-5 days for initial adaptation",
                    "Increase hydration (dry air + increased respiration)"
                ])

            elif altitude > 1000:  # Mild altitude
                impact.update({
                    'load_multiplier': 1.1 + oxygen_reduction * 0.3,
                    'recovery_modifier': 1.05 + oxygen_reduction * 0.2,
                    'performance_impact': performance_loss * 0.5
                })
                impact['specific_stressors'].append(f"Mild altitude stress ({altitude:.0f}m)")
                impact['recommendations'].append("Minor adaptation period (1-2 days)")

        return impact

    def _analyze_air_quality_stress(self, conditions: EnvironmentalProfile) -> Dict[str, any]:
        """Analyze air quality impact on training"""
        if conditions.air_quality_index is None:
            return {'load_multiplier': 1.0, 'recovery_modifier': 1.0, 'performance_impact': 0.0}

        aqi = conditions.air_quality_index

        impact = {
            'load_multiplier': 1.0,
            'recovery_modifier': 1.0,
            'performance_impact': 0.0,
            'specific_stressors': [],
            'recommendations': []
        }

        if aqi > 300:  # Hazardous
            impact.update({
                'load_multiplier': 2.0,
                'recovery_modifier': 1.8,
                'performance_impact': 30.0
            })
            impact['specific_stressors'].append(f"Hazardous air quality (AQI {aqi})")
            impact['recommendations'].append("AVOID outdoor training completely")

        elif aqi > 200:  # Very unhealthy
            impact.update({
                'load_multiplier': 1.6,
                'recovery_modifier': 1.4,
                'performance_impact': 20.0
            })
            impact['specific_stressors'].append(f"Very unhealthy air quality (AQI {aqi})")
            impact['recommendations'].append("Move training indoors if possible")

        elif aqi > 150:  # Unhealthy
            impact.update({
                'load_multiplier': 1.3,
                'recovery_modifier': 1.2,
                'performance_impact': 12.0
            })
            impact['specific_stressors'].append(f"Unhealthy air quality (AQI {aqi})")
            impact['recommendations'].append("Reduce outdoor exercise intensity")

        elif aqi > 100:  # Unhealthy for sensitive groups
            impact.update({
                'load_multiplier': 1.15,
                'recovery_modifier': 1.1,
                'performance_impact': 5.0
            })
            impact['specific_stressors'].append(f"Moderate air quality (AQI {aqi})")
            impact['recommendations'].append("Sensitive individuals should reduce intensity")

        elif aqi > 50:  # Moderate
            impact.update({
                'load_multiplier': 1.05,
                'recovery_modifier': 1.02,
                'performance_impact': 2.0
            })

        return impact

    def _analyze_heat_stress(self, conditions: EnvironmentalProfile) -> Dict[str, any]:
        """Analyze combined heat and humidity stress (Heat Index)"""
        if conditions.temperature_celsius is None or conditions.humidity_percent is None:
            return {'load_multiplier': 1.0, 'recovery_modifier': 1.0, 'performance_impact': 0.0}

        # Calculate Heat Index (apparent temperature)
        temp_f = conditions.temperature_celsius * 9/5 + 32
        humidity = conditions.humidity_percent

        # Simplified Heat Index calculation
        heat_index_f = temp_f
        if temp_f >= 80 and humidity >= 40:
            heat_index_f = (
                -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
                - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
                - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
                + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
            )

        heat_index_c = (heat_index_f - 32) * 5/9
        conditions.heat_index = heat_index_c

        impact = {
            'load_multiplier': 1.0,
            'recovery_modifier': 1.0,
            'performance_impact': 0.0,
            'specific_stressors': [],
            'recommendations': []
        }

        # Heat Index categories (based on NOAA guidelines)
        if heat_index_c > 54:  # Extreme danger
            impact.update({
                'load_multiplier': 2.5,
                'recovery_modifier': 2.0,
                'performance_impact': 40.0
            })
            impact['specific_stressors'].append(f"Extreme heat index ({heat_index_c:.1f}°C)")
            impact['recommendations'].append("DANGEROUS - Cancel outdoor training")

        elif heat_index_c > 40:  # Danger
            impact.update({
                'load_multiplier': 1.8,
                'recovery_modifier': 1.6,
                'performance_impact': 25.0
            })
            impact['specific_stressors'].append(f"Dangerous heat index ({heat_index_c:.1f}°C)")
            impact['recommendations'].append("High risk of heat exhaustion")

        elif heat_index_c > 32:  # Extreme caution
            impact.update({
                'load_multiplier': 1.4,
                'recovery_modifier': 1.3,
                'performance_impact': 15.0
            })
            impact['specific_stressors'].append(f"High heat index ({heat_index_c:.1f}°C)")
            impact['recommendations'].append("Caution - possible heat exhaustion")

        elif heat_index_c > 27:  # Caution
            impact.update({
                'load_multiplier': 1.2,
                'recovery_modifier': 1.15,
                'performance_impact': 8.0
            })
            impact['specific_stressors'].append(f"Moderate heat index ({heat_index_c:.1f}°C)")
            impact['recommendations'].append("Fatigue possible with prolonged exposure")

        return impact

    def _combine_impacts(self, main_analysis: Dict, additional_impact: Dict):
        """Combine multiple environmental impacts"""
        # Multiply load multipliers (cumulative stress)
        main_analysis['load_multiplier'] *= additional_impact.get('load_multiplier', 1.0)

        # Multiply recovery modifiers
        main_analysis['recovery_modifier'] *= additional_impact.get('recovery_modifier', 1.0)

        # Add performance impacts (cumulative degradation)
        main_analysis['performance_impact'] += additional_impact.get('performance_impact', 0.0)

        # Combine stressors and recommendations
        main_analysis['specific_stressors'].extend(additional_impact.get('specific_stressors', []))
        main_analysis['recommendations'].extend(additional_impact.get('recommendations', []))

    def _determine_overall_stress_level(self, analysis: Dict) -> str:
        """Determine overall environmental stress level"""
        load_multiplier = analysis['load_multiplier']
        performance_impact = analysis['performance_impact']

        if load_multiplier >= 2.0 or performance_impact >= 25:
            return EnvironmentalCondition.EXTREME.value
        elif load_multiplier >= 1.5 or performance_impact >= 15:
            return EnvironmentalCondition.HIGH_STRESS.value
        elif load_multiplier >= 1.2 or performance_impact >= 8:
            return EnvironmentalCondition.MODERATE_STRESS.value
        elif load_multiplier >= 1.05 or performance_impact >= 2:
            return EnvironmentalCondition.MILD_STRESS.value
        else:
            return EnvironmentalCondition.OPTIMAL.value

    def adjust_training_load_for_environment(self, base_training_load: float,
                                           conditions: EnvironmentalProfile,
                                           activity_type: str = "endurance") -> Dict[str, any]:
        """
        Adjust training load based on environmental conditions.

        Args:
            base_training_load: Original planned training load
            conditions: Environmental conditions
            activity_type: Type of training activity

        Returns:
            Adjusted training load and environmental recommendations
        """
        environmental_analysis = self.analyze_environmental_impact(conditions, activity_type)

        adjusted_load = base_training_load * environmental_analysis['load_multiplier']

        return {
            'original_load': base_training_load,
            'adjusted_load': adjusted_load,
            'load_increase_percent': ((adjusted_load - base_training_load) / base_training_load) * 100,
            'environmental_analysis': environmental_analysis,
            'training_modifications': self._get_training_modifications(environmental_analysis),
            'safety_status': self._assess_safety_status(environmental_analysis)
        }

    def _get_training_modifications(self, analysis: Dict) -> List[str]:
        """Generate specific training modifications based on environmental stress"""
        modifications = []
        stress_level = analysis['overall_stress_level']

        if stress_level == EnvironmentalCondition.EXTREME.value:
            modifications.extend([
                "Consider canceling outdoor training",
                "Move to climate-controlled environment",
                "Postpone to better conditions"
            ])
        elif stress_level == EnvironmentalCondition.HIGH_STRESS.value:
            modifications.extend([
                "Reduce training intensity by 20-30%",
                "Increase rest intervals",
                "Monitor vital signs closely"
            ])
        elif stress_level == EnvironmentalCondition.MODERATE_STRESS.value:
            modifications.extend([
                "Reduce training intensity by 10-15%",
                "Extra hydration breaks",
                "Monitor athlete comfort"
            ])
        elif stress_level == EnvironmentalCondition.MILD_STRESS.value:
            modifications.extend([
                "Slight intensity adjustment",
                "Additional preparation time"
            ])

        return modifications

    def _assess_safety_status(self, analysis: Dict) -> str:
        """Assess overall safety status for training"""
        stress_level = analysis['overall_stress_level']

        if stress_level == EnvironmentalCondition.EXTREME.value:
            return "UNSAFE - Training not recommended"
        elif stress_level == EnvironmentalCondition.HIGH_STRESS.value:
            return "HIGH RISK - Extreme caution required"
        elif stress_level == EnvironmentalCondition.MODERATE_STRESS.value:
            return "MODERATE RISK - Adjustments needed"
        elif stress_level == EnvironmentalCondition.MILD_STRESS.value:
            return "LOW RISK - Minor adjustments"
        else:
            return "SAFE - Optimal conditions"


def create_environmental_report(conditions: EnvironmentalProfile,
                              activity_type: str = "endurance") -> Dict:
    """
    Create comprehensive environmental analysis report.

    Args:
        conditions: Environmental conditions
        activity_type: Type of training activity

    Returns:
        Comprehensive environmental analysis report
    """
    analyzer = EnvironmentalAnalyzer()

    # Perform environmental analysis
    analysis = analyzer.analyze_environmental_impact(conditions, activity_type)

    # Generate training modifications
    modifications = analyzer._get_training_modifications(analysis)
    safety_status = analyzer._assess_safety_status(analysis)

    return {
        'timestamp': datetime.now().isoformat(),
        'conditions': {
            'temperature_c': conditions.temperature_celsius,
            'humidity_percent': conditions.humidity_percent,
            'altitude_m': conditions.altitude_meters,
            'air_quality_aqi': conditions.air_quality_index,
            'heat_index_c': getattr(conditions, 'heat_index', None)
        },
        'analysis': analysis,
        'training_modifications': modifications,
        'safety_status': safety_status,
        'activity_type': activity_type
    }