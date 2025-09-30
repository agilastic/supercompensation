"""
Strength Training Planner Module

Provides periodized strength training programs that align with endurance phases.
Uses streprogen library for scientific strength periodization.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging

# Check if streprogen is available
try:
    from streprogen import Program
    STREPROGEN_AVAILABLE = True
except ImportError:
    STREPROGEN_AVAILABLE = False
    Program = None

logger = logging.getLogger(__name__)


class StrengthPhase(Enum):
    """Strength training phases aligned with endurance periodization."""
    ANATOMICAL_ADAPTATION = "anatomical_adaptation"  # Base phase - high volume, low intensity
    MAX_STRENGTH = "max_strength"                    # Build phase - moderate volume, high intensity
    POWER_MAINTENANCE = "power_maintenance"          # Peak/Taper - low volume, explosive focus
    RECOVERY = "recovery"                             # Active recovery - minimal load


@dataclass
class Exercise:
    """Individual exercise prescription."""
    name: str
    sets: int
    reps: int
    intensity_percent: float  # Percentage of 1RM
    rest_seconds: int
    notes: str = ""


@dataclass
class StrengthWorkout:
    """Complete strength training workout prescription."""
    phase: StrengthPhase
    week_number: int
    day_number: int
    exercises: List[Exercise]
    volume_tss: float      # Equivalent TSS for strength session
    duration_min: int
    warmup_protocol: str
    cooldown_protocol: str
    notes: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'phase': self.phase.value,
            'week_number': self.week_number,
            'day_number': self.day_number,
            'exercises': [
                {
                    'name': ex.name,
                    'sets': ex.sets,
                    'reps': ex.reps,
                    'intensity': f"{ex.intensity_percent}%",
                    'rest': f"{ex.rest_seconds}s"
                } for ex in self.exercises
            ],
            'volume_tss': self.volume_tss,
            'duration_min': self.duration_min,
            'warmup': self.warmup_protocol,
            'cooldown': self.cooldown_protocol,
            'notes': self.notes
        }


class StrengthPlanner:
    """
    Plans periodized strength training programs for endurance athletes.

    Key principles:
    - Aligns strength phases with endurance periodization
    - Prevents interference with endurance adaptations
    - Focuses on injury prevention and power development
    """

    def __init__(self):
        """Initialize the strength planner."""
        self.logger = logging.getLogger(__name__)

        if not STREPROGEN_AVAILABLE:
            self.logger.warning("streprogen not available - using simplified strength planning")

        # Load strength configuration from environment
        self._load_strength_config()

        # Exercise database for endurance athletes
        self.exercise_database = {
            'lower_power': ['Squat', 'Deadlift', 'Bulgarian Split Squat', 'Box Jump'],
            'lower_strength': ['Front Squat', 'Romanian Deadlift', 'Leg Press', 'Calf Raise'],
            'upper_pull': ['Pull-up', 'Bent Over Row', 'Cable Row', 'Face Pull'],
            'upper_push': ['Bench Press', 'Overhead Press', 'Dips', 'Push-up'],
            'core': ['Plank', 'Dead Bug', 'Bird Dog', 'Pallof Press'],
            'plyometric': ['Jump Squat', 'Bounding', 'Depth Jump', 'Medicine Ball Slam']
        }

    def align_with_endurance_phase(self,
                                   endurance_phase: str,
                                   week_in_phase: int) -> StrengthPhase:
        """
        Map endurance periodization phase to appropriate strength phase.

        Args:
            endurance_phase: Current endurance training phase
            week_in_phase: Week number within the phase

        Returns:
            Appropriate strength training phase
        """
        phase_mapping = {
            "RECOVERY": StrengthPhase.RECOVERY,
            "BASE": StrengthPhase.ANATOMICAL_ADAPTATION,
            "BUILD": StrengthPhase.MAX_STRENGTH,
            "INTENSIFICATION": StrengthPhase.MAX_STRENGTH,
            "PEAK": StrengthPhase.POWER_MAINTENANCE,
            "TAPER": StrengthPhase.POWER_MAINTENANCE,
            "TRANSITION": StrengthPhase.RECOVERY
        }

        # Default to anatomical adaptation if phase unknown
        strength_phase = phase_mapping.get(endurance_phase.upper(), StrengthPhase.ANATOMICAL_ADAPTATION)

        # Adjust for week in phase (transition weeks)
        if week_in_phase == 1 and strength_phase != StrengthPhase.RECOVERY:
            # First week of new phase - use transition loading
            self.logger.info(f"Transition week detected - reducing strength volume")

        return strength_phase

    def calculate_strength_tss(self, exercises: List[Exercise], phase: StrengthPhase) -> float:
        """
        Calculate equivalent TSS for strength workout.

        Based on:
        - Total volume (sets x reps x intensity)
        - Phase-specific multipliers
        - Exercise complexity
        """
        total_volume = 0

        for exercise in exercises:
            # Volume = sets × reps × intensity
            volume = exercise.sets * exercise.reps * (exercise.intensity_percent / 100)

            # Compound exercises count more
            if any(compound in exercise.name.lower() for compound in ['squat', 'deadlift', 'press']):
                volume *= 1.5

            total_volume += volume

        # Phase-specific TSS multipliers
        phase_multipliers = {
            StrengthPhase.ANATOMICAL_ADAPTATION: 0.8,  # Lower intensity
            StrengthPhase.MAX_STRENGTH: 1.2,           # High neuromuscular stress
            StrengthPhase.POWER_MAINTENANCE: 1.0,      # Moderate stress
            StrengthPhase.RECOVERY: 0.5                # Minimal stress
        }

        base_tss = total_volume * 2  # Base conversion factor
        phase_adjusted_tss = base_tss * phase_multipliers.get(phase, 1.0)

        # Cap at reasonable maximum
        return min(phase_adjusted_tss, 80)  # Max 80 TSS for strength session

    def estimate_duration(self, exercises: List[Exercise]) -> int:
        """Estimate workout duration based on exercises and rest periods."""
        total_time = 10  # Warmup

        for exercise in exercises:
            # Time per set = reps × 3 seconds + rest
            time_per_set = (exercise.reps * 3 + exercise.rest_seconds) / 60  # Convert to minutes
            total_time += exercise.sets * time_per_set
            total_time += 2  # Transition between exercises

        total_time += 10  # Cooldown

        return int(total_time)

    def generate_workout(self,
                        strength_phase: StrengthPhase,
                        week_number: int,
                        day_of_week: int,
                        athlete_level: str = "intermediate") -> Optional[StrengthWorkout]:
        """
        Generate specific strength workout for given parameters.

        Args:
            strength_phase: Current strength training phase
            week_number: Week within the training block
            day_of_week: Day of week (0=Monday, 6=Sunday)
            athlete_level: Athlete experience level

        Returns:
            Complete strength workout prescription or None if rest day
        """

        # Determine if this should be a strength day based on phase
        strength_days = self._get_strength_days(strength_phase, day_of_week)

        if day_of_week not in strength_days:
            return None

        # Generate phase-specific workout
        if strength_phase == StrengthPhase.ANATOMICAL_ADAPTATION:
            exercises = self._generate_adaptation_workout(week_number, day_of_week)
        elif strength_phase == StrengthPhase.MAX_STRENGTH:
            exercises = self._generate_max_strength_workout(week_number, day_of_week)
        elif strength_phase == StrengthPhase.POWER_MAINTENANCE:
            exercises = self._generate_power_workout(week_number, day_of_week)
        else:  # Recovery
            exercises = self._generate_recovery_workout(week_number, day_of_week)

        if not exercises:
            return None

        # Calculate workout metrics
        volume_tss = self.calculate_strength_tss(exercises, strength_phase)
        duration = self.estimate_duration(exercises)

        # Create workout prescription
        return StrengthWorkout(
            phase=strength_phase,
            week_number=week_number,
            day_number=day_of_week,
            exercises=exercises,
            volume_tss=volume_tss,
            duration_min=duration,
            warmup_protocol=self._get_warmup_protocol(strength_phase),
            cooldown_protocol=self._get_cooldown_protocol(),
            notes=self._generate_workout_notes(strength_phase, week_number)
        )

    def _get_strength_days(self, phase: StrengthPhase, day_of_week: int) -> List[int]:
        """Determine which days should include strength training."""
        if phase == StrengthPhase.ANATOMICAL_ADAPTATION:
            return [1, 4]  # Tuesday, Friday - 2x per week
        elif phase == StrengthPhase.MAX_STRENGTH:
            return [1, 5]  # Tuesday, Saturday - avoid hard endurance days
        elif phase == StrengthPhase.POWER_MAINTENANCE:
            return [2]     # Wednesday only - minimal volume
        else:  # Recovery
            return [3]     # Thursday - light movement

    def _generate_adaptation_workout(self, week: int, day: int) -> List[Exercise]:
        """Generate anatomical adaptation phase workout (3x12-15 @ 60-70%)."""
        if day == 1:  # Tuesday - Lower body focus
            return [
                Exercise("Goblet Squat", 3, 15, 60, 60, "Focus on form"),
                Exercise("Romanian Deadlift", 3, 12, 65, 60, "Control the eccentric"),
                Exercise("Walking Lunge", 3, 12, 0, 45, "Bodyweight"),
                Exercise("Plank", 3, 60, 0, 30, "60 second holds"),
                Exercise("Calf Raise", 3, 20, 60, 45, "Single leg if possible")
            ]
        else:  # Friday - Upper body focus
            return [
                Exercise("Push-up", 3, 15, 0, 45, "Bodyweight"),
                Exercise("Bent Over Row", 3, 12, 65, 60, "Squeeze at top"),
                Exercise("Overhead Press", 3, 12, 60, 60, "Core engaged"),
                Exercise("Face Pull", 3, 15, 50, 45, "Rear delts"),
                Exercise("Bird Dog", 3, 10, 0, 30, "10 per side")
            ]

    def _generate_max_strength_workout(self, week: int, day: int) -> List[Exercise]:
        """Generate max strength phase workout (4-5x3-5 @ 80-90%)."""
        intensity = 80 + (week * 2.5)  # Progressive overload
        intensity = min(intensity, 90)  # Cap at 90%

        if day == 1:  # Tuesday - Lower body power
            return [
                Exercise("Back Squat", 5, 5, intensity, 180, "Full depth"),
                Exercise("Deadlift", 4, 3, intensity + 5, 180, "Reset each rep"),
                Exercise("Box Jump", 4, 5, 0, 90, "Focus on landing"),
                Exercise("Hanging Leg Raise", 3, 10, 0, 60, "Control the movement")
            ]
        else:  # Saturday - Upper body power
            return [
                Exercise("Bench Press", 5, 5, intensity, 150, "Control descent"),
                Exercise("Weighted Pull-up", 4, 5, 70, 120, "Add weight if needed"),
                Exercise("Barbell Row", 4, 5, intensity - 5, 120, "Explosive pull"),
                Exercise("Pallof Press", 3, 12, 60, 60, "Anti-rotation")
            ]

    def _generate_power_workout(self, week: int, day: int) -> List[Exercise]:
        """Generate power maintenance workout (3-4x3-6 @ 70-80%, explosive)."""
        return [
            Exercise("Jump Squat", 4, 6, 30, 120, "Explosive, use barbell"),
            Exercise("Power Clean", 4, 3, 70, 150, "Focus on speed"),
            Exercise("Medicine Ball Slam", 3, 8, 0, 90, "Maximum effort"),
            Exercise("Plank to Push-up", 3, 10, 0, 60, "Dynamic stability")
        ]

    def _generate_recovery_workout(self, week: int, day: int) -> List[Exercise]:
        """Generate recovery phase workout (2x15-20 @ 40-50%)."""
        return [
            Exercise("Bodyweight Squat", 2, 20, 0, 45, "Mobility focus"),
            Exercise("Band Pull-Apart", 2, 20, 0, 30, "Light resistance"),
            Exercise("Dead Bug", 2, 15, 0, 30, "Core activation"),
            Exercise("Walking", 1, 10, 0, 0, "10 minutes easy")
        ]

    def _get_warmup_protocol(self, phase: StrengthPhase) -> str:
        """Get phase-specific warmup protocol."""
        if phase == StrengthPhase.MAX_STRENGTH:
            return "5min easy cardio, dynamic stretching, 3 warm-up sets with progressive loading"
        elif phase == StrengthPhase.POWER_MAINTENANCE:
            return "5min cardio, dynamic movements, plyometric prep, 2-3 explosive warm-up sets"
        else:
            return "5min easy cardio, dynamic stretching, mobility work"

    def _get_cooldown_protocol(self) -> str:
        """Get standard cooldown protocol."""
        return "5min easy cardio, static stretching, foam rolling focus areas"

    def _generate_workout_notes(self, phase: StrengthPhase, week: int) -> str:
        """Generate phase and week-specific training notes."""
        notes = {
            StrengthPhase.ANATOMICAL_ADAPTATION: f"Week {week}: Focus on perfect form and building work capacity",
            StrengthPhase.MAX_STRENGTH: f"Week {week}: Heavy loads, full recovery between sets",
            StrengthPhase.POWER_MAINTENANCE: f"Week {week}: Explosive movement, quality over quantity",
            StrengthPhase.RECOVERY: f"Week {week}: Light movement, mobility and activation"
        }
        return notes.get(phase, "")

    def generate_streprogen_program(self,
                                   phase: StrengthPhase,
                                   duration_weeks: int = 4,
                                   athlete_level: str = "intermediate",
                                   current_maxes: Optional[Dict[str, float]] = None) -> Optional[Program]:
        """
        Generate a comprehensive periodized program using streprogen.

        Args:
            phase: Current strength training phase
            duration_weeks: Program duration in weeks
            athlete_level: beginner/intermediate/advanced
            current_maxes: Dictionary of current 1RM estimates {exercise_name: weight}

        Returns:
            Complete streprogen Program with DynamicExercise progression
        """
        if not STREPROGEN_AVAILABLE:
            self.logger.warning("streprogen not installed - cannot generate full program")
            return None

        try:
            from streprogen import Program, DynamicExercise, StaticExercise, Day
            from streprogen import progression_sinusoidal, progression_sawtooth
            import warnings

            # Create program with phase-specific parameters
            program_name = f"{phase.value.replace('_', ' ').title()} Program ({duration_weeks}w)"

            # Phase-specific program configuration
            if phase == StrengthPhase.ANATOMICAL_ADAPTATION:
                program = Program(
                    name=program_name,
                    duration=duration_weeks,
                    reps_per_exercise=35,  # Reduced volume to avoid warnings
                    min_reps=8,
                    max_reps=12,
                    intensity=70,  # Within streprogen's recommended range
                    percent_inc_per_week=2.0,  # Conservative progression
                    units="kg",
                    round_to=1.25  # Smaller increments for technique work
                    # Remove progression_func to use default
                )

            elif phase == StrengthPhase.MAX_STRENGTH:
                program = Program(
                    name=program_name,
                    duration=duration_weeks,
                    reps_per_exercise=25,  # Standard volume for streprogen
                    min_reps=5,
                    max_reps=8,  # Consistent rep range
                    intensity=75,  # Lower intensity to match rep range perfectly
                    percent_inc_per_week=1.0,  # Conservative progression
                    units="kg",
                    round_to=2.5  # Standard increments
                )

            elif phase == StrengthPhase.POWER_MAINTENANCE:
                program = Program(
                    name=program_name,
                    duration=duration_weeks,
                    reps_per_exercise=15,  # Very low volume
                    min_reps=2,
                    max_reps=4,
                    intensity=75,  # Moderate-high for speed
                    percent_inc_per_week=0.5,  # Minimal progression - focus on speed
                    units="kg",
                    round_to=2.5,
                    progression_func=progression_sinusoidal  # Maintain consistency
                )

            else:  # Recovery phase
                program = Program(
                    name=program_name,
                    duration=duration_weeks,
                    reps_per_exercise=30,  # Moderate volume for movement quality
                    min_reps=8,
                    max_reps=12,
                    intensity=50,  # Very light
                    percent_inc_per_week=0.0,  # No progression - recovery focus
                    units="kg",
                    round_to=1.25,
                    progression_func=progression_sinusoidal
                )

            # Set up default starting weights based on athlete level if not provided
            if current_maxes is None:
                level_multipliers = {
                    "beginner": {"squat": 1.0, "deadlift": 1.2, "bench": 0.8, "press": 0.6},
                    "intermediate": {"squat": 1.5, "deadlift": 1.8, "bench": 1.2, "press": 0.9},
                    "advanced": {"squat": 2.0, "deadlift": 2.3, "bench": 1.5, "press": 1.1}
                }
                base_weight = 70  # kg baseline for average athlete
                multipliers = level_multipliers.get(athlete_level, level_multipliers["intermediate"])
                current_maxes = {
                    "squat": base_weight * multipliers["squat"],
                    "deadlift": base_weight * multipliers["deadlift"],
                    "bench": base_weight * multipliers["bench"],
                    "press": base_weight * multipliers["press"]
                }

            # Create training days based on phase (with warning suppression)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._add_training_days_to_program(program, phase, current_maxes, duration_weeks)

            return program

        except Exception as e:
            self.logger.error(f"Failed to generate streprogen program: {e}")
            return None

    def _add_training_days_to_program(self, program: Program, phase: StrengthPhase,
                                     current_maxes: Dict[str, float], duration_weeks: int):
        """Add phase-specific training days with DynamicExercise progression."""
        from streprogen import DynamicExercise, StaticExercise, Day

        if phase == StrengthPhase.ANATOMICAL_ADAPTATION:
            # 3x per week full body for adaptation

            # Day A - Lower body focus
            day_a = Day(name="Lower Body Focus")
            day_a.add_exercises(
                DynamicExercise("Goblet Squat",
                              start_weight=current_maxes["squat"] * 0.4,
                              final_weight=current_maxes["squat"] * 0.6),  # Increased progression
                DynamicExercise("Romanian Deadlift",
                              start_weight=current_maxes["deadlift"] * 0.4,
                              final_weight=current_maxes["deadlift"] * 0.6),  # Increased progression
                StaticExercise("Walking Lunge", "3 x 12 each leg"),
                StaticExercise("Plank", "3 x 45-60s"),
                StaticExercise("Glute Bridge", "3 x 15-20")
            )
            program.add_days(day_a)

            # Day B - Upper body focus
            day_b = Day(name="Upper Body Focus")
            day_b.add_exercises(
                DynamicExercise("Push-up Progression",
                              start_weight=0,  # Bodyweight
                              final_weight=5,  # Add weight if needed
                              min_reps=10, max_reps=15),
                DynamicExercise("Dumbbell Row",
                              start_weight=current_maxes["bench"] * 0.3,
                              final_weight=current_maxes["bench"] * 0.4,
                              min_reps=10, max_reps=12),
                DynamicExercise("Overhead Press",
                              start_weight=current_maxes["press"] * 0.6,
                              final_weight=current_maxes["press"] * 0.7,
                              min_reps=8, max_reps=12),
                StaticExercise("Face Pull", "3 x 15-20"),
                StaticExercise("Dead Bug", "3 x 10 each side")
            )
            program.add_days(day_b)

        elif phase == StrengthPhase.MAX_STRENGTH:
            # 2x per week with heavy compounds

            # Day A - Lower body power
            day_a = Day(name="Lower Power Day")
            day_a.add_exercises(
                DynamicExercise("Back Squat",
                              start_weight=current_maxes["squat"] * 0.70,
                              final_weight=current_maxes["squat"] * 0.78),  # Only 8% increase
                DynamicExercise("Romanian Deadlift",
                              start_weight=current_maxes["deadlift"] * 0.65,
                              final_weight=current_maxes["deadlift"] * 0.72),  # Only 7% increase
                StaticExercise("Box Jump", "4 x 5 explosive"),
                StaticExercise("Hanging Leg Raise", "3 x 8-12")
            )
            program.add_days(day_a)

            # Day B - Upper body power
            day_b = Day(name="Upper Power Day")
            day_b.add_exercises(
                DynamicExercise("Bench Press",
                              start_weight=current_maxes["bench"] * 0.75,
                              final_weight=current_maxes["bench"] * 0.82),  # Only 7% increase
                DynamicExercise("Barbell Row",  # Uses your rowing equipment strength
                              start_weight=current_maxes["bench"] * 0.70,
                              final_weight=current_maxes["bench"] * 0.77),  # Only 7% increase
                DynamicExercise("Overhead Press",
                              start_weight=current_maxes["press"] * 0.70,
                              final_weight=current_maxes["press"] * 0.77),  # Only 7% increase
                StaticExercise("Pallof Press", "3 x 10-12 each side")
            )
            program.add_days(day_b)

        elif phase == StrengthPhase.POWER_MAINTENANCE:
            # 1x per week explosive focus
            day_power = Day(name="Power Maintenance")
            day_power.add_exercises(
                DynamicExercise("Jump Squat",
                              start_weight=current_maxes["squat"] * 0.25,
                              final_weight=current_maxes["squat"] * 0.30,
                              min_reps=3, max_reps=5),
                DynamicExercise("Power Clean",
                              start_weight=current_maxes["deadlift"] * 0.5,
                              final_weight=current_maxes["deadlift"] * 0.6,
                              min_reps=2, max_reps=4),
                StaticExercise("Medicine Ball Slam", "4 x 6 explosive"),
                StaticExercise("Plank to Push-up", "3 x 8-10")
            )
            program.add_days(day_power)

        else:  # Recovery
            day_recovery = Day(name="Active Recovery")
            day_recovery.add_exercises(
                StaticExercise("Bodyweight Squat", "2 x 15-20"),
                StaticExercise("Band Pull-Apart", "2 x 20"),
                StaticExercise("Dead Bug", "2 x 10 each side"),
                StaticExercise("Walking", "10-15 minutes easy")
            )
            program.add_days(day_recovery)

    def export_program_to_file(self, program: Program, filepath: str, format: str = "txt") -> bool:
        """
        Export streprogen program to file in various formats.

        Args:
            program: Generated streprogen Program
            filepath: Output file path
            format: Export format ("txt", "html", "tex")

        Returns:
            True if export successful, False otherwise
        """
        if not STREPROGEN_AVAILABLE or program is None:
            return False

        try:
            import os
            import warnings

            # Suppress streprogen warnings during export
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Ensure program is rendered first
                program.render()

                # Create directory if needed
                if os.path.dirname(filepath):
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # Export based on format
                if format.lower() == "html":
                    content = program.to_html()
                elif format.lower() == "tex":
                    content = program.to_tex()
                else:  # Default to txt
                    content = program.to_txt()

                # Write to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            self.logger.info(f"Program exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export program: {e}")
            return False

    def create_autoregulated_program(self,
                                   phase: StrengthPhase,
                                   duration_weeks: int = 4,
                                   rpe_target: float = 8.0,
                                   current_maxes: Optional[Dict[str, float]] = None) -> Optional[Program]:
        """
        Create an autoregulated program using RPE-based progression.

        Args:
            phase: Training phase
            duration_weeks: Program duration
            rpe_target: Target RPE (Rate of Perceived Exertion)
            current_maxes: Current 1RM estimates

        Returns:
            Program with autoregulated progression
        """
        if not STREPROGEN_AVAILABLE:
            return None

        try:
            from streprogen import Program, DynamicExercise

            # Convert RPE to intensity percentage (approximate)
            rpe_to_intensity = {
                6.0: 60, 6.5: 65, 7.0: 70, 7.5: 75,
                8.0: 80, 8.5: 85, 9.0: 90, 9.5: 95, 10.0: 100
            }
            intensity = rpe_to_intensity.get(rpe_target, 80)

            # Create intensity scaling function for autoregulation
            def intensity_scaler(week):
                """Vary intensity slightly each week for autoregulation."""
                import math
                base_variation = 0.95 + 0.1 * math.sin(week * math.pi / 4)  # ±5% variation
                return base_variation

            program = self.generate_streprogen_program(
                phase=phase,
                duration_weeks=duration_weeks,
                current_maxes=current_maxes
            )

            # Modify program for autoregulation
            if program:
                program.name += " (RPE-based)"
                # Note: Advanced autoregulation would require custom streprogen extensions
                self.logger.info(f"Created autoregulated program targeting RPE {rpe_target}")

            return program

        except Exception as e:
            self.logger.error(f"Failed to create autoregulated program: {e}")
            return None

    def generate_deload_week_program(self,
                                   base_phase: StrengthPhase,
                                   current_maxes: Dict[str, float]) -> Optional[Program]:
        """
        Generate a specialized deload week program.

        Args:
            base_phase: The phase this deload follows
            current_maxes: Current strength levels

        Returns:
            Single week deload program
        """
        if not STREPROGEN_AVAILABLE:
            return None

        try:
            from streprogen import Program, DynamicExercise, StaticExercise, Day

            # Deload parameters: 50-60% intensity, 50% volume
            program = Program(
                name=f"Deload Week ({base_phase.value.replace('_', ' ').title()})",
                duration=1,  # One week only
                reps_per_exercise=15,  # Reduced volume
                min_reps=6,
                max_reps=10,
                intensity=55,  # Very light
                percent_inc_per_week=0,  # No progression during deload
                units="kg",
                round_to=2.5
            )

            # Create deload day focusing on movement quality
            deload_day = Day(name="Deload - Movement Quality")

            if base_phase == StrengthPhase.MAX_STRENGTH:
                # Light versions of heavy lifts
                deload_day.add_exercises(
                    DynamicExercise("Squat (light)",
                                  start_weight=current_maxes["squat"] * 0.5,
                                  final_weight=current_maxes["squat"] * 0.5,
                                  min_reps=8, max_reps=10),
                    DynamicExercise("Bench Press (light)",
                                  start_weight=current_maxes["bench"] * 0.5,
                                  final_weight=current_maxes["bench"] * 0.5,
                                  min_reps=8, max_reps=10),
                    StaticExercise("Mobility Flow", "15 minutes"),
                    StaticExercise("Core Activation", "3 x 10")
                )
            else:
                # General movement restoration
                deload_day.add_exercises(
                    StaticExercise("Bodyweight Movement", "20 minutes"),
                    StaticExercise("Flexibility Work", "15 minutes"),
                    StaticExercise("Activation Exercises", "3 x 12"),
                    StaticExercise("Light Cardio", "10 minutes easy")
                )

            program.add_days(deload_day)
            return program

        except Exception as e:
            self.logger.error(f"Failed to generate deload program: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def analyze_program_metrics(self, program: Program) -> Dict[str, Any]:
        """
        Analyze streprogen program metrics and provide insights.

        Args:
            program: Generated streprogen program

        Returns:
            Dictionary of program analysis metrics
        """
        if not STREPROGEN_AVAILABLE or program is None:
            return {}

        try:
            analysis = {
                "program_name": program.name,
                "duration_weeks": program.duration,
                "total_days": len(program.days),
                "training_frequency": len(program.days),  # Days per week
                "phase_characteristics": {},
                "volume_progression": [],
                "intensity_progression": []
            }

            # Analyze each day
            day_analysis = []
            total_exercises = 0

            for day in program.days:
                day_info = {
                    "name": day.name,
                    "exercises": len(day.exercises),
                    "dynamic_exercises": sum(1 for ex in day.exercises if hasattr(ex, 'start_weight')),
                    "static_exercises": sum(1 for ex in day.exercises if not hasattr(ex, 'start_weight'))
                }
                day_analysis.append(day_info)
                total_exercises += len(day.exercises)

            analysis["days"] = day_analysis
            analysis["total_exercises"] = total_exercises
            analysis["avg_exercises_per_day"] = total_exercises / len(program.days) if program.days else 0

            # Program characteristics based on parameters
            if hasattr(program, 'intensity'):
                if program.intensity >= 85:
                    analysis["intensity_classification"] = "High Intensity"
                elif program.intensity >= 70:
                    analysis["intensity_classification"] = "Moderate Intensity"
                else:
                    analysis["intensity_classification"] = "Low Intensity"

            if hasattr(program, 'reps_per_exercise'):
                if program.reps_per_exercise >= 35:
                    analysis["volume_classification"] = "High Volume"
                elif program.reps_per_exercise >= 20:
                    analysis["volume_classification"] = "Moderate Volume"
                else:
                    analysis["volume_classification"] = "Low Volume"

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze program: {e}")
            return {}

    def get_program_summary(self, program: Program) -> str:
        """
        Generate a human-readable summary of the streprogen program.

        Args:
            program: Generated streprogen program

        Returns:
            Formatted string summary
        """
        if not program:
            return "No program available"

        analysis = self.analyze_program_metrics(program)

        summary = f"""
📋 Strength Program Summary: {analysis.get('program_name', 'Unknown')}

⏱️  Duration: {analysis.get('duration_weeks', 0)} weeks
🏋️  Training Days: {analysis.get('total_days', 0)} per week
💪 Total Exercises: {analysis.get('total_exercises', 0)}
📊 Volume: {analysis.get('volume_classification', 'Unknown')}
⚡ Intensity: {analysis.get('intensity_classification', 'Unknown')}

📅 Weekly Structure:"""

        for day in analysis.get('days', []):
            summary += f"""
   • {day['name']}: {day['exercises']} exercises ({day['dynamic_exercises']} progressive, {day['static_exercises']} static)"""

        return summary

    def _load_strength_config(self):
        """Load strength training configuration from environment variables."""
        import os

        # Load 1RM values from .env (with fallbacks)
        self.current_maxes = {
            "bench": float(os.getenv("STRENGTH_BENCH_PRESS_1RM", 43.2)),
            "squat": float(os.getenv("STRENGTH_SQUAT_1RM", 100.0)),
            "deadlift": float(os.getenv("STRENGTH_DEADLIFT_1RM", 120.0)),
            "press": float(os.getenv("STRENGTH_OVERHEAD_PRESS_1RM", 30.0))
        }

        # Load preferences
        self.athlete_level = os.getenv("STRENGTH_ATHLETE_LEVEL", "intermediate")
        self.units = os.getenv("STRENGTH_PREFERRED_UNITS", "kg")
        self.round_to = float(os.getenv("STRENGTH_ROUND_TO", 2.5))

        # Load equipment-specific weights
        self.butterfly_weight = float(os.getenv("STRENGTH_BUTTERFLY_3RM", 65.0))
        self.rowing_weight = float(os.getenv("STRENGTH_ROWING_TRACTOR_3RM", 65.0))

        self.logger.info(f"Loaded strength config: Bench {self.current_maxes['bench']}kg, "
                        f"Squat {self.current_maxes['squat']}kg, "
                        f"Deadlift {self.current_maxes['deadlift']}kg, "
                        f"Press {self.current_maxes['press']}kg")

    def get_current_maxes(self) -> Dict[str, float]:
        """Get current 1RM estimates from configuration."""
        return self.current_maxes.copy()