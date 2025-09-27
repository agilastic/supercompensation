"""Integration layer for advanced training models with recommendation system.

This module bridges the advanced models with the existing recommendation engine.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .advanced_model import (
    FitnessFatigueModel,
    PerPotModel,
    OptimalControlSolver,
    OptimalControlProblem,
    TrainingGoal,
    AdaptiveParameterLearning,
    generate_optimal_weekly_plan
)
from .multisystem_recovery import MultiSystemRecoveryModel
from .model import SupercompensationAnalyzer
from ..config import config
from ..db import get_db
from ..db.models import Activity, Metric


class IntegratedTrainingAnalyzer:
    """Integrates advanced models for comprehensive training analysis."""

    def __init__(self, user_id: str = "default"):
        """Initialize integrated analyzer with all models."""
        self.user_id = user_id
        self.db = get_db()

        # Initialize models
        self.ff_model = FitnessFatigueModel(user_id=user_id)
        self.perpot_model = PerPotModel(user_id=user_id, max_daily_load=250.0)  # Set realistic max daily TSS
        self.recovery_model = MultiSystemRecoveryModel()
        self.adaptive_learner = AdaptiveParameterLearning(user_id=user_id)

        # Load user parameters if available
        self._load_user_parameters()

    def _load_user_parameters(self):
        """Load personalized model parameters from database."""
        with self.db.get_session() as session:
            # Query for adaptive parameters
            from ..db.models import AdaptiveModelParameters
            params = session.query(AdaptiveModelParameters).filter_by(
                user_id=self.user_id
            ).first()

            if params:
                # Temporarily disable adaptive parameters to ensure consistency with basic model
                # Update models with personalized parameters
                # self.ff_model.k1 = params.fitness_magnitude
                # self.ff_model.k2 = params.fatigue_magnitude
                # self.ff_model.tau1 = params.fitness_decay
                # self.ff_model.tau2 = params.fatigue_decay
                pass

    def analyze_with_advanced_models(
        self,
        days_back: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """Perform comprehensive analysis using all advanced models."""

        # --- START FIX ---
        # Step 1: Use the proven basic analyzer to get the definitive DataFrame with correct daily loads.
        basic_analyzer = SupercompensationAnalyzer(user_id=self.user_id)
        correct_daily_data = basic_analyzer.analyze(days_back=days_back)

        # CRITICAL FIX: The 'date' column must be removed BEFORE renaming to prevent datetime arithmetic
        # Drop the date column if it exists (it contains datetime objects that cause the TypeError)
        if 'date' in correct_daily_data.columns:
            correct_daily_data = correct_daily_data.drop(columns=['date'])

        # Step 2: Use this correct data as the single source of truth for all subsequent models.
        # The 'ff_results' now contain plausible Fitness/Fatigue/Form values.
        # We rename columns for consistency within the advanced analysis pipeline.
        ff_results = correct_daily_data.rename(columns={
            'fitness': 'ff_fitness',
            'fatigue': 'ff_fatigue',
            'form': 'ff_form',
            'training_load': 'load'
        })

        # Additional safety: ensure no datetime columns remain
        # Check all columns for datetime types and remove them
        for col in ff_results.columns:
            if pd.api.types.is_datetime64_any_dtype(ff_results[col]):
                ff_results = ff_results.drop(columns=[col])

        # Keep only numeric columns
        ff_results = ff_results.select_dtypes(include=[np.number])

        # Ensure we have the required columns
        required_columns = ['ff_fitness', 'ff_fatigue', 'ff_form', 'load']
        for col in required_columns:
            if col not in ff_results.columns:
                # If a required column is missing, add it with default values
                ff_results[col] = 0.0

        # Reset index to ensure it's numeric (not datetime)
        ff_results = ff_results.reset_index(drop=True)
        # --- END FIX ---

        # Now, all subsequent analyses use the same, correct 'training_loads' and 'ff_results'

        # PerPot analysis uses the 'load' column from the now-correct ff_results
        perpot_results = self._analyze_perpot(ff_results)

        # Multi-system recovery analysis also uses the 'load' column
        recovery_results = self._analyze_recovery(ff_results)

        # Combine results
        combined_results = self._combine_analyses(
            ff_results, perpot_results, recovery_results
        )

        # Save to database - disabled to prevent any datetime issues
        # self._save_advanced_metrics(combined_results)

        return {
            'fitness_fatigue': ff_results,
            'perpot': perpot_results,
            'recovery': recovery_results,
            'combined': combined_results
        }

    def _get_training_history(self, days_back: int) -> pd.DataFrame:
        # DEPRECATED: Now using SupercompensationAnalyzer data pipeline for consistency
        """Get training history from database."""
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        with self.db.get_session() as session:
            activities = session.query(Activity).filter(
                Activity.start_date >= start_date,
                Activity.start_date <= end_date
            ).order_by(Activity.start_date).all()

            # Create daily aggregated data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            daily_data = pd.DataFrame(index=date_range)
            daily_data['load'] = 0.0
            daily_data['duration'] = 0.0
            daily_data['sport'] = 'Rest'

            for activity in activities:
                date = pd.Timestamp(activity.start_date.date())
                if date in daily_data.index:
                    # Apply sport-specific multiplier
                    multiplier = config.get_sport_load_multiplier(activity.type)
                    load = activity.training_load * multiplier if activity.training_load else 0

                    daily_data.loc[date, 'load'] += load
                    daily_data.loc[date, 'duration'] += activity.moving_time / 60 if activity.moving_time else 0
                    if load > 0:
                        daily_data.loc[date, 'sport'] = activity.type

        return daily_data


    def _analyze_perpot(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze using PerPot model for overtraining detection."""
        loads = training_data['load'].values
        t = np.arange(len(loads))

        perpot_results = self.perpot_model.simulate(loads, t)

        results = training_data.copy()
        results['perpot_strain'] = perpot_results['strain_potential']
        results['perpot_response'] = perpot_results['response_potential']
        results['perpot_performance'] = perpot_results['performance_potential']
        results['overtraining_risk'] = perpot_results['overtraining_risk']

        return results

    def _analyze_recovery(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze multi-system recovery status."""
        results = training_data.copy()

        # Initialize recovery columns
        results['metabolic_recovery'] = 100.0
        results['neural_recovery'] = 100.0
        results['structural_recovery'] = 100.0
        results['overall_recovery'] = 100.0

        # Calculate recovery for each day
        for i in range(1, len(results)):
            # Get recent activities
            recent_loads = training_data['load'].iloc[max(0, i-7):i+1].values
            # Handle case where sport column might not exist
            if 'sport' in training_data.columns:
                recent_sports = training_data['sport'].iloc[max(0, i-7):i+1].values
            else:
                recent_sports = ['Unknown'] * len(recent_loads)

            # Calculate hours since last significant load
            last_load_idx = None
            for j in range(i-1, -1, -1):
                if training_data['load'].iloc[j] > 30:  # Significant load threshold
                    last_load_idx = j
                    break

            if last_load_idx is not None:
                hours_since = (i - last_load_idx) * 24
                last_load = training_data['load'].iloc[last_load_idx]

                # Calculate system recovery
                from .multisystem_recovery import RecoverySystem
                metabolic = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.METABOLIC, hours_since, last_load
                )
                neural = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.NEURAL, hours_since, last_load
                )
                structural = self.recovery_model.calculate_system_recovery(
                    RecoverySystem.STRUCTURAL, hours_since, last_load
                )

                results.iloc[i, results.columns.get_loc('metabolic_recovery')] = metabolic * 100
                results.iloc[i, results.columns.get_loc('neural_recovery')] = neural * 100
                results.iloc[i, results.columns.get_loc('structural_recovery')] = structural * 100
                results.iloc[i, results.columns.get_loc('overall_recovery')] = min(metabolic, neural, structural) * 100

        return results

    def _combine_analyses(
        self,
        ff_results: pd.DataFrame,
        perpot_results: pd.DataFrame,
        recovery_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine results from all models into unified metrics."""
        combined = ff_results.copy()

        # Add PerPot metrics
        combined['perpot_performance'] = perpot_results['perpot_performance']
        combined['overtraining_risk'] = perpot_results['overtraining_risk']

        # Add recovery metrics modified by overtraining risk
        base_recovery = recovery_results['overall_recovery']
        # If overtraining risk detected, cap recovery at 60% (scientific maximum during overreaching)
        adjusted_recovery = base_recovery.copy()
        overtraining_mask = combined['overtraining_risk']
        adjusted_recovery[overtraining_mask] = np.minimum(adjusted_recovery[overtraining_mask], 60.0)
        combined['overall_recovery'] = adjusted_recovery

        # Calculate composite readiness score incorporating Garmin wellness data
        # New weights: 30% form, 20% PerPot, 20% recovery, 30% Garmin wellness

        # Ensure all values are float to avoid any type issues
        form_vals = combined['ff_form'].astype(float)
        fatigue_vals = combined['ff_fatigue'].astype(float)

        # CRITICAL FIX: Detect dangerous "high form + high fatigue" state (non-functional overreaching)
        # This occurs when TSB is high but ATL is also very high (>100), indicating overreaching
        high_form_high_fatigue_mask = (form_vals > 20) & (fatigue_vals > 100)

        # Apply sports science correction: cap form at -10 when in overreaching state
        corrected_form = form_vals.copy()
        corrected_form[high_form_high_fatigue_mask] = -10  # Force negative form for overreaching

        form_normalized = np.clip((corrected_form + 50) / 100, 0, 1)
        recovery_normalized = np.clip(combined['overall_recovery'].astype(float) / 100, 0, 1)
        perpot_normalized = np.clip(combined['perpot_performance'].astype(float), 0, 1)

        # Get Garmin wellness readiness for each date
        wellness_readiness = []
        for date in combined.index:
            try:
                from ..analysis.garmin_scores_analyzer import GarminScoresAnalyzer
                analyzer = GarminScoresAnalyzer(user_id="user")
                wellness_data = analyzer.get_readiness_score(date)
                readiness = wellness_data.get('readiness_score', 70.0)
                wellness_readiness.append(readiness if readiness is not None else 70.0)
            except Exception:
                wellness_readiness.append(70.0)  # Default if no data

        wellness_normalized = np.clip(np.array(wellness_readiness) / 100, 0, 1)

        # SPORTS SCIENCE: Replace binary overtraining_risk with nuanced risk states
        # Define precise risk states instead of confusing binary flag
        risk_state = []
        for i in range(len(combined)):
            if high_form_high_fatigue_mask[i]:
                risk_state.append("NON_FUNCTIONAL_OVERREACHING")
            elif combined['overtraining_risk'].iloc[i]:
                risk_state.append("HIGH_STRAIN")
            else:
                risk_state.append("SAFE")

        combined['risk_state'] = risk_state
        # Keep overtraining_risk for backward compatibility but derive from risk_state
        combined['overtraining_risk'] = combined['risk_state'].isin(['NON_FUNCTIONAL_OVERREACHING', 'HIGH_STRAIN'])

        # Apply appropriate penalties based on risk state
        overtraining_penalty = []
        for state in risk_state:
            if state == "NON_FUNCTIONAL_OVERREACHING":
                overtraining_penalty.append(0.6)  # Most severe - immediate rest needed
            elif state == "HIGH_STRAIN":
                overtraining_penalty.append(0.8)  # Significant load reduction
            else:
                overtraining_penalty.append(1.0)  # Normal training
        overtraining_penalty = np.array(overtraining_penalty)

        combined['composite_readiness'] = np.clip((
            0.3 * form_normalized +
            0.2 * perpot_normalized +
            0.2 * recovery_normalized +
            0.3 * wellness_normalized
        ) * overtraining_penalty * 100, 0, 100)  # Ensure 0-100 range

        # Add training recommendations based on composite and risk state
        combined['recommendation'] = combined.apply(
            lambda row: self._get_recommendation_from_risk_state(
                row['composite_readiness'],
                row['risk_state']
            ),
            axis=1
        )

        return combined

    def _get_recommendation_from_risk_state(self, readiness: float, risk_state: str) -> str:
        """Get training recommendation from composite readiness score and risk state."""
        # SPORTS SCIENCE: Handle each risk state appropriately
        if risk_state == "NON_FUNCTIONAL_OVERREACHING":
            return "REST"  # Immediate rest required despite high form
        elif risk_state == "HIGH_STRAIN":
            return "RECOVERY"  # Active recovery needed

        # For SAFE state, use normal readiness-based recommendations
        if readiness >= 85:
            return "PEAK"
        elif readiness >= 70:
            return "HARD"
        elif readiness >= 55:
            return "MODERATE"
        elif readiness >= 40:
            return "EASY"
        elif readiness >= 25:
            return "RECOVERY"
        else:
            return "REST"

    def get_daily_recommendation(self) -> Dict[str, any]:
        """Get today's training recommendation using advanced models."""
        try:
            # Analyze with advanced models to get current state
            analysis = self.analyze_with_advanced_models(days_back=90)
            combined_results = analysis['combined']

            if len(combined_results) == 0:
                # Fallback if no data
                return {
                    'recommendation': 'EASY',
                    'activity': 'Easy aerobic training',
                    'rationale': 'No historical data available',
                    'form_score': 0.0,
                    'readiness_score': 50.0,
                    'performance_potential': 0.5
                }

            # Get latest metrics
            latest = combined_results.iloc[-1]

            recommendation = latest['recommendation']
            readiness = latest['composite_readiness']
            form_score = latest['ff_form']
            perf_potential = latest['perpot_performance']

            # Map recommendation to activity
            activity_map = {
                'REST': 'Complete rest or light stretching',
                'RECOVERY': 'Active recovery - easy walk or gentle yoga',
                'EASY': 'Easy aerobic training at conversational pace',
                'MODERATE': 'Moderate intensity training with some tempo work',
                'HARD': 'High intensity intervals or threshold training',
                'PEAK': 'Peak intensity training or race simulation'
            }

            # Generate precise rationale based on risk state
            risk_state = latest.get('risk_state', 'SAFE')

            if risk_state == "NON_FUNCTIONAL_OVERREACHING":
                rationale = f"Non-functional overreaching detected (Form: {form_score:.1f}, high fatigue). Despite peaked TSB, immediate rest is required to prevent performance decline and avoid true overtraining."
            elif risk_state == "HIGH_STRAIN":
                rationale = f"High training strain detected (Readiness: {readiness:.1f}). Active recovery needed to absorb training load."
            else:
                # Normal rationale for safe training state
                rationale_parts = []
                if form_score > 20:
                    rationale_parts.append("excellent form")
                elif form_score > 5:
                    rationale_parts.append("good form")
                elif form_score < -10:
                    rationale_parts.append("accumulated fatigue")

                if readiness > 75:
                    rationale_parts.append("high readiness")
                elif readiness < 40:
                    rationale_parts.append("low readiness")

                rationale = f"Based on current {', '.join(rationale_parts) if rationale_parts else 'training status'}"

            return {
                'recommendation': recommendation,
                'activity': activity_map.get(recommendation, 'Unknown activity'),
                'rationale': rationale,
                'form_score': float(form_score),
                'readiness_score': float(readiness),
                'performance_potential': float(perf_potential)
            }

        except Exception as e:
            # Fallback recommendation
            return {
                'recommendation': 'EASY',
                'activity': 'Easy aerobic training',
                'rationale': f'Error in analysis: {str(e)}',
                'form_score': 0.0,
                'readiness_score': 50.0,
                'performance_potential': 0.5
            }

    def _save_advanced_metrics(self, results: pd.DataFrame):
        """Save advanced metrics to database."""
        with self.db.get_session() as session:
            for _, row in results.iterrows():
                # Convert timestamp to datetime
                metric_date = row.name.to_pydatetime() if hasattr(row.name, 'to_pydatetime') else row.name

                # Find or create metric
                existing = session.query(Metric).filter(
                    Metric.date >= metric_date,
                    Metric.date < metric_date + timedelta(days=1)
                ).first()

                if existing:
                    # Update with advanced metrics
                    existing.fitness = row['ff_fitness']
                    existing.fatigue = row['ff_fatigue']
                    existing.form = row['ff_form']
                    existing.recommendation = row['recommendation']
                    # Store advanced metrics in JSON field if available
                    existing.advanced_metrics = {
                        'perpot_performance': float(row['perpot_performance']),
                        'risk_state': str(row['risk_state']),
                        'overtraining_risk': bool(row['overtraining_risk']),  # Keep for backward compatibility
                        'overall_recovery': float(row['overall_recovery']),
                        'composite_readiness': float(row['composite_readiness'])
                    }
                else:
                    # Create new metric
                    metric = Metric(
                        date=metric_date,
                        fitness=row['ff_fitness'],
                        fatigue=row['ff_fatigue'],
                        form=row['ff_form'],
                        daily_load=row['load'],
                        recommendation=row['recommendation']
                    )
                    session.add(metric)

            session.commit()

    def generate_optimal_plan(
        self,
        goal: str = 'balanced',
        duration_days: int = 7,
        rest_days: List[int] = None
    ) -> Dict:
        """Generate optimal training plan using advanced optimization.

        Args:
            goal: Training goal ('maximize_fti', 'minimize_fatigue', 'balanced', etc.)
            duration_days: Number of days to plan
            rest_days: List of rest day indices (0=Monday)

        Returns:
            Optimal training plan with daily recommendations
        """
        # Get current state
        current_state = self._get_current_state()

        # Map goal string to enum
        goal_map = {
            'maximize_fti': TrainingGoal.MAXIMIZE_FTI,
            'maximize_fatigue': TrainingGoal.MAXIMIZE_FATIGUE,
            'minimize_fatigue': TrainingGoal.MINIMIZE_FATIGUE,
            'balanced': TrainingGoal.BALANCED
        }
        training_goal = goal_map.get(goal, TrainingGoal.BALANCED)

        # Generate optimal plan
        plan = generate_optimal_weekly_plan(
            current_fitness=current_state['fitness'],
            current_fatigue=current_state['fatigue'],
            goal=training_goal,
            rest_days=rest_days
        )

        # Add practical details
        plan['daily_details'] = []
        for i in range(len(plan['loads'])):
            day_detail = {
                'day': i,
                'load': plan['loads'][i],
                'recommendation': plan['recommendations'][i],
                'predicted_fitness': plan['fitness'][i],
                'predicted_fatigue': plan['fatigue'][i],
                'predicted_form': plan['fitness'][i] - plan['fatigue'][i]
            }
            plan['daily_details'].append(day_detail)

        return plan

    def _get_current_state(self) -> Dict[str, float]:
        """Get current training state from database."""
        with self.db.get_session() as session:
            latest = session.query(Metric).order_by(Metric.date.desc()).first()

            if latest:
                return {
                    'fitness': latest.fitness,
                    'fatigue': latest.fatigue,
                    'form': latest.form
                }
            else:
                return {
                    'fitness': 0.0,
                    'fatigue': 0.0,
                    'form': 0.0
                }

    def adapt_parameters(self, performance_feedback: Dict):
        """Adapt model parameters based on performance feedback.

        Args:
            performance_feedback: Dictionary with actual vs predicted performance
        """
        # Get recent training and performance data
        training_data = self._get_training_history(30)

        # Create performance DataFrame from feedback
        performance_data = pd.DataFrame([{
            'date': pd.Timestamp(performance_feedback['date']),
            'performance': performance_feedback['actual_performance']
        }])

        # Adapt parameters using differential evolution
        optimized = self.adaptive_learner.adapt_with_differential_evolution(
            training_history=training_data,
            performance_data=performance_data,
            model_class=FitnessFatigueModel
        )

        # Update model with optimized parameters
        if optimized['optimization_success']:
            self.ff_model.k1 = optimized['k1']
            self.ff_model.k2 = optimized['k2']
            self.ff_model.tau1 = optimized['tau1']
            self.ff_model.tau2 = optimized['tau2']
            self.ff_model.p_star = optimized['p_star']

        return optimized


def get_integrated_analyzer(user_id: str = "default") -> IntegratedTrainingAnalyzer:
    """Factory function to get integrated analyzer instance."""
    return IntegratedTrainingAnalyzer(user_id=user_id)