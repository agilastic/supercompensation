"""Integration layer for advanced training models with recommendation system.

This module bridges the advanced models with the existing recommendation engine.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .advanced_model import (
    EnhancedFitnessFatigueModel,
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
        self.ff_model = EnhancedFitnessFatigueModel(user_id=user_id)
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

        # Use the proven basic analyzer to get a DataFrame with correct daily loads
        basic_analyzer = SupercompensationAnalyzer(user_id=self.user_id)
        correct_daily_data = basic_analyzer.analyze(days_back=days_back)

        # Extract ONLY the raw loads and date range to prevent TypeErrors
        training_loads = correct_daily_data['training_load'].values
        date_index = correct_daily_data['date']
        t = np.arange(len(training_loads))

        # Create a base DataFrame for this analysis run using the correct data
        base_df = pd.DataFrame(index=date_index)
        base_df['load'] = training_loads

        # Enhanced Fitness-Fatigue analysis
        ff_results = self._analyze_fitness_fatigue(base_df)

        # PerPot analysis with overtraining detection
        perpot_results = self._analyze_perpot(base_df)

        # Multi-system recovery analysis
        recovery_results = self._analyze_recovery(base_df)

        # Combine results
        combined_results = self._combine_analyses(
            ff_results, perpot_results, recovery_results
        )

        # Save to database
        self._save_advanced_metrics(combined_results)

        return {
            'fitness_fatigue': ff_results,
            'perpot': perpot_results,
            'recovery': recovery_results,
            'combined': combined_results
        }

    def _get_training_history(self, days_back: int) -> pd.DataFrame:
        # DEPRECATED: Now using SupercompensationAnalyzer data pipeline for consistency
        """Get training history from database."""
        end_date = datetime.utcnow().date()
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

    def _analyze_fitness_fatigue(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze using enhanced Fitness-Fatigue model."""
        loads = training_data['load'].values
        t = np.arange(len(loads))

        fitness, fatigue, performance = self.ff_model.impulse_response(loads, t)

        results = training_data.copy()
        results['ff_fitness'] = fitness
        results['ff_fatigue'] = fatigue
        results['ff_performance'] = performance
        # Form is now correctly calculated by the enhanced model
        results['ff_form'] = performance

        return results

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

                results.loc[results.index[i], 'metabolic_recovery'] = metabolic * 100
                results.loc[results.index[i], 'neural_recovery'] = neural * 100
                results.loc[results.index[i], 'structural_recovery'] = structural * 100
                results.loc[results.index[i], 'overall_recovery'] = min(metabolic, neural, structural) * 100

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

        # Calculate composite readiness score
        # Weight: 40% form, 30% PerPot, 30% recovery
        # Form normalization: Scale form to 0-1 range (form typically ranges from -50 to +50 for realistic training)
        form_normalized = np.clip((combined['ff_form'] + 50) / 100, 0, 1)
        recovery_normalized = np.clip(combined['overall_recovery'] / 100, 0, 1)
        perpot_normalized = np.clip(combined['perpot_performance'], 0, 1)

        # Heavy penalty for overtraining risk
        overtraining_penalty = combined['overtraining_risk'].apply(lambda x: 0.5 if x else 1.0)

        combined['composite_readiness'] = np.clip((
            0.4 * form_normalized +
            0.3 * perpot_normalized +
            0.3 * recovery_normalized
        ) * overtraining_penalty * 100, 0, 100)  # Ensure 0-100 range

        # Add training recommendations based on composite
        combined['recommendation'] = combined.apply(
            lambda row: self._get_recommendation_from_composite(
                row['composite_readiness'],
                row['overtraining_risk']
            ),
            axis=1
        )

        return combined

    def _get_recommendation_from_composite(self, readiness: float, overtraining_risk: bool = False) -> str:
        """Get training recommendation from composite readiness score."""
        # Override with rest if overtraining detected
        if overtraining_risk:
            return "REST"

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

            # Generate rationale
            rationale_parts = []
            if latest['overtraining_risk']:
                rationale_parts.append("overtraining risk detected")
            if form_score > 20:
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
                        'overtraining_risk': bool(row['overtraining_risk']),
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
            model_class=EnhancedFitnessFatigueModel
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