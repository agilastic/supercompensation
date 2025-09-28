#!/usr/bin/env python3
"""
Training Load Model Validation Script

Validates our custom BanisterModel implementation against established
CTL/ATL calculation methods used by TrainingPeaks and other platforms.

This script implements the gold-standard exponential weighted average
formulas to ensure our training load calculations are accurate.

Phase 1: Training Load Validation
Author: Claude Code Assistant
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from strava_supercompensation.db import get_db
    from strava_supercompensation.db.models import Activity, Metric
    from strava_supercompensation.analysis.model import BanisterModel
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from project root: python scripts/validate_load_model.py")
    sys.exit(1)


class TrainingLoadValidator:
    """
    Validates CTL/ATL calculations using multiple methods:
    1. Our custom BanisterModel
    2. Simple rolling averages (basic method)
    3. Proper exponential weighted averages (TrainingPeaks method)
    """

    def __init__(self):
        self.console = Console()
        self.db = get_db()

        # Standard time constants (TrainingPeaks compatible)
        self.ctl_time_constant = 42  # days (fitness decay)
        self.atl_time_constant = 7   # days (fatigue decay)

        # Calculate lambda values for exponential weighting
        self.lambda_ctl = 1 - np.exp(-1 / self.ctl_time_constant)
        self.lambda_atl = 1 - np.exp(-1 / self.atl_time_constant)

    def get_training_data(self, days: int = 90) -> pd.DataFrame:
        """Fetch recent training data for validation."""
        try:
            with self.db.get_session() as session:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                # Get daily aggregated TSS values
                activities = session.query(Activity).filter(
                    Activity.start_date >= start_date,
                    Activity.start_date <= end_date
                ).order_by(Activity.start_date).all()

                # Create daily TSS DataFrame
                date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
                df = pd.DataFrame({'date': date_range, 'tss': 0.0})

                # Aggregate TSS by date
                for activity in activities:
                    date = activity.start_date.date()
                    if date in df['date'].dt.date.values:
                        idx = df[df['date'].dt.date == date].index[0]
                        df.loc[idx, 'tss'] += activity.training_load or 0

                df.set_index('date', inplace=True)
                return df

        except Exception as e:
            self.console.print(f"[red]Error fetching training data: {e}[/red]")
            return pd.DataFrame()

    def calculate_our_model(self, tss_series: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate CTL/ATL using our BanisterModel."""
        try:
            model = BanisterModel()

            # Convert to numpy array
            daily_loads = tss_series.values
            t = np.arange(len(daily_loads))

            # Use our model's impulse_response method
            fitness, fatigue, performance = model.impulse_response(daily_loads, t)

            return {
                'ctl': fitness,
                'atl': fatigue,
                'tsb': fitness - fatigue,
                'method': 'Our BanisterModel'
            }

        except Exception as e:
            self.console.print(f"[red]Error in our model: {e}[/red]")
            return {'ctl': np.array([]), 'atl': np.array([]), 'tsb': np.array([])}

    def calculate_rolling_average(self, tss_series: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate CTL/ATL using simple rolling averages (basic method)."""
        try:
            # Simple rolling averages as baseline comparison
            ctl = tss_series.rolling(window=self.ctl_time_constant, min_periods=1).mean().values
            atl = tss_series.rolling(window=self.atl_time_constant, min_periods=1).mean().values
            tsb = ctl - atl

            return {
                'ctl': ctl,
                'atl': atl,
                'tsb': tsb,
                'method': 'Simple Rolling Average'
            }

        except Exception as e:
            self.console.print(f"[red]Error in rolling average: {e}[/red]")
            return {'ctl': np.array([]), 'atl': np.array([]), 'tsb': np.array([])}

    def calculate_exponential_weighted(self, tss_series: pd.Series) -> Dict[str, np.ndarray]:
        """
        Calculate CTL/ATL using proper exponential weighted averages.

        This is the TrainingPeaks/industry standard method.
        """
        try:
            daily_tss = tss_series.values
            n_days = len(daily_tss)

            ctl = np.zeros(n_days)
            atl = np.zeros(n_days)

            # Initialize with first day
            if n_days > 0:
                ctl[0] = daily_tss[0] * self.lambda_ctl
                atl[0] = daily_tss[0] * self.lambda_atl

            # Exponential weighted calculation
            for i in range(1, n_days):
                ctl[i] = daily_tss[i] * self.lambda_ctl + (1 - self.lambda_ctl) * ctl[i-1]
                atl[i] = daily_tss[i] * self.lambda_atl + (1 - self.lambda_atl) * atl[i-1]

            tsb = ctl - atl

            return {
                'ctl': ctl,
                'atl': atl,
                'tsb': tsb,
                'method': 'Exponential Weighted (TrainingPeaks Standard)'
            }

        except Exception as e:
            self.console.print(f"[red]Error in exponential weighted: {e}[/red]")
            return {'ctl': np.array([]), 'atl': np.array([]), 'tsb': np.array([])}

    def calculate_statistics(self, results: List[Dict]) -> pd.DataFrame:
        """Calculate comparison statistics between methods."""
        if len(results) < 2:
            return pd.DataFrame()

        # Use the exponential weighted method as the reference (gold standard)
        reference = next((r for r in results if 'Exponential' in r['method']), results[0])

        stats = []
        for result in results:
            if result['method'] == reference['method']:
                continue  # Skip comparing with itself

            for metric in ['ctl', 'atl', 'tsb']:
                ref_values = reference[metric]
                test_values = result[metric]

                if len(ref_values) > 0 and len(test_values) > 0:
                    # Calculate differences
                    min_len = min(len(ref_values), len(test_values))
                    ref_subset = ref_values[-min_len:]
                    test_subset = test_values[-min_len:]

                    abs_diff = np.abs(ref_subset - test_subset)
                    rel_diff = abs_diff / (ref_subset + 1e-10) * 100  # Avoid division by zero

                    stats.append({
                        'comparison': f"{result['method']} vs {reference['method']}",
                        'metric': metric.upper(),
                        'mean_abs_error': np.mean(abs_diff),
                        'max_abs_error': np.max(abs_diff),
                        'mean_rel_error_pct': np.mean(rel_diff),
                        'max_rel_error_pct': np.max(rel_diff),
                        'final_value_ref': ref_subset[-1],
                        'final_value_test': test_subset[-1],
                        'final_diff': abs(ref_subset[-1] - test_subset[-1]),
                        'final_rel_diff_pct': abs(ref_subset[-1] - test_subset[-1]) / (ref_subset[-1] + 1e-10) * 100
                    })

        return pd.DataFrame(stats)

    def print_validation_results(self, stats_df: pd.DataFrame, results: List[Dict]):
        """Print comprehensive validation results."""

        # Title panel
        title = Panel.fit(
            "[bold blue]Training Load Model Validation Results[/bold blue]\n" +
            "[dim]Comparing our BanisterModel against industry standards[/dim]",
            border_style="blue"
        )
        self.console.print(title)

        if stats_df.empty:
            self.console.print("[red]No validation data available[/red]")
            return

        # Model configuration table
        config_table = Table(title="Model Configuration", border_style="dim")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_column("Note", style="dim")

        config_table.add_row("CTL Time Constant (τ1)", f"{self.ctl_time_constant} days", "Fitness decay rate")
        config_table.add_row("ATL Time Constant (τ2)", f"{self.atl_time_constant} days", "Fatigue decay rate")
        config_table.add_row("Lambda CTL (λ1)", f"{self.lambda_ctl:.6f}", "Exponential weight for fitness")
        config_table.add_row("Lambda ATL (λ2)", f"{self.lambda_atl:.6f}", "Exponential weight for fatigue")

        self.console.print(config_table)
        self.console.print()

        # Validation results table
        results_table = Table(title="Validation Results", border_style="green")
        results_table.add_column("Comparison", style="white")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Mean Error", style="yellow")
        results_table.add_column("Max Error", style="red")
        results_table.add_column("Mean %", style="yellow")
        results_table.add_column("Max %", style="red")
        results_table.add_column("Final Diff", style="magenta")
        results_table.add_column("Status", style="bold")

        for _, row in stats_df.iterrows():
            # Determine status based on error thresholds
            if row['mean_rel_error_pct'] < 2.0:
                status = "[green]✓ EXCELLENT[/green]"
            elif row['mean_rel_error_pct'] < 5.0:
                status = "[yellow]⚠ GOOD[/yellow]"
            elif row['mean_rel_error_pct'] < 10.0:
                status = "[orange3]△ FAIR[/orange3]"
            else:
                status = "[red]✗ NEEDS FIX[/red]"

            results_table.add_row(
                row['comparison'].replace(' vs ', '\nvs '),
                row['metric'],
                f"{row['mean_abs_error']:.2f}",
                f"{row['max_abs_error']:.2f}",
                f"{row['mean_rel_error_pct']:.1f}%",
                f"{row['max_rel_error_pct']:.1f}%",
                f"{row['final_diff']:.2f}",
                status
            )

        self.console.print(results_table)
        self.console.print()

        # Sample final values comparison
        final_table = Table(title="Final Values Comparison (Most Recent Day)", border_style="blue")
        final_table.add_column("Method", style="white")
        final_table.add_column("CTL", style="green")
        final_table.add_column("ATL", style="yellow")
        final_table.add_column("TSB", style="cyan")

        for result in results:
            if len(result['ctl']) > 0:
                final_table.add_row(
                    result['method'],
                    f"{result['ctl'][-1]:.1f}",
                    f"{result['atl'][-1]:.1f}",
                    f"{result['tsb'][-1]:.1f}"
                )

        self.console.print(final_table)

        # Interpretation and recommendations
        self.console.print()
        interpretation = Panel(
            self._get_interpretation(stats_df),
            title="[bold]Interpretation & Recommendations[/bold]",
            border_style="cyan"
        )
        self.console.print(interpretation)

    def _get_interpretation(self, stats_df: pd.DataFrame) -> str:
        """Generate interpretation of validation results."""
        if stats_df.empty:
            return "No data available for interpretation."

        # Find our model's performance
        our_model_stats = stats_df[stats_df['comparison'].str.contains('BanisterModel', na=False)]

        if our_model_stats.empty:
            return "BanisterModel comparison not found."

        avg_error = our_model_stats['mean_rel_error_pct'].mean()
        max_error = our_model_stats['max_rel_error_pct'].max()

        interpretation = ""

        if avg_error < 2.0:
            interpretation += "🎯 [green]VALIDATION PASSED[/green]: Your BanisterModel implementation is EXCELLENT!\n"
            interpretation += f"• Average error: {avg_error:.1f}% (Target: <2%)\n"
            interpretation += "• Your model matches TrainingPeaks standards\n"
            interpretation += "• No code changes needed - proceed with confidence\n"

        elif avg_error < 5.0:
            interpretation += "✅ [yellow]VALIDATION GOOD[/yellow]: Your BanisterModel is highly accurate\n"
            interpretation += f"• Average error: {avg_error:.1f}% (Acceptable: <5%)\n"
            interpretation += "• Minor discrepancies detected but acceptable for production\n"
            interpretation += "• Consider reviewing edge cases in initialization\n"

        elif avg_error < 10.0:
            interpretation += "⚠️ [orange3]VALIDATION FAIR[/orange3]: Your BanisterModel needs improvement\n"
            interpretation += f"• Average error: {avg_error:.1f}% (Warning: >5%)\n"
            interpretation += "• Review exponential decay calculations\n"
            interpretation += "• Check initialization values and formula implementation\n"

        else:
            interpretation += "🚨 [red]VALIDATION FAILED[/red]: Your BanisterModel requires fixes\n"
            interpretation += f"• Average error: {avg_error:.1f}% (Critical: >10%)\n"
            interpretation += "• Significant discrepancy from industry standards\n"
            interpretation += "• Review core CTL/ATL calculation formulas\n"
            interpretation += "• Compare with reference implementation line by line\n"

        interpretation += f"\n[bold]Technical Details:[/bold]\n"
        interpretation += f"• Maximum error observed: {max_error:.1f}%\n"
        interpretation += f"• Reference: Exponential Weighted Average (TrainingPeaks standard)\n"
        interpretation += f"• Formula: CTL(t) = TSS(t) × λ + CTL(t-1) × (1-λ)\n"
        interpretation += f"• Where λ = 1 - exp(-1/τ), τ = {self.ctl_time_constant} days for CTL"

        return interpretation

    def run_validation(self, days: int = 90) -> bool:
        """Run complete validation suite."""
        self.console.print(f"[blue]Fetching {days} days of training data...[/blue]")

        df = self.get_training_data(days)
        if df.empty or df['tss'].sum() == 0:
            self.console.print("[red]No training data found for validation[/red]")
            return False

        self.console.print(f"[green]Found {len(df)} days with {df['tss'].sum():.0f} total TSS[/green]")
        self.console.print()

        # Calculate using all methods
        results = []

        self.console.print("[blue]Calculating with our BanisterModel...[/blue]")
        results.append(self.calculate_our_model(df['tss']))

        self.console.print("[blue]Calculating with rolling averages...[/blue]")
        results.append(self.calculate_rolling_average(df['tss']))

        self.console.print("[blue]Calculating with exponential weighted averages...[/blue]")
        results.append(self.calculate_exponential_weighted(df['tss']))

        # Calculate statistics
        stats_df = self.calculate_statistics(results)

        # Print results
        self.console.print()
        self.print_validation_results(stats_df, results)

        # Return validation success
        if not stats_df.empty:
            our_model_avg_error = stats_df[
                stats_df['comparison'].str.contains('BanisterModel', na=False)
            ]['mean_rel_error_pct'].mean()
            return our_model_avg_error < 5.0  # 5% threshold for validation success

        return False


def main():
    """Main validation script entry point."""
    console = Console()

    console.print(Panel.fit(
        "[bold blue]Training Load Model Validation[/bold blue]\n" +
        "[dim]Phase 1: Validating CTL/ATL calculations against industry standards[/dim]",
        border_style="blue"
    ))

    validator = TrainingLoadValidator()

    try:
        success = validator.run_validation(days=90)

        if success:
            console.print("\n[green bold]✓ VALIDATION SUCCESSFUL[/green bold]")
            console.print("[green]Your BanisterModel implementation is validated and ready for production![/green]")
        else:
            console.print("\n[yellow bold]⚠ VALIDATION NEEDS ATTENTION[/yellow bold]")
            console.print("[yellow]Review the recommendations above to improve model accuracy.[/yellow]")

    except Exception as e:
        console.print(f"\n[red bold]✗ VALIDATION ERROR[/red bold]")
        console.print(f"[red]Error during validation: {e}[/red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())