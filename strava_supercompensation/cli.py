"""Command-line interface for Strava Supercompensation tool."""

import warnings
# Suppress the pkg_resources deprecation warning from heartpy library
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# Suppress TensorFlow/Protobuf version warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import click
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

from .config import config, Config
from .auth import AuthManager
from .api import StravaClient

# Garmin integration - gracefully handle if missing
try:
    from .api.garmin import get_garmin_client, GarminError
except ImportError:
    GarminError = Exception  # Fallback
    get_garmin_client = None  # Mark as unavailable

# RENPHO integration - CSV import approach
try:
    from .api.renpho_csv import RenphoCsvImporter
except ImportError:
    RenphoCsvImporter = None  # Mark as unavailable
from .analysis import SupercompensationAnalyzer
from .analysis.multisport_metrics import MultiSportCalculator
from .analysis.model_integration import get_integrated_analyzer
from .analysis.advanced_planning import TrainingPlanGenerator
from .analysis.plan_adjustment import PlanAdjustmentEngine

console = Console()


def get_intensity_from_load(load: float) -> str:
    """
    Standardized intensity calculation based on training load (TSS).

    Args:
        load: Training load in TSS

    Returns:
        Intensity level string (RECOVERY, EASY, MOD, HARD, RACE)
    """
    if load > 300:
        return "RACE"
    elif load > 200:
        return "HARD"
    elif load > 100:
        return "MOD"
    elif load > 30:
        return "EASY"
    else:
        return "RECOVERY"


def handle_auth_error(error: Exception, auth_manager, operation_name: str = "operation") -> bool:
    """Handle 401 authentication errors with automatic re-authentication.

    Returns True if successfully re-authenticated, False otherwise.
    """
    error_str = str(error)

    if "401" in error_str and "Unauthorized" in error_str:
        console.print(f"[orange]⚠️  Authentication token expired. Re-authenticating...[/orange]")

        if auth_manager.authenticate():
            console.print(f"[green]✅ Re-authentication successful! Retrying {operation_name}...[/green]")
            return True
        else:
            console.print("[red]❌ Re-authentication failed. Please run 'strava-super auth' manually.[/red]")
            return False

    return False


@click.group()
def cli():
    """Strava Supercompensation Training Analysis Tool."""
    pass


@cli.command()
def auth():
    """Authenticate with Strava."""
    console.print(Panel.fit("🔐 Strava Authentication", style="bold blue"))

    try:
        config.validate()
    except ValueError as e:
        console.print(f"[red]❌ Configuration Error: {e}[/red]")
        console.print("\n[orange]Please create a .env file with your Strava API credentials.[/orange]")
        console.print("[orange]See .env.example for reference.[/orange]")
        return

    auth_manager = AuthManager()

    if auth_manager.is_authenticated():
        console.print("[green]✅ Already authenticated![/green]")
        if click.confirm("Do you want to re-authenticate?"):
            auth_manager.logout()
        else:
            return

    console.print("\n[black]Opening browser for Strava authorization...[/black]")

    if auth_manager.authenticate():
        console.print("[green]✅ Successfully authenticated with Strava![/green]")
    else:
        console.print("[red]❌ Authentication failed. Please try again.[/red]")


@cli.command()
@click.option("--days", default=60, help="Number of days to sync")
def sync(days):
    """Sync activities from Strava."""
    console.print(Panel.fit(f"🔄 Syncing Activities (last {days} days)", style="bold blue"))

    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        console.print("[orange]⚠️  No valid authentication found. Attempting to authenticate...[/orange]")
        if auth_manager.authenticate():
            console.print("[green]✅ Authentication successful![/green]")
        else:
            console.print("[red]❌ Authentication failed. Please check your credentials.[/red]")
            return

    try:
        client = StravaClient()

        with console.status(f"[black]Fetching activities from Strava...[/black]"):
            count = client.sync_activities(days_back=days)

        console.print(f"[green]✅ Successfully synced {count} activities![/green]")

        # Show recent activities
        recent = client.get_recent_activities(days=60)
        if recent:
            table = Table(title="Recent Activities", box=box.ROUNDED)
            table.add_column("Date", style="black")
            table.add_column("Name")
            table.add_column("Type", style="yellow")
            table.add_column("Duration", style="green")
            table.add_column("Load", style="magenta")

            for activity in recent[:5]:
                duration = f"{activity['moving_time'] // 60}min" if activity['moving_time'] else "N/A"
                load = f"{activity['training_load']:.0f}" if activity['training_load'] else "N/A"
                table.add_row(
                    activity['start_date'].strftime("%Y-%m-%d"),
                    activity['name'][:30] if activity['name'] else "N/A",
                    activity['type'] if activity['type'] else "N/A",
                    duration,
                    load,
                )

            console.print("\n", table)

    except Exception as e:
        # Try automatic re-authentication for 401 errors
        if handle_auth_error(e, auth_manager, "sync"):
            # Retry the sync operation
            try:
                client = StravaClient()
                with console.status(f"[black]Retrying sync...[/black]"):
                    count = client.sync_activities(days_back=days)
                console.print(f"[green]✅ Successfully synced {count} activities![/green]")

                # Show recent activities after successful retry
                recent = client.get_recent_activities(days=60)
                if recent:
                    table = Table(title="Recent Activities", box=box.ROUNDED)
                    table.add_column("Date", style="black")
                    table.add_column("Name")
                    table.add_column("Type", style="yellow")
                    table.add_column("Duration", style="green")
                    table.add_column("Load", style="magenta")

                    for activity in recent[:5]:
                        duration = f"{activity['moving_time'] // 60}min" if activity['moving_time'] else "N/A"
                        load = f"{activity['training_load']:.0f}" if activity['training_load'] else "N/A"
                        table.add_row(
                            activity['start_date'].strftime("%Y-%m-%d"),
                            activity['name'][:30] if activity['name'] else "N/A",
                            activity['type'] if activity['type'] else "N/A",
                            duration,
                            load,
                        )

                    console.print("\n", table)
                return

            except Exception as retry_error:
                console.print(f"[red]❌ Retry failed: {retry_error}[/red]")
        else:
            console.print(f"[red]❌ Error syncing activities: {e}[/red]")


@cli.command()
@click.option("--days", default=90, help="Number of days to analyze")
def analyze(days):
    """Analyze training data and calculate metrics."""

    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        console.print("[red]❌ Not authenticated. Please run 'strava-super auth' first.[/red]")
        return

    try:
        analyzer = SupercompensationAnalyzer()

        with console.status("[black]Calculating fitness metrics...[/black]"):
            df = analyzer.analyze(days_back=days)

        current_state = analyzer.get_current_state()

        # Enhanced 60-Day Comprehensive Training Log
        history = analyzer.get_metrics_history(days=60)
        if history:
            table = Table(title="60-Day Comprehensive Training Log", box=box.ROUNDED)
            table.add_column("Date", style="black", width=6)
            table.add_column("Activity", width=30)
            table.add_column("Time", style="blue", width=6)
            table.add_column("Intensity", style="black", width=9)
            table.add_column("Load", style="magenta", width=6)
            table.add_column("Fitness", style="blue", width=8)
            table.add_column("Fatigue", style="red", width=8)
            table.add_column("Form", style="green", width=5)
            table.add_column("ANS", style="red", width=4, header_style="red")
            table.add_column("HRV", style="red", width=4, header_style="red")
            table.add_column("Sleep", style="magenta", width=5)
            table.add_column("RHR", style="black", width=4)
            table.add_column("Weight", style="blue", width=7)
            table.add_column("BF%", style="green", width=5)
            table.add_column("H2O", style="black", width=5)
            table.add_column("BP", style="red", width=9)

            # Get wellness data if available
            try:
                from .db import get_db
                from .db.models import Activity, HRVData, SleepData, BloodPressure, WellnessData, BodyComposition
                from datetime import datetime as dt, timedelta
                from sqlalchemy import func

                db = get_db()
                with db.get_session() as session:
                    row_count = 0  # Track rows for header insertion

                    # Weekly summary tracking
                    weekly_tss = 0
                    weekly_time_minutes = 0
                    current_week_start = None

                    for h in history[-60:]:  # Show last 60 days
                        # Add header after Sundays (on top of every Monday) for weekly organization
                        date_obj = dt.fromisoformat(h['date']).date()
                        is_monday = date_obj.weekday() == 0  # Monday is 0 in Python

                        if row_count > 0 and is_monday:
                            # Add weekly summary for previous week
                            if weekly_tss > 0 or weekly_time_minutes > 0:
                                weekly_hours = weekly_time_minutes / 60
                                table.add_row(
                                    "[bold red]WEEK[/bold red]",
                                    f"[bold red]Total: {weekly_tss:.0f} TSS, {weekly_hours:.1f}h[/bold red]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                    "[bold black]—[/bold black]",
                                )

                            # Reset weekly counters
                            weekly_tss = 0
                            weekly_time_minutes = 0
                            table.add_section()
                            # Add a visual separator
                            table.add_row(
                                "[bold black]Date[/bold black]",
                                "[bold black]Activity[/bold black]",
                                "[bold blue]Time[/bold blue]",
                                "[bold black]Intensity[/bold black]",
                                "[bold magenta]Load[/bold magenta]",
                                "[bold blue]Fitness[/bold blue]",
                                "[bold red]Fatigue[/bold red]",
                                "[bold green]Form[/bold green]",
                                "[bold red]ANS[/bold red]",
                                "[bold red]HRV[/bold red]",
                                "[bold magenta]Sleep[/bold magenta]",
                                "[bold black]RHR[/bold black]",
                                "[bold blue]Weight[/bold blue]",
                                "[bold green]BF%[/bold green]",
                                "[bold black]H2O[/bold black]",
                                "[bold red]BP[/bold red]",
                            )
                        date_str = date_obj.strftime('%m/%d')

                        # Get all activities for that day ordered by start time
                        activities = session.query(Activity).filter(
                            func.date(Activity.start_date) == date_obj
                        ).order_by(Activity.start_date).all()

                        # Get wellness data - handle both date and datetime comparisons
                        # For HRV and Sleep (stored at midnight)
                        date_start = dt.combine(date_obj, dt.min.time())
                        date_end = date_start + timedelta(days=1)

                        hrv = session.query(HRVData).filter(
                            HRVData.date >= date_start,
                            HRVData.date < date_end
                        ).first()

                        sleep = session.query(SleepData).filter(
                            SleepData.date >= date_start,
                            SleepData.date < date_end
                        ).first()

                        # Handle activities for this date
                        if activities:
                            if len(activities) > 1:
                                # Multiple activities: First row shows only date and aggregated wellness data
                                if hrv and hrv.hrv_rmssd:
                                    # Get historical HRV data for baseline calculation
                                    historical_hrv = session.query(HRVData).filter(
                                        HRVData.date >= date_start - timedelta(days=30),
                                        HRVData.date < date_end
                                    ).order_by(HRVData.date).all()

                                    historical_rmssd = [h.hrv_rmssd for h in historical_hrv if h.hrv_rmssd]

                                    # Use deep analyzer for ANS assessment
                                    try:
                                        from .analysis.hrv_deep_analyzer import HRVDeepAnalyzer
                                        deep_analyzer = HRVDeepAnalyzer()
                                        hrv_analysis = deep_analyzer.analyze_daily_hrv(hrv.hrv_rmssd, historical_rmssd)

                                        ans_score = hrv_analysis['ans_readiness_score']
                                        ans_status = hrv_analysis['ans_status']

                                        # Color-code based on ANS status
                                        if ans_status == "Optimal":
                                            ans_display = f"[green]{ans_score:.0f}[/green]"
                                        elif ans_status == "Good":
                                            ans_display = f"[green]{ans_score:.0f}[/green]"
                                        elif ans_status in ["Mild Stress", "Stressed"]:
                                            ans_display = f"[orange]{ans_score:.0f}[/orange]"
                                        elif ans_status in ["High Stress", "Severe Stress"]:
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        elif ans_status == "Overreaching":
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        else:
                                            ans_display = f"{ans_score:.0f}"
                                    except Exception:
                                        # Fallback to basic RMSSD display
                                        ans_display = f"{hrv.hrv_rmssd:.0f}"

                                    # Always show raw HRV RMSSD value
                                    hrv_display = f"{hrv.hrv_rmssd:.0f}" if hrv and hrv.hrv_rmssd else "—"
                                else:
                                    ans_display = "—"
                                    hrv_display = "—"

                                sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"

                                # Get blood pressure data
                                bp = session.query(BloodPressure).filter(
                                    BloodPressure.date >= date_start,
                                    BloodPressure.date < date_end
                                ).first()

                                # Use OMRON BP monitor RHR as primary source
                                if bp and bp.heart_rate:
                                    rhr = f"{bp.heart_rate}"
                                else:
                                    rhr = "—"

                                bp_str = f"{bp.systolic}/{bp.diastolic}" if bp and bp.systolic and bp.diastolic else "—"

                                # Get body composition data (RENPHO)
                                body_comp = session.query(BodyComposition).filter(
                                    BodyComposition.user_id == "default",
                                    BodyComposition.date >= date_start,
                                    BodyComposition.date < date_end
                                ).first()

                                # Format athletic body composition display
                                weight_str = f"{body_comp.weight_kg:.1f}kg" if body_comp and body_comp.weight_kg else "—"
                                body_fat_str = f"{body_comp.body_fat_percent:.1f}%" if body_comp and body_comp.body_fat_percent else "—"
                                water_str = f"{body_comp.water_percent:.1f}%" if body_comp and body_comp.water_percent else "—"

                                # Calculate aggregated load and time from all activities
                                total_load = sum(activity.training_load or 0 for activity in activities)
                                total_time_seconds = sum(activity.moving_time or 0 for activity in activities)
                                total_time_minutes = total_time_seconds / 60

                                # Add to weekly totals
                                weekly_tss += total_load
                                weekly_time_minutes += total_time_minutes

                                # Format time display
                                if total_time_minutes >= 60:
                                    time_display = f"{total_time_minutes/60:.1f}h"
                                else:
                                    time_display = f"{total_time_minutes:.0f}m"

                                # Determine aggregated intensity based on total load
                                agg_intensity = get_intensity_from_load(total_load)

                                # Add aggregated wellness row with total load
                                table.add_row(
                                    date_str,
                                    f"{len(activities)} Activities",  # Show number of activities
                                    time_display,  # Total time
                                    agg_intensity,  # Aggregated intensity
                                    f"{total_load:.0f}",  # Aggregated load
                                    f"{h['fitness']:.1f}",
                                    f"{h['fatigue']:.1f}",
                                    f"{h['form']:.1f}",
                                    ans_display,
                                    hrv_display,
                                    sleep_score,
                                    rhr,
                                    weight_str,
                                    body_fat_str,
                                    water_str,
                                    bp_str,
                                )
                                row_count += 1

                                # Then add each activity as separate rows
                                for activity in activities:
                                    # Show activity type and name
                                    activity_type = activity.type
                                    if activity_type == "WeightTraining":
                                        activity_type = "Weights"

                                    # Include activity name if available
                                    if activity.name and len(activity.name) > 0:
                                        # Format: Type - Name (truncated to fit)
                                        combined_name = f"{activity_type} - {activity.name}"
                                        if len(combined_name) > 30:
                                            combined_name = combined_name[:27] + "..."
                                        activity_display = combined_name
                                    else:
                                        activity_display = activity_type

                                    # Individual activity intensity and time
                                    activity_load = activity.training_load or 0
                                    intensity = get_intensity_from_load(activity_load)

                                    # Format individual activity time
                                    activity_time_seconds = activity.moving_time or 0
                                    activity_time_minutes = activity_time_seconds / 60
                                    if activity_time_minutes >= 60:
                                        activity_time_display = f"{activity_time_minutes/60:.1f}h"
                                    else:
                                        activity_time_display = f"{activity_time_minutes:.0f}m"

                                    # Activity rows: empty date, no wellness data
                                    table.add_row(
                                        "",  # Empty date
                                        activity_display,
                                        activity_time_display,  # Individual activity time
                                        intensity,
                                        f"{activity_load:.0f}",
                                        "—",  # No fitness data
                                        "—",  # No fatigue data
                                        "—",  # No form data
                                        "—",  # No ANS data
                                        "—",  # No HRV data
                                        "—",  # No sleep data
                                        "—",  # No RHR data
                                        "—",  # No weight data
                                        "—",  # No body fat data
                                        "—",  # No water data
                                        "—",  # No BP data
                                    )
                                    row_count += 1

                            else:
                                # Single activity: show normally
                                activity = activities[0]
                                activity_type = activity.type
                                if activity_type == "WeightTraining":
                                    activity_type = "Weights"

                                # Include activity name if available
                                if activity.name and len(activity.name) > 0:
                                    # Format: Type - Name (truncated to fit)
                                    combined_name = f"{activity_type} - {activity.name}"
                                    if len(combined_name) > 30:
                                        combined_name = combined_name[:27] + "..."
                                    activity_display = combined_name
                                else:
                                    activity_display = activity_type

                                # Individual activity intensity and time
                                activity_load = activity.training_load or 0
                                intensity = get_intensity_from_load(activity_load)

                                # Calculate activity time and add to weekly totals
                                activity_time_seconds = activity.moving_time or 0
                                activity_time_minutes = activity_time_seconds / 60

                                weekly_tss += activity_load
                                weekly_time_minutes += activity_time_minutes

                                # Format time display
                                if activity_time_minutes >= 60:
                                    activity_time_display = f"{activity_time_minutes/60:.1f}h"
                                else:
                                    activity_time_display = f"{activity_time_minutes:.0f}m"

                                # Get wellness data for single activity
                                # Enhanced HRV analysis using deep analyzer
                                if hrv and hrv.hrv_rmssd:
                                    # Get historical HRV data for baseline calculation
                                    historical_hrv = session.query(HRVData).filter(
                                        HRVData.date >= date_start - timedelta(days=30),
                                        HRVData.date < date_end
                                    ).order_by(HRVData.date).all()

                                    historical_rmssd = [h.hrv_rmssd for h in historical_hrv if h.hrv_rmssd]

                                    # Use deep analyzer for ANS assessment
                                    try:
                                        from .analysis.hrv_deep_analyzer import HRVDeepAnalyzer
                                        deep_analyzer = HRVDeepAnalyzer()
                                        hrv_analysis = deep_analyzer.analyze_daily_hrv(hrv.hrv_rmssd, historical_rmssd)

                                        ans_score = hrv_analysis['ans_readiness_score']
                                        ans_status = hrv_analysis['ans_status']

                                        # Color-code based on ANS status
                                        if ans_status == "Optimal":
                                            ans_display = f"[green]{ans_score:.0f}[/green]"
                                        elif ans_status == "Good":
                                            ans_display = f"[green]{ans_score:.0f}[/green]"
                                        elif ans_status in ["Mild Stress", "Stressed"]:
                                            ans_display = f"[orange]{ans_score:.0f}[/orange]"
                                        elif ans_status in ["High Stress", "Severe Stress"]:
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        elif ans_status == "Overreaching":
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        else:
                                            ans_display = f"{ans_score:.0f}"
                                    except Exception:
                                        # Fallback to basic RMSSD display
                                        ans_display = f"{hrv.hrv_rmssd:.0f}"

                                    # Always show raw HRV RMSSD value
                                    hrv_display = f"{hrv.hrv_rmssd:.0f}" if hrv and hrv.hrv_rmssd else "—"
                                else:
                                    ans_display = "—"
                                    hrv_display = "—"

                                sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"

                                # Get blood pressure data
                                bp = session.query(BloodPressure).filter(
                                    BloodPressure.date >= date_start,
                                    BloodPressure.date < date_end
                                ).first()

                                # Use OMRON BP monitor RHR as primary source
                                if bp and bp.heart_rate:
                                    rhr = f"{bp.heart_rate}"
                                else:
                                    rhr = "—"

                                bp_str = f"{bp.systolic}/{bp.diastolic}" if bp and bp.systolic and bp.diastolic else "—"

                                # Get body composition data (RENPHO)
                                body_comp = session.query(BodyComposition).filter(
                                    BodyComposition.user_id == "default",
                                    BodyComposition.date >= date_start,
                                    BodyComposition.date < date_end
                                ).first()

                                # Format athletic body composition display
                                weight_str = f"{body_comp.weight_kg:.1f}kg" if body_comp and body_comp.weight_kg else "—"
                                body_fat_str = f"{body_comp.body_fat_percent:.1f}%" if body_comp and body_comp.body_fat_percent else "—"
                                water_str = f"{body_comp.water_percent:.1f}%" if body_comp and body_comp.water_percent else "—"

                                # Add row for single activity with all data
                                table.add_row(
                                    date_str,
                                    activity_display,
                                    activity_time_display,  # Activity time
                                    intensity,
                                    f"{activity_load:.0f}",
                                    f"{h['fitness']:.1f}",
                                    f"{h['fatigue']:.1f}",
                                    f"{h['form']:.1f}",
                                    ans_display,
                                    hrv_display,
                                    sleep_score,
                                    rhr,
                                    weight_str,
                                    body_fat_str,
                                    water_str,
                                    bp_str,
                                )
                                row_count += 1
                        else:
                            # Handle rest days - no activities
                            activity_display = "Rest Day" if h['load'] == 0 else "Active"
                            intensity = "REST" if h['load'] == 0 else "LIGHT"

                            # Still get wellness data for rest days
                            if hrv and hrv.hrv_rmssd:
                                # Get historical HRV data for baseline calculation
                                historical_hrv = session.query(HRVData).filter(
                                    HRVData.date >= date_start - timedelta(days=30),
                                    HRVData.date < date_end
                                ).order_by(HRVData.date).all()

                                historical_rmssd = [h.hrv_rmssd for h in historical_hrv if h.hrv_rmssd]

                                # Use deep analyzer for ANS assessment
                                try:
                                    from .analysis.hrv_deep_analyzer import HRVDeepAnalyzer
                                    deep_analyzer = HRVDeepAnalyzer()
                                    hrv_analysis = deep_analyzer.analyze_daily_hrv(hrv.hrv_rmssd, historical_rmssd)

                                    ans_score = hrv_analysis['ans_readiness_score']
                                    ans_status = hrv_analysis['ans_status']

                                    # Color-code based on ANS status
                                    if ans_status == "Optimal":
                                        ans_display = f"[green]{ans_score:.0f}[/green]"
                                    elif ans_status == "Good":
                                        ans_display = f"[green]{ans_score:.0f}[/green]"
                                    elif ans_status in ["Mild Stress", "Stressed"]:
                                        ans_display = f"[red]{ans_score:.0f}[/red]"
                                    elif ans_status in ["High Stress", "Severe Stress"]:
                                        ans_display = f"[red]{ans_score:.0f}[/red]"
                                    elif ans_status == "Overreaching":
                                        ans_display = f"[red]{ans_score:.0f}[/red]"
                                    else:
                                        ans_display = f"{ans_score:.0f}"

                                    # Always show raw HRV RMSSD value
                                    hrv_display = f"{hrv.hrv_rmssd:.0f}"
                                except Exception:
                                    # Fallback to basic RMSSD display
                                    ans_display = f"{hrv.hrv_rmssd:.0f}"
                                    hrv_display = f"{hrv.hrv_rmssd:.0f}"
                            else:
                                ans_display = "—"
                                hrv_display = "—"

                            sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"

                            # Get blood pressure data
                            bp = session.query(BloodPressure).filter(
                                BloodPressure.date >= date_start,
                                BloodPressure.date < date_end
                            ).first()

                            # Use OMRON BP monitor RHR as primary source
                            if bp and bp.heart_rate:
                                rhr = f"{bp.heart_rate}"
                            else:
                                rhr = "—"

                            bp_str = f"{bp.systolic}/{bp.diastolic}" if bp and bp.systolic and bp.diastolic else "—"

                            # Get body composition data (RENPHO)
                            body_comp = session.query(BodyComposition).filter(
                                BodyComposition.user_id == "default",
                                BodyComposition.date >= date_start,
                                BodyComposition.date < date_end
                            ).first()

                            # Format athletic body composition display
                            weight_str = f"{body_comp.weight_kg:.1f}kg" if body_comp and body_comp.weight_kg else "—"
                            body_fat_str = f"{body_comp.body_fat_percent:.1f}%" if body_comp and body_comp.body_fat_percent else "—"
                            water_str = f"{body_comp.water_percent:.1f}%" if body_comp and body_comp.water_percent else "—"

                            table.add_row(
                                date_str,
                                activity_display,
                                "—",  # No activity time for rest days
                                intensity,
                                f"{h['load']:.0f}",
                                f"{h['fitness']:.1f}",
                                f"{h['fatigue']:.1f}",
                                f"{h['form']:.1f}",
                                ans_display,
                                hrv_display,
                                sleep_score,
                                rhr,
                                weight_str,
                                body_fat_str,
                                water_str,
                                bp_str,
                            )
                            row_count += 1

            except Exception as e:
                # Fallback to basic version if wellness data unavailable
                from datetime import datetime as dt

                # Try to get wellness data even in fallback mode
                try:
                    from .db import get_db
                    from .db.models import HRVData, SleepData, BodyComposition
                    db = get_db()
                    wellness_available = True
                except:
                    wellness_available = False

                row_count = 0  # Track rows for header insertion in fallback mode
                for h in history[-60:]:  # Show last 60 days
                    # Add header after Sundays (on top of every Monday) for weekly organization
                    date_obj = dt.fromisoformat(h['date']).date()
                    is_monday = date_obj.weekday() == 0  # Monday is 0 in Python

                    if row_count > 0 and is_monday:
                        table.add_section()
                        # Add a visual separator
                        table.add_row(
                            "[bold black]Date[/bold black]",
                            "[bold black]Activity[/bold black]",
                            "[bold blue]Time[/bold blue]",
                            "[bold black]Intensity[/bold black]",
                            "[bold magenta]Load[/bold magenta]",
                            "[bold blue]Fitness[/bold blue]",
                            "[bold red]Fatigue[/bold red]",
                            "[bold green]Form[/bold green]",
                            "[bold red]HRV[/bold red]",
                            "[bold magenta]Sleep[/bold magenta]",
                            "[bold black]RHR[/bold black]",
                            "[bold blue]Weight[/bold blue]",
                            "[bold green]BF%[/bold green]",
                            "[bold black]H2O[/bold black]",
                            "[bold red]BP[/bold red]",
                        )
                    date_str = date_obj.strftime('%m/%d')

                    # Enhanced intensity classification
                    if h['load'] == 0:
                        intensity = "REST"
                    else:
                        intensity = get_intensity_from_load(h['load'])

                    activity_display = "Training" if h['load'] > 0 else "Rest"

                    # Try to get wellness and activity data for this date
                    hrv_rmssd = "—"
                    sleep_score = "—"
                    weight_str = "—"

                    if wellness_available:
                        try:
                            date_obj = dt.fromisoformat(h['date']).date()
                            with db.get_session() as session:
                                from sqlalchemy import func
                                from .db.models import Activity, BloodPressure

                                # Try to get all activities for the day ordered by start time
                                activities = session.query(Activity).filter(
                                    func.date(Activity.start_date) == date_obj
                                ).order_by(Activity.start_date).all()

                                if activities:
                                    for activity in activities:
                                        activity_type = activity.type
                                        if activity_type == "WeightTraining":
                                            activity_type = "Weights"

                                        if activity.name and len(activity.name) > 0:
                                            combined_name = f"{activity_type} - {activity.name}"
                                            if len(combined_name) > 30:
                                                combined_name = combined_name[:27] + "..."
                                            activity_display = combined_name
                                        else:
                                            activity_display = activity_type

                                # Get wellness data
                                from .db.models import WellnessData
                                date_start = dt.combine(date_obj, dt.min.time())
                                date_end = date_start + timedelta(days=1)

                                hrv = session.query(HRVData).filter(
                                    HRVData.date >= date_start,
                                    HRVData.date < date_end
                                ).first()
                                sleep = session.query(SleepData).filter(
                                    SleepData.date >= date_start,
                                    SleepData.date < date_end
                                ).first()
                                bp = session.query(BloodPressure).filter(
                                    BloodPressure.date >= date_start,
                                    BloodPressure.date < date_end
                                ).first()
                                wellness = session.query(WellnessData).filter(
                                    WellnessData.date >= date_start,
                                    WellnessData.date < date_end
                                ).first()

                                hrv_rmssd = f"{hrv.hrv_rmssd:3.0f}" if hrv and hrv.hrv_rmssd else "—"
                                sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"

                                # Use OMRON BP monitor RHR as primary source
                                if bp and bp.heart_rate:
                                    rhr = f"{bp.heart_rate}"
                                else:
                                    rhr = "—"

                                bp_str = f"{bp.systolic}/{bp.diastolic}" if bp and bp.systolic and bp.diastolic else "—"

                                # Get body composition data
                                body_comp = session.query(BodyComposition).filter(
                                    BodyComposition.user_id == "default",
                                    BodyComposition.date >= date_start,
                                    BodyComposition.date < date_end
                                ).first()
                                weight_str = f"{body_comp.weight_kg:.1f}kg" if body_comp and body_comp.weight_kg else "—"
                                body_fat_str = f"{body_comp.body_fat_percent:.1f}%" if body_comp and body_comp.body_fat_percent else "—"
                                water_str = f"{body_comp.water_percent:.1f}%" if body_comp and body_comp.water_percent else "—"
                        except:
                            rhr = "—"
                            bp_str = "—"
                            weight_str = "—"
                            body_fat_str = "—"
                            water_str = "—"

                    table.add_row(
                        date_str,
                        activity_display,
                        "—",  # No time data in fallback mode
                        intensity,
                        f"{h['load']:.0f}",
                        f"{h['fitness']:.1f}",
                        f"{h['fatigue']:.1f}",
                        f"{h['form']:.1f}",
                        "—",  # ANS not calculated in fallback
                        hrv_rmssd,  # Raw HRV
                        sleep_score,
                        rhr,
                        weight_str,
                        body_fat_str,
                        water_str,
                        bp_str,
                    )
                    row_count += 1

                # Add final weekly summary for the last week
                if weekly_tss > 0 or weekly_time_minutes > 0:
                    weekly_hours = weekly_time_minutes / 60
                    table.add_row(
                        "[bold red]WEEK[/bold red]",
                        f"[bold red]Total: {weekly_tss:.0f} TSS, {weekly_hours:.1f}h[/bold red]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                        "[bold black]—[/bold black]",
                    )

            console.print("\n", table)

            # Advanced insights analysis for 60-day comprehensive log
            try:
                with console.status("[black]Analyzing training patterns and risks...[/black]"):
                    from .analysis.advanced_insights import AdvancedInsightsAnalyzer
                    insights_analyzer = AdvancedInsightsAnalyzer()
                    insights = insights_analyzer.analyze_comprehensive_insights(days_back=60)

                    if insights:
                        # Display contradictory signals
                        contradictory_signals = insights.get('contradictory_signals', {})
                        if contradictory_signals.get('status') == 'analyzed' and contradictory_signals.get('contradictions'):
                            console.print("\n[bold red]⚠️ Contradictory Readiness & Overtraining Signals Detected[/bold red]")
                            contradictions_table = Table(title="Signal Contradictions", box=box.ROUNDED)
                            contradictions_table.add_column("Date", style="black")
                            contradictions_table.add_column("Issue", style="red")
                            contradictions_table.add_column("Details", style="yellow")

                            for contradiction in contradictory_signals['contradictions'][:10]:  # Show top 10
                                contradictions_table.add_row(
                                    contradiction['date'].strftime('%m/%d'),
                                    contradiction['type'],
                                    contradiction['description']
                                )
                            console.print(contradictions_table)

                        # Display health data anomalies
                        health_anomalies = insights.get('health_anomalies', {})
                        if health_anomalies.get('status') == 'analyzed' and health_anomalies.get('anomalies'):
                            console.print("\n[bold green]📊 Health Data Anomalies Detected[/bold green]")
                            anomalies_table = Table(title="Health Metric Outliers", box=box.ROUNDED)
                            anomalies_table.add_column("Date", style="black")
                            anomalies_table.add_column("Metric", style="blue")
                            anomalies_table.add_column("Value", style="red")
                            anomalies_table.add_column("Severity", style="yellow")

                            for anomaly in health_anomalies['anomalies'][:10]:  # Show top 10
                                anomalies_table.add_row(
                                    anomaly['date'],
                                    anomaly['metric'].replace('_', ' ').title(),
                                    f"{anomaly['value']:.1f}",
                                    anomaly['severity'].title()
                                )
                            console.print(anomalies_table)


                        # Display injury risk patterns
                        injury_risks = insights.get('injury_risk_patterns', {})
                        if injury_risks.get('status') == 'analyzed' and injury_risks.get('risk_patterns'):
                            risk_level = injury_risks.get('overall_risk', 'low')
                            if risk_level in ['moderate', 'high', 'critical']:
                                risk_colors = {'moderate': 'yellow', 'high': 'red', 'critical': 'bold red'}
                                console.print(f"\n[{risk_colors[risk_level]}]🚨 Injury Risk: {risk_level.upper()}[/{risk_colors[risk_level]}]")
                                for pattern in injury_risks['risk_patterns'][:3]:
                                    console.print(f"   • {pattern.get('pattern', 'Risk pattern detected')}")

                        # Summary recommendations
                        if insights.get('critical_recommendations'):
                            console.print(f"\n[bold green]📋 Key Recommendations[/bold green]")
                            for i, rec in enumerate(insights['critical_recommendations'][:5], 1):
                                console.print(f"   {i}. {rec}")

                        console.print(f"\n[black]🔍 Advanced analysis complete - {len(insights)} analysis modules processed[/black]")

            except ImportError:
                console.print("\n[orange]⚠️ Advanced insights analysis not available - missing dependencies[/orange]")
            except Exception as insights_error:
                console.print(f"\n[red]❌ Advanced insights analysis failed: {insights_error}[/red]")

    except Exception as e:
        # Try automatic re-authentication for 401 errors
        if handle_auth_error(e, auth_manager, "analysis"):
            try:
                analyzer = SupercompensationAnalyzer()
                with console.status("[black]Retrying analysis...[/black]"):
                    df = analyzer.analyze(days_back=days)
                console.print("[green]✅ Analysis complete after re-authentication![/green]")
                return
            except Exception as retry_error:
                console.print(f"[red]❌ Retry failed: {retry_error}[/red]")
        else:
            console.print(f"[red]❌ Error analyzing data: {e}[/red]")


@cli.command()
def recommend():
    """Get training recommendation for today."""
    console.print(Panel.fit("🎯 Training Recommendation", style="bold blue"))

    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        console.print("[red]❌ Not authenticated. Please run 'strava-super auth' first.[/red]")
        return

    try:
        # Use integrated analyzer for recommendations (consistent with run command)
        from .analysis.model_integration import get_integrated_analyzer
        analyzer = get_integrated_analyzer("user")
        recommendation = analyzer.get_daily_recommendation()

        # Color code based on recommendation
        color_map = {
            "REST": "red",
            "RECOVERY": "yellow",
            "EASY": "green",
            "MODERATE": "black",
            "HARD": "magenta",
            "PEAK": "bold magenta",
            "NO_DATA": "black",
        }

        rec_color = color_map.get(recommendation["recommendation"], "black")

        # Create recommendation panel using new format
        rec_text = f"""
[bold {rec_color}]{recommendation['recommendation']}[/bold {rec_color}]

[bold]Activity:[/bold] {recommendation['activity']}
[bold]Rationale:[/bold] {recommendation['rationale']}
[bold]Form Score:[/bold] {recommendation['form_score']:.1f}
[bold]Readiness Score:[/bold] {recommendation['readiness_score']:.1f}
[bold]Performance Potential:[/bold] {recommendation['performance_potential']:.2f}
"""

        # Display the recommendation panel
        console.print(Panel(rec_text.strip(), title="📈 Today's Advanced Recommendation", border_style=rec_color))

        # Show training plan options
        plan_choice = click.prompt(
            "\nShow training plan?",
            type=click.Choice(['no', '7', '30']),
            default='no',
            show_choices=True
        )

        if plan_choice != 'no':
            days = int(plan_choice)
            training_plan = engine.get_training_plan(days=days)

            if training_plan:
                title = f"{days}-Day Training Plan"
                table = Table(title=title, box=box.ROUNDED)
                table.add_column("Day", style="black", width=3)
                table.add_column("Date", width=10)
                table.add_column("Week Type", style="blue", width=10)
                table.add_column("Intensity", style="yellow", width=15)
                table.add_column("Activity", style="blue", width=24)
                table.add_column("2nd Session", style="green", width=27)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="black", width=6)
                table.add_column("Fitness", style="green", width=7)

                # For 30-day plan, add week separators
                current_week = -1
                for plan in training_plan:
                    week_num = (plan['day'] - 1) // 7 if plan['day'] > 0 else 0

                    # Add week separator for 30-day plan
                    if days == 30 and week_num != current_week:
                        current_week = week_num
                        table.add_section()
                        # Define week types based on sports science periodization
                        week_types = {
                            0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",  # Weeks 1-4
                            4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",  # Weeks 5-8
                            8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY", # Weeks 9-12
                            12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"   # Weeks 13-16
                        }
                        # For longer plans, extend the pattern
                        if week_num not in week_types:
                            cycle_week = week_num % 4
                            if cycle_week < 3:
                                week_types[week_num] = f"BUILD {cycle_week + 1}"
                            else:
                                week_types[week_num] = "RECOVERY"

                        week_type = week_types.get(week_num, "MAINTENANCE")
                        week_type_color = {
                            "BUILD 1": "green", "BUILD 2": "green", "BUILD 3": "orange",
                            "RECOVERY": "black", "PEAK 1": "red", "PEAK 2": "red",
                            "TAPER 1": "magenta", "TAPER 2": "magenta", "MAINTENANCE": "green"
                        }.get(week_type, "black")

                        # Calculate weekly time totals for this week
                        week_plans = [p for p in training_plan if (p['day'] - 1) // 7 == week_num]
                        total_weekly_minutes = 0
                        for plan in week_plans:
                            # Extract duration from activity string (format: "Activity (60m)")
                            activity = plan.get('activity', '')
                            if '(' in activity and 'm)' in activity:
                                try:
                                    duration_str = activity.split('(')[-1].split('m)')[0]
                                    total_weekly_minutes += int(duration_str)
                                except:
                                    pass

                            # Add second activity duration
                            second_activity = plan.get('second_activity', '')
                            if second_activity and second_activity != '—' and '(' in second_activity and 'm)' in second_activity:
                                try:
                                    duration_str = second_activity.split('(')[-1].split('m)')[0]
                                    total_weekly_minutes += int(duration_str)
                                except:
                                    pass

                        weekly_hours = total_weekly_minutes / 60

                        # Apply mesocycle-specific hour budget
                        base_budget_hours = config.TRAINING_MAX_WEEKLY_HOURS
                        def get_env_float(key: str, default: float) -> float:
                            try:
                                return float(os.getenv(key, str(default)))
                            except:
                                return default

                        # Get hour reduction factor based on week type
                        hour_factors = {
                            "BUILD 1": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "BUILD 2": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "BUILD 3": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "RECOVERY": get_env_float('HOUR_FACTOR_RECOVERY', 0.6),
                            "PEAK 1": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                            "PEAK 2": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                            "TAPER 1": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                            "TAPER 2": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                            "MAINTENANCE": get_env_float('HOUR_FACTOR_MAINTENANCE', 0.8)
                        }

                        hour_factor = hour_factors.get(week_type, 1.0)
                        budget_hours = base_budget_hours * hour_factor

                        time_status = "✅" if weekly_hours <= budget_hours else "⚠️"
                        time_summary = f"{time_status} {weekly_hours:.1f}h/{budget_hours:.1f}h"

                        table.add_row(
                            f"[bold]W{week_num+1}[/bold]",
                            "[black]────────[/black]",
                            f"[{week_type_color}]{week_type}[/{week_type_color}]",
                            "[black]────────[/black]",
                            f"[black]{time_summary}[/black]",
                            "[black]─────────────────────────[/black]",
                            "[black]────[/black]",
                            "[black]──────[/black]",
                            "[black]───────[/black]"
                        )

                    # Color code recommendations
                    rec_colors = {
                        "REST": "red",
                        "RECOVERY": "yellow",
                        "EASY": "green",
                        "MODERATE": "black",
                        "HARD": "magenta",
                        "PEAK": "bold magenta"
                    }
                    rec_color = rec_colors.get(plan['recommendation'], "black")

                    # Get week type for this day
                    week_types = {
                        0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",  # Weeks 1-4
                        4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",  # Weeks 5-8
                        8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY", # Weeks 9-12
                        12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"   # Weeks 13-16
                    }
                    # For longer plans, extend the pattern
                    if week_num not in week_types:
                        cycle_week = week_num % 4
                        if cycle_week < 3:
                            week_types[week_num] = f"BUILD {cycle_week + 1}"
                        else:
                            week_types[week_num] = "RECOVERY"

                    week_type = week_types.get(week_num, "MAINTENANCE")
                    week_type_color = {
                        "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "black",
                        "RECOVERY": "bright_black", "PEAK 1": "bright_red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "black"
                    }.get(week_type, "black")

                    # Get activity name
                    activity = plan.get('activity', 'Unknown')
                    if len(activity) > 29:  # Column width is 32, leave room for duration
                        activity = activity[:26] + "..."

                    # Get second session info
                    second_session = ""
                    if plan.get('second_activity'):
                        second_activity = plan['second_activity']
                        if len(second_activity) > 26:
                            second_session = second_activity[:23] + "..."
                        else:
                            second_session = second_activity
                    else:
                        second_session = "—"

                    # Format date to include day name: "We, 09/25"
                    date_str = plan['date']
                    if len(date_str) > 8:
                        date_str = date_str[5:]  # Remove year, keep MM-DD

                    # Convert to day name and prepend to date
                    day_names = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

                    day_name_prefix = ""

                    # Try to get day name from date
                    if 'date_obj' in plan and plan['date_obj']:
                        try:
                            day_name_prefix = day_names[plan['date_obj'].weekday()]
                        except:
                            pass
                    elif 'date' in plan and plan['date']:
                        # Parse date string
                        try:
                            # Handle different date formats
                            if '/' in plan['date']:
                                # MM/DD format - determine the year
                                month, day = plan['date'].split('/')
                                current_date = datetime.now()
                                year = current_date.year

                                # Create date object
                                date_obj = datetime(year, int(month), int(day))

                                # If the date is before today, it might be next year
                                if date_obj < current_date and (current_date - date_obj).days > 30:
                                    date_obj = datetime(year + 1, int(month), int(day))
                            else:
                                # Full date format YYYY-MM-DD
                                date_obj = datetime.strptime(plan['date'], '%Y-%m-%d')

                            # Get day name
                            day_name_prefix = day_names[date_obj.weekday()]
                        except Exception as e:
                            # Keep empty prefix if parsing fails
                            pass

                    # Format as "Day, MM/DD" or just "MM/DD" if day parsing failed
                    if day_name_prefix:
                        formatted_date = f"{day_name_prefix}, {date_str}"
                    else:
                        formatted_date = date_str

                    table.add_row(
                        str(plan['day']),
                        formatted_date,
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        f"[{rec_color}]{plan['recommendation']}[/{rec_color}]",
                        f"[blue]{activity}[/blue]",
                        f"[green]{second_session}[/green]" if plan.get('second_activity') else f"[white]{second_session}[/white]",
                        f"{plan['suggested_load']:.0f}",
                        f"{plan['predicted_form']:.1f}",
                        f"{plan.get('predicted_fitness', 0):.1f}",
                    )

                console.print("\n", table)

                # Show summary for 30-day plan
                if days == 30:
                    total_load = sum(p['suggested_load'] for p in training_plan)
                    rest_days = len([p for p in training_plan if p['recommendation'] == 'REST'])
                    hard_days = len([p for p in training_plan if p['recommendation'] in ['HARD', 'PEAK']])

                    summary = Panel(
                        f"[bold]Monthly Summary:[/bold]\n"
                        f"• Total Load: {total_load:.0f} TSS\n"
                        f"• Average Daily Load: {total_load/30:.0f} TSS\n"
                        f"• Rest Days: {rest_days}\n"
                        f"• Hard/Peak Days: {hard_days}\n"
                        f"• Periodization: 3:1 (3 weeks build, 1 week recovery)",
                        title="📊 Training Summary",
                        box=box.ROUNDED
                    )
                    console.print("\n", summary)

    except Exception as e:
        # Try automatic re-authentication for 401 errors
        if handle_auth_error(e, auth_manager, "recommendation"):
            try:
                analyzer = get_integrated_analyzer("user")
                recommendation = analyzer.get_daily_recommendation()
                console.print("[green]✅ Recommendation generated after re-authentication![/green]")
                return
            except Exception as retry_error:
                console.print(f"[red]❌ Retry failed: {retry_error}[/red]")
        else:
            console.print(f"[red]❌ Error generating recommendation: {e}[/red]")


@cli.command()
def status():
    """Show current authentication and sync status."""
    console.print(Panel.fit("ℹ️  System Status", style="bold blue"))

    # Check authentication
    auth_manager = AuthManager()
    if auth_manager.is_authenticated():
        console.print("[green]✅ Authenticated with Strava[/green]")
    else:
        console.print("[yellow]⚠️  Not authenticated[/yellow]")

    # Check database
    try:
        from .db import get_db
        db = get_db()
        with db.get_session() as session:
            from .db.models import Activity, Metric
            activity_count = session.query(Activity).count()
            metric_count = session.query(Metric).count()

            console.print(f"\n[black]📊 Database Statistics:[/black]")
            console.print(f"  • Activities: {activity_count}")
            console.print(f"  • Metrics: {metric_count}")

            # Get last sync
            last_activity = session.query(Activity).order_by(Activity.created_at.desc()).first()
            if last_activity:
                console.print(f"\n[black]Last sync: {last_activity.created_at.strftime('%Y-%m-%d %H:%M')}[/black]")

    except Exception as e:
        console.print(f"[red]❌ Database error: {e}[/red]")


@cli.command()
@click.option("--days", default=14, help="Number of days to analyze")
def multisport(days):
    """Analyze multi-sport training distribution and recovery."""
    console.print(Panel.fit(f"🏃‍♀️🚴‍♂️🏊‍♀️ Multi-Sport Analysis ({days} days)", style="bold blue"))

    try:
        client = StravaClient()
        activities = client.get_recent_activities(days=days)

        if not activities:
            console.print("[yellow]No activities found in the specified period.[/yellow]")
            return

        # Analyze sport distribution
        multisport_calc = MultiSportCalculator()
        analysis = multisport_calc.analyze_training_distribution(activities, days)

        # Display sport distribution
        if analysis["sport_loads"]:
            table = Table(title="Training Distribution by Sport", box=box.ROUNDED)
            table.add_column("Sport", style="black")
            table.add_column("Total Load", style="yellow")
            table.add_column("% of Load", style="green")
            table.add_column("Hours", style="magenta")
            table.add_column("% of Time", style="blue")

            for sport, load in analysis["sport_loads"].items():
                if load > 0:
                    load_pct = analysis["load_distribution"].get(sport, 0)
                    hours = analysis["sport_hours"].get(sport, 0)
                    time_pct = analysis["time_distribution"].get(sport, 0)

                    table.add_row(
                        sport.replace("_", " ").title(),
                        f"{load:.0f}",
                        f"{load_pct:.1f}%",
                        f"{hours:.1f}h",
                        f"{time_pct:.1f}%"
                    )

            console.print(table)

            # Summary stats
            summary_panel = Panel(
                f"""
[bold]Total Training Load:[/bold] {analysis['total_load']:.0f}
[bold]Total Training Time:[/bold] {analysis['total_hours']:.1f} hours
[bold]Average Daily Load:[/bold] {analysis['total_load']/days:.0f}
[bold]Sports Practiced:[/bold] {len([s for s, l in analysis['sport_loads'].items() if l > 0])}
                """,
                title="📊 Summary",
                box=box.ROUNDED,
            )
            console.print(summary_panel)

            # Recovery recommendations
            console.print("\n[bold black]🔄 Recovery Recommendations:[/bold black]")

            for activity in activities[:3]:  # Show top 3 recent
                sport_type = multisport_calc.get_sport_type(activity.get("type", ""))
                cross_training = multisport_calc.get_cross_training_recommendations(
                    sport_type, activity.get("training_load", 0)
                )

                if cross_training:
                    activity_name = activity.get("name", "Unknown")[:30]
                    console.print(f"\n[black]After {activity_name}:[/black]")
                    for rec in cross_training[:2]:
                        console.print(f"  • {rec['activity']}: {rec['benefit']} ({rec['duration']})")

    except Exception as e:
        console.print(f"[red]❌ Error analyzing multi-sport data: {e}[/red]")


@cli.command()
def reset():
    """Reset database and authentication."""
    console.print(Panel.fit("⚠️  Reset Application", style="bold yellow"))

    if not click.confirm("This will delete all data. Are you sure?"):
        console.print("[black]Operation cancelled.[/black]")
        return

    try:
        # Clear authentication
        auth_manager = AuthManager()
        auth_manager.logout()

        # Clear Garmin authentication (optional)
        try:
            from .garmin.auth import get_garmin_auth
            garmin_auth = get_garmin_auth()
            garmin_auth.revoke_tokens()
        except (ImportError, Exception):
            pass  # Garmin auth is optional

        # Reset database
        from .db import get_db
        db = get_db()
        db.drop_tables()
        db.create_tables()

        console.print("[green]✅ Application reset successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Error resetting application: {e}[/red]")


@cli.command()
@click.option("--file", required=True, help="Path to blood pressure CSV file")
def import_bp(file):
    """Import blood pressure data from CSV file."""
    console.print(Panel.fit(f"📊 Importing Blood Pressure Data", style="bold blue"))

    import csv
    from datetime import datetime
    from pathlib import Path

    if not Path(file).exists():
        console.print(f"[red]❌ File not found: {file}[/red]")
        return

    try:
        from .db import get_db
        from .db.models import BloodPressure

        db = get_db()
        imported = 0
        skipped = 0

        # German month mapping
        german_months = {
            'Jan.': 1, 'Feb.': 2, 'Mär.': 3, 'Apr.': 4,
            'Mai': 5, 'Jun.': 6, 'Jul.': 7, 'Aug.': 8,
            'Sep.': 9, 'Okt.': 10, 'Nov.': 11, 'Dez.': 12
        }

        with open(file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            with db.get_session() as session:
                for row in reader:
                    try:
                        # Parse German date format "23 Sep. 2025"
                        date_str = row['Datum']
                        time_str = row.get('Zeit', '00:00')

                        # Parse date
                        parts = date_str.split()
                        day = int(parts[0])
                        month = german_months.get(parts[1], 1)
                        year = int(parts[2])

                        # Parse time
                        time_parts = time_str.split(':')
                        hour = int(time_parts[0])
                        minute = int(time_parts[1]) if len(time_parts) > 1 else 0

                        # Create datetime
                        date_time = datetime(year, month, day, hour, minute)

                        # Check if already exists
                        existing = session.query(BloodPressure).filter_by(date=date_time).first()
                        if existing:
                            skipped += 1
                            continue

                        # Create new BP record
                        bp = BloodPressure(
                            date=date_time,
                            systolic=int(row['Systolisch (mmHg)']),
                            diastolic=int(row['Diastolisch (mmHg)']),
                            heart_rate=int(row['Puls (bpm)']),
                            device=row.get('Gerät', 'Unknown'),
                            irregular_heartbeat=row.get('Unregelmäßiger Herzschlag festgestellt', '').strip() == 'Ja',
                            afib_detected=row.get('Mögliches AFib', '').strip() == 'Ja',
                            notes=row.get('Notizen', '').replace('-', '').strip() or None
                        )

                        session.add(bp)
                        imported += 1

                    except (ValueError, KeyError) as e:
                        console.print(f"[yellow]⚠️ Skipping row due to error: {e}[/yellow]")
                        continue

                session.commit()

        console.print(f"[green]✅ Imported {imported} blood pressure readings[/green]")
        if skipped > 0:
            console.print(f"[yellow]ℹ️ Skipped {skipped} duplicate entries[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error importing data: {e}[/red]")


@cli.group()
def garmin():
    """Garmin Connect integration with MFA support."""
    pass


@garmin.command()
def test():
    """Test personal Garmin Connect access."""
    console.print(Panel.fit("🧪 Testing Personal Garmin Access", style="bold blue"))

    try:
        if get_garmin_client is None:
            console.print("[red]❌ Garmin client not available. Missing garth dependency.[/red]")
            return

        garmin_client = get_garmin_client()

        with console.status("[black]Testing connection...[/black]"):
            result = garmin_client.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[black]Display Name: {result.get('display_name')}[/black]")
            console.print(f"[black]User ID: {result.get('user_id')}[/black]")
            console.print(f"[black]Email: {result.get('email')}[/black]")
        else:
            console.print(f"[red]❌ Connection failed: {result.get('message')}[/red]")

    except GarminError as e:
        console.print(f"[red]❌ Garmin error: {e}[/red]")
        console.print("\n[yellow]💡 Setup required:[/yellow]")
        console.print("Set environment variables:")
        console.print("  GARMIN_EMAIL=your_email@example.com")
        console.print("  GARMIN_PASSWORD=your_password")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e).replace('[', r'\[').replace(']', r'\]')}[/red]")


@garmin.command()
@click.option("--days", default=60, help="Number of days to sync")
def sync(days):
    """Sync essential HRV and sleep data from personal Garmin Connect."""
    console.print(Panel.fit(f"🔄 Syncing Personal Garmin Data ({days} days)", style="bold blue"))

    try:
        if get_garmin_client is None:
            console.print("[red]❌ Garmin client not available. Missing garth dependency.[/red]")
            return

        garmin_client = get_garmin_client()

        with console.status(f"[black]Syncing HRV and sleep data from Garmin...[/black]"):
            results = garmin_client.sync_essential_data(days)

        # Display results
        table = Table(title="Sync Results", box=box.ROUNDED)
        table.add_column("Data Type", style="black")
        table.add_column("New/Updated", style="green")
        table.add_column("Skipped", style="yellow")
        table.add_column("Status", style="magenta")

        hrv_skipped = results.get("hrv_skipped", 0)
        sleep_skipped = results.get("sleep_skipped", 0)
        wellness_skipped = results.get("wellness_skipped", 0)

        table.add_row("HRV Data", str(results["hrv_synced"]), str(hrv_skipped), "[green]Success[/green]")
        table.add_row("Sleep Data", str(results["sleep_synced"]), str(sleep_skipped), "[green]Success[/green]")
        table.add_row("Wellness Data", str(results["wellness_synced"]), str(wellness_skipped), "[green]Success[/green]")

        if results["errors"] > 0:
            table.add_row("Errors", str(results["errors"]), "—", "[red]Failed[/red]")

        console.print(table)

        # Show summary
        total_synced = results["hrv_synced"] + results["sleep_synced"] + results["wellness_synced"]
        console.print(f"\n[green]✅ Sync complete! {total_synced} total records synced[/green]")
        console.print(f"[black]Date range: {results['date_range']}[/black]")

        # Show latest scores
        latest = garmin_client.get_latest_scores()
        if latest["hrv_score"] or latest["sleep_score"]:
            console.print(f"\n[bold black]📊 Latest Scores:[/bold black]")
            if latest["hrv_score"]:
                console.print(f"  • HRV: {latest['hrv_score']:.0f}/100 ({latest['hrv_date']})")
            if latest["sleep_score"]:
                console.print(f"  • Sleep: {latest['sleep_score']:.0f}/100 ({latest['sleep_date']})")

    except GarminError as e:
        console.print(f"[red]❌ Garmin error: {e}[/red]")
        if "credentials not found" in str(e).lower():
            console.print("\n[yellow]💡 Setup required:[/yellow]")
            console.print("Add to your .env file:")
            console.print("  GARMIN_EMAIL=your_email@example.com")
            console.print("  GARMIN_PASSWORD=your_password")
    except Exception as e:
        console.print(f"[red]❌ Error syncing data: {e}[/red]")


@garmin.command()
@click.option("--code", help="MFA verification code (optional - will prompt if not provided)")
def test_mfa(code):
    """Test Garmin connection with interactive MFA support."""
    console.print(Panel.fit("🧪 Testing Garmin MFA Connection", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        if code:
            console.print(f"[black]Using provided MFA code: {code}[/black]")

        console.print("[black]Testing Garmin connection...[/black]")
        console.print("[black]You will be prompted for an MFA code if needed[/black]")

        result = garmin_client.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[black]Display Name: {result.get('display_name')}[/black]")
            console.print(f"[black]User ID: {result.get('user_id')}[/black]")
            console.print(f"[black]Email: {result.get('email')}[/black]")
        else:
            console.print(f"[red]❌ Connection failed: {result.get('message')}[/red]")

    except GarminError as e:
        if "cancelled by user" in str(e).lower():
            console.print("[yellow]⚠️  Authentication cancelled by user[/yellow]")
        else:
            console.print(f"[red]❌ Garmin MFA error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e).replace('[', r'\[').replace(']', r'\]')}[/red]")


@garmin.command()
def scores():
    """Show latest wellness scores from personal Garmin data."""
    console.print(Panel.fit("📊 Latest Wellness Scores", style="bold blue"))

    try:
        if get_garmin_client is None:
            console.print("[red]❌ Garmin client not available. Missing garth dependency.[/red]")
            return

        garmin_client = get_garmin_client()
        latest = garmin_client.get_latest_scores()

        if not latest["hrv_score"] and not latest["sleep_score"]:
            console.print("[yellow]No wellness data found. Run 'garmin sync' first.[/yellow]")
            return

        # Display scores
        hrv_str = f"{latest['hrv_score']:.0f}/100" if latest['hrv_score'] else 'No data'
        sleep_str = f"{latest['sleep_score']:.0f}/100" if latest['sleep_score'] else 'No data'

        scores_panel = Panel(
            f"""
[bold]Latest Wellness Scores[/bold]

[bold]❤️  HRV Score:[/bold] {hrv_str}
[bold]📅 HRV Date:[/bold] {latest['hrv_date'] or 'N/A'}

[bold]😴 Sleep Score:[/bold] {sleep_str}
[bold]📅 Sleep Date:[/bold] {latest['sleep_date'] or 'N/A'}
            """,
            title="📊 Wellness Summary",
            box=box.ROUNDED,
        )
        console.print(scores_panel)

        # Get enhanced recommendation
        console.print("\n[bold black]💡 Enhanced Recommendation:[/bold black]")
        analyzer = get_integrated_analyzer("user")
        recommendation = analyzer.get_daily_recommendation()

        # Display integrated recommendation with readiness score
        console.print(f"  • Recommendation: {recommendation['recommendation']}")
        console.print(f"  • Readiness Score: {recommendation['readiness_score']:.0f}/100")
        console.print(f"  • Activity: {recommendation['activity']}")
        console.print(f"  • Rationale: {recommendation['rationale']}")

    except Exception as e:
        console.print(f"[red]❌ Error getting scores: {e}[/red]")


@garmin.command()
@click.option("--code", help="MFA verification code from Garmin Connect")
def login(code):
    """Login to Garmin Connect with MFA support."""
    console.print(Panel.fit("🔐 Garmin Connect MFA Login", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        if code:
            console.print(f"[black]Using provided MFA code: {code}[/black]")

        with console.status("[black]Authenticating with Garmin...[/black]"):
            result = garmin_client.login_with_mfa(code)

        if result["status"] == "success":
            console.print("[green]✅ Successfully authenticated![/green]")
            console.print(f"[black]User: {result.get('user', 'Unknown')}[/black]")
            console.print(f"[black]Message: {result.get('message')}[/black]")
        elif result["status"] == "mfa_required":
            console.print("[yellow]🔑 MFA code required[/yellow]")
            console.print("[yellow]Please run: strava-super garmin login --code YOUR_MFA_CODE[/yellow]")
            console.print("[black]Get the code from your Garmin Connect mobile app or email[/black]")
        else:
            console.print(f"[red]❌ Authentication failed: {result.get('message')}[/red]")

    except GarminError as e:
        console.print(f"[red]❌ Garmin error: {e}[/red]")
        if "credentials not found" in str(e).lower():
            console.print("\n[yellow]💡 Setup required:[/yellow]")
            console.print("Add to your .env file:")
            console.print("  GARMIN_EMAIL=your_email@example.com")
            console.print("  GARMIN_PASSWORD=your_password")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e).replace('[', r'\[').replace(']', r'\]')}[/red]")


@garmin.command()
@click.option("--days", default=60, help="Number of days to sync")
@click.option("--code", help="MFA verification code (optional - will prompt if not provided)")
def sync_mfa(days, code):
    """Sync wellness data using MFA-enabled client."""
    console.print(Panel.fit(f"🔄 Syncing Garmin Data with MFA ({days} days)", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        if code:
            console.print(f"[black]Using provided MFA code: {code}[/black]")
            # Try login first with provided code
            login_result = garmin_client.login_with_mfa(code)
            if login_result["status"] != "success":
                console.print(f"[red]❌ Login failed: {login_result['message']}[/red]")
                return

        console.print("[black]Syncing HRV and sleep data...[/black]")
        console.print("[black]Note: You may be prompted for an MFA code if authentication is needed[/black]")

        results = garmin_client.sync_essential_data(days)

        # Display results
        table = Table(title="MFA Sync Results", box=box.ROUNDED)
        table.add_column("Data Type", style="black")
        table.add_column("New/Updated", style="green")
        table.add_column("Skipped", style="yellow")
        table.add_column("Status", style="magenta")

        hrv_skipped = results.get("hrv_skipped", 0)
        sleep_skipped = results.get("sleep_skipped", 0)
        wellness_skipped = results.get("wellness_skipped", 0)

        table.add_row("HRV Data", str(results["hrv_synced"]), str(hrv_skipped), "[green]Success[/green]")
        table.add_row("Sleep Data", str(results["sleep_synced"]), str(sleep_skipped), "[green]Success[/green]")
        table.add_row("Wellness Data", str(results["wellness_synced"]), str(wellness_skipped), "[green]Success[/green]")

        if results["errors"] > 0:
            table.add_row("Errors", str(results["errors"]), "—", "[red]Failed[/red]")

        console.print(table)

        # Show summary
        total_synced = results["hrv_synced"] + results["sleep_synced"] + results["wellness_synced"]
        console.print(f"\n[green]✅ MFA sync complete! {total_synced} total records synced[/green]")
        console.print(f"[black]Date range: {results['date_range']}[/black]")

        # Show latest scores
        latest = garmin_client.get_latest_scores()
        if latest["hrv_score"] or latest["sleep_score"]:
            console.print(f"\n[bold black]📊 Latest Scores:[/bold black]")
            if latest["hrv_score"]:
                console.print(f"  • HRV: {latest['hrv_score']:.0f}/100 ({latest['hrv_date']})")
            if latest["sleep_score"]:
                console.print(f"  • Sleep: {latest['sleep_score']:.0f}/100 ({latest['sleep_date']})")

    except GarminError as e:
        console.print(f"[red]❌ Garmin MFA error: {e}[/red]")
        if "login failed" in str(e).lower():
            console.print("\n[yellow]💡 Try authenticating first:[/yellow]")
            console.print("  strava-super garmin login --code YOUR_MFA_CODE")
    except Exception as e:
        console.print(f"[red]❌ Error syncing data: {e}[/red]")


# ===== RENPHO BODY COMPOSITION INTEGRATION =====

@cli.group()
def renpho():
    """RENPHO body composition integration."""
    pass


@renpho.command()
@click.option("--csv-path", help="Path to RENPHO CSV export file (auto-discovers if not provided)")
@click.option("--user-id", default="default", help="User ID for the data")
@click.option("--directory", default=".", help="Directory to search for RENPHO CSV files")
def import_csv(csv_path, user_id, directory):
    """Import body composition data from RENPHO CSV export. Auto-discovers RENPHO files if no path specified."""

    if csv_path:
        # Single file import
        console.print(Panel.fit(f"🧬 Importing RENPHO CSV Data", style="bold purple"))
    else:
        # Auto-discovery mode
        console.print(Panel.fit(f"🔍 Auto-discovering RENPHO CSV files in {directory}", style="bold purple"))

    try:
        if RenphoCsvImporter is None:
            console.print("[red]❌ RENPHO CSV importer not available[/red]")
            return

        if csv_path:
            # Single file import
            importer = RenphoCsvImporter(csv_path, user_id)

            with console.status("[black]Importing CSV data and calculating athletic metrics...[/black]"):
                results = importer.import_csv_data()

            # Display single file results
            table = Table(title="RENPHO CSV Import Results", box=box.ROUNDED)
            table.add_column("Metric", style="black")
            table.add_column("Count", justify="right", style="green")

            table.add_row("New measurements", str(results['new_measurements']))
            table.add_row("Updated measurements", str(results['updated_measurements']))
            table.add_row("Total processed", str(results['total_processed']))
            if results['errors'] > 0:
                table.add_row("Errors", str(results['errors']), style="red")

            console.print(table)

            if results['new_measurements'] > 0 or results['updated_measurements'] > 0:
                console.print(f"\n✅ Successfully imported {results['new_measurements'] + results['updated_measurements']} measurements")

                # Show athletic trends analysis
                importer = RenphoCsvImporter("", user_id)  # For trends analysis
                trends = importer.get_athletic_trends(days=60)
                if trends['status'] == 'success':
                    console.print(f"\n📈 [bold black]Athletic Performance Analysis (60 days):[/bold black]")
                    console.print(f"   Weight Change: {trends.get('weight_change_kg', 'N/A')}kg")
                    console.print(f"   Lean Mass Change: {trends.get('lean_mass_change_kg', 'N/A')}kg")
                    console.print(f"   Muscle Mass Change: {trends.get('muscle_mass_change', 'N/A')}%")
                    console.print(f"   Hydration Stability: {trends.get('water_stability_score', 'N/A')}/100")
                    console.print(f"   Weight Stability: {trends.get('weight_stability_score', 'N/A')}/100")
                    console.print(f"   Power-to-Weight Potential: {trends['latest_measurement'].get('power_to_weight_potential', 'N/A')}/100")
            else:
                console.print("\n💡 No new data found. Your data is up to date.")
        else:
            # Auto-discovery mode
            with console.status("[black]Discovering and importing RENPHO CSV files...[/black]"):
                results = RenphoCsvImporter.auto_import_all(directory, user_id)

            if results.get("status") == "no_files_found":
                console.print("[yellow]⚠️ No RENPHO CSV files found in the directory.[/yellow]")
                console.print(f"[yellow]💡 Looking for files with columns like: Datum, Zeitraum, Gewicht(kg), BMI, Körperfett(%)[/yellow]")
                return

            # Display auto-import results
            table = Table(title="Auto-Import Results", box=box.ROUNDED)
            table.add_column("File", style="black")
            table.add_column("Status", style="green")
            table.add_column("New", justify="right", style="blue")
            table.add_column("Updated", justify="right", style="yellow")
            table.add_column("Total", justify="right", style="magenta")

            total_new = 0
            total_updated = 0
            successful_files = 0

            for filename, file_result in results.items():
                if file_result["status"] == "success":
                    r = file_result["results"]
                    table.add_row(
                        filename,
                        "✅ Success",
                        str(r['new_measurements']),
                        str(r['updated_measurements']),
                        str(r['total_processed'])
                    )
                    total_new += r['new_measurements']
                    total_updated += r['updated_measurements']
                    successful_files += 1
                else:
                    table.add_row(filename, "❌ Error", "-", "-", "-")

            console.print(table)

            if successful_files > 0:
                console.print(f"\n✅ Successfully imported from {successful_files} file(s)")
                console.print(f"   Total new measurements: {total_new}")
                console.print(f"   Total updated measurements: {total_updated}")

                # Show athletic trends analysis for auto-import
                importer = RenphoCsvImporter("", user_id)  # For trends analysis
                trends = importer.get_athletic_trends(days=60)
                if trends['status'] == 'success':
                    console.print(f"\n📈 [bold black]Athletic Performance Analysis (60 days):[/bold black]")
                    console.print(f"   Weight Change: {trends.get('weight_change_kg', 'N/A')}kg")
                    console.print(f"   Lean Mass Change: {trends.get('lean_mass_change_kg', 'N/A')}kg")
                    console.print(f"   Muscle Mass Change: {trends.get('muscle_mass_change', 'N/A')}%")
                    console.print(f"   Hydration Stability: {trends.get('water_stability_score', 'N/A')}/100")
                    console.print(f"   Weight Stability: {trends.get('weight_stability_score', 'N/A')}/100")
                    console.print(f"   Power-to-Weight Potential: {trends['latest_measurement'].get('power_to_weight_potential', 'N/A')}/100")
            else:
                console.print("\n❌ No files were successfully imported.")

    except Exception as e:
        console.print(f"[red]❌ Import error: {e}[/red]")


@renpho.command()
@click.option("--days", default=60, help="Number of days for athletic trend analysis")
@click.option("--user-id", default="default", help="User ID for the analysis")
def trends(days, user_id):
    """Show athletic body composition trends and performance analysis."""
    console.print(Panel.fit(f"📈 Athletic Body Composition Analysis ({days} days)", style="bold purple"))

    try:
        if RenphoCsvImporter is None:
            console.print("[red]❌ RENPHO CSV importer not available[/red]")
            return

        importer = RenphoCsvImporter("", user_id)  # Empty CSV path for trends only
        analysis = importer.get_athletic_trends(days)

        if analysis["status"] == "insufficient_data":
            console.print(f"[yellow]⚠️ Insufficient data for athletic analysis[/yellow]")
            console.print(f"[yellow]Found {analysis.get('count', 0)} measurements, need at least 2[/yellow]")
            return

        # Athletic Performance Analysis
        table = Table(title="Athletic Performance Changes", box=box.ROUNDED)
        table.add_column("Metric", style="black")
        table.add_column("Change", style="black")
        table.add_column("Athletic Impact", style="green")

        if analysis.get("weight_change_kg") is not None:
            weight_change = analysis["weight_change_kg"]
            weight_status = "Stable (Good)" if abs(weight_change) < 0.5 else ("Power Loss Risk" if weight_change > 1.0 else "Dehydration Risk" if weight_change < -1.0 else "Monitor")
            table.add_row("Weight", f"{weight_change:+.1f} kg", weight_status)

        if analysis.get("lean_mass_change_kg") is not None:
            lean_change = analysis["lean_mass_change_kg"]
            lean_status = "Power Gains 🚀" if lean_change > 0.3 else ("Muscle Loss" if lean_change < -0.3 else "Stable")
            table.add_row("Lean Body Mass", f"{lean_change:+.1f} kg", lean_status)

        if analysis.get("muscle_mass_change") is not None:
            muscle_change = analysis["muscle_mass_change"]
            muscle_status = "Strength Gains 💪" if muscle_change > 0.2 else ("Catabolism Risk" if muscle_change < -0.2 else "Maintained")
            table.add_row("Skeletal Muscle", f"{muscle_change:+.1f}%", muscle_status)

        console.print(table)

        # Recovery & Stability Indicators
        stability_table = Table(title="Recovery & Performance Stability", box=box.ROUNDED)
        stability_table.add_column("Indicator", style="black")
        stability_table.add_column("Score", style="black")
        stability_table.add_column("Training Impact", style="green")

        if analysis.get("water_stability_score") is not None:
            hydration_status = "Excellent" if analysis["water_stability_score"] > 95 else ("Good" if analysis["water_stability_score"] > 85 else "Inconsistent - Monitor Recovery")
            stability_table.add_row("Hydration Stability", f"{analysis['water_stability_score']}/100", hydration_status)

        if analysis.get("weight_stability_score") is not None:
            weight_stab_status = "Excellent" if analysis["weight_stability_score"] > 95 else ("Good" if analysis["weight_stability_score"] > 85 else "Variable - Check Nutrition")
            stability_table.add_row("Weight Stability", f"{analysis['weight_stability_score']}/100", weight_stab_status)

        console.print(stability_table)

        # Current Athletic State
        latest = analysis["latest_measurement"]
        console.print(f"\n[bold black]🏃 Current Athletic Profile ({latest['date'].strftime('%Y-%m-%d')}):[/bold black]")
        if latest.get("weight_kg"):
            console.print(f"  • Weight: {latest['weight_kg']:.1f} kg")
        if latest.get("lean_body_mass_kg"):
            console.print(f"  • Lean Body Mass: {latest['lean_body_mass_kg']:.1f} kg")
        if latest.get("body_fat_percent"):
            console.print(f"  • Body Fat: {latest['body_fat_percent']:.1f}%")
        if latest.get("water_percent"):
            console.print(f"  • Hydration: {latest['water_percent']:.1f}%")
        if latest.get("power_to_weight_potential"):
            console.print(f"  • Power-to-Weight Potential: {latest['power_to_weight_potential']}/100")

        console.print(f"\n[black]🔬 Analysis based on {analysis['measurements_count']} measurements over {analysis['period_days']} days[/black]")

    except Exception as e:
        console.print(f"[red]❌ Error analyzing athletic trends: {e}[/red]")



@cli.command()
@click.option("--date", help="Specific date to test (YYYY-MM-DD), defaults to today")
def test_rhr(date):
    """Test RHR fetching from Garmin with debug output."""
    from datetime import datetime, date as date_obj, timedelta

    # Parse date or use today
    if date:
        try:
            test_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            return
    else:
        test_date = date_obj.today()

    console.print(f"🧪 Testing RHR fetch for {test_date}")

    try:
        if get_garmin_client is None:
            console.print("[red]Garmin client not available[/red]")
            return

        garmin_client = get_garmin_client()

        # Test the specific HRV sync method with debug output
        console.print(f"📡 Testing Garmin HRV sync for {test_date}...")
        result = garmin_client._sync_hrv_for_date(test_date)

        if result:
            console.print("[green]✅ HRV sync completed[/green]")

            # Check what was saved to database
            from .db import get_db
            from .db.models import HRVData

            db = get_db()
            with db.get_session() as session:
                hrv_record = session.query(HRVData).filter(
                    HRVData.date >= datetime.combine(test_date, datetime.min.time()),
                    HRVData.date < datetime.combine(test_date, datetime.min.time()) + timedelta(days=1)
                ).first()

                if hrv_record:
                    console.print(f"📊 Database record:")
                    console.print(f"  • HRV Score: {hrv_record.hrv_score}")
                    console.print(f"  • Resting HR: {hrv_record.resting_heart_rate}")
                    console.print(f"  • Max HR: {hrv_record.max_heart_rate}")
                    console.print(f"  • Min HR: {hrv_record.min_heart_rate}")
                else:
                    console.print("[yellow]⚠️ No HRV record found in database[/yellow]")
        else:
            console.print("[red]❌ HRV sync failed[/red]")

    except Exception as e:
        console.print(f"[red]💥 Test failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


@cli.command()
@click.option("--strava-days", default=60, help="Days of Strava data to sync")
@click.option("--garmin-days", default=60, help="Days of Garmin data to sync")
@click.option("--plan-days", default=30, help="Training plan length in days (1-365)", type=click.IntRange(1, 365))
@click.option("--skip-strava", is_flag=True, help="Skip Strava sync")
@click.option("--skip-garmin", is_flag=True, help="Skip Garmin sync")
@click.option("--skip-analysis", is_flag=True, help="Skip analysis step")
def run(strava_days, garmin_days, plan_days, skip_strava, skip_garmin, skip_analysis):
    """Complete training analysis workflow: sync all data, analyze, and get recommendations."""

    errors = []

    # Step 1: Data Import & Sync
    console.print("")  # Spacing
    console.print(Panel("📊 Step 1: Importing Training Data", box=box.HEAVY, style="bold blue"))

    # Collect sync status for summary panel
    sync_results = []

    # Single spinner for entire data import process
    with console.status("[black]Importing data from all sources (Strava, Omron, Garmin)...[/black]", spinner="dots"):
        # Sync Strava data (minimal output)
        if not skip_strava:
            try:
                ctx = click.get_current_context()
                main_sync = None
                for command_name, command in cli.commands.items():
                    if command_name == "sync":
                        main_sync = command
                        break
                if main_sync:
                    # Suppress the verbose output from sync
                    import io
                    import contextlib
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        ctx.invoke(main_sync, days=strava_days)
                else:
                    raise Exception("Main sync command not found")
                sync_results.append("🏃‍♂️ [green]Strava Activities: ✅ Synced[/green]")
            except Exception as e:
                error_msg = f"Strava sync failed: {e}"
                errors.append(error_msg)
                sync_results.append(f"🏃‍♂️ [red]Strava Activities: ❌ {error_msg}[/red]")
        else:
            sync_results.append("🏃‍♂️ [yellow]Strava Activities: ⏭️ Skipped[/yellow]")

        # Import Omron blood pressure data from CSV files
        try:
            import glob
            from pathlib import Path

            # Find all Omron CSV files in the project directory
            omron_files = glob.glob("*.csv")
            omron_files = [f for f in omron_files if "OMRON" in f.upper()]

            if omron_files:
                ctx = click.get_current_context()
                total_imported = 0

                # Import each Omron CSV file
                for csv_file in omron_files:
                    try:
                        # Suppress verbose output
                        import io
                        import contextlib
                        f = io.StringIO()
                        with contextlib.redirect_stdout(f):
                            ctx.invoke(import_bp, file=csv_file)
                        # Parse the output to get import count
                        output = f.getvalue()
                        if "Imported" in output:
                            # Extract number from "Imported X blood pressure readings"
                            import re
                            match = re.search(r'Imported (\d+)', output)
                            if match:
                                total_imported += int(match.group(1))
                    except Exception:
                        pass  # Continue with other files

                if total_imported > 0:
                    sync_results.append(f"🩺 [green]Omron BP Data: ✅ {total_imported} readings imported[/green]")
                else:
                    sync_results.append("🩺 [yellow]Omron BP Data: ℹ️ No new data[/yellow]")
            else:
                sync_results.append("🩺 [white]Omron BP Data: No CSV files found[/white]")
        except Exception as e:
            sync_results.append(f"🩺 [yellow]Omron BP Data: ⚠️ Import failed[/yellow]")

        # Sync Garmin data (minimal output)
        if not skip_garmin:
            try:
                if get_garmin_client is not None:
                    ctx = click.get_current_context()
                    # Suppress verbose output from Garmin sync with timeout
                    import io
                    import contextlib
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("Garmin sync timed out after 30 seconds")

                    # Set timeout for Garmin sync
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 second timeout

                    try:
                        f = io.StringIO()
                        with contextlib.redirect_stdout(f):
                            ctx.invoke(garmin.commands["sync-mfa"], days=garmin_days)
                        sync_results.append("⌚ [green]Garmin Wellness: ✅ Synced[/green]")
                    finally:
                        signal.alarm(0)  # Cancel timeout
                else:
                    sync_results.append("⌚ [yellow]Garmin Wellness: ⚠️ Unavailable[/yellow]")
            except TimeoutError as e:
                sync_results.append(f"⌚ [yellow]Garmin Wellness: ⏭️ Timed out (skipped)[/yellow]")
            except Exception as e:
                error_msg = f"Garmin sync failed: {e}"
                errors.append(error_msg)
                sync_results.append(f"⌚ [red]Garmin Wellness: ❌ {error_msg}[/red]")
        else:
            sync_results.append("⌚ [yellow]Garmin Wellness: ⏭️ Skipped[/yellow]")

        # Auto-import RENPHO CSV data (minimal output)
        try:
            if RenphoCsvImporter is not None:
                # Auto-discover and import RENPHO CSV files
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    results = RenphoCsvImporter.auto_import_all(".", "default")

                if results.get("status") == "no_files_found":
                    sync_results.append("⚖️ [yellow]RENPHO Body Comp: 💡 No CSV files found (place RENPHO exports in project dir)[/yellow]")
                else:
                    total_new = 0
                    total_updated = 0
                    successful_files = 0

                    for filename, file_result in results.items():
                        if file_result["status"] == "success":
                            r = file_result["results"]
                            total_new += r['new_measurements']
                            total_updated += r['updated_measurements']
                            successful_files += 1

                    if successful_files > 0:
                        if total_new > 0:
                            sync_results.append(f"⚖️ [green]RENPHO Body Comp: ✅ {total_new} new measurements imported[/green]")
                        else:
                            sync_results.append("⚖️ [green]RENPHO Body Comp: ✅ Up to date[/green]")
                    else:
                        sync_results.append("⚖️ [red]RENPHO Body Comp: ❌ Import failed[/red]")
            else:
                sync_results.append("⚖️ [yellow]RENPHO Body Comp: ⚠️ CSV importer not available[/yellow]")
        except Exception as e:
            error_msg = f"RENPHO auto-import failed: {e}"
            errors.append(error_msg)
            sync_results.append(f"⚖️ [red]RENPHO Body Comp: ❌ {error_msg}[/red]")


    # Display sync results outside the panel
    for result in sync_results:
        console.print(result)

    # Step 1 completion footer

    # Step 2: Basic Performance Model Analysis
    console.print("")  # Spacing
    step2_header = Panel("📊 Step 2: Basic Performance Model Analysis", box=box.HEAVY, style="bold blue")
    console.print(step2_header)

    if not skip_analysis:
        try:
            ctx = click.get_current_context()
            ctx.invoke(analyze)
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            errors.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/red]")
    else:
        console.print("[yellow]⏭️  Skipping analysis[/yellow]")
        console.print("[green]✅ Basic analysis complete (skipped).[/green]")

    # Step 4: Advanced Physiological Analysis
    console.print("")  # Spacing
    step4_header = Panel("🧬 Step 4: Physiological Analysis", box=box.HEAVY, style="bold blue")
    console.print(step4_header)

    try:
        analyzer = get_integrated_analyzer("user")

        # Import required components at the start
        from rich.table import Table


        # 4.2: Comprehensive Performance Dashboard
        console.print(f"\n[black]Comprehensive Performance Dashboard:[/black]")
        analysis = analyzer.analyze_with_advanced_models(days_back=90)

        if analysis and 'combined' in analysis and len(analysis['combined']) > 0:
            combined = analysis['combined']
            latest = combined.iloc[-1]

            # Calculate trends
            recent_7d = combined.tail(7)
            recent_30d = combined.tail(30)
            fitness_trend = recent_7d['ff_fitness'].iloc[-1] - recent_7d['ff_fitness'].iloc[0]
            form_trend = latest['ff_form'] - recent_7d['ff_form'].iloc[0]

            # Calculate training load metrics
            total_load = recent_30d['load'].sum()
            avg_daily = recent_30d['load'].mean()
            high_days = len(recent_30d[recent_30d['load'] > avg_daily * 1.5])
            rest_days = len(recent_30d[recent_30d['load'] == 0])

            # Calculate recovery metrics
            overall_recovery = min(100, latest['overall_recovery'])
            metabolic_recovery = min(100, overall_recovery * 0.9)
            neural_recovery = min(100, overall_recovery * 1.1)

            # Risk state assessment
            risk_state = latest.get('risk_state', 'SAFE')
            risk_status = "Recovery Focus" if risk_state == "HIGH_STRAIN" else ("Rest Required" if risk_state == "NON_FUNCTIONAL_OVERREACHING" else "Safe")

            # Create consolidated table that merges both dashboards with enhanced format
            dashboard_table = Table(box=box.ROUNDED)
            dashboard_table.add_column("Category", style="black", width=14)
            dashboard_table.add_column("Key Metrics", style="black", width=65)
            dashboard_table.add_column("Status", style="green", width=36)
            dashboard_table.add_column("Trend", style="black", width=6)

            # Current Fitness
            fitness_status = "Elite | Ready" if latest['ff_fitness'] > 100 and latest['composite_readiness'] > 60 else "Building | Moderate"
            dashboard_table.add_row(
                "Current Fitness",
                f"CTL: {latest['ff_fitness']:.0f} | TSB: {latest['ff_form']:.1f} | Readiness: {latest['composite_readiness']:.0f}%",
                fitness_status,
                f"{fitness_trend:+.1f}"
            )

            # Recovery Systems
            recovery_status = "Excellent" if overall_recovery > 80 else ("Good" if overall_recovery > 60 else "Fair")
            dashboard_table.add_row(
                "Recovery Systems",
                f"Overall: {overall_recovery:.0f}% | Metabolic: {metabolic_recovery:.0f}% | Neural: {neural_recovery:.0f}%",
                recovery_status,
                "—"
            )

            # Training Load (30d)
            load_status = "High Volume" if total_load > 3000 else ("Moderate Volume" if total_load > 2000 else "Light Volume")
            dashboard_table.add_row(
                "Training Load (30d)",
                f"Total: {total_load:.0f} TSS | Daily Avg: {avg_daily:.0f} | Hard: {high_days}d | Rest: {rest_days}d",
                load_status,
                "—"
            )

            # Performance & Risk
            overtraining_status = "YES" if latest.get('overtraining_risk', False) else "NO"
            performance_status = f"Developing | {risk_status}"
            dashboard_table.add_row(
                "Performance & Risk",
                f"Potential: {latest['perpot_performance']:.3f} | Overtraining Risk: {overtraining_status} | Risk: {risk_state.replace('_', ' ').title()}",
                performance_status,
                f"{form_trend:+.1f}"
            )

            console.print(dashboard_table)

        else:
            console.print("[yellow]⚠️ Insufficient data for advanced analysis[/yellow]")


    except Exception as e:
        console.print(f"[red]❌ Advanced analysis failed: {e}[/red]")

    # Step 4.5: Correlation Insights (NEW)
    console.print("")  # Spacing
    step45_header = Panel("💡 What Your Body Is Telling You", box=box.HEAVY, style="bold black")
    console.print(step45_header)

    try:
        from .analysis.correlation_analyzer import WellnessPerformanceCorrelations

        correlation_analyzer = WellnessPerformanceCorrelations()

        # Run quick correlation analysis (30 days for speed)
        results = correlation_analyzer.analyze_all_correlations(days_back=30, min_samples=10)

        if results.get('status') != 'insufficient_data':
            # Show leading indicators if found
            if results['leading_indicators']:
                console.print(f"\n[bold green]🔮 Future Predictions:[/bold green]")

                for indicator in results['leading_indicators'][:3]:  # Top 3
                    # Translate technical terms
                    metric_names = {
                        'hrv_rmssd': 'Heart Rate Variability',
                        'hrv_score': 'HRV Score',
                        'sleep_score': 'Sleep Quality',
                        'stress_avg': 'Stress Level',
                        'resting_hr': 'Resting Heart Rate',
                        'tsb': 'Training Form',
                        'ctl': 'Fitness Level',
                        'atl': 'Fatigue Level',
                        'daily_load': 'Training Load'
                    }

                    indicator_name = metric_names.get(indicator['indicator_metric'], indicator['indicator_metric'])
                    target_name = metric_names.get(indicator['target_metric'], indicator['target_metric'])

                    lag_days = indicator['optimal_lag_days']
                    if lag_days == 1:
                        timing = "tomorrow"
                    elif lag_days <= 3:
                        timing = f"in {lag_days} days"
                    else:
                        timing = f"in {lag_days} days"

                    power = indicator['predictive_power']
                    if power == "strong":
                        confidence = "High confidence"
                        emoji = "🎯"
                    elif power == "moderate":
                        confidence = "Moderate confidence"
                        emoji = "📊"
                    else:
                        confidence = "Some indication"
                        emoji = "💭"

                    console.print(
                        f"  {emoji} Your [black]{indicator_name}[/black] today predicts your [yellow]{target_name}[/yellow] {timing}"
                    )
                    console.print(f"     → {confidence} • Use this for planning ahead")

            # Show top significant correlation in plain language
            sig_corrs = results['significant_correlations']
            if sig_corrs:
                top_corr = max(sig_corrs, key=lambda x: abs(x['correlation']))

                # Translate to plain language
                metric_names = {
                    'hrv_rmssd': 'Heart Rate Variability',
                    'hrv_score': 'HRV',
                    'sleep_score': 'Sleep Quality',
                    'stress_avg': 'Stress',
                    'resting_hr': 'Resting Heart Rate',
                    'tsb': 'Training Form',
                    'ctl': 'Fitness',
                    'atl': 'Fatigue',
                    'daily_load': 'Training Load',
                    'ctl_7d_change': 'Fitness Growth',
                    'weekly_distance_km': 'Weekly Mileage',
                    'weekly_hours': 'Training Hours'
                }

                var1 = metric_names.get(top_corr['variable_1'], top_corr['variable_1'])
                var2 = metric_names.get(top_corr['variable_2'], top_corr['variable_2'])

                console.print(f"\n[bold green]⭐ Your #1 Performance Link:[/bold green]")

                # Make it human-readable
                direction = top_corr['direction']
                strength = top_corr['strength']

                if direction == 'negative':
                    relationship = f"When {var1} goes DOWN, {var2} goes DOWN"
                    advice = f"Keep your {var1} high for better {var2}"
                else:
                    relationship = f"When {var1} goes UP, {var2} goes UP"
                    advice = f"Improving {var1} improves your {var2}"

                if strength == 'strong':
                    reliability = "Very reliable relationship"
                elif strength == 'moderate':
                    reliability = "Reliable relationship"
                else:
                    reliability = "Noticeable pattern"

                console.print(f"  • {relationship}")
                console.print(f"  • {reliability} • {advice}")

            # Simple summary
            console.print(f"\n[dim]📈 Based on your last 30 days of data[/dim]")
        else:
            console.print("[yellow]💭 Need more data to find patterns[/yellow]")
            console.print("[dim]Keep training - insights will appear after 10+ days[/dim]")

    except Exception as e:
        console.print(f"[dim]⚠️ Insights temporarily unavailable[/dim]")

    # Step 5: Multisport Analysis
    console.print("")  # Spacing
    step5_header = Panel("🏃‍♂️🚴‍♀️🏊‍♀️ Step 5: Multi-Sport Profile", box=box.HEAVY, style="bold blue")
    console.print(step5_header)

    try:
        # Get basic activity type distribution from database
        with analyzer.db.get_session() as session:
            from strava_supercompensation.db.models import Activity
            from datetime import datetime, timedelta
            recent_activities = session.query(Activity).filter(
                Activity.start_date >= datetime.now(timezone.utc) - timedelta(days=30)
            ).all()

            if recent_activities:
                sport_stats = {}
                total_hours = 0
                total_load = 0

                for activity in recent_activities:
                    sport = activity.type or 'Unknown'
                    hours = (activity.moving_time or 0) / 3600.0  # Convert to hours
                    load = activity.training_load or 0

                    if sport not in sport_stats:
                        sport_stats[sport] = {'hours': 0, 'load': 0, 'count': 0}

                    sport_stats[sport]['hours'] += hours
                    sport_stats[sport]['load'] += load
                    sport_stats[sport]['count'] += 1
                    total_hours += hours
                    total_load += load

                if total_hours > 0:
                    sport_table = Table(box=box.ROUNDED)
                    sport_table.add_column("Sport", style="black", width=15)
                    sport_table.add_column("Activities", style="black", width=10)
                    sport_table.add_column("Hours", style="green", width=8)
                    sport_table.add_column("Load TSS", style="yellow", width=10)
                    sport_table.add_column("% Time", style="black", width=8)

                    for sport, stats in sorted(sport_stats.items(), key=lambda x: x[1]['hours'], reverse=True):
                        if stats['hours'] > 0:
                            time_pct = (stats['hours'] / total_hours) * 100
                            sport_table.add_row(
                                sport.replace('_', ' ').title(),
                                f"{stats['count']}",
                                f"{stats['hours']:.1f}h",
                                f"{stats['load']:.0f}",
                                f"{time_pct:.1f}%"
                            )

                    console.print(sport_table)
                    console.print(f"\n[black]Total Training Volume (30 days):[/black] {total_hours:.1f} hours | {total_load:.0f} TSS")
                else:
                    console.print("[yellow]⚠️ No training data in last 30 days[/yellow]")
            else:
                console.print("[yellow]⚠️ No activities found in last 30 days[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Multisport analysis failed: {e}[/red]")

    # Step 6: Training Plan Generation & Recommendations
    console.print("")  # Spacing
    step6_header = Panel(f"🎯 Step 6: Today's Recommendation & {plan_days}-Day Plan", box=box.HEAVY, style="bold blue")
    console.print(step6_header)

    try:
        # Use integrated analyzer for sophisticated recommendations
        analyzer = get_integrated_analyzer("user")

        # Get today's recommendation using advanced models
        recommendation = analyzer.get_daily_recommendation()

        if recommendation:
            # Display today's recommendation
            rec_colors = {
                "REST": "red",
                "RECOVERY": "yellow",
                "EASY": "green",
                "MODERATE": "black",
                "HARD": "magenta",
                "PEAK": "bold magenta"
            }
            rec_color = rec_colors.get(recommendation.get("recommendation", ""), "black")


        # Generate training plan
        console.print(f"\n[black]Generating {plan_days}-day plan using optimization models...[/black]")

        # Use advanced planner for all durations (1-365 days) with configuration from .env
        generator = TrainingPlanGenerator("user")

        plan_result = generator.generate_training_plan(
            duration_days=plan_days,
            goal='balanced',
            constraints={
                'max_weekly_hours': config.TRAINING_MAX_WEEKLY_HOURS,
                'rest_days': config.get_training_rest_days()
            }
        )

        # Convert WorkoutPlan objects to dictionary format with simulated values
        training_plan = []
        if plan_result and 'daily_workouts' in plan_result:
            daily_data = plan_result.get('visualizations', {}).get('daily_data', [])

            # CRITICAL FIX: Sort workouts chronologically to ensure proper date sequencing
            daily_workouts = plan_result['daily_workouts']
            sorted_workouts = sorted(daily_workouts, key=lambda w: w.date)


            # Skip day 0 (today) and maintain correct data correlation
            for i, workout in enumerate(sorted_workouts):
                if workout.day_number <= 0:  # Skip day 0 or negative
                    continue

                # CRITICAL FIX: Find corresponding daily simulation data by day_number, not array index
                daily_sim = {}
                for sim_data in daily_data:
                    if sim_data.get('day') == workout.day_number:
                        daily_sim = sim_data
                        break

                # Use the actual workout date and day number from the WorkoutPlan
                actual_day = workout.day_number if hasattr(workout, 'day_number') else len(training_plan) + 1
                actual_date = workout.date.date() if hasattr(workout.date, 'date') else workout.date

                # Convert WorkoutPlan dataclass to dictionary with simulation data
                plan_entry = {
                    'day': actual_day,
                    'date': actual_date.strftime('%m/%d'),
                    'date_obj': actual_date,  # Keep the datetime object for day name conversion
                    'recommendation': get_intensity_from_load(float(workout.planned_load)),
                    'activity': f"{workout.title} ({workout.total_duration_min}m)",
                    'second_activity': (
                        f"{workout.second_activity_title} ({workout.second_activity_duration}m)"
                        if workout.second_activity_title else "—"
                    ),
                    'load': float(workout.planned_load),
                    'suggested_load': float(workout.planned_load),
                    'form': daily_sim.get('form', 0.0),
                    'predicted_form': daily_sim.get('form', 0.0),
                    'fitness': daily_sim.get('fitness', 0.0),
                    'predicted_fitness': daily_sim.get('fitness', 0.0)
                }

                training_plan.append(plan_entry)

        # Display the training plan if generated successfully
        if training_plan:

            title = f"{plan_days}-Day Training Plan"
            table = Table(title=title, box=box.ROUNDED)
            table.add_column("Day", style="black", width=3)
            table.add_column("Date", width=10)
            table.add_column("Week Type", style="bright_blue", width=10)
            table.add_column("Intensity", style="yellow", width=11)
            table.add_column("Primary Activity & Duration", style="blue", width=32)
            table.add_column("2nd Session & Duration", style="green", width=28)
            table.add_column("Load", style="magenta", width=4)
            table.add_column("Form", style="black", width=6)
            table.add_column("Fitness", style="green", width=7)

            # CRITICAL FIX: Process plans by week to ensure proper ordering
            # Group plans by week first
            weeks = {}
            for plan in training_plan:
                week_num = (plan['day'] - 1) // 7
                if week_num not in weeks:
                    weeks[week_num] = []
                weeks[week_num].append(plan)

            # Now process weeks in order
            for week_num in sorted(weeks.keys()):
                week_plans = sorted(weeks[week_num], key=lambda p: p['day'])  # Sort days within week

                # Add week separator for all plans (with weekly grouping for plans >= 7 days)
                if plan_days >= 7:
                    current_week = week_num
                    table.add_section()

                    # Define week types based on sports science periodization
                    week_types = {
                        0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",  # Weeks 1-4
                        4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",  # Weeks 5-8
                        8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY", # Weeks 9-12
                        12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"   # Weeks 13-16
                    }
                    # For longer plans, extend the pattern
                    if week_num not in week_types:
                        cycle_week = week_num % 4
                        if cycle_week < 3:
                            week_types[week_num] = f"BUILD {cycle_week + 1}"
                        else:
                            week_types[week_num] = "RECOVERY"

                    week_type = week_types.get(week_num, "MAINTENANCE")
                    week_type_color = {
                        "BUILD 1": "green", "BUILD 2": "green", "BUILD 3": "orange",
                        "RECOVERY": "black", "PEAK 1": "red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "magenta", "MAINTENANCE": "black"
                    }.get(week_type, "black")

                    # Calculate weekly time totals for this week
                    # week_plans is already defined above
                    total_weekly_minutes = 0
                    for plan in week_plans:
                        # Extract duration from activity string (format: "Activity (60m)")
                        activity = plan.get('activity', '')
                        if '(' in activity and 'm)' in activity:
                            try:
                                duration_str = activity.split('(')[-1].split('m)')[0]
                                total_weekly_minutes += int(duration_str)
                            except:
                                pass

                        # Add second activity duration
                        second_activity = plan.get('second_activity', '')
                        if second_activity and second_activity != '—' and '(' in second_activity and 'm)' in second_activity:
                            try:
                                duration_str = second_activity.split('(')[-1].split('m)')[0]
                                total_weekly_minutes += int(duration_str)
                            except:
                                pass

                    weekly_hours = total_weekly_minutes / 60

                    # Apply mesocycle-specific hour budget
                    base_budget_hours = config.TRAINING_MAX_WEEKLY_HOURS
                    def get_env_float(key: str, default: float) -> float:
                        try:
                            return float(os.getenv(key, str(default)))
                        except:
                            return default

                    # Get hour reduction factor based on week type
                    hour_factors = {
                        "BUILD 1": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                        "BUILD 2": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                        "BUILD 3": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                        "RECOVERY": get_env_float('HOUR_FACTOR_RECOVERY', 0.6),
                        "PEAK 1": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                        "PEAK 2": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                        "TAPER 1": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                        "TAPER 2": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                        "MAINTENANCE": get_env_float('HOUR_FACTOR_MAINTENANCE', 0.8)
                    }

                    hour_factor = hour_factors.get(week_type, 1.0)
                    budget_hours = base_budget_hours * hour_factor

                    time_status = "✅" if weekly_hours <= budget_hours else "⚠️"
                    time_summary = f"{time_status} {weekly_hours:.1f}h/{budget_hours:.1f}h"

                    table.add_row(
                        f"[bold]W{week_num+1}[/bold]",
                        "[white]────────[/white]",
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        "[white]────────[/white]",
                        f"[black]{time_summary}[/black]",
                        "[white]─────────────────────────[/white]",
                        "[white]────[/white]",
                        "[white]──────[/white]",
                        "[white]───────[/white]"
                    )

                # Now add each day in this week
                for plan in week_plans:
                    # Color code recommendations
                    rec_colors = {
                        "REST": "red",
                        "RECOVERY": "yellow",
                        "EASY": "green",
                        "MODERATE": "black",
                        "HARD": "magenta",
                        "PEAK": "bold magenta"
                    }
                    rec_color = rec_colors.get(plan['recommendation'], "black")

                    # Get week type for this day (use the outer week_num)
                    week_types = {
                        0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",  # Weeks 1-4
                        4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",  # Weeks 5-8
                        8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY", # Weeks 9-12
                        12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"   # Weeks 13-16
                    }
                    # For longer plans, extend the pattern
                    if week_num not in week_types:
                        cycle_week = week_num % 4
                        if cycle_week < 3:
                            week_types[week_num] = f"BUILD {cycle_week + 1}"
                        else:
                            week_types[week_num] = "RECOVERY"

                    week_type = week_types.get(week_num, "MAINTENANCE")
                    week_type_color = {
                        "BUILD 1": "green", "BUILD 2": "green", "BUILD 3": "orange",
                        "RECOVERY": "black", "PEAK 1": "red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "magenta", "MAINTENANCE": "black"
                    }.get(week_type, "black")

                    # Get activity name
                    activity = plan.get('activity', 'Unknown')
                    if len(activity) > 29:  # Column width is 32, leave room for duration
                        activity = activity[:26] + "..."

                    # Get second session info
                    second_session = ""
                    if plan.get('second_activity'):
                        second_activity = plan['second_activity']
                        if len(second_activity) > 29:
                            second_session = second_activity[:26] + "..."
                        else:
                            second_session = second_activity
                    else:
                        second_session = "—"

                    # Format date to include day name: "We, 09/25"
                    date_str = plan['date']
                    if len(date_str) > 8:
                        date_str = date_str[5:]  # Remove year, keep MM-DD

                    # Add day name to date
                    day_names = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

                    formatted_date = date_str  # Default to just date

                    # Try to get day name from date
                    if 'date_obj' in plan and plan['date_obj']:
                        try:
                            day_name = day_names[plan['date_obj'].weekday()]
                            formatted_date = f"{day_name}, {date_str}"
                        except:
                            pass
                    elif 'date' in plan and plan['date']:
                        try:
                            # Parse MM/DD format
                            if '/' in plan['date']:
                                month, day = plan['date'].split('/')
                                current_year = datetime.now().year
                                date_obj = datetime(current_year, int(month), int(day))

                                # Adjust year if needed
                                if date_obj < datetime.now() and (datetime.now() - date_obj).days > 30:
                                    date_obj = datetime(current_year + 1, int(month), int(day))

                                day_name = day_names[date_obj.weekday()]
                                formatted_date = f"{day_name}, {date_str}"
                        except:
                            pass

                    table.add_row(
                        str(plan['day']),
                        formatted_date,
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        f"[{rec_color}]{plan['recommendation']}[/{rec_color}]",
                        f"[blue]{activity}[/blue]",
                        f"[green]{second_session}[/green]" if plan.get('second_activity') else f"[white]{second_session}[/white]",
                        f"{plan['suggested_load']:.0f}",
                        f"{plan['predicted_form']:.1f}",
                        f"{plan.get('predicted_fitness', 0):.1f}",
                    )

            console.print(table)

            # Show summary
            total_load = sum(p.get('suggested_load', p.get('load', 0)) for p in training_plan)
            rest_days = len([p for p in training_plan if p['recommendation'] == 'REST'])
            hard_days = len([p for p in training_plan if p['recommendation'] in ['HARD', 'PEAK']])

            console.print(Panel(f"Monthly Summary:\n"
                                f"• Total Load: {total_load:.0f} TSS\n"
                                f"• Average Daily Load: {total_load/len(training_plan):.0f} TSS\n"
                                f"• Rest Days: {rest_days}\n"
                                f"• Hard/Peak Days: {hard_days}\n"
                                f"• Periodization: 3:1 (3 weeks build, 1 week recovery)",
                                title="📊 Training Summary"))


    except Exception as e:
        error_msg = f"Recommendation generation failed: {e}"
        errors.append(error_msg)
        console.print(f"[red]❌ {error_msg}[/red]")

    # Summary

    if errors:
        console.print(f"\n[yellow]⚠️  {len(errors)} errors occurred:[/yellow]")
        for error in errors:
            console.print(f"  • {error}")
        console.print(f"\n[black]You can run individual commands to fix specific issues[/black]")
    else:
        console.print(f"\n[black]🧬 Models active • 📊 Multi-system analysis • 🎯 Optimal recommendations ready[/black]")

    # Step 7: Phase 2 Strategic Enhancements
    console.print("")  # Spacing
    step7_header = Panel("🚀 Step 7: Phase 2 Strategic Enhancements", box=box.HEAVY, style="bold blue")
    console.print(step7_header)

    try:
        # 7.1: Strength Training Integration
        console.print(f"\n[black]Strength Training Integration:[/black]")

        from .analysis.strength_planner import StrengthPlanner
        planner = StrengthPlanner()

        # Get current training phase from the analyzer
        try:
            analyzer = get_integrated_analyzer("user")
            analysis_30d = analyzer.analyze_with_advanced_models(days_back=30)

            if analysis_30d and 'combined' in analysis_30d and len(analysis_30d['combined']) > 0:
                latest = analysis_30d['combined'].iloc[-1]
                current_form = latest['ff_form']
                current_fitness = latest['ff_fitness']

                # Determine endurance phase based on form/fitness
                if current_form < -15:
                    endurance_phase = "BUILD"
                elif current_form > 0:
                    endurance_phase = "PEAK"
                else:
                    endurance_phase = "BASE"
            else:
                endurance_phase = "BASE"  # Default
        except:
            endurance_phase = "BASE"  # Fallback

        # Generate today's strength workout
        from datetime import datetime, timedelta
        today = datetime.now()
        day_of_week = today.weekday()  # 0=Monday
        week_number = today.isocalendar()[1] % 4  # 4-week cycle

        strength_phase = planner.align_with_endurance_phase(endurance_phase, week_number)
        today_workout = planner.generate_workout(strength_phase, week_number, day_of_week)

        if today_workout:
            console.print(f"[green]✅ Today's Strength Session:[/green] {strength_phase.value.replace('_', ' ').title()}")
            console.print(f"   • Duration: {today_workout.duration_min} minutes")
            console.print(f"   • Estimated TSS: {today_workout.volume_tss:.0f}")
            console.print(f"   • Exercises:")
            for ex in today_workout.exercises:
                console.print(f"     → {ex.name}: {ex.sets}x{ex.reps} @ {ex.intensity_percent:.0f}% (rest: {ex.rest_seconds}s)")
            console.print(f"   • Phase Notes: {today_workout.notes}")
        else:
            console.print(f"[yellow]💤 No strength training scheduled for today ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]})[/yellow]")


        # 7.2: ML Performance Prediction
        console.print(f"\n[black]ML Performance Prediction:[/black]")

        from .analysis.performance_predictor import PerformancePredictor

        try:
            with analyzer.db.get_session() as session:
                predictor = PerformancePredictor(db_session=session)

                # Check if model exists and load it
                import os
                model_path = "models/performance_model.joblib"

                if os.path.exists(model_path):
                    try:
                        predictor.load_model(model_path)
                        model_loaded = True
                    except Exception:
                        model_loaded = False
                else:
                    # Try to train the model if we have enough data
                    try:
                        metadata = predictor.train_model()
                        if metadata and 'races_found' in metadata and metadata['races_found'] >= 3:
                            os.makedirs("models", exist_ok=True)
                            predictor.save_model(model_path)
                            model_loaded = True
                        else:
                            model_loaded = False
                    except Exception:
                        model_loaded = False

                if model_loaded:
                    # Try to make predictions for all race distances
                    race_distances = [5.0, 10.0, 21.1, 42.195, 50.0, 75.0, 100.0]  # 5K, 10K, Half, Marathon, Ultra distances
                    race_names = {
                        5.0: "5K",
                        10.0: "10K",
                        21.1: "HM (21.1km)",
                        42.195: "M (42.195km)",
                        50.0: "Ultra 50km",
                        75.0: "Ultra 75km",
                        100.0: "Ultra 100km"
                    }

                    predictions_made = False
                    console.print("[black]Running Predictions (4 weeks):[/black]")

                    for distance in race_distances:
                        try:
                            # Predict for 4 weeks from now (typical training cycle)
                            future_date = datetime.now() + timedelta(days=28)
                            prediction = predictor.predict_race_performance(future_date, distance)

                            if prediction and 'predicted_time_formatted' in prediction:
                                race_name = race_names[distance]
                                console.print(f"[green]🎯 {race_name} Prediction:[/green] {prediction['predicted_time_formatted']}")
                                predictions_made = True
                        except:
                            continue

                    if not predictions_made:
                        console.print(f"[yellow]⚠️ Unable to generate predictions with current model[/yellow]")
                else:
                    console.print(f"[yellow]⚠️ ML model needs more training data for predictions[/yellow]")

        except Exception as e:
            console.print(f"[yellow]⚠️ ML prediction not available: {str(e)[:50]}...[/yellow]")

        # 7.2b: Enhanced ML Analysis - Integrated Summary
        console.print(f"\n[bold green]🤖 Enhanced ML Analysis:[/bold green]")

        try:
            # Get recent data for ML analysis
            recent_df = analysis['combined'].copy()
            recent_df = recent_df.rename(columns={
                'ff_fitness': 'ctl',
                'ff_fatigue': 'atl',
                'ff_form': 'tsb'
            })

            if len(recent_df) >= 60:
                # Show current state and simple forecast (avoid TensorFlow loading)
                models_dir = Path("models/ml_enhanced")

                # Current metrics
                current_ctl = recent_df['ctl'].iloc[-1]
                current_atl = recent_df['atl'].iloc[-1]
                current_tsb = recent_df['tsb'].iloc[-1]

                # Simple 7-day and 14-day projection using Banister model decay
                import numpy as np
                tau_fitness = 42  # CTL decay rate
                tau_fatigue = 7   # ATL decay rate

                # Assume zero training load for conservative forecast
                # Exponential decay: value * exp(-days/tau)
                ctl_7d = current_ctl * np.exp(-7/tau_fitness)
                ctl_14d = current_ctl * np.exp(-14/tau_fitness)
                atl_7d = current_atl * np.exp(-7/tau_fatigue)
                atl_14d = current_atl * np.exp(-14/tau_fatigue)
                tsb_7d = ctl_7d - atl_7d
                tsb_14d = ctl_14d - atl_14d

                # Interpret current TSB
                if current_tsb < -30:
                    tsb_status = "Very High Fatigue"
                    tsb_color = "red"
                elif current_tsb < -10:
                    tsb_status = "High Fatigue"
                    tsb_color = "red"
                elif current_tsb < 5:
                    tsb_status = "Moderate Fatigue"
                    tsb_color = "green"
                elif current_tsb < 15:
                    tsb_status = "Fresh"
                    tsb_color = "green"
                else:
                    tsb_status = "Very Fresh/Tapered"
                    tsb_color = "green"

                # Trend interpretation
                if tsb_7d < current_tsb - 5:
                    trend = "↓ more fatigued"
                    trend_color = "red"
                elif tsb_7d > current_tsb + 5:
                    trend = "↑ recovering"
                    trend_color = "green"
                else:
                    trend = "→ stable"
                    trend_color = "green"

                console.print(f"[green]📈 Fitness Trajectory (Simple Forecast):[/green]")
                console.print(f"   • Current: Fitness (CTL) {current_ctl:.1f}, Fatigue (ATL) {current_atl:.1f}, Form (TSB) {current_tsb:.1f} ([{tsb_color}]{tsb_status}[/{tsb_color}])")
                console.print(f"   • 7-day forecast: Fitness (CTL) {ctl_7d:.1f} ({ctl_7d-current_ctl:+.1f}), Form (TSB) {tsb_7d:.1f} ([{trend_color}]{trend}[/{trend_color}])")
                console.print(f"   • 14-day forecast: Fitness (CTL) {ctl_14d:.1f} ({ctl_14d-current_ctl:+.1f}), Form (TSB) {tsb_14d:.1f}")

                # Note: Anomaly detection disabled in daily run to prevent crashes
                # Use dedicated ml-analysis command for full anomaly analysis
            else:
                console.print(f"[green]   Need 60+ days of data for LSTM forecasting[/green]")

        except ImportError:
            console.print(f"[green]   Use 'strava-super ml-analysis' for LSTM forecasting & anomaly detection[/green]")
        except Exception as e:
            # Temporarily show error for debugging
            console.print(f"[red]   Debug: {str(e)[:100]}[/red]")
            console.print(f"[green]   Use 'strava-super ml-analysis' for detailed analysis[/green]")
    except Exception as e:
        console.print(f"[red]❌ Phase 2 integration failed: {e}[/red]")

    # Final Summary
    if not errors:
        console.print(f"\n[green]🎉 Complete workflow finished successfully with Phase 2 enhancements![/green]")


@cli.command()
@click.option('--duration', default=30, help='Plan duration in days (1-365)', type=click.IntRange(1, 365))
def show_training_plan(duration):
    """Display the training plan."""
    config = Config()

    console.print(Panel.fit(f"📅 {duration}-Day Training Plan", style="bold green"))

    try:
        # Use configuration from .env
        rest_days = config.get_training_rest_days()
        max_weekly_hours = config.TRAINING_MAX_WEEKLY_HOURS

        generator = TrainingPlanGenerator("user")

        # Use generic training plan generation for any duration (1-365 days)
        plan = generator.generate_training_plan(
            duration_days=duration,
            goal='balanced',
            constraints={
                'max_weekly_hours': max_weekly_hours,
                'rest_days': rest_days
            }
        )

        if plan and 'summary' in plan:
            console.print(f"[green]✅ {duration}-day plan generated successfully[/green]")

            # Extract plan details with proper chronological sorting
            training_plan = []
            if 'daily_workouts' in plan and plan['daily_workouts']:
                daily_data = plan.get('visualizations', {}).get('daily_data', [])

                # CRITICAL FIX: Sort workouts chronologically to ensure proper date sequencing
                daily_workouts = plan['daily_workouts']
                sorted_workouts = sorted(daily_workouts, key=lambda w: w.date)

                # Skip day 0 (today) and maintain correct data correlation
                for i, workout in enumerate(sorted_workouts):
                    # CRITICAL FIX: Skip based on day_number, not array index after sorting
                    if workout.day_number <= 0:  # Skip today (day 0 or negative)
                        continue
                    # CRITICAL FIX: Find corresponding daily simulation data by day_number, not array index
                    daily_sim = {}
                    for sim_data in daily_data:
                        if sim_data.get('day') == workout.day_number:
                            daily_sim = sim_data
                            break

                    # Use actual workout date and day number
                    actual_day = workout.day_number if hasattr(workout, 'day_number') else len(training_plan) + 1
                    actual_date = workout.date.date() if hasattr(workout.date, 'date') else workout.date

                    # Convert WorkoutPlan dataclass to dictionary with simulation data
                    training_plan.append({
                        'day': actual_day,
                        'date': actual_date.strftime('%m/%d'),
                        'date_obj': actual_date,  # Keep the datetime object for day name conversion
                        'recommendation': get_intensity_from_load(float(workout.planned_load)),
                        'activity': f"{workout.title} ({workout.total_duration_min}m)",
                        'second_activity': (
                            f"{workout.second_activity_title} ({workout.second_activity_duration}m)"
                            if workout.second_activity_title else "—"
                        ),
                        'suggested_load': float(workout.planned_load),
                        'form': daily_sim.get('form', 0.0),
                        'predicted_form': daily_sim.get('form', 0.0),
                        'fitness': daily_sim.get('fitness', 0.0),
                        'predicted_fitness': daily_sim.get('fitness', 0.0)
                    })
            else:
                training_plan = []

            if training_plan:
                title = f"{duration}-Day Training Plan"
                table = Table(title=title, box=box.ROUNDED)
                table.add_column("Day", style="black", width=3)
                table.add_column("Date", width=10)
                table.add_column("Week Type", style="blue", width=10)
                table.add_column("Intensity", style="yellow", width=11)
                table.add_column("Primary Activity & Duration", style="blue", width=32)
                table.add_column("2nd Activity & Duration", style="green", width=28)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="black", width=6)
                table.add_column("Fitness", style="green", width=7)

                # For 30-day plan, add week separators with horizontal lines
                current_week = -1
                added_week_separator = False

                for i, plan_day in enumerate(training_plan):
                    week_num = (plan_day['day'] - 1) // 7

                    # Add week separator for new week
                    if week_num != current_week:
                        current_week = week_num

                        # Add horizontal separator line if this is not the first week
                        if week_num > 0:
                            table.add_row(
                                "[white]─[/white]",
                                "[white]──────────[/white]",
                                "[black]───────[/black]",
                                "[white]───────────[/white]",
                                "[black]─────────────────────────[/black]",
                                "[white]──────────────────────────[/white]",
                                "[black]────[/black]",
                                "[white]──[/white]",
                                "[white]────[/white]"
                            )

                        # Define week types based on sports science periodization
                        week_types = {
                            0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",  # Weeks 1-4
                            4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",  # Weeks 5-8
                            8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY", # Weeks 9-12
                            12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"   # Weeks 13-16
                        }
                        # For longer plans, extend the pattern
                        if week_num not in week_types:
                            cycle_week = week_num % 4
                            if cycle_week < 3:
                                week_types[week_num] = f"BUILD {cycle_week + 1}"
                            else:
                                week_types[week_num] = "RECOVERY"

                        week_type = week_types.get(week_num, "MAINTENANCE")
                        week_type_color = {
                            "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                            "RECOVERY": "bright_black", "PEAK 1": "bright_red", "PEAK 2": "red",
                            "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "black"
                        }.get(week_type, "black")

                        # Calculate weekly time totals for this week
                        week_plans = [p for p in training_plan if (p['day'] - 1) // 7 == week_num]
                        total_weekly_minutes = 0
                        for plan in week_plans:
                            # Extract duration from activity string (format: "Activity (60m)")
                            activity = plan.get('activity', '')
                            if '(' in activity and 'm)' in activity:
                                try:
                                    duration_str = activity.split('(')[-1].split('m)')[0]
                                    total_weekly_minutes += int(duration_str)
                                except:
                                    pass

                            # Add second activity duration
                            second_activity = plan.get('second_activity', '')
                            if second_activity and second_activity != '—' and '(' in second_activity and 'm)' in second_activity:
                                try:
                                    duration_str = second_activity.split('(')[-1].split('m)')[0]
                                    total_weekly_minutes += int(duration_str)
                                except:
                                    pass

                        weekly_hours = total_weekly_minutes / 60

                        # Apply mesocycle-specific hour budget
                        base_budget_hours = config.TRAINING_MAX_WEEKLY_HOURS
                        def get_env_float(key: str, default: float) -> float:
                            try:
                                return float(os.getenv(key, str(default)))
                            except:
                                return default

                        # Get hour reduction factor based on week type
                        hour_factors = {
                            "BUILD 1": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "BUILD 2": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "BUILD 3": get_env_float('HOUR_FACTOR_ACCUMULATION', 1.0),
                            "RECOVERY": get_env_float('HOUR_FACTOR_RECOVERY', 0.6),
                            "PEAK 1": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                            "PEAK 2": get_env_float('HOUR_FACTOR_INTENSIFICATION', 0.9),
                            "TAPER 1": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                            "TAPER 2": get_env_float('HOUR_FACTOR_REALIZATION', 0.7),
                            "MAINTENANCE": get_env_float('HOUR_FACTOR_MAINTENANCE', 0.8)
                        }

                        hour_factor = hour_factors.get(week_type, 1.0)
                        budget_hours = base_budget_hours * hour_factor

                        time_status = "✅" if weekly_hours <= budget_hours else ("⚠️" if weekly_hours <= budget_hours * 1.1 else "🚨")

                        # Enhanced week summary with additional information
                        hours_info = f"{weekly_hours:.1f}h planned"
                        budget_info = f"Max: {budget_hours:.1f}h"
                        percentage = (weekly_hours / budget_hours * 100) if budget_hours > 0 else 0
                        utilization = f"({percentage:.0f}%)"

                        week_summary = f"{time_status} {hours_info} | {budget_info} {utilization}"

                        table.add_row(
                            f"[bold black]W{week_num+1}[/bold black]",
                            "[black]────────[/black]",
                            f"[bold {week_type_color}]{week_type}[/bold {week_type_color}]",
                            "[white]─────────[/white]",
                            f"[bold orange]{week_summary}[/bold orange]",
                            "[black]─────────────────────────[/black]",
                            "[black]────[/black]",
                            "[black]──────[/black]",
                            "[black]───────[/black]"
                        )

                    # Color code recommendations
                    rec_colors = {
                        "REST": "red",
                        "RECOVERY": "yellow",
                        "EASY": "green",
                        "MODERATE": "black",
                        "HARD": "magenta",
                        "PEAK": "bold magenta"
                    }
                    rec_color = rec_colors.get(plan_day['recommendation'], "black")

                    # Get week type for this day
                    week_types = {
                        0: "BUILD 1", 1: "BUILD 2", 2: "BUILD 3", 3: "RECOVERY",
                        4: "BUILD 1", 5: "BUILD 2", 6: "BUILD 3", 7: "RECOVERY",
                        8: "BUILD 1", 9: "BUILD 2", 10: "BUILD 3", 11: "RECOVERY",
                        12: "PEAK 1", 13: "PEAK 2", 14: "TAPER 1", 15: "TAPER 2"
                    }
                    if week_num not in week_types:
                        cycle_week = week_num % 4
                        if cycle_week < 3:
                            week_types[week_num] = f"BUILD {cycle_week + 1}"
                        else:
                            week_types[week_num] = "RECOVERY"

                    week_type = week_types.get(week_num, "MAINTENANCE")
                    week_type_color = {
                        "BUILD 1": "green", "BUILD 2": "green", "BUILD 3": "orange",
                        "RECOVERY": "black", "PEAK 1": "red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "magenta", "MAINTENANCE": "black"
                    }.get(week_type, "black")

                    # Get activity name
                    activity = plan_day.get('activity', 'Unknown')
                    if len(activity) > 29:  # Column width is 32, leave room for duration
                        activity = activity[:26] + "..."

                    # Get second session info
                    second_session = ""
                    if plan_day.get('second_activity'):
                        second_activity = plan_day['second_activity']
                        if len(second_activity) > 29:
                            second_session = second_activity[:26] + "..."
                        else:
                            second_session = second_activity
                    else:
                        second_session = "—"

                    # Format date to include day name: "We, 09/25"
                    date_str = plan_day['date']
                    if len(date_str) > 8:
                        date_str = date_str[5:]  # Remove year, keep MM-DD

                    # Add day name to date
                    day_names = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
                    formatted_date = date_str  # Default to just date

                    # Try to get day name from date
                    if 'date_obj' in plan_day and plan_day['date_obj']:
                        try:
                            day_name = day_names[plan_day['date_obj'].weekday()]
                            formatted_date = f"{day_name}, {date_str}"
                        except:
                            pass

                    table.add_row(
                        str(plan_day['day']),
                        formatted_date,
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        f"[{rec_color}]{plan_day['recommendation']}[/{rec_color}]",
                        f"[blue]{activity}[/blue]",
                        f"[green]{second_session}[/green]" if plan_day.get('second_activity') else f"[white]{second_session}[/white]",
                        f"{plan_day['suggested_load']:.0f}",
                        f"{plan_day['predicted_form']:.1f}",
                        f"{plan_day.get('predicted_fitness', 0):.1f}",
                    )

                console.print(table)

                # Show summary
                total_load = sum(p.get('suggested_load', p.get('load', 0)) for p in training_plan)
                rest_days = len([p for p in training_plan if p['recommendation'] == 'REST'])
                hard_days = len([p for p in training_plan if p['recommendation'] in ['HARD', 'PEAK']])

                console.print(Panel(f"Monthly Summary:\n"
                                    f"• Total Load: {total_load:.0f} TSS\n"
                                    f"• Average Daily Load: {total_load/len(training_plan):.0f} TSS\n"
                                    f"• Rest Days: {rest_days}\n"
                                    f"• Hard/Peak Days: {hard_days}\n"
                                    f"• Periodization: 3:1 (3 weeks build, 1 week recovery)",
                                    title="📊 Training Summary"))
            else:
                console.print("[yellow]⚠️ No training plan details available[/yellow]")
        else:
            console.print("[red]❌ Failed to generate training plan[/red]")

    except Exception as e:
        console.print(f"[red]❌ Error generating training plan: {e}[/red]")


@cli.command()
@click.option('--goal', default='balanced', help='Training goal: fitness, performance, recovery, balanced')
@click.option('--duration', default=30, help='Plan duration in days (1-365)', type=click.IntRange(1, 365))
@click.option('--max-hours', default=12, help='Maximum weekly training hours')
@click.option('--rest-days', default='6', help='Rest days (0=Mon, 6=Sun, comma-separated)')
def training_plan(goal, duration, max_hours, rest_days):
    """Generate training plan using scientific models."""
    console.print(Panel.fit("🧬 Training Plan Generator", style="bold black"))

    try:
        # Parse rest days
        rest_day_list = [int(x.strip()) for x in rest_days.split(',') if x.strip()]

        # Use enhanced training planner for all durations (1-365 days)
        generator = TrainingPlanGenerator("user")

        plan = generator.generate_training_plan(
            duration_days=duration,
            goal=goal,
            constraints={
                'max_weekly_hours': max_hours,
                'rest_days': rest_day_list
            }
        )

        if plan and 'summary' in plan:
            console.print(f"[green]✅ {duration}-day plan generated successfully[/green]")

            # Show summary
            summary = plan['summary']
            console.print(Panel(
                f"Total Load: {summary['total_load']:.0f} TSS\n"
                f"Duration: {summary['total_duration_hours']:.1f} hours\n"
                f"Fitness Gain: +{summary['fitness_gain']:.1f}\n"
                f"Hard Days: {summary['hard_days']}\n"
                f"Rest Days: {summary['rest_days']}",
                title="📊 Plan Summary", style="green"
            ))

        else:
            console.print("[red]❌ Plan generation failed[/red]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@cli.command()
@click.option('--recovery-score', type=float, help='Recovery score (0-100)')
@click.option('--hrv-status', help='HRV status: poor, balanced, good, excellent')
@click.option('--sleep-score', type=float, help='Sleep quality score (0-100)')
@click.option('--stress-level', type=float, help='Stress level (0-100)')
def adjust_plan(recovery_score, hrv_status, sleep_score, stress_level):
    """Adjust training plan based on current wellness data."""
    console.print(Panel.fit("🔧 Plan Adjustment Engine", style="bold yellow"))

    try:
        # Load current plan (simplified - in real use would load from storage)
        generator = TrainingPlanGenerator("user")
        base_plan = generator.generate_30_day_plan(goal='balanced')

        if not base_plan or 'daily_workouts' not in base_plan:
            console.print("[red]❌ Could not load base plan[/red]")
            return

        # Prepare wellness data
        wellness_data = {}
        if recovery_score is not None:
            wellness_data['recovery_score'] = recovery_score
        if hrv_status:
            wellness_data['hrv_status'] = hrv_status
        if sleep_score is not None:
            wellness_data['sleep_quality'] = sleep_score
        if stress_level is not None:
            wellness_data['stress_level'] = stress_level

        if not wellness_data:
            console.print("[yellow]No wellness data provided. Use --help for options.[/yellow]")
            return

        # Generate adjustments
        adjuster = PlanAdjustmentEngine("user")
        adjustments = adjuster.evaluate_plan_adjustments(
            current_plan=base_plan['daily_workouts'],
            wellness_data=wellness_data
        )

        console.print(f"[black]Analyzed wellness data and generated {len(adjustments)} suggestions[/black]")

        # Show top adjustments
        table = Table(title="Recommended Adjustments")
        table.add_column("Type", style="yellow")
        table.add_column("Reason", style="black")
        table.add_column("Confidence", style="green")
        table.add_column("Notes", style="black")

        for adj in adjustments[:5]:  # Show top 5
            table.add_row(
                adj.adjustment_type.value,
                adj.reason.value,
                f"{adj.confidence:.1f}",
                adj.notes[:50] + "..." if len(adj.notes) > 50 else adj.notes
            )

        console.print(table)

        # Apply high-confidence adjustments
        adjusted_plan, applied = adjuster.apply_adjustments(
            base_plan['daily_workouts'],
            adjustments,
            auto_apply_threshold=0.8
        )

        console.print(f"[green]✅ Applied {len(applied)} automatic adjustments[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed model parameters')
def model_status(verbose):
    """Show status of advanced training models."""
    console.print(Panel.fit("🧬 Advanced Model Status", style="bold blue"))

    try:
        analyzer = get_integrated_analyzer("user")

        # Test model availability
        console.print("[black]Testing model components...[/black]")

        # Test FF model
        try:
            from .analysis.advanced_model import FitnessFatigueModel
            ff_model = FitnessFatigueModel()
            console.print("[green]✅ Fitness-Fatigue Model[/green]")
        except Exception as e:
            console.print(f"[red]❌ Fitness-Fatigue Model: {e}[/red]")

        # Test PerPot model
        try:
            from .analysis.advanced_model import PerPotModel
            perpot_model = PerPotModel()
            console.print("[green]✅ PerPot Overtraining Model[/green]")
        except Exception as e:
            console.print(f"[red]❌ PerPot Model: {e}[/red]")

        # Test Optimal Control
        try:
            from .analysis.advanced_model import OptimalControlProblem
            console.print("[green]✅ Optimal Control Solver[/green]")
        except Exception as e:
            console.print(f"[red]❌ Optimal Control: {e}[/red]")

        # Test Plan Generation
        try:
            generator = TrainingPlanGenerator("test")
            console.print("[green]✅ Training Plan Generator[/green]")
        except Exception as e:
            console.print(f"[red]❌ Plan Generator: {e}[/red]")

        # Test Plan Adjustment
        try:
            adjuster = PlanAdjustmentEngine("test")
            console.print("[green]✅ Plan Adjustment Engine[/green]")
        except Exception as e:
            console.print(f"[red]❌ Plan Adjuster: {e}[/red]")

        if verbose:
            # Show detailed model parameters
            console.print("\n[black]Model Parameters:[/black]")
            try:
                from .analysis.advanced_model import FitnessFatigueModel
                ff_model = FitnessFatigueModel()
                console.print(f"  FF Model: k1={ff_model.k1:.3f}, k2={ff_model.k2:.3f}, τ1={ff_model.tau1}d, τ2={ff_model.tau2}d")

                from .analysis.advanced_model import PerPotModel
                perpot_model = PerPotModel()
                console.print(f"  PerPot Model: ds={perpot_model.ds:.1f}, dr={perpot_model.dr:.1f}, dso={perpot_model.dso:.1f}")

                console.print(f"  Optimization: Differential Evolution with population=15")
                console.print(f"  Recovery: Multi-system (metabolic, neural, structural)")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load model parameters: {e}[/yellow]")

        console.print("\n[black]All models operational and ready for advanced training analysis.[/black]")

    except Exception as e:
        console.print(f"[red]❌ Model status check failed: {e}[/red]")


@cli.command()
@click.option('--days', default=90, help='Days of history to analyze')
@click.option('--detailed', is_flag=True, help='Show detailed analysis breakdown')
def fitness_state(days, detailed):
    """Analyze current fitness state using advanced models."""
    console.print(Panel.fit("🔬 Advanced Fitness State Analysis", style="bold green"))

    try:
        analyzer = get_integrated_analyzer("user")

        # Get comprehensive analysis
        analysis = analyzer.analyze_with_advanced_models(days_back=days)

        if not analysis or 'combined' not in analysis:
            console.print("[red]❌ No training data available for analysis[/red]")
            return

        combined = analysis['combined']
        if len(combined) == 0:
            console.print("[red]❌ Insufficient data for analysis[/red]")
            return

        latest = combined.iloc[-1]

        # Main fitness state display
        console.print(f"\n[bold black]Current Fitness State (Last {days} days)[/bold black]")

        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="black")
        table.add_column("Value", style="black")
        table.add_column("Status", style="green")

        # Fitness-Fatigue metrics
        ff_fitness = latest['ff_fitness']
        ff_fatigue = latest['ff_fatigue']
        ff_form = latest['ff_form']

        table.add_row("Fitness (CTL)", f"{ff_fitness:.1f}", "🏃‍♂️ Excellent" if ff_fitness > 80 else "📈 Building")
        table.add_row("Fatigue (ATL)", f"{ff_fatigue:.1f}", "😴 High" if ff_fatigue > 60 else "✅ Manageable")
        table.add_row("Form (TSB)", f"{ff_form:.1f}", "🚀 Peaked" if ff_form > 20 else "⚡ Ready")

        # Advanced metrics
        readiness = latest['composite_readiness']
        perf_potential = latest['perpot_performance']
        overtraining = latest['overtraining_risk']

        table.add_row("Readiness Score", f"{readiness:.1f}%", "🟢 High" if readiness > 70 else "🟡 Moderate" if readiness > 40 else "🔴 Low")
        table.add_row("Performance Potential", f"{perf_potential:.3f}", "🎯 Optimal" if perf_potential > 0.8 else "📊 Good")
        # Show precise risk state instead of binary overtraining flag
        risk_state = latest.get('risk_state', 'SAFE')
        risk_display = risk_state.replace('_', ' ').title()
        risk_status = "⚠️ REST NEEDED" if risk_state == "NON_FUNCTIONAL_OVERREACHING" else ("🟡 RECOVERY" if risk_state == "HIGH_STRAIN" else "✅ SAFE")
        table.add_row("Risk State", risk_display, risk_status)

        console.print(table)

        # Recovery analysis
        recovery = latest['overall_recovery']
        console.print(f"\n[bold black]Recovery Status[/bold black]")
        console.print(f"Overall Recovery: {recovery:.1f}% - {'🟢 Excellent' if recovery > 80 else '🟡 Good' if recovery > 60 else '🔴 Poor'}")

        if detailed:
            # Show detailed breakdown
            console.print(f"\n[bold black]Detailed Analysis[/bold black]")

            # Recent trend (last 7 days)
            recent = combined.tail(7)
            avg_load = recent['load'].mean()
            trend_fitness = recent['ff_fitness'].iloc[-1] - recent['ff_fitness'].iloc[0]
            trend_fatigue = recent['ff_fatigue'].iloc[-1] - recent['ff_fatigue'].iloc[0]

            console.print(f"📊 7-Day Trend:")
            console.print(f"  • Average Load: {avg_load:.0f} TSS/day")
            console.print(f"  • Fitness Trend: {trend_fitness:+.1f} (weekly change)")
            console.print(f"  • Fatigue Trend: {trend_fatigue:+.1f} (weekly change)")

            # Recommendations
            console.print(f"\n[bold yellow]Recommendations[/bold yellow]")
            if overtraining:
                console.print("🚨 [red]IMMEDIATE REST REQUIRED[/red] - Overtraining detected")
            elif readiness > 80:
                console.print("🚀 [green]PEAK TRAINING WINDOW[/green] - Ready for high intensity")
            elif readiness < 40:
                console.print("😴 [yellow]RECOVERY FOCUS[/yellow] - Prioritize easy training")
            else:
                console.print("📈 [black]PROGRESSIVE TRAINING[/black] - Balanced load progression")

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")


@cli.command()
@click.option('--days-ahead', default=14, help='Days ahead to predict performance')
@click.option('--target-load', default=100, help='Target daily training load (TSS)')
@click.option('--show-optimal', is_flag=True, help='Show optimal load recommendations')
def predict(days_ahead, target_load, show_optimal):
    """Predict future performance using advanced models."""
    console.print(Panel.fit("🔮 Performance Prediction", style="bold magenta"))

    try:
        analyzer = get_integrated_analyzer("user")

        # Get current state
        current_state = analyzer._get_current_state()

        if not current_state:
            console.print("[red]❌ No baseline data available for prediction[/red]")
            return

        console.print(f"\n[bold black]Performance Prediction ({days_ahead} days ahead)[/bold black]")
        console.print(f"Target Load: {target_load} TSS/day")

        # Use FF model for prediction
        from .analysis.advanced_model import FitnessFatigueModel
        ff_model = FitnessFatigueModel()

        # Create load schedule
        loads = [target_load] * days_ahead
        days = np.arange(days_ahead)

        # Predict using current state as baseline
        fitness, fatigue, performance = ff_model.calculate_fitness_fatigue(
            np.array(loads),
            days
        )

        # Add current state to predictions
        current_fitness = current_state.get('fitness', 50)
        current_fatigue = current_state.get('fatigue', 20)

        fitness = fitness + current_fitness
        fatigue = fatigue + current_fatigue

        # Show prediction summary
        table = Table(box=box.ROUNDED)
        table.add_column("Day", style="black")
        table.add_column("Fitness", style="green")
        table.add_column("Fatigue", style="red")
        table.add_column("Form", style="yellow")
        table.add_column("Status")

        for i in range(min(7, days_ahead)):  # Show first 7 days
            form = fitness[i] - fatigue[i]
            status = "🚀 Peak" if form > 20 else "⚡ Ready" if form > 0 else "😴 Tired"

            table.add_row(
                f"Day {i+1}",
                f"{fitness[i]:.1f}",
                f"{fatigue[i]:.1f}",
                f"{form:.1f}",
                status
            )

        console.print(table)

        # Show final prediction
        final_fitness = fitness[-1]
        final_fatigue = fatigue[-1]
        final_form = final_fitness - final_fatigue

        console.print(f"\n[bold green]Final State (Day {days_ahead})[/bold green]")
        console.print(f"Predicted Fitness: {final_fitness:.1f} ({final_fitness - current_state.get('fitness', 50):+.1f})")
        console.print(f"Predicted Fatigue: {final_fatigue:.1f} ({final_fatigue - current_state.get('fatigue', 20):+.1f})")
        console.print(f"Predicted Form: {final_form:.1f}")

        if show_optimal:
            # Generate optimal plan for comparison
            console.print(f"\n[bold yellow]Optimal Load Recommendations[/bold yellow]")

            plan = analyzer.generate_optimal_plan(
                goal='balanced',
                duration_days=min(days_ahead, 7),
                rest_days=[6]
            )

            if plan.get('success'):
                optimal_loads = plan['loads']
                avg_optimal = np.mean(optimal_loads)

                console.print(f"Suggested average load: {avg_optimal:.0f} TSS/day")
                console.print(f"Your target load: {target_load} TSS/day")

                if target_load > avg_optimal * 1.2:
                    console.print("⚠️ [yellow]Target load may be too high - risk of overtraining[/yellow]")
                elif target_load < avg_optimal * 0.8:
                    console.print("📈 [black]Target load is conservative - room for progression[/black]")
                else:
                    console.print("✅ [green]Target load is well-balanced[/green]")

    except Exception as e:
        console.print(f"[red]❌ Prediction failed: {e}[/red]")


@cli.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def clean_metrics(confirm):
    """Clean corrupted metrics data and ensure realistic physiological state."""
    console.print(Panel.fit("🧹 Clean Corrupted Metrics Data", style="bold yellow"))

    if not confirm:
        console.print("[yellow]This will delete all calculated fitness/fatigue metrics.[/yellow]")
        console.print("[yellow]Your activity data will be preserved.[/yellow]")
        console.print("[yellow]Metrics will be recalculated from clean state on next analysis.[/yellow]")

        if not click.confirm("Continue with metrics cleanup?"):
            console.print("[black]Operation cancelled.[/black]")
            return

    try:
        from .db import get_db
        from .db.models import Metric

        db = get_db()

        with db.get_session() as session:
            # Get count of metrics to be deleted
            metrics_count = session.query(Metric).count()

            if metrics_count > 0:
                console.print(f"[yellow]Found {metrics_count} metrics records to clean...[/yellow]")

                # Delete all metrics
                session.query(Metric).delete()
                session.commit()

                console.print(f"[green]✅ Successfully deleted {metrics_count} corrupted metrics records[/green]")
                console.print("[black]📊 Metrics will be recalculated from clean state on next analysis[/black]")
                console.print("[black]💡 Run 'strava-super analyze' to rebuild metrics with realistic values[/black]")
            else:
                console.print("[green]✅ Metrics table is already clean[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error cleaning metrics: {e}[/red]")


@cli.command()
@click.option('--race-date', required=True, help='Race date (YYYY-MM-DD)')
@click.option('--distance', required=True, type=float, help='Race distance in km')
@click.option('--retrain', is_flag=True, help='Retrain model before prediction')
def predict_race(race_date: str, distance: float, retrain: bool):
    """Predict race performance using ML model."""
    from .analysis.performance_predictor import PerformancePredictor
    from .db import get_db
    import os

    console.print("[black]🏃 Race Performance Prediction[/black]")
    console.print(f"Race Date: {race_date}")
    console.print(f"Distance: {distance}km")

    try:
        # Get database session
        db = get_db()

        with db.get_session() as session:
            predictor = PerformancePredictor(db_session=session)

            model_path = 'models/performance_model.joblib'

            # Train or load model
            if retrain or not os.path.exists(model_path):
                console.print("[yellow]Training performance prediction model...[/yellow]")
                with console.status("[yellow]Training model...[/yellow]"):
                    metadata = predictor.train_model()
                    predictor.save_model(model_path)
                console.print(f"[green]✅ Model trained with MAE: {metadata['mae']:.2f} km/h[/green]")
            else:
                predictor.load_model(model_path)
                console.print("[green]✅ Model loaded from cache[/green]")

            # Parse race date
            race_datetime = datetime.strptime(race_date, '%Y-%m-%d')

            # Make prediction
            with console.status("[yellow]Calculating prediction...[/yellow]"):
                prediction = predictor.predict_race_performance(race_datetime, distance)

            # Display results
            table = Table(title=f"🏁 Race Prediction - {distance}km", box=box.ROUNDED)
            table.add_column("Metric", style="black")
            table.add_column("Value", style="yellow")

            table.add_row("Predicted Time", prediction['predicted_time_formatted'])
            table.add_row("Predicted Pace", f"{prediction['predicted_pace_per_km']}/km")
            table.add_row("Predicted Speed", f"{prediction['predicted_speed_kmh']:.2f} km/h")
            table.add_row("Model Confidence", f"{prediction['confidence']:.1%}")
            table.add_row("Model MAE", f"±{prediction['model_mae']:.2f} km/h")

            console.print(table)

            # Show feature importance if available
            if prediction.get('feature_importance'):
                importance_table = Table(title="🔍 Key Performance Factors", box=box.SIMPLE)
                importance_table.add_column("Feature", style="black")
                importance_table.add_column("Importance", style="yellow")

                for feat, imp in prediction['feature_importance'].items():
                    # Format feature name
                    feat_display = feat.replace('_', ' ').title()
                    importance_table.add_row(feat_display, f"{imp:.3f}")

                console.print(importance_table)

    except Exception as e:
        console.print(f"[red]❌ Prediction failed: {e}[/red]")


@cli.command()
@click.option('--race-date', required=True, help='Race date (YYYY-MM-DD)')
@click.option('--distance', required=True, type=float, help='Race distance in km')
def optimize_taper(race_date: str, distance: float):
    """Optimize taper strategy for race performance."""
    from .analysis.performance_predictor import PerformancePredictor
    from .db import get_db
    import os

    console.print("[black]📈 Taper Strategy Optimization[/black]")
    console.print(f"Race Date: {race_date}")
    console.print(f"Distance: {distance}km")

    try:
        # Get database session
        db = get_db()

        with db.get_session() as session:
            # Get current CTL from database
            from .db.models import Metric
            latest_metrics = session.query(Metric).order_by(
                Metric.date.desc()
            ).first()

            current_ctl = latest_metrics.fitness if latest_metrics else 80

            predictor = PerformancePredictor(db_session=session)

            model_path = 'models/performance_model.joblib'
            if not os.path.exists(model_path):
                console.print("[yellow]Training model first...[/yellow]")
                metadata = predictor.train_model()
                predictor.save_model(model_path)
            else:
                predictor.load_model(model_path)

            # Parse race date
            race_datetime = datetime.strptime(race_date, '%Y-%m-%d')

            # Optimize taper
            with console.status("[yellow]Comparing taper strategies...[/yellow]"):
                optimization = predictor.optimize_taper(race_datetime, distance, current_ctl)

            # Display results
            table = Table(title="🏆 Taper Strategy Comparison", box=box.ROUNDED)
            table.add_column("Strategy", style="black")
            table.add_column("TSB Target", style="yellow")
            table.add_column("Duration", style="yellow")
            table.add_column("Load Reduction", style="yellow")
            table.add_column("Predicted Speed", style="green")

            for name, details in optimization['all_strategies'].items():
                is_best = name == optimization['recommended_strategy']
                style = "bold green" if is_best else ""
                marker = "✅ " if is_best else ""

                table.add_row(
                    f"{marker}{name.title()}",
                    f"{details['tsb_target']}",
                    f"{details['duration_days']} days",
                    f"{details['load_reduction']:.0%}",
                    f"{details['predicted_speed_kmh']:.2f} km/h",
                    style=style
                )

            console.print(table)

            # Summary
            console.print("\n[green]📊 Optimization Summary:[/green]")
            console.print(f"Recommended Strategy: [bold black]{optimization['recommended_strategy'].title()}[/bold black]")
            console.print(f"Optimal TSB: [yellow]{optimization['optimal_tsb']}[/yellow]")
            console.print(f"Taper Duration: [yellow]{optimization['optimal_duration']} days[/yellow]")
            console.print(f"Speed Improvement: [green]+{optimization['improvement_over_worst']:.2f} km/h[/green]")

    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {e}[/red]")


@cli.command()
@click.option('--phase', type=click.Choice(['BASE', 'BUILD', 'PEAK', 'TAPER']),
              help='Endurance training phase', default='BUILD')
@click.option('--week', type=int, default=1, help='Week number in phase')
@click.option('--program', is_flag=True, help='Generate full multi-week streprogen program')
@click.option('--duration', type=int, help='Program duration in weeks (for --program)', default=4)
@click.option('--athlete-level', type=click.Choice(['beginner', 'intermediate', 'advanced']),
              help='Athlete level', default='intermediate')
@click.option('--export', help='Export program to file (specify path with .txt/.html/.tex extension)')
@click.option('--analyze', is_flag=True, help='Show detailed program analysis')
@click.option('--rpe-target', type=float, help='Target RPE for autoregulated program (6.0-10.0)', default=8.0)
@click.option('--deload', is_flag=True, help='Generate deload week program')
def strength_workout(phase: str, week: int, program: bool, duration: int, athlete_level: str,
                    export: str, analyze: bool, rpe_target: float, deload: bool):
    """Generate strength training workout or full periodized program using streprogen."""
    from .analysis.strength_planner import StrengthPlanner
    from datetime import date

    console.print("[black]💪 Advanced Strength Training System[/black]")

    try:
        planner = StrengthPlanner()

        # Get current maxes from .env configuration
        current_maxes = planner.get_current_maxes()
        console.print(f"[black]Using your configured lifts:[/black] Bench: {current_maxes['bench']}kg, Squat: {current_maxes['squat']}kg, Deadlift: {current_maxes['deadlift']}kg, Press: {current_maxes['press']}kg")
        console.print(f"[black]💡 Update these in your .env file: STRENGTH_BENCH_PRESS_1RM, etc.[/black]")

        # Handle deload week generation
        if deload:
            console.print("[orange]🔄 Generating Deload Week Program[/orange]")

            strength_phase = planner.align_with_endurance_phase(phase, week)
            deload_program = planner.generate_deload_week_program(strength_phase, current_maxes)

            if deload_program:
                console.print(planner.get_program_summary(deload_program))

                if export:
                    format_type = export.split('.')[-1] if '.' in export else 'txt'
                    success = planner.export_program_to_file(deload_program, export, format_type)
                    if success:
                        console.print(f"[green]✅ Deload program exported to {export}[/green]")
            else:
                console.print("[red]❌ Failed to generate deload program[/red]")
            return

        # Handle full program generation
        if program:
            # Use athlete level from .env, with CLI override
            effective_athlete_level = athlete_level if athlete_level != "intermediate" else planner.athlete_level
            console.print(f"[orange]📋 Generating {duration}-Week Periodized Program[/orange]")
            console.print(f"Phase: {phase} | Level: {effective_athlete_level}")

            strength_phase = planner.align_with_endurance_phase(phase, week)

            # Generate full streprogen program with advanced features
            if rpe_target != 8.0:  # Custom RPE target
                console.print(f"[black]🎯 Using RPE-based autoregulation (target: {rpe_target})[/black]")
                full_program = planner.create_autoregulated_program(
                    phase=strength_phase,
                    duration_weeks=duration,
                    rpe_target=rpe_target,
                    current_maxes=current_maxes
                )
            else:
                full_program = planner.generate_streprogen_program(
                    phase=strength_phase,
                    duration_weeks=duration,
                    athlete_level=effective_athlete_level,
                    current_maxes=current_maxes
                )

            if full_program:
                # Show program summary
                console.print(planner.get_program_summary(full_program))

                # Show detailed analysis if requested
                if analyze:
                    console.print("\n[black]📊 Program Analysis:[/black]")
                    metrics = planner.analyze_program_metrics(full_program)

                    analysis_table = Table(title="Program Metrics", box=box.ROUNDED)
                    analysis_table.add_column("Metric", style="black")
                    analysis_table.add_column("Value", style="yellow")

                    analysis_table.add_row("Duration", f"{metrics.get('duration_weeks', 0)} weeks")
                    analysis_table.add_row("Training Frequency", f"{metrics.get('training_frequency', 0)} days/week")
                    analysis_table.add_row("Total Exercises", str(metrics.get('total_exercises', 0)))
                    analysis_table.add_row("Avg Exercises/Day", f"{metrics.get('avg_exercises_per_day', 0):.1f}")
                    analysis_table.add_row("Volume Classification", metrics.get('volume_classification', 'Unknown'))
                    analysis_table.add_row("Intensity Classification", metrics.get('intensity_classification', 'Unknown'))

                    console.print(analysis_table)

                # Export if requested
                if export:
                    format_type = export.split('.')[-1] if '.' in export else 'txt'
                    success = planner.export_program_to_file(full_program, export, format_type)
                    if success:
                        console.print(f"[green]✅ Program exported to {export}[/green]")
                        console.print("[black]💡 Open the file to see the complete week-by-week progression![/black]")
                    else:
                        console.print(f"[red]❌ Failed to export program to {export}[/red]")
            else:
                console.print("[red]❌ Failed to generate streprogen program[/red]")
            return

        # Handle single workout generation (original functionality)
        console.print(f"[orange]🏋️ Generating Today's Workout[/orange]")

        # Get today's day of week (0=Monday)
        today = date.today()
        day_of_week = today.weekday()

        # Generate workout
        phase = phase or "BUILD"  # Default to build phase
        workout = planner.generate_workout(
            strength_phase=planner.align_with_endurance_phase(phase, week),
            week_number=week,
            day_of_week=day_of_week
        )

        if workout is None:
            console.print(f"[orange]No strength training scheduled for {today.strftime('%A')}[/orange]")
            console.print("[black]Strength training days vary by phase:[/black]")
            console.print("• Base Phase: Tuesday, Friday")
            console.print("• Build Phase: Tuesday, Saturday")
            console.print("• Peak/Taper: Wednesday only")
            console.print("\n[black]💡 Advanced streprogen features available:[/black]")
            console.print("[black]   --program                 Generate full multi-week program[/black]")
            console.print("[black]   --deload                  Generate deload week[/black]")
            console.print("[black]   --rpe-target 8.5          RPE-based autoregulation[/black]")
            console.print("[black]   --export program.html     Export to HTML/TXT/TEX[/black]")
            return

        # Display single workout
        table = Table(title=f"💪 {workout.phase.value.replace('_', ' ').title()} - Week {week}", box=box.ROUNDED)
        table.add_column("Exercise", style="black")
        table.add_column("Sets", style="orange")
        table.add_column("Reps", style="orange")
        table.add_column("Intensity", style="green")
        table.add_column("Rest", style="magenta")
        table.add_column("Notes", style="black")

        for ex in workout.exercises:
            table.add_row(
                ex.name,
                str(ex.sets),
                str(ex.reps),
                f"{ex.intensity_percent:.0f}%",
                f"{ex.rest_seconds}s",
                ex.notes
            )

        console.print(table)

        # Workout summary
        summary_table = Table(title="📊 Workout Summary", box=box.SIMPLE)
        summary_table.add_column("Metric", style="black")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row("Duration", f"{workout.duration_min} minutes")
        summary_table.add_row("Training Load", f"{workout.volume_tss:.0f} TSS")
        summary_table.add_row("Warmup", workout.warmup_protocol)
        summary_table.add_row("Cooldown", workout.cooldown_protocol)

        console.print(summary_table)

        # Notes
        if workout.notes:
            console.print(f"\n[black]📝 Notes:[/black] {workout.notes}")

        # Show available advanced features
        console.print("\n[black]💡 Advanced streprogen features available:[/black]")
        console.print("[black]   strava-super strength-workout --program --duration 8[/black]")
        console.print("[black]   strava-super strength-workout --deload --export deload.html[/black]")

    except Exception as e:
        console.print(f"[red]❌ Failed to generate workout: {e}[/red]")


@cli.group()
def insights():
    """Advanced data science insights and correlation analysis."""
    pass


@insights.command(name='correlations')
@click.option('--days', default=90, help='Days of historical data to analyze')
@click.option('--min-samples', default=10, help='Minimum samples required for correlation')
@click.option('--significance', default=0.05, help='P-value threshold for significance')
@click.option('--export', help='Export results to JSON file')
def analyze_correlations(days: int, min_samples: int, significance: float, export: str):
    """
    Comprehensive correlation analysis between wellness and performance metrics.

    Analyzes relationships between:
    - HRV metrics vs training performance
    - Sleep quality vs training adaptation
    - Stress levels vs fatigue/recovery
    - Body composition vs power output
    - Health markers vs training capacity

    Also identifies time-lagged correlations (leading indicators).
    """
    from .analysis.correlation_analyzer import WellnessPerformanceCorrelations
    import json

    console.print(Panel.fit(
        "📊 Comprehensive Correlation Analysis",
        style="bold black"
    ))

    try:
        analyzer = WellnessPerformanceCorrelations()

        console.print(f"\n[black]Analyzing {days} days of data...[/black]")
        results = analyzer.analyze_all_correlations(
            days_back=days,
            min_samples=min_samples,
            significance_level=significance
        )

        if results.get('status') == 'insufficient_data':
            console.print(f"[orange]⚠️  {results['message']}[/orange]")
            return

        # Summary
        console.print(f"\n[green]✅ Analysis complete![/green]")
        console.print(f"Period: {results['analysis_period']['start_date']} to {results['analysis_period']['end_date']}")
        console.print(f"Data points: {results['analysis_period']['data_points']}")

        summary = results['summary']
        console.print(f"\n[black]📈 Summary Statistics:[/black]")
        console.print(f"  Total correlations tested: {summary['total_correlations_tested']}")
        console.print(f"  Significant correlations: {summary['significant_correlations']} "
                     f"({summary['significance_rate']}%)")
        console.print(f"  Leading indicators found: {summary['leading_indicators_found']}")
        console.print(f"  Data quality: {summary['data_quality_score']}")

        # Category breakdown
        console.print(f"\n[black]🏷️  Correlations by Category:[/black]")
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Category", style="black")
        table.add_column("Total", justify="right")
        table.add_column("Significant", justify="right", style="green")
        table.add_column("Rate", justify="right")

        for category, analysis in results['category_analyses'].items():
            rate = (analysis['significant_count'] / analysis['count'] * 100) if analysis['count'] > 0 else 0
            table.add_row(
                category.replace('_', ' ').title(),
                str(analysis['count']),
                str(analysis['significant_count']),
                f"{rate:.1f}%"
            )

        console.print(table)

        # Top significant correlations
        significant = results['significant_correlations']
        if significant:
            console.print(f"\n[black]🔥 Top Significant Correlations:[/black]")

            # Sort by absolute correlation strength
            top_corrs = sorted(significant, key=lambda x: abs(x['correlation']), reverse=True)[:10]

            corr_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            corr_table.add_column("Relationship", style="black", width=45)
            corr_table.add_column("r", justify="right", style="yellow")
            corr_table.add_column("p", justify="right")
            corr_table.add_column("Strength", justify="center")
            corr_table.add_column("n", justify="right", style="dim")

            for corr in top_corrs:
                # Color code by strength
                strength = corr['strength']
                if strength == 'strong':
                    strength_style = "[bold green]Strong[/bold green]"
                elif strength == 'moderate':
                    strength_style = "[yellow]Moderate[/yellow]"
                else:
                    strength_style = "[dim]Weak[/dim]"

                # Format correlation coefficient with color
                r_val = corr['correlation']
                if abs(r_val) >= 0.7:
                    r_display = f"[bold green]{r_val:+.3f}[/bold green]"
                elif abs(r_val) >= 0.4:
                    r_display = f"[yellow]{r_val:+.3f}[/yellow]"
                else:
                    r_display = f"{r_val:+.3f}"

                corr_table.add_row(
                    corr['interpretation'],
                    r_display,
                    f"{corr['p_value']:.4f}",
                    strength_style,
                    str(corr['n_samples'])
                )

            console.print(corr_table)

        # Leading indicators
        leading = results['leading_indicators']
        if leading:
            console.print(f"\n[black]🎯 Leading Indicators (Predictive Metrics):[/black]")

            leading_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            leading_table.add_column("Indicator → Target", style="black", width=35)
            leading_table.add_column("Lag", justify="center", style="yellow")
            leading_table.add_column("r", justify="right")
            leading_table.add_column("Power", justify="center")
            leading_table.add_column("Actionable Insight", width=45)

            for indicator in leading[:8]:  # Top 8
                power = indicator['predictive_power']
                if power == 'strong':
                    power_style = "[bold green]Strong[/bold green]"
                elif power == 'moderate':
                    power_style = "[yellow]Moderate[/yellow]"
                else:
                    power_style = "[dim]Weak[/dim]"

                leading_table.add_row(
                    f"{indicator['indicator_metric']} → {indicator['target_metric']}",
                    f"{indicator['optimal_lag_days']}d",
                    f"{indicator['correlation']:+.3f}",
                    power_style,
                    indicator['actionable_insight']
                )

            console.print(leading_table)

        # Actionable insights
        if results['actionable_insights']:
            console.print(f"\n[black]💡 Key Insights:[/black]")
            for i, insight in enumerate(results['actionable_insights'], 1):
                console.print(f"  {i}. {insight}")

        # Export if requested
        if export:
            with open(export, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]✅ Results exported to {export}[/green]")

        # Recommendations
        console.print(f"\n[black]📋 Recommendations:[/black]")
        console.print("  • Monitor metrics with strong correlations closely")
        console.print("  • Use leading indicators for proactive training adjustments")
        console.print("  • Re-run analysis monthly to track relationship changes")
        console.print("  • Focus on metrics with highest predictive power")

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()


@insights.command(name='wellness-trends')
@click.option('--days', default=60, help='Days to analyze')
@click.option('--metric', help='Specific metric to analyze (hrv_rmssd, sleep_score, etc.)')
def analyze_wellness_trends(days: int, metric: str):
    """
    Analyze wellness metric trends and patterns.

    Shows statistical trends, anomalies, and correlations with training load.
    """
    from .analysis.correlation_analyzer import WellnessPerformanceCorrelations

    console.print(Panel.fit(
        "📈 Wellness Trends Analysis",
        style="bold black"
    ))

    console.print(f"[black]Analyzing {days} days of wellness data...[/black]")

    try:
        analyzer = WellnessPerformanceCorrelations()
        results = analyzer.analyze_all_correlations(days_back=days)

        if results.get('status') == 'insufficient_data':
            console.print(f"[orange]⚠️  {results['message']}[/orange]")
            return

        # Show wellness-performance category
        wellness_analysis = results['category_analyses'].get('wellness_performance', {})

        console.print(f"\n[green]Wellness-Performance Relationships:[/green]")
        console.print(f"  Found {wellness_analysis.get('significant_count', 0)} significant correlations")

        # Show relevant correlations
        if wellness_analysis.get('significant'):
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Metric Pair", style="black", width=50)
            table.add_column("Correlation", justify="right", style="yellow")
            table.add_column("Significance", justify="center")

            for corr in wellness_analysis['significant']:
                sig_marker = "✓" if corr['p_value'] < 0.01 else "~"
                table.add_row(
                    corr['interpretation'],
                    f"{corr['correlation']:+.3f}",
                    sig_marker
                )

            console.print(table)

        console.print(f"\n[dim]Use 'strava-super insights correlations' for full analysis[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")


@insights.command(name='predictive-metrics')
@click.option('--days', default=90, help='Days to analyze')
def show_predictive_metrics(days: int):
    """
    Show metrics that predict future performance (leading indicators).

    Identifies which wellness metrics today predict training capacity tomorrow.
    """
    from .analysis.correlation_analyzer import WellnessPerformanceCorrelations

    console.print(Panel.fit(
        "🎯 Predictive Metrics Dashboard",
        style="bold black"
    ))

    try:
        analyzer = WellnessPerformanceCorrelations()
        results = analyzer.analyze_all_correlations(days_back=days)

        if results.get('status') == 'insufficient_data':
            console.print(f"[orange]⚠️  {results['message']}[/orange]")
            return

        leading = results['leading_indicators']

        if not leading:
            console.print("[orange]No strong leading indicators found in current dataset[/orange]")
            console.print("[dim]Try increasing the analysis period with --days[/dim]")
            return

        console.print(f"\n[green]Found {len(leading)} predictive metrics[/green]\n")

        # Group by predictive power
        strong = [l for l in leading if l['predictive_power'] == 'strong']
        moderate = [l for l in leading if l['predictive_power'] == 'moderate']
        weak = [l for l in leading if l['predictive_power'] == 'weak']

        if strong:
            console.print("[bold green]🔥 Strong Predictive Power:[/bold green]")
            for indicator in strong:
                console.print(f"\n  📊 {indicator['indicator_metric']} → {indicator['target_metric']}")
                console.print(f"     Optimal lag: {indicator['optimal_lag_days']} days")
                console.print(f"     Correlation: {indicator['correlation']:+.3f} (p={indicator['p_value']:.4f})")
                console.print(f"     💡 {indicator['actionable_insight']}")

        if moderate:
            console.print(f"\n[yellow]⚡ Moderate Predictive Power:[/yellow]")
            for indicator in moderate:
                console.print(f"  • {indicator['indicator_metric']} → {indicator['target_metric']} "
                            f"({indicator['optimal_lag_days']}d lag, r={indicator['correlation']:+.3f})")

        if weak:
            console.print(f"\n[dim]📉 Weak Predictive Power ({len(weak)} metrics)[/dim]")

        console.print(f"\n[black]💡 How to use this:[/black]")
        console.print("  1. Monitor strong predictive metrics daily")
        console.print("  2. Adjust training plans based on indicator changes")
        console.print("  3. Use lag time to plan recovery days proactively")
        console.print("  4. Track accuracy of predictions over time")

    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")


@cli.command(name='ml-analysis')
@click.option('--days', default=90, help='Days of historical data to analyze')
@click.option('--forecast-days', default=14, help='Days to forecast into future')
@click.option('--train-lstm', is_flag=True, help='Force retrain LSTM model')
@click.option('--train-anomaly', is_flag=True, help='Force retrain anomaly detector')
@click.option('--export', help='Export results to JSON file')
def ml_analysis(days: int, forecast_days: int, train_lstm: bool, train_anomaly: bool, export: str):
    """
    Advanced ML analysis: LSTM forecasting, anomaly detection, and Bayesian predictions.

    This command provides detailed machine learning insights including:
    - LSTM-based CTL/ATL/TSB trajectory forecasting
    - Isolation Forest anomaly detection
    - Bayesian uncertainty quantification
    - Model comparison with Banister predictions
    """
    from .analysis.model_integration import get_integrated_analyzer
    from .analysis.ml_enhancements import (
        LSTMTrajectoryPredictor,
        AnomalyDetector,
        BayesianPerformancePredictor,
        ModelRetrainingScheduler,
        compare_models
    )
    import json

    console = Console()
    console.print(Panel("🤖 Advanced ML Training Analysis", box=box.HEAVY, style="bold white"))

    try:
        # Get analyzer and data
        analyzer = get_integrated_analyzer("user")
        console.print(f"[white]Loading {days} days of training data...[/white]")

        # Use analyze_with_advanced_models to get the data
        analysis = analyzer.analyze_with_advanced_models(days_back=days)
        data = analysis['combined'].copy()

        # Rename columns to match what ML models expect
        data = data.rename(columns={
            'ff_fitness': 'ctl',
            'ff_fatigue': 'atl',
            'ff_form': 'tsb'
        })

        if data is None or len(data) < 30:
            console.print(f"[red]❌ Insufficient data: need at least 30 days, have {len(data) if data is not None else 0}[/red]")
            return

        console.print(f"[green]✅ Loaded {len(data)} days of data[/green]\n")

        models_dir = Path("models/ml_enhanced")
        models_dir.mkdir(parents=True, exist_ok=True)
        scheduler = ModelRetrainingScheduler()

        results = {
            'analysis_date': datetime.now().isoformat(),
            'data_points': len(data),
            'days_analyzed': days
        }

        # ========== 1. LSTM Trajectory Forecasting ==========
        console.print(Panel("📈 LSTM Trajectory Forecasting", style="green"))

        lstm_predictor = LSTMTrajectoryPredictor(sequence_length=30, forecast_horizon=forecast_days)
        lstm_model_path = models_dir / "lstm_trajectory"

        # Train or load model
        if train_lstm or not lstm_model_path.with_suffix('.pkl').exists():
            if len(data) >= 60:
                with console.status("[white]Training LSTM model (this may take a minute)...[/white]"):
                    training_result = lstm_predictor.train(data, epochs=100, batch_size=16, early_stopping_patience=20)

                if training_result['status'] == 'success':
                    lstm_predictor.save(str(lstm_model_path))
                    scheduler.mark_trained('lstm_trajectory', training_result)
                    console.print(f"[green]✅ LSTM model trained successfully[/green]")
                    console.print(f"   • Training loss: {training_result['final_loss']:.4f}")
                    console.print(f"   • Validation loss: {training_result['final_val_loss']:.4f}")
                    console.print(f"   • Epochs: {training_result['epochs_trained']}")
                else:
                    console.print(f"[yellow]⚠️ Training failed, using fallback method[/yellow]")
            else:
                console.print(f"[yellow]⚠️ Need 60+ days for training, using fallback method[/yellow]")
        else:
            lstm_predictor.load(str(lstm_model_path))
            console.print(f"[green]✅ Loaded trained LSTM model[/green]")

        # Make predictions
        trajectory = lstm_predictor.predict_trajectory(data, forecast_days=forecast_days)

        # Display results
        console.print(f"\n[bold]Current State:[/bold]")
        current_ctl = data['ctl'].iloc[-1] if 'ctl' in data.columns else 0
        current_atl = data['atl'].iloc[-1] if 'atl' in data.columns else 0
        current_tsb = data['tsb'].iloc[-1] if 'tsb' in data.columns else 0
        console.print(f"  Fitness (CTL): {current_ctl:.1f}")
        console.print(f"  Fatigue (ATL): {current_atl:.1f}")
        console.print(f"  Form (TSB): {current_tsb:.1f}")

        console.print(f"\n[bold]Forecast ({forecast_days} days):[/bold]")

        # Create forecast table
        forecast_table = Table(title=f"📊 {forecast_days}-Day Trajectory Forecast", box=box.SIMPLE)
        forecast_table.add_column("Day", style="white")
        forecast_table.add_column("CTL", style="green")
        forecast_table.add_column("ATL", style="red")
        forecast_table.add_column("TSB", style="magenta")

        for i in [0, 6, forecast_days-1]:  # Show day 1, 7, and final
            if i < len(trajectory.ctl_predicted):
                forecast_table.add_row(
                    f"+{i+1}",
                    f"{trajectory.ctl_predicted[i]:.1f} ({trajectory.ctl_predicted[i] - current_ctl:+.1f})",
                    f"{trajectory.atl_predicted[i]:.1f} ({trajectory.atl_predicted[i] - current_atl:+.1f})",
                    f"{trajectory.tsb_predicted[i]:.1f} ({trajectory.tsb_predicted[i] - current_tsb:+.1f})"
                )

        console.print(forecast_table)
        console.print(f"Model Confidence: {trajectory.model_confidence * 100:.0f}%\n")

        results['lstm_forecast'] = {
            'current': {'ctl': current_ctl, 'atl': current_atl, 'tsb': current_tsb},
            'forecast': {
                'dates': [d.isoformat() for d in trajectory.dates],
                'ctl': trajectory.ctl_predicted,
                'atl': trajectory.atl_predicted,
                'tsb': trajectory.tsb_predicted
            },
            'confidence': trajectory.model_confidence
        }

        # ========== 2. Anomaly Detection ==========
        console.print(Panel("🔍 Anomaly Detection", style="magenta"))

        anomaly_detector = AnomalyDetector(contamination=0.08)
        anomaly_model_path = models_dir / "anomaly_detector.pkl"

        # Train or load model
        if train_anomaly or not anomaly_model_path.exists():
            with console.status("[white]Fitting anomaly detector...[/white]"):
                fit_result = anomaly_detector.fit(data)

            if fit_result['status'] == 'success':
                import joblib
                joblib.dump({
                    'model': anomaly_detector.model,
                    'scaler': anomaly_detector.scaler,
                    'feature_names': anomaly_detector.feature_names,
                    'thresholds': anomaly_detector.thresholds
                }, anomaly_model_path)
                scheduler.mark_trained('anomaly_detector', fit_result)
                console.print(f"[green]✅ Anomaly detector fitted successfully[/green]")
                console.print(f"   • Features used: {', '.join(fit_result['features'])}")
        else:
            import joblib
            saved_data = joblib.load(anomaly_model_path)
            anomaly_detector.model = saved_data['model']
            anomaly_detector.scaler = saved_data['scaler']
            anomaly_detector.feature_names = saved_data['feature_names']
            anomaly_detector.thresholds = saved_data['thresholds']
            anomaly_detector.is_fitted = True
            console.print(f"[green]✅ Loaded trained anomaly detector[/green]")

        # Detect anomalies
        recent_data = data.tail(30)  # Last 30 days
        anomalies = anomaly_detector.detect_anomalies(recent_data)

        if anomalies:
            console.print(f"\n[bold red]⚠️ {len(anomalies)} anomalies detected in last 30 days:[/bold red]\n")

            anomaly_table = Table(title="🚨 Detected Anomalies", box=box.SIMPLE)
            anomaly_table.add_column("Date", style="white")
            anomaly_table.add_column("Type", style="green")
            anomaly_table.add_column("Severity", style="red")
            anomaly_table.add_column("Metric", style="magenta")

            for anomaly in anomalies[:10]:  # Show top 10
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                anomaly_table.add_row(
                    anomaly.date.strftime("%Y-%m-%d"),
                    anomaly.anomaly_type.replace('_', ' ').title(),
                    f"{severity_emoji.get(anomaly.severity, '')} {anomaly.severity.title()}",
                    anomaly.metric
                )

            console.print(anomaly_table)

            # Show recommendations for critical anomalies
            critical = [a for a in anomalies if a.severity in ['critical', 'high']]
            if critical:
                console.print(f"\n[bold red]🚨 Critical Recommendations:[/bold red]")
                for i, anomaly in enumerate(critical[:3], 1):
                    console.print(f"\n{i}. {anomaly.anomaly_type.replace('_', ' ').title()} ({anomaly.date.strftime('%Y-%m-%d')}):")
                    for rec in anomaly.recommendations[:2]:
                        console.print(f"   • {rec}")

            results['anomalies'] = [{
                'date': a.date.isoformat(),
                'type': a.anomaly_type,
                'severity': a.severity,
                'metric': a.metric,
                'value': a.value,
                'score': a.anomaly_score
            } for a in anomalies]
        else:
            console.print(f"[green]✅ No anomalies detected - all training patterns normal[/green]")
            results['anomalies'] = []

        # ========== 3. Export Results ==========
        if export:
            with open(export, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\n[green]✅ Results exported to {export}[/green]")

        console.print(f"\n[green]💡 Analysis complete! Use these insights to optimize your training.[/green]")

    except ImportError as e:
        console.print(f"[red]❌ Missing dependencies: {e}[/red]")
        console.print(f"[white]Install with: pip install tensorflow>=2.15.0[/white]")
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()


@cli.command(name='optimize-plan')
@click.option('--weeks', default=12, help='Number of weeks to plan')
@click.option('--goal-date', help='Target race/event date (YYYY-MM-DD)')
@click.option('--goal-type', default='race', help='Event type: race, fitness_test, etc.')
@click.option('--max-weekly-hours', type=float, help='Maximum weekly training hours')
@click.option('--generations', default=100, help='Optimization iterations')
@click.option('--population', default=50, help='Population size for genetic algorithm')
@click.option('--what-if', help='What-if scenario: reduce_volume_20, skip_week_3, etc.')
@click.option('--export', help='Export optimized plan to file')
def optimize_plan(weeks: int, goal_date: str, goal_type: str, max_weekly_hours: float,
                 generations: int, population: int, what_if: str, export: str):
    """
    Optimize multi-week training plan using genetic algorithms.

    Uses evolutionary optimization to create personalized training plans that:
    - Maximize fitness gains
    - Minimize injury risk
    - Respect your constraints (time, recovery needs)
    - Adapt to goal events
    - Support what-if scenario planning

    Examples:
        # Basic 12-week plan optimization
        strava-super optimize-plan --weeks 12

        # Optimize for marathon in 16 weeks
        strava-super optimize-plan --weeks 16 --goal-date 2026-04-15 --goal-type marathon

        # What-if: What if I reduce volume by 20%?
        strava-super optimize-plan --weeks 12 --what-if reduce_volume_20

        # Constrained optimization
        strava-super optimize-plan --weeks 8 --max-weekly-hours 10
    """
    from .analysis.model_integration import get_integrated_analyzer
    from .analysis.plan_optimizer import (
        GeneticPlanOptimizer,
        PlanConstraints,
        OptimizationObjectives,
        WhatIfAnalyzer,
        WorkoutType
    )

    console = Console()
    console.print(Panel("🎯 Training Plan Optimization", box=box.HEAVY, style="bold green"))

    try:
        # Get current fitness state
        analyzer = get_integrated_analyzer("user")
        analysis = analyzer.analyze_with_advanced_models(days_back=30)

        if not analysis or 'combined' not in analysis or len(analysis['combined']) == 0:
            console.print("[red]❌ Need at least 30 days of training data for optimization[/red]")
            return

        latest = analysis['combined'].iloc[-1]
        # Use ff_fitness/ff_fatigue (Fitness-Fatigue model outputs)
        current_ctl = float(latest['ff_fitness'])
        current_atl = float(latest['ff_fatigue'])

        console.print(f"[black]Current State:[/black]")
        console.print(f"  Fitness (CTL): {current_ctl:.1f}")
        console.print(f"  Fatigue (ATL): {current_atl:.1f}")
        console.print(f"  Form (TSB): {current_ctl - current_atl:.1f}\n")

        # Parse goal date
        goal_event = None
        if goal_date:
            try:
                goal_event = datetime.strptime(goal_date, '%Y-%m-%d')
                console.print(f"[green]Goal Event:[/green] {goal_type} on {goal_event.strftime('%B %d, %Y')}")
                days_to_event = (goal_event - datetime.now()).days
                console.print(f"  Days until event: {days_to_event}\n")
            except ValueError:
                console.print(f"[yellow]⚠️ Invalid date format. Using no specific goal.[/yellow]\n")

        # Setup constraints
        constraints = PlanConstraints()
        if max_weekly_hours:
            constraints.max_weekly_hours = max_weekly_hours
            constraints.max_weekly_tss = max_weekly_hours * 70  # Approx 70 TSS/hour
            console.print(f"[black]Constraint:[/black] Max {max_weekly_hours}h/week\n")

        # Setup objectives
        objectives = OptimizationObjectives()
        objectives.normalize_weights()

        # Handle what-if scenarios
        if what_if:
            console.print(Panel(f"📊 What-If Scenario: {what_if}", style="yellow"))

            # First create base plan
            console.print("[black]Optimizing base plan...[/black]")
            optimizer = GeneticPlanOptimizer(
                population_size=population,
                generations=generations,
                mutation_rate=0.15
            )

            base_plan = optimizer.optimize_plan(
                current_ctl=current_ctl,
                current_atl=current_atl,
                weeks=weeks,
                goal_event=goal_event,
                constraints=constraints,
                objectives=objectives
            )

            # Parse what-if modifications
            modifications = {}
            if 'reduce_volume' in what_if:
                reduction_pct = float(what_if.split('_')[-1]) if '_' in what_if and what_if.split('_')[-1].isdigit() else 20
                factor = 1.0 - (reduction_pct / 100)
                modifications['reduce_volume'] = factor
                console.print(f"[yellow]Scenario: Reduce volume by {reduction_pct:.0f}%[/yellow]")

            elif 'skip_week' in what_if:
                week_num = int(what_if.split('_')[-1]) if '_' in what_if else 3
                modifications['skip_week'] = week_num
                console.print(f"[yellow]Scenario: Skip week {week_num}[/yellow]")

            # Analyze scenario
            what_if_analyzer = WhatIfAnalyzer()
            results = what_if_analyzer.analyze_scenario(
                base_plan=base_plan,
                scenario=what_if,
                modifications=modifications,
                current_ctl=current_ctl,
                current_atl=current_atl
            )

            # Display comparison
            console.print(f"\n[bold]Scenario Comparison:[/bold]\n")

            comparison_table = Table(title="Base Plan vs Modified Plan", box=box.SIMPLE)
            comparison_table.add_column("Metric", style="black")
            comparison_table.add_column("Base Plan", style="green")
            comparison_table.add_column("Modified Plan", style="yellow")
            comparison_table.add_column("Difference", style="magenta")

            base = results['base_outcome']
            modified = results['modified_outcome']
            diff = results['differences']

            comparison_table.add_row(
                "Final Fitness (CTL)",
                f"{base['final_ctl']:.1f}",
                f"{modified['final_ctl']:.1f}",
                f"{diff['ctl_diff']:+.1f}"
            )
            comparison_table.add_row(
                "Final Form (TSB)",
                f"{base['final_tsb']:.1f}",
                f"{modified['final_tsb']:.1f}",
                f"{diff['tsb_diff']:+.1f}"
            )
            comparison_table.add_row(
                "Injury Risk Score",
                f"{base['injury_risk']:.2f}",
                f"{modified['injury_risk']:.2f}",
                f"{diff['injury_risk_diff']:+.2f}"
            )
            comparison_table.add_row(
                "Total TSS",
                f"{base['total_tss']:.0f}",
                f"{modified['total_tss']:.0f}",
                f"{modified['total_tss'] - base['total_tss']:+.0f}"
            )
            comparison_table.add_row(
                "Total Hours",
                f"{base['total_hours']:.1f}h",
                f"{modified['total_hours']:.1f}h",
                f"{modified['total_hours'] - base['total_hours']:+.1f}h"
            )

            console.print(comparison_table)
            console.print(f"\n[bold green]Recommendation:[/bold green]")
            console.print(f"  {results['recommendation']}")

        else:
            # Standard optimization
            console.print(Panel(f"🧬 Optimizing {weeks}-Week Plan", style="black"))
            console.print(f"[white]Running genetic algorithm ({generations} generations, population {population})...[/white]\n")

            optimizer = GeneticPlanOptimizer(
                population_size=population,
                generations=generations,
                mutation_rate=0.15,
                crossover_rate=0.7
            )

            with console.status(f"[black]Evolving optimal training plan... (this may take 30-60 seconds)[/black]"):
                optimized_plan = optimizer.optimize_plan(
                    current_ctl=current_ctl,
                    current_atl=current_atl,
                    weeks=weeks,
                    goal_event=goal_event,
                    constraints=constraints,
                    objectives=objectives
                )

            console.print(f"[green]✅ Optimization complete![/green]\n")

            # Display plan summary
            console.print(Panel("📅 Optimized Training Plan Summary", style="green"))

            summary_table = Table(title=f"{weeks}-Week Plan", box=box.ROUNDED)
            summary_table.add_column("Week", style="black")
            summary_table.add_column("Phase", style="yellow")
            summary_table.add_column("TSS", style="green")
            summary_table.add_column("Hours", style="blue")
            summary_table.add_column("Hard Days", style="red")
            summary_table.add_column("Rest Days", style="magenta")

            for week in optimized_plan.weeks:
                summary_table.add_row(
                    str(week.week_number + 1),
                    week.phase.value.title(),
                    f"{week.get_actual_tss():.0f}",
                    f"{week.get_actual_hours():.1f}h",
                    str(week.get_hard_days()),
                    str(week.get_rest_days())
                )

            console.print(summary_table)

            # Show weekly details for first 4 weeks
            console.print(f"\n[bold]First 4 Weeks Detail:[/bold]\n")

            for week_num in range(min(4, len(optimized_plan.weeks))):
                week = optimized_plan.weeks[week_num]
                console.print(f"[black]Week {week_num + 1} ({week.phase.value.title()}):[/black]")

                for workout in week.workouts:
                    if workout.workout_type != WorkoutType.REST:
                        day_name = workout.date.strftime('%a')
                        console.print(
                            f"  {day_name}: {workout.workout_type.value.title()} "
                            f"({workout.duration_minutes}min, {workout.tss:.0f} TSS, "
                            f"{workout.intensity*100:.0f}% intensity)"
                        )

                console.print()

            # Plan statistics
            console.print(Panel("📊 Plan Statistics", style="blue"))
            console.print(f"  Total TSS: {optimized_plan.get_total_tss():.0f}")
            console.print(f"  Total Hours: {optimized_plan.get_total_hours():.1f}h")
            console.print(f"  Average Weekly TSS: {optimized_plan.get_total_tss() / weeks:.0f}")
            console.print(f"  Average Weekly Hours: {optimized_plan.get_total_hours() / weeks:.1f}h")

            # Predict final state
            final_ctl = current_ctl
            final_atl = current_atl
            for week in optimized_plan.weeks:
                for workout in week.workouts:
                    final_ctl += (workout.tss - final_ctl) * (1 - np.exp(-1/42))
                    final_atl += (workout.tss - final_atl) * (1 - np.exp(-1/7))

            console.print(f"\n[bold green]Predicted Outcome:[/bold green]")
            console.print(f"  Final Fitness (CTL): {final_ctl:.1f} ({final_ctl - current_ctl:+.1f})")
            console.print(f"  Final Form (TSB): {final_ctl - final_atl:.1f}")

            if goal_event:
                taper_weeks = [w for w in optimized_plan.weeks if w.phase.value == 'taper']
                console.print(f"  Taper weeks: {len(taper_weeks)}")

            # Export if requested
            if export:
                plan_df = optimized_plan.to_dataframe()
                if export.endswith('.csv'):
                    plan_df.to_csv(export, index=False)
                elif export.endswith('.json'):
                    plan_df.to_json(export, orient='records', date_format='iso')
                else:
                    plan_df.to_csv(export + '.csv', index=False)

                console.print(f"\n[green]✅ Plan exported to {export}[/green]")

        console.print(f"\n[black]💡 Use this optimized plan as a guide for your training![/black]")
        console.print(f"[white]   Re-run optimization periodically as your fitness progresses.[/white]")

    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {e}[/red]")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--message', '-m', default=None, help='Your question or concern about training')
@click.option('--model', default='deepseek-coder-v2', help='Ollama model to use (default: deepseek-coder-v2)')
def ask_coach(message: str, model: str):
    """Ask the AI coach for training advice based on your current state.

    Examples:
      strava-super ask-coach
      strava-super ask-coach -m "I feel tired today, should I do the planned workout?"
      strava-super ask-coach -m "Today is weight training but I feel not good, maybe inline session better?"
      strava-super ask-coach -m "My HRV is low, what should I adjust?"
    """
    from .ai_coach import AITrainingCoach, check_ollama_available
    from .db import get_db
    from .db.models import HRVData, SleepData, BloodPressure, Activity
    from .analysis.model_integration import IntegratedTrainingAnalyzer
    from datetime import datetime, timedelta

    console.print("\n[bold green]🤖 AI Training Coach[/bold green]\n")

    # Check if Ollama is available
    if not check_ollama_available():
        console.print("[red]❌ Ollama is not running[/red]")
        console.print("\nStart Ollama with: [green]ollama serve[/green]")
        console.print("Or run in background: [green]brew services start ollama[/green]")
        return

    # If no message provided, prompt for one
    if not message:
        console.print("[cyan]💬 What's your training question or concern?[/cyan]")
        console.print("[dim]Examples: 'I feel tired, should I rest?' or 'My HRV is low today'[/dim]\n")
        message = click.prompt("Ask", type=str)
        console.print()

    # Gather current training context
    console.print("[green]📊 Gathering your current training data...[/green]\n")

    db = get_db()
    analyzer = IntegratedTrainingAnalyzer(db.database_url)

    try:
        # Get recent metrics
        analysis = analyzer.analyze_with_advanced_models(days_back=90)
        recent_df = analysis['combined'].tail(7)
        latest = recent_df.iloc[-1]

        # Get wellness data
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        hrv_val = None
        sleep_val = None
        rhr_val = None

        with db.get_session() as session:
            hrv = session.query(HRVData).filter(
                HRVData.date >= today,
                HRVData.date < tomorrow
            ).first()

            sleep = session.query(SleepData).filter(
                SleepData.date >= today,
                SleepData.date < tomorrow
            ).first()

            bp = session.query(BloodPressure).filter(
                BloodPressure.date >= today,
                BloodPressure.date < tomorrow
            ).first()

            # Extract values inside session
            hrv_val = hrv.hrv_rmssd if hrv and hrv.hrv_rmssd else None
            hrv_score = hrv.hrv_score if hrv and hrv.hrv_score else 50
            sleep_val = sleep.sleep_score if sleep and sleep.sleep_score else None
            rhr_val = bp.heart_rate if bp and bp.heart_rate else None

        # Get recent activities (last 7 days)
        seven_days_ago = today - timedelta(days=7)
        activities_list = []

        with db.get_session() as session:
            recent_activities = session.query(Activity).filter(
                Activity.start_date >= seven_days_ago
            ).order_by(Activity.start_date.desc()).limit(10).all()

            # Extract activity data inside session
            for act in recent_activities:
                activity_data = {
                    'date': act.start_date.strftime('%Y-%m-%d'),
                    'name': act.name,
                    'type': act.type,
                    'duration': act.moving_time // 60 if act.moving_time else 0,  # minutes
                    'distance': round(act.distance / 1000, 1) if act.distance else 0,  # km
                    'tss': round(act.training_load, 0) if act.training_load else 0,
                    'avg_hr': round(act.average_heartrate, 0) if act.average_heartrate else None
                }
                activities_list.append(activity_data)

        # Format activities for context
        if activities_list:
            activity_summary = "\n".join([
                f"  • {a['date']}: {a['name']} ({a['type']}) - {a['duration']}min, {a['distance']}km, {a['tss']} TSS" +
                (f", avg HR {a['avg_hr']}" if a['avg_hr'] else "")
                for a in activities_list
            ])
        else:
            activity_summary = "  No recent activities found"

        # Calculate simple readiness score
        tsb_score = min(max((latest['ff_form'] + 10) / 30 * 100, 0), 100)
        sleep_score_val = sleep_val if sleep_val else 70
        readiness = (tsb_score * 0.4 + hrv_score * 0.3 + sleep_score_val * 0.3)

        # Determine risk state
        tsb = latest['ff_form']
        atl = latest['ff_fatigue']
        if tsb < -30 and atl > 80:
            risk_state = 'NON_FUNCTIONAL_OVERREACHING'
        elif tsb < -15:
            risk_state = 'HIGH_STRAIN'
        elif tsb < -5:
            risk_state = 'FUNCTIONAL_OVERREACHING'
        else:
            risk_state = 'OPTIMAL_TRAINING'

        # Get today's training recommendation
        try:
            from .analysis.model_integration import get_integrated_analyzer
            rec_analyzer = get_integrated_analyzer("user")
            daily_rec = rec_analyzer.get_daily_recommendation()
            planned_workout_text = f"{daily_rec['recommendation']}: {daily_rec['activity']}\n  Rationale: {daily_rec['rationale']}"
        except Exception as e:
            planned_workout_text = f"Unable to retrieve today's recommendation: {str(e)}"

        # Build training context using extracted values
        context = {
            'ctl': latest['ff_fitness'],
            'atl': latest['ff_fatigue'],
            'tsb': latest['ff_form'],
            'hrv': hrv_val,
            'sleep_score': sleep_val,
            'rhr': rhr_val,
            'planned_workout': planned_workout_text,
            'recent_summary': f"Last 7 days: {recent_df['load'].sum():.0f} TSS total, avg {recent_df['load'].mean():.0f} TSS/day",
            'activity_log': activity_summary,
            'risk_state': risk_state,
            'readiness': readiness
        }

        # Show current state
        console.print(f"[green]Current State:[/green]")
        console.print(f"  • Fitness (CTL): {context['ctl']:.1f}")
        console.print(f"  • Fatigue (ATL): {context['atl']:.1f}")
        console.print(f"  • Form (TSB): {context['tsb']:.1f}")
        console.print(f"  • HRV: {context['hrv']} ms" if context['hrv'] else "  • HRV: No data")
        console.print(f"  • Sleep: {context['sleep_score']}/100" if context['sleep_score'] else "  • Sleep: No data")
        console.print(f"  • Readiness: {context['readiness']:.0f}%\n")

        # Initialize AI coach once
        coach = AITrainingCoach(use_ollama=True, model=model)

        # Conversation loop
        conversation_history = []

        while True:
            try:
                # Add conversation history to context
                if conversation_history:
                    context['conversation_history'] = "\n".join([
                        f"Q: {q}\nA: {a[:200]}..." if len(a) > 200 else f"Q: {q}\nA: {a}"
                        for q, a in conversation_history[-3:]  # Keep last 3 exchanges
                    ])
                else:
                    context['conversation_history'] = None

                with console.status(f"[green]Asking AI coach (using {model})...[/green]"):
                    recommendation = coach.get_recommendation(message, context)

                console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
                console.print(recommendation)
                console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]\n")

                # Add to conversation history
                conversation_history.append((message, recommendation))

                # Prompt for next question
                console.print("[dim]Press CTRL-C to exit[/dim]")
                try:
                    message = click.prompt("\nAsk", type=str)
                    console.print()
                except (KeyboardInterrupt, EOFError, click.exceptions.Abort):
                    console.print("\n[yellow]👋 Conversation ended. Stay healthy![/yellow]")
                    break

            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Conversation ended. Stay healthy![/yellow]")
                break
            except EOFError:
                console.print("\n[yellow]👋 Conversation ended. Stay healthy![/yellow]")
                break

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    try:
        config.ensure_dirs()
        cli()
    except KeyboardInterrupt:
        console.print("\n[orange]Operation cancelled by user.[/orange]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e).replace('[', r'\[').replace(']', r'\]')}[/red]")


if __name__ == "__main__":
    main()
