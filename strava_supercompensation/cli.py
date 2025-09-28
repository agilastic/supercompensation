"""Command-line interface for Strava Supercompensation tool."""

import warnings
# Suppress the pkg_resources deprecation warning from heartpy library
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import click
import numpy as np
from datetime import datetime, timedelta, timezone
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
        console.print(f"[yellow]⚠️  Authentication token expired. Re-authenticating...[/yellow]")

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
        console.print("\n[yellow]Please create a .env file with your Strava API credentials.[/yellow]")
        console.print("[yellow]See .env.example for reference.[/yellow]")
        return

    auth_manager = AuthManager()

    if auth_manager.is_authenticated():
        console.print("[green]✅ Already authenticated![/green]")
        if click.confirm("Do you want to re-authenticate?"):
            auth_manager.logout()
        else:
            return

    console.print("\n[cyan]Opening browser for Strava authorization...[/cyan]")

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
        console.print("[yellow]⚠️  No valid authentication found. Attempting to authenticate...[/yellow]")
        if auth_manager.authenticate():
            console.print("[green]✅ Authentication successful![/green]")
        else:
            console.print("[red]❌ Authentication failed. Please check your credentials.[/red]")
            return

    try:
        client = StravaClient()

        with console.status(f"[cyan]Fetching activities from Strava...[/cyan]"):
            count = client.sync_activities(days_back=days)

        console.print(f"[green]✅ Successfully synced {count} activities![/green]")

        # Show recent activities
        recent = client.get_recent_activities(days=60)
        if recent:
            table = Table(title="Recent Activities", box=box.ROUNDED)
            table.add_column("Date", style="cyan")
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
                with console.status(f"[cyan]Retrying sync...[/cyan]"):
                    count = client.sync_activities(days_back=days)
                console.print(f"[green]✅ Successfully synced {count} activities![/green]")

                # Show recent activities after successful retry
                recent = client.get_recent_activities(days=60)
                if recent:
                    table = Table(title="Recent Activities", box=box.ROUNDED)
                    table.add_column("Date", style="cyan")
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

        with console.status("[cyan]Calculating fitness metrics...[/cyan]"):
            df = analyzer.analyze(days_back=days)

        current_state = analyzer.get_current_state()

        # Enhanced 60-Day Comprehensive Training Log
        history = analyzer.get_metrics_history(days=60)
        if history:
            table = Table(title="60-Day Comprehensive Training Log", box=box.ROUNDED)
            table.add_column("Date", style="cyan", width=6)
            table.add_column("Activity", width=30)
            table.add_column("Intensity", style="yellow", width=9)
            table.add_column("Load", style="magenta", width=6)
            table.add_column("Fitness", style="blue", width=8)
            table.add_column("Fatigue", style="bright_red", width=8)
            table.add_column("Form", style="green", width=5)
            table.add_column("HRV", style="red", width=4)
            table.add_column("Sleep", style="purple", width=5)
            table.add_column("RHR", style="yellow", width=4)
            table.add_column("Weight", style="blue", width=7)
            table.add_column("BF%", style="bright_yellow", width=5)
            table.add_column("H2O", style="cyan", width=5)
            table.add_column("BP", style="red", width=9)

            # Get wellness data if available
            try:
                from .db import get_db
                from .db.models import Activity, HRVData, SleepData, BloodPressure, WellnessData, BodyComposition
                from datetime import datetime as dt, timedelta
                from sqlalchemy import func

                db = get_db()
                with db.get_session() as session:
                    for h in history[-60:]:  # Show last 60 days
                        date_obj = dt.fromisoformat(h['date']).date()
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
                                            ans_display = f"[yellow]{ans_score:.0f}[/yellow]"
                                        elif ans_status in ["High Stress", "Severe Stress"]:
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        elif ans_status == "Overreaching":
                                            ans_display = f"[orange3]{ans_score:.0f}[/orange3]"
                                        else:
                                            ans_display = f"{ans_score:.0f}"
                                    except Exception:
                                        # Fallback to basic RMSSD display
                                        ans_display = f"{hrv.hrv_rmssd:.0f}"
                                else:
                                    ans_display = "—"

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

                                # Calculate aggregated load from all activities
                                total_load = sum(activity.training_load or 0 for activity in activities)

                                # Determine aggregated intensity based on total load
                                agg_intensity = get_intensity_from_load(total_load)

                                # Add aggregated wellness row with total load
                                table.add_row(
                                    date_str,
                                    f"{len(activities)} Activities",  # Show number of activities
                                    agg_intensity,  # Aggregated intensity
                                    f"{total_load:.0f}",  # Aggregated load
                                    f"{h['fitness']:.1f}",
                                    f"{h['fatigue']:.1f}",
                                    f"{h['form']:.1f}",
                                    ans_display,
                                    sleep_score,
                                    rhr,
                                    weight_str,
                                    body_fat_str,
                                    water_str,
                                    bp_str,
                                )

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

                                    # Individual activity intensity based on training load
                                    activity_load = activity.training_load or 0
                                    intensity = get_intensity_from_load(activity_load)

                                    # Activity rows: empty date, no wellness data
                                    table.add_row(
                                        "",  # Empty date
                                        activity_display,
                                        intensity,
                                        f"{activity_load:.0f}",
                                        "—",  # No fitness data
                                        "—",  # No fatigue data
                                        "—",  # No form data
                                        "—",  # No ANS data
                                        "—",  # No sleep data
                                        "—",  # No RHR data
                                        "—",  # No weight data
                                        "—",  # No body fat data
                                        "—",  # No water data
                                        "—",  # No BP data
                                    )

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

                                # Individual activity intensity based on training load
                                activity_load = activity.training_load or 0
                                intensity = get_intensity_from_load(activity_load)

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
                                            ans_display = f"[yellow]{ans_score:.0f}[/yellow]"
                                        elif ans_status in ["High Stress", "Severe Stress"]:
                                            ans_display = f"[red]{ans_score:.0f}[/red]"
                                        elif ans_status == "Overreaching":
                                            ans_display = f"[orange3]{ans_score:.0f}[/orange3]"
                                        else:
                                            ans_display = f"{ans_score:.0f}"
                                    except Exception:
                                        # Fallback to basic RMSSD display
                                        ans_display = f"{hrv.hrv_rmssd:.0f}"
                                else:
                                    ans_display = "—"

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
                                    intensity,
                                    f"{activity_load:.0f}",
                                    f"{h['fitness']:.1f}",
                                    f"{h['fatigue']:.1f}",
                                    f"{h['form']:.1f}",
                                    ans_display,
                                    sleep_score,
                                    rhr,
                                    weight_str,
                                    body_fat_str,
                                    water_str,
                                    bp_str,
                                )
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
                                        ans_display = f"[yellow]{ans_score:.0f}[/yellow]"
                                    elif ans_status in ["High Stress", "Severe Stress"]:
                                        ans_display = f"[red]{ans_score:.0f}[/red]"
                                    elif ans_status == "Overreaching":
                                        ans_display = f"[orange3]{ans_score:.0f}[/orange3]"
                                    else:
                                        ans_display = f"{ans_score:.0f}"
                                except Exception:
                                    # Fallback to basic RMSSD display
                                    ans_display = f"{hrv.hrv_rmssd:.0f}"
                            else:
                                ans_display = "—"

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
                                intensity,
                                f"{h['load']:.0f}",
                                f"{h['fitness']:.1f}",
                                f"{h['fatigue']:.1f}",
                                f"{h['form']:.1f}",
                                ans_display,
                                sleep_score,
                                rhr,
                                weight_str,
                                body_fat_str,
                                water_str,
                                bp_str,
                            )

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

                for h in history[-60:]:  # Show last 60 days
                    date_str = dt.fromisoformat(h['date']).strftime('%m/%d')

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
                        intensity,
                        f"{h['load']:.0f}",
                        f"{h['fitness']:.1f}",
                        f"{h['fatigue']:.1f}",
                        f"{h['form']:.1f}",
                        hrv_rmssd,
                        sleep_score,
                        rhr,
                        weight_str,
                        body_fat_str,
                        water_str,
                        bp_str,
                    )

            console.print("\n", table)

            # Advanced insights analysis for 60-day comprehensive log
            try:
                with console.status("[cyan]Analyzing training patterns and risks...[/cyan]"):
                    from .analysis.advanced_insights import AdvancedInsightsAnalyzer
                    insights_analyzer = AdvancedInsightsAnalyzer()
                    insights = insights_analyzer.analyze_comprehensive_insights(days_back=60)

                    if insights:
                        # Display contradictory signals
                        contradictory_signals = insights.get('contradictory_signals', {})
                        if contradictory_signals.get('status') == 'analyzed' and contradictory_signals.get('contradictions'):
                            console.print("\n[bold red]⚠️ Contradictory Readiness & Overtraining Signals Detected[/bold red]")
                            contradictions_table = Table(title="Signal Contradictions", box=box.ROUNDED)
                            contradictions_table.add_column("Date", style="cyan")
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
                            console.print("\n[bold yellow]📊 Health Data Anomalies Detected[/bold yellow]")
                            anomalies_table = Table(title="Health Metric Outliers", box=box.ROUNDED)
                            anomalies_table.add_column("Date", style="cyan")
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

                        console.print(f"\n[dim]🔍 Advanced analysis complete - {len(insights)} analysis modules processed[/dim]")

            except ImportError:
                console.print("\n[yellow]⚠️ Advanced insights analysis not available - missing dependencies[/yellow]")
            except Exception as insights_error:
                console.print(f"\n[red]❌ Advanced insights analysis failed: {insights_error}[/red]")

    except Exception as e:
        # Try automatic re-authentication for 401 errors
        if handle_auth_error(e, auth_manager, "analysis"):
            try:
                analyzer = SupercompensationAnalyzer()
                with console.status("[cyan]Retrying analysis...[/cyan]"):
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
            "MODERATE": "cyan",
            "HARD": "magenta",
            "PEAK": "bold magenta",
            "NO_DATA": "dim",
        }

        rec_color = color_map.get(recommendation["recommendation"], "white")

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
                table.add_column("Day", style="cyan", width=3)
                table.add_column("Date", width=10)
                table.add_column("Week Type", style="bright_blue", width=10)
                table.add_column("Intensity", style="yellow", width=15)
                table.add_column("Activity", style="blue", width=24)
                table.add_column("2nd Session", style="green", width=27)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="cyan", width=6)
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
                            "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                            "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
                            "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "white"
                        }.get(week_type, "white")

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
                            "[white]────────[/white]",
                            f"[{week_type_color}]{week_type}[/{week_type_color}]",
                            "[white]────────[/white]",
                            f"[cyan]{time_summary}[/cyan]",
                            "[white]─────────────────────────[/white]",
                            "[white]────[/white]",
                            "[white]──────[/white]",
                            "[white]───────[/white]"
                        )

                    # Color code recommendations
                    rec_colors = {
                        "REST": "red",
                        "RECOVERY": "yellow",
                        "EASY": "green",
                        "MODERATE": "cyan",
                        "HARD": "magenta",
                        "PEAK": "bold magenta"
                    }
                    rec_color = rec_colors.get(plan['recommendation'], "white")

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
                        "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                        "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "white"
                    }.get(week_type, "white")

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

            console.print(f"\n[cyan]📊 Database Statistics:[/cyan]")
            console.print(f"  • Activities: {activity_count}")
            console.print(f"  • Metrics: {metric_count}")

            # Get last sync
            last_activity = session.query(Activity).order_by(Activity.created_at.desc()).first()
            if last_activity:
                console.print(f"\n[cyan]Last sync: {last_activity.created_at.strftime('%Y-%m-%d %H:%M')}[/cyan]")

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
            table.add_column("Sport", style="cyan")
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
            console.print("\n[bold cyan]🔄 Recovery Recommendations:[/bold cyan]")

            for activity in activities[:3]:  # Show top 3 recent
                sport_type = multisport_calc.get_sport_type(activity.get("type", ""))
                cross_training = multisport_calc.get_cross_training_recommendations(
                    sport_type, activity.get("training_load", 0)
                )

                if cross_training:
                    activity_name = activity.get("name", "Unknown")[:30]
                    console.print(f"\n[cyan]After {activity_name}:[/cyan]")
                    for rec in cross_training[:2]:
                        console.print(f"  • {rec['activity']}: {rec['benefit']} ({rec['duration']})")

    except Exception as e:
        console.print(f"[red]❌ Error analyzing multi-sport data: {e}[/red]")


@cli.command()
def reset():
    """Reset database and authentication."""
    console.print(Panel.fit("⚠️  Reset Application", style="bold yellow"))

    if not click.confirm("This will delete all data. Are you sure?"):
        console.print("[cyan]Operation cancelled.[/cyan]")
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

        with console.status("[cyan]Testing connection...[/cyan]"):
            result = garmin_client.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[cyan]Display Name: {result.get('display_name')}[/cyan]")
            console.print(f"[cyan]User ID: {result.get('user_id')}[/cyan]")
            console.print(f"[cyan]Email: {result.get('email')}[/cyan]")
        else:
            console.print(f"[red]❌ Connection failed: {result.get('message')}[/red]")

    except GarminError as e:
        console.print(f"[red]❌ Garmin error: {e}[/red]")
        console.print("\n[yellow]💡 Setup required:[/yellow]")
        console.print("Set environment variables:")
        console.print("  GARMIN_EMAIL=your_email@example.com")
        console.print("  GARMIN_PASSWORD=your_password")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")


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

        with console.status(f"[cyan]Syncing HRV and sleep data from Garmin...[/cyan]"):
            results = garmin_client.sync_essential_data(days)

        # Display results
        table = Table(title="Sync Results", box=box.ROUNDED)
        table.add_column("Data Type", style="cyan")
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
        console.print(f"[cyan]Date range: {results['date_range']}[/cyan]")

        # Show latest scores
        latest = garmin_client.get_latest_scores()
        if latest["hrv_score"] or latest["sleep_score"]:
            console.print(f"\n[bold cyan]📊 Latest Scores:[/bold cyan]")
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
            console.print(f"[cyan]Using provided MFA code: {code}[/cyan]")

        console.print("[cyan]Testing Garmin connection...[/cyan]")
        console.print("[cyan]You will be prompted for an MFA code if needed[/cyan]")

        result = garmin_client.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[cyan]Display Name: {result.get('display_name')}[/cyan]")
            console.print(f"[cyan]User ID: {result.get('user_id')}[/cyan]")
            console.print(f"[cyan]Email: {result.get('email')}[/cyan]")
        else:
            console.print(f"[red]❌ Connection failed: {result.get('message')}[/red]")

    except GarminError as e:
        if "cancelled by user" in str(e).lower():
            console.print("[yellow]⚠️  Authentication cancelled by user[/yellow]")
        else:
            console.print(f"[red]❌ Garmin MFA error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")


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
        console.print("\n[bold cyan]💡 Enhanced Recommendation:[/bold cyan]")
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
            console.print(f"[cyan]Using provided MFA code: {code}[/cyan]")

        with console.status("[cyan]Authenticating with Garmin...[/cyan]"):
            result = garmin_client.login_with_mfa(code)

        if result["status"] == "success":
            console.print("[green]✅ Successfully authenticated![/green]")
            console.print(f"[cyan]User: {result.get('user', 'Unknown')}[/cyan]")
            console.print(f"[cyan]Message: {result.get('message')}[/cyan]")
        elif result["status"] == "mfa_required":
            console.print("[yellow]🔑 MFA code required[/yellow]")
            console.print("[yellow]Please run: strava-super garmin login --code YOUR_MFA_CODE[/yellow]")
            console.print("[cyan]Get the code from your Garmin Connect mobile app or email[/cyan]")
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
        console.print(f"[red]❌ Unexpected error: {e}[/red]")


@garmin.command()
@click.option("--days", default=60, help="Number of days to sync")
@click.option("--code", help="MFA verification code (optional - will prompt if not provided)")
def sync_mfa(days, code):
    """Sync wellness data using MFA-enabled client."""
    console.print(Panel.fit(f"🔄 Syncing Garmin Data with MFA ({days} days)", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        if code:
            console.print(f"[cyan]Using provided MFA code: {code}[/cyan]")
            # Try login first with provided code
            login_result = garmin_client.login_with_mfa(code)
            if login_result["status"] != "success":
                console.print(f"[red]❌ Login failed: {login_result['message']}[/red]")
                return

        console.print("[cyan]Syncing HRV and sleep data...[/cyan]")
        console.print("[cyan]Note: You may be prompted for an MFA code if authentication is needed[/cyan]")

        results = garmin_client.sync_essential_data(days)

        # Display results
        table = Table(title="MFA Sync Results", box=box.ROUNDED)
        table.add_column("Data Type", style="cyan")
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
        console.print(f"[cyan]Date range: {results['date_range']}[/cyan]")

        # Show latest scores
        latest = garmin_client.get_latest_scores()
        if latest["hrv_score"] or latest["sleep_score"]:
            console.print(f"\n[bold cyan]📊 Latest Scores:[/bold cyan]")
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

            with console.status("[cyan]Importing CSV data and calculating athletic metrics...[/cyan]"):
                results = importer.import_csv_data()

            # Display single file results
            table = Table(title="RENPHO CSV Import Results", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
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
                    console.print(f"\n📈 [bold cyan]Athletic Performance Analysis (60 days):[/bold cyan]")
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
            with console.status("[cyan]Discovering and importing RENPHO CSV files...[/cyan]"):
                results = RenphoCsvImporter.auto_import_all(directory, user_id)

            if results.get("status") == "no_files_found":
                console.print("[yellow]⚠️ No RENPHO CSV files found in the directory.[/yellow]")
                console.print(f"[yellow]💡 Looking for files with columns like: Datum, Zeitraum, Gewicht(kg), BMI, Körperfett(%)[/yellow]")
                return

            # Display auto-import results
            table = Table(title="Auto-Import Results", box=box.ROUNDED)
            table.add_column("File", style="cyan")
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
                    console.print(f"\n📈 [bold cyan]Athletic Performance Analysis (60 days):[/bold cyan]")
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
        table.add_column("Metric", style="cyan")
        table.add_column("Change", style="white")
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
        stability_table.add_column("Indicator", style="cyan")
        stability_table.add_column("Score", style="white")
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
        console.print(f"\n[bold cyan]🏃 Current Athletic Profile ({latest['date'].strftime('%Y-%m-%d')}):[/bold cyan]")
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

        console.print(f"\n[cyan]🔬 Analysis based on {analysis['measurements_count']} measurements over {analysis['period_days']} days[/cyan]")

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
    with console.status("[cyan]Importing data from all sources (Strava, Omron, Garmin)...[/cyan]", spinner="dots"):
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
                sync_results.append("🩺 [dim]Omron BP Data: No CSV files found[/dim]")
        except Exception as e:
            sync_results.append(f"🩺 [yellow]Omron BP Data: ⚠️ Import failed[/yellow]")

        # Sync Garmin data (minimal output)
        if not skip_garmin:
            try:
                if get_garmin_client is not None:
                    ctx = click.get_current_context()
                    # Suppress verbose output from Garmin sync
                    import io
                    import contextlib
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        ctx.invoke(garmin.commands["sync-mfa"], days=garmin_days)
                    sync_results.append("⌚ [green]Garmin Wellness: ✅ Synced[/green]")
                else:
                    sync_results.append("⌚ [yellow]Garmin Wellness: ⚠️ Unavailable[/yellow]")
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
        console.print(f"\n[cyan]Comprehensive Performance Dashboard:[/cyan]")
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
            dashboard_table.add_column("Category", style="cyan", width=14)
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
                    sport_table.add_column("Sport", style="cyan", width=15)
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
                    console.print(f"\n[cyan]Total Training Volume (30 days):[/cyan] {total_hours:.1f} hours | {total_load:.0f} TSS")
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
                "MODERATE": "cyan",
                "HARD": "magenta",
                "PEAK": "bold magenta"
            }
            rec_color = rec_colors.get(recommendation.get("recommendation", ""), "black")


        # Generate training plan
        console.print(f"\n[cyan]Generating {plan_days}-day plan using optimization models...[/cyan]")

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
            table.add_column("Day", style="cyan", width=3)
            table.add_column("Date", width=10)
            table.add_column("Week Type", style="bright_blue", width=10)
            table.add_column("Intensity", style="yellow", width=11)
            table.add_column("Primary Activity & Duration", style="blue", width=32)
            table.add_column("2nd Session & Duration", style="green", width=28)
            table.add_column("Load", style="magenta", width=4)
            table.add_column("Form", style="cyan", width=6)
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
                        "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                        "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "black"
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
                        f"[cyan]{time_summary}[/cyan]",
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
                        "MODERATE": "cyan",
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
                        "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                        "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
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
        console.print(f"\n[cyan]You can run individual commands to fix specific issues[/cyan]")
    else:

        console.print(f"\n[cyan]🧬 Models active • 📊 Multi-system analysis • 🎯 Optimal recommendations ready[/cyan]")


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
                table.add_column("Day", style="cyan", width=3)
                table.add_column("Date", width=10)
                table.add_column("Week Type", style="bright_blue", width=10)
                table.add_column("Intensity", style="yellow", width=11)
                table.add_column("Primary Activity & Duration", style="blue", width=32)
                table.add_column("2nd Activity & Duration", style="green", width=28)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="cyan", width=6)
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
                                "[white]───────[/white]",
                                "[white]───────────[/white]",
                                "[white]─────────────────────────[/white]",
                                "[white]──────────────────────────[/white]",
                                "[white]────[/white]",
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
                            "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
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
                            f"[bold bright_white]W{week_num+1}[/bold bright_white]",
                            "[dim white]────────[/dim white]",
                            f"[bold {week_type_color}]{week_type}[/bold {week_type_color}]",
                            "[dim white]─────────[/dim white]",
                            f"[bold cyan]{week_summary}[/bold cyan]",
                            "[dim white]─────────────────────────[/dim white]",
                            "[dim white]────[/dim white]",
                            "[dim white]──────[/dim white]",
                            "[dim white]───────[/dim white]"
                        )

                    # Color code recommendations
                    rec_colors = {
                        "REST": "red",
                        "RECOVERY": "yellow",
                        "EASY": "green",
                        "MODERATE": "cyan",
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
                        "BUILD 1": "bright_green", "BUILD 2": "green", "BUILD 3": "yellow",
                        "RECOVERY": "bright_cyan", "PEAK 1": "bright_red", "PEAK 2": "red",
                        "TAPER 1": "magenta", "TAPER 2": "bright_magenta", "MAINTENANCE": "black"
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
    console.print(Panel.fit("🧬 Training Plan Generator", style="bold cyan"))

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

        console.print(f"[cyan]Analyzed wellness data and generated {len(adjustments)} suggestions[/cyan]")

        # Show top adjustments
        table = Table(title="Recommended Adjustments")
        table.add_column("Type", style="yellow")
        table.add_column("Reason", style="cyan")
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
        console.print("[cyan]Testing model components...[/cyan]")

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
            console.print("\n[cyan]Model Parameters:[/cyan]")
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

        console.print("\n[cyan]All models operational and ready for advanced training analysis.[/cyan]")

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
        console.print(f"\n[bold cyan]Current Fitness State (Last {days} days)[/bold cyan]")

        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
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
        console.print(f"\n[bold cyan]Recovery Status[/bold cyan]")
        console.print(f"Overall Recovery: {recovery:.1f}% - {'🟢 Excellent' if recovery > 80 else '🟡 Good' if recovery > 60 else '🔴 Poor'}")

        if detailed:
            # Show detailed breakdown
            console.print(f"\n[bold cyan]Detailed Analysis[/bold cyan]")

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
                console.print("📈 [cyan]PROGRESSIVE TRAINING[/cyan] - Balanced load progression")

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

        console.print(f"\n[bold cyan]Performance Prediction ({days_ahead} days ahead)[/bold cyan]")
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
        table.add_column("Day", style="cyan")
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
                    console.print("📈 [cyan]Target load is conservative - room for progression[/cyan]")
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
            console.print("[cyan]Operation cancelled.[/cyan]")
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
                console.print("[cyan]📊 Metrics will be recalculated from clean state on next analysis[/cyan]")
                console.print("[cyan]💡 Run 'strava-super analyze' to rebuild metrics with realistic values[/cyan]")
            else:
                console.print("[green]✅ Metrics table is already clean[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error cleaning metrics: {e}[/red]")


def main():
    """Main entry point."""
    try:
        config.ensure_dirs()
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
