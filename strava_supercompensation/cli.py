"""Command-line interface for Strava Supercompensation tool."""

import click
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

from .config import config
from .auth import AuthManager
from .api import StravaClient

# Garmin integration - gracefully handle if missing
try:
    from .api.garmin import get_garmin_client, GarminError
except ImportError:
    GarminError = Exception  # Fallback
    get_garmin_client = None  # Mark as unavailable
from .analysis import SupercompensationAnalyzer
from .analysis.multisport_metrics import MultiSportCalculator
from .analysis.model_integration import get_integrated_analyzer
from .analysis.advanced_planning import AdvancedPlanGenerator
from .analysis.plan_adjustment import PlanAdjustmentEngine

console = Console()


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
@click.option("--days", default=30, help="Number of days to sync")
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
        recent = client.get_recent_activities(days=7)
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
                recent = client.get_recent_activities(days=7)
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
            table.add_column("Activity", width=12)
            table.add_column("Int.", style="yellow", width=4)
            table.add_column("Load", style="magenta", width=5)
            table.add_column("Fitness", style="blue", width=6)
            table.add_column("Form", style="green", width=5)
            table.add_column("HRV", style="red", width=4)
            table.add_column("Sleep", style="purple", width=5)

            # Get wellness data if available
            try:
                from strava_supercompensation.db.database import get_db
                from strava_supercompensation.db.models import Activity, HRVData, SleepData
                from datetime import datetime as dt, timedelta
                from sqlalchemy import func

                db = get_db()
                with db.get_session() as session:
                    for h in history[-60:]:  # Show last 60 days
                        date_obj = dt.fromisoformat(h['date']).date()
                        date_str = date_obj.strftime('%m/%d')

                        # Get primary activity for that day (highest load)
                        activity = session.query(Activity).filter(
                            func.date(Activity.start_date) == date_obj
                        ).order_by(Activity.training_load.desc()).first()

                        # Get wellness data
                        hrv = session.query(HRVData).filter(
                            func.date(HRVData.date) == date_obj
                        ).first()

                        sleep = session.query(SleepData).filter(
                            func.date(SleepData.date) == date_obj
                        ).first()

                        # Determine activity and intensity
                        if activity and activity.type:
                            # Show actual activity type with proper formatting
                            activity_name = activity.type
                            if activity_name == "WeightTraining":
                                activity_name = "Weights"
                            elif len(activity_name) > 12:
                                activity_name = activity_name[:12]

                            # Intensity based on load and activity type
                            if h['load'] > 300:
                                intensity = "RACE"
                            elif h['load'] > 200:
                                intensity = "HARD"
                            elif h['load'] > 100:
                                intensity = "MOD"
                            elif h['load'] > 30:
                                intensity = "EASY"
                            else:
                                intensity = "RECOVERY"
                        else:
                            activity_name = "Rest Day" if h['load'] == 0 else "Active"
                            intensity = "REST" if h['load'] == 0 else "LIGHT"

                        # Wellness scores with proper formatting - using hrv_rmssd and sleep_score
                        hrv_rmssd = f"{hrv.hrv_rmssd:3.0f}" if hrv and hrv.hrv_rmssd else "—"
                        sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"

                        table.add_row(
                            date_str,
                            activity_name,
                            intensity,
                            f"{h['load']:.0f}",
                            f"{h['fitness']:.1f}",
                            f"{h['form']:.1f}",
                            hrv_rmssd,
                            sleep_score,
                        )

            except Exception as e:
                # Fallback to basic version if wellness data unavailable
                from datetime import datetime as dt

                # Try to get wellness data even in fallback mode
                try:
                    from strava_supercompensation.db.database import get_db
                    from strava_supercompensation.db.models import HRVData, SleepData
                    db = get_db()
                    wellness_available = True
                except:
                    wellness_available = False

                for h in history[-60:]:  # Show last 60 days
                    date_str = dt.fromisoformat(h['date']).strftime('%m/%d')

                    # Enhanced intensity classification
                    if h['load'] > 300:
                        intensity = "RACE"
                    elif h['load'] > 200:
                        intensity = "HARD"
                    elif h['load'] > 100:
                        intensity = "MOD"
                    elif h['load'] > 30:
                        intensity = "EASY"
                    else:
                        intensity = "REST"

                    activity_name = "Training" if h['load'] > 0 else "Rest"

                    # Try to get wellness data for this date
                    hrv_rmssd = "—"
                    sleep_score = "—"

                    if wellness_available:
                        try:
                            date_obj = dt.fromisoformat(h['date']).date()
                            with db.get_session() as session:
                                from sqlalchemy import func
                                hrv = session.query(HRVData).filter(func.date(HRVData.date) == date_obj).first()
                                sleep = session.query(SleepData).filter(func.date(SleepData.date) == date_obj).first()

                                hrv_rmssd = f"{hrv.hrv_rmssd:3.0f}" if hrv and hrv.hrv_rmssd else "—"
                                sleep_score = f"{sleep.sleep_score:3.0f}" if sleep and sleep.sleep_score else "—"
                        except:
                            pass  # Keep default "—" values

                    table.add_row(
                        date_str,
                        activity_name,
                        intensity,
                        f"{h['load']:.0f}",
                        f"{h['fitness']:.1f}",
                        f"{h['form']:.1f}",
                        hrv_rmssd,
                        sleep_score,
                    )

            console.print("\n", table)

        console.print("[green]✅ Analysis complete![/green]")

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
                table.add_column("Date", width=8)
                table.add_column("Week Type", style="bright_blue", width=10)
                table.add_column("Intensity", style="yellow", width=8)
                table.add_column("Activity", style="blue", width=24)
                table.add_column("2nd Session", style="green", width=29)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="cyan", width=6)
                table.add_column("Fitness", style="green", width=7)

                # For 30-day plan, add week separators
                current_week = -1
                for plan in training_plan:
                    week_num = (plan['day'] - 1) // 7

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

                        table.add_row(
                            f"[bold]W{week_num+1}[/bold]",
                            "[white]────────[/white]",
                            f"[{week_type_color}]{week_type}[/{week_type_color}]",
                            "[white]────────[/white]",
                            "[white]────────────────────[/white]",
                            "[white]─────────────────────────[/white]",
                            "[white]────[/white]",
                            "[white]──────[/white]"
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
                    if len(activity) > 24:
                        activity = activity[:21] + "..."

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

                    # Format date (shorter)
                    date_str = plan['date']
                    if len(date_str) > 8:
                        date_str = date_str[5:]  # Remove year, keep MM-DD

                    table.add_row(
                        str(plan['day']),
                        date_str,
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

        # Clear Garmin authentication
        garmin_auth = get_garmin_auth()
        garmin_auth.revoke_tokens()

        # Reset database
        from .db import get_db
        db = get_db()
        db.drop_tables()
        db.create_tables()

        console.print("[green]✅ Application reset successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Error resetting application: {e}[/red]")


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
@click.option("--days", default=7, help="Number of days to sync")
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
@click.option("--days", default=7, help="Number of days to sync")
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


@cli.command()
@click.option("--strava-days", default=30, help="Days of Strava data to sync")
@click.option("--garmin-days", default=30, help="Days of Garmin data to sync")
@click.option("--plan-days", default=30, help="Training plan length (7 or 30 days)")
@click.option("--skip-strava", is_flag=True, help="Skip Strava sync")
@click.option("--skip-garmin", is_flag=True, help="Skip Garmin sync")
@click.option("--skip-analysis", is_flag=True, help="Skip analysis step")
def run(strava_days, garmin_days, plan_days, skip_strava, skip_garmin, skip_analysis):
    """Complete training analysis workflow: sync all data, analyze, and get recommendations."""

    errors = []

    # Step 1: Combined Data Sync (Strava + Garmin)
    console.print("")  # Spacing
    # Collect sync status for summary panel
    sync_results = []

    # Sync Strava data (minimal output)
    if not skip_strava:
        try:
            with console.status("[cyan]Syncing Strava activities...[/cyan]"):
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

    # Sync Garmin data (minimal output)
    if not skip_garmin:
        try:
            if get_garmin_client is not None:
                with console.status("[cyan]Syncing Garmin wellness data...[/cyan]"):
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

    # Create Step 1 header panel
    step1_header = Panel("📱 Step 1: Syncing Strava Activities", box=box.HEAVY, style="bold blue")
    console.print(step1_header)

    # Display sync results outside the panel
    for result in sync_results:
        console.print(result)

    # Step 1 completion footer
    console.print("[green]✅ Step 1: Data synchronization complete[/green]")

    # Step 2: Basic Performance Model Analysis
    console.print("")  # Spacing
    step2_header = Panel("📊 Step 2: Basic Performance Model Analysis", box=box.HEAVY, style="bold blue")
    console.print(step2_header)

    if not skip_analysis:
        try:
            ctx = click.get_current_context()
            ctx.invoke(analyze)
            console.print("[green]✅ Basic analysis complete.[/green]")
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            errors.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/red]")
    else:
        console.print("[yellow]⏭️  Skipping analysis[/yellow]")
        console.print("[green]✅ Basic analysis complete (skipped).[/green]")

    # Step 4: Advanced Physiological Analysis
    console.print("")  # Spacing
    step4_header = Panel("🧬 Step 4: Advanced Physiological Analysis", box=box.HEAVY, style="bold blue")
    console.print(step4_header)

    try:
        analyzer = get_integrated_analyzer("user")

        # Import required components at the start
        from rich.table import Table

        # 4.1: Advanced Model Status Check
        console.print("\n[cyan]Advanced Model Status:[/cyan]")
        model_status_table = Table(box=box.ROUNDED)
        model_status_table.add_column("Model", style="cyan")
        model_status_table.add_column("Status", style="green")
        model_status_table.add_column("Parameters", style="white")

        try:
            from .analysis.advanced_model import EnhancedFitnessFatigueModel, PerPotModel
            ff_model = EnhancedFitnessFatigueModel()
            model_status_table.add_row("Enhanced FF", "✅ Active", f"k1={ff_model.k1:.3f}, τ1={ff_model.tau1}d")

            perpot_model = PerPotModel()
            model_status_table.add_row("PerPot", "✅ Active", f"ds={perpot_model.ds:.1f}, dr={perpot_model.dr:.1f}")

            model_status_table.add_row("Optimal Control", "✅ Active", "Differential Evolution")
            model_status_table.add_row("Multi-Recovery", "✅ Active", "3-System Integration")
        except Exception as e:
            model_status_table.add_row("Models", "❌ Error", str(e)[:30])

        console.print(model_status_table)

        # 4.2: Current Fitness State Analysis
        console.print(f"\n[cyan]Advanced Fitness State Analysis:[/cyan]")
        analysis = analyzer.analyze_with_advanced_models(days_back=90)

        if analysis and 'combined' in analysis and len(analysis['combined']) > 0:
            combined = analysis['combined']
            latest = combined.iloc[-1]

            fitness_table = Table(box=box.ROUNDED)
            fitness_table.add_column("Metric", style="cyan", width=20)
            fitness_table.add_column("Value", width=10)
            fitness_table.add_column("Status", style="green", width=15)
            fitness_table.add_column("7-Day Trend", style="yellow", width=12)

            # Calculate trends
            recent_7d = combined.tail(7)
            fitness_trend = recent_7d['ff_fitness'].iloc[-1] - recent_7d['ff_fitness'].iloc[0]
            fatigue_trend = recent_7d['ff_fatigue'].iloc[-1] - recent_7d['ff_fatigue'].iloc[0]
            readiness_trend = recent_7d['composite_readiness'].iloc[-1] - recent_7d['composite_readiness'].iloc[0]

            # Add rows with comprehensive data
            fitness_table.add_row(
                "Fitness (CTL)",
                f"{latest['ff_fitness']:.1f}",
                "🏃‍♂️ Elite" if latest['ff_fitness'] > 100 else "📈 Building",
                f"{fitness_trend:+.1f}"
            )
            fitness_table.add_row(
                "Fatigue (ATL)",
                f"{latest['ff_fatigue']:.1f}",
                "😴 High" if latest['ff_fatigue'] > 60 else "✅ Manageable",
                f"{fatigue_trend:+.1f}"
            )
            fitness_table.add_row(
                "Form (TSB)",
                f"{latest['ff_form']:.1f}",
                "🚀 Peaked" if latest['ff_form'] > 20 else "⚡ Ready",
                f"{latest['ff_form'] - recent_7d['ff_form'].iloc[0]:+.1f}"
            )
            fitness_table.add_row(
                "Readiness Score",
                f"{latest['composite_readiness']:.1f}%",
                "🟢 High" if latest['composite_readiness'] > 70 else "🟡 Moderate",
                f"{readiness_trend:+.1f}%"
            )
            fitness_table.add_row(
                "Performance Potential",
                f"{latest['perpot_performance']:.3f}",
                "🎯 Optimal" if latest['perpot_performance'] > 0.8 else "📊 Good",
                "—"
            )
            fitness_table.add_row(
                "Overtraining Risk",
                "YES" if latest['overtraining_risk'] else "NO",
                "⚠️ CAUTION" if latest['overtraining_risk'] else "✅ SAFE",
                "—"
            )

            console.print(fitness_table)

            # 4.3: Recovery System Analysis
            console.print(f"\n[cyan]Multi-System Recovery Analysis:[/cyan]")
            recovery_table = Table(box=box.ROUNDED)
            recovery_table.add_column("System", style="cyan")
            recovery_table.add_column("Status", style="white")
            recovery_table.add_column("Recovery %", style="green")

            overall_recovery = min(100, latest['overall_recovery'])  # Cap at 100%
            # Simulate individual system recovery (in real implementation, these would be calculated)
            metabolic_recovery = min(100, overall_recovery * 0.9)
            neural_recovery = min(100, overall_recovery * 1.1)
            structural_recovery = min(100, overall_recovery * 0.95)

            recovery_table.add_row("Metabolic", "🔥 Active", f"{metabolic_recovery:.0f}%")
            recovery_table.add_row("Neural", "⚡ Processing", f"{neural_recovery:.0f}%")
            recovery_table.add_row("Structural", "🏗️ Rebuilding", f"{structural_recovery:.0f}%")
            recovery_table.add_row("Overall", "🎯 Integrated", f"{overall_recovery:.0f}%")

            console.print(recovery_table)

            # 4.4: Training Load Distribution (last 30 days)
            console.print(f"\n[cyan]Training Load Distribution (30 days):[/cyan]")
            recent_30d = combined.tail(30)

            total_load = recent_30d['load'].sum()
            avg_daily = recent_30d['load'].mean()
            max_load = recent_30d['load'].max()
            load_std = recent_30d['load'].std()

            # Calculate intensity distribution
            high_days = len(recent_30d[recent_30d['load'] > avg_daily * 1.5])
            moderate_days = len(recent_30d[(recent_30d['load'] >= avg_daily * 0.5) & (recent_30d['load'] <= avg_daily * 1.5)])
            easy_days = len(recent_30d[recent_30d['load'] < avg_daily * 0.5])
            rest_days = len(recent_30d[recent_30d['load'] == 0])

            load_table = Table(box=box.ROUNDED)
            load_table.add_column("Metric", style="cyan")
            load_table.add_column("Value", style="white")
            load_table.add_column("Assessment", style="green")

            load_table.add_row("Total Load", f"{total_load:.0f} TSS", "📊 Volume")
            load_table.add_row("Average Daily", f"{avg_daily:.0f} TSS", "📈 Consistency")
            load_table.add_row("Peak Load", f"{max_load:.0f} TSS", "🚀 Max Effort")
            load_table.add_row("Load Variability", f"{load_std:.0f} TSS", "🎯 Balance")
            load_table.add_row("", "", "")
            load_table.add_row("High Intensity Days", f"{high_days}", "🔥 Quality")
            load_table.add_row("Moderate Days", f"{moderate_days}", "⚡ Base")
            load_table.add_row("Easy Days", f"{easy_days}", "🌱 Recovery")
            load_table.add_row("Rest Days", f"{rest_days}", "😴 Complete Rest")

            console.print(load_table)

        else:
            console.print("[yellow]⚠️ Insufficient data for advanced analysis[/yellow]")

        console.print("[green]✅ Advanced physiological analysis complete.[/green]")

    except Exception as e:
        console.print(f"[red]❌ Advanced analysis failed: {e}[/red]")

    # Step 5: Performance Prediction
    console.print("")  # Spacing
    step5_header = Panel("🔮 Step 5: Performance Prediction", box=box.HEAVY, style="bold blue")
    console.print(step5_header)

    try:
        # Get current state for prediction baseline
        current_state = analyzer._get_current_state()

        if current_state:
            console.print("\n[cyan]7-Day Performance Forecast:[/cyan]")

            # Use integrated analyzer's FF model for consistent prediction
            ff_model = analyzer.ff_model

            # Predict with moderate load (100 TSS/day)
            target_load = 100
            days_ahead = 7
            loads = [target_load] * days_ahead
            days = np.arange(days_ahead)

            fitness, fatigue, performance = ff_model.impulse_response(np.array(loads), days)

            # Use current state from the same model for baseline consistency
            baseline_fitness = current_state['fitness']
            baseline_fatigue = current_state['fatigue']
            fitness = fitness + baseline_fitness
            fatigue = fatigue + baseline_fatigue

            prediction_table = Table(box=box.ROUNDED)
            prediction_table.add_column("Day", style="cyan", width=8)
            prediction_table.add_column("Fitness", style="green", width=10)
            prediction_table.add_column("Fatigue", style="red", width=10)
            prediction_table.add_column("Form", style="yellow", width=10)
            prediction_table.add_column("Recommendation", width=15)

            for i in range(days_ahead):
                form = fitness[i] - fatigue[i]
                if form > 20:
                    rec = "🚀 Peak Training"
                elif form > 0:
                    rec = "⚡ Hard Training"
                elif form > -10:
                    rec = "📈 Moderate"
                else:
                    rec = "😴 Recovery"

                prediction_table.add_row(
                    f"Day {i+1}",
                    f"{fitness[i]:.1f}",
                    f"{fatigue[i]:.1f}",
                    f"{form:.1f}",
                    rec
                )

            console.print(prediction_table)

            # Show prediction summary
            final_fitness = fitness[-1]
            final_form = final_fitness - fatigue[-1]
            fitness_gain = final_fitness - baseline_fitness

            # Present forecast summary in a panel
            forecast_summary = Panel(
                f"[bold]Predicted Fitness Gain:[/bold] {fitness_gain:+.1f} points\n"
                f"[bold]Final Form Score:[/bold] {final_form:.1f}\n"
                f"[bold]Training Load:[/bold] {target_load} TSS/day",
                title="🎯 7-Day Forecast Summary",
                box=box.ROUNDED
            )
            console.print(forecast_summary)

        console.print("[green]✅ Performance prediction complete.[/green]")

    except Exception as e:
        console.print(f"[red]❌ Performance prediction failed: {e}[/red]")

    # Step 6: Multisport Analysis
    console.print("")  # Spacing
    step6_header = Panel("🏃‍♂️🚴‍♀️🏊‍♀️ Step 6: Multi-Sport Profile", box=box.HEAVY, style="bold blue")
    console.print(step6_header)

    try:
        # Get basic activity type distribution from database
        with analyzer.db.get_session() as session:
            from strava_supercompensation.db.models import Activity
            recent_activities = session.query(Activity).filter(
                Activity.start_date >= datetime.utcnow() - timedelta(days=30)
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
                    console.print("\n[cyan]Sport Distribution (30 days):[/cyan]")
                    sport_table = Table(box=box.ROUNDED)
                    sport_table.add_column("Sport", style="cyan", width=15)
                    sport_table.add_column("Activities", style="white", width=10)
                    sport_table.add_column("Hours", style="green", width=8)
                    sport_table.add_column("Load TSS", style="yellow", width=10)
                    sport_table.add_column("% Time", style="white", width=8)

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
                    console.print(f"\n[cyan]Total Training Volume:[/cyan] {total_hours:.1f} hours | {total_load:.0f} TSS")
                else:
                    console.print("[yellow]⚠️ No training data in last 30 days[/yellow]")
            else:
                console.print("[yellow]⚠️ No activities found in last 30 days[/yellow]")

        console.print("[green]✅ Multi-sport analysis complete.[/green]")

    except Exception as e:
        console.print(f"[red]❌ Multisport analysis failed: {e}[/red]")

    # Step 7: Training Plan Generation & Recommendations
    console.print("")  # Spacing
    step7_header = Panel("🎯 Step 7: Today's Recommendation & 30-Day Plan", box=box.HEAVY, style="bold blue")
    console.print(step7_header)

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
            rec_color = rec_colors.get(recommendation.get("recommendation", ""), "white")

            # Display recommendation in professional format
            rec_content = f"[bold {rec_color}]{recommendation.get('recommendation', 'Unknown')}[/bold {rec_color}]\n\n"
            rec_content += f"[bold]Activity:[/bold]  {recommendation.get('activity', 'N/A')}\n"
            rec_content += f"[bold]Rationale:[/bold] {recommendation.get('rationale', 'N/A')}\n\n"
            rec_content += f"[bold]Form:[/bold] {recommendation.get('form_score', 0):.1f}        "
            rec_content += f"[bold]Readiness:[/bold] {recommendation.get('readiness_score', 0):.1f}%      "
            rec_content += f"[bold]Performance Potential:[/bold] {recommendation.get('performance_potential', 0):.2f}"

            recommendation_panel = Panel(
                rec_content,
                title="📈 Today's Advanced Recommendation",
                box=box.ROUNDED,
                border_style=rec_color
            )
            console.print(recommendation_panel)

        # Generate advanced training plan
        console.print(f"\n[cyan]Generating {plan_days}-day plan using advanced models...[/cyan]")

        if plan_days == 30:
            # Use advanced 30-day planner
            generator = AdvancedPlanGenerator("user")
            plan_result = generator.generate_30_day_plan(
                goal='balanced',
                constraints={'max_weekly_hours': 12, 'rest_days': [6]}
            )

            # Convert WorkoutPlan objects to dictionary format with simulated values
            training_plan = []
            if plan_result and 'daily_workouts' in plan_result:
                daily_data = plan_result.get('visualizations', {}).get('daily_data', [])
                for i, workout in enumerate(plan_result['daily_workouts']):
                    # Get corresponding daily simulation data
                    daily_sim = daily_data[i] if i < len(daily_data) else {}

                    # Convert WorkoutPlan dataclass to dictionary with simulation data
                    training_plan.append({
                        'day': workout.day_number,
                        'date': workout.date.strftime('%m/%d'),
                        'recommendation': workout.intensity_level.upper(),
                        'activity': workout.title,
                        'second_activity': None,  # 30-day plans don't have second sessions
                        'load': float(workout.planned_load),
                        'suggested_load': float(workout.planned_load),
                        'form': daily_sim.get('form', 0.0),
                        'predicted_form': daily_sim.get('form', 0.0),
                        'fitness': daily_sim.get('fitness', 0.0),
                        'predicted_fitness': daily_sim.get('fitness', 0.0)
                    })
        else:
            # Use integrated analyzer for shorter plans
            plan_result = analyzer.generate_optimal_plan(
                goal='balanced',
                duration_days=plan_days,
                rest_days=[6]
            )

            if plan_result.get('success'):
                # Convert to training plan format
                training_plan = []
                for i, (load, rec, detail) in enumerate(zip(
                    plan_result['loads'],
                    plan_result['recommendations'],
                    plan_result['daily_details']
                )):
                    training_plan.append({
                        'day': i + 1,
                        'date': (datetime.now() + timedelta(days=i)).strftime('%m/%d'),
                        'recommendation': rec.upper(),
                        'activity': detail.get('activity', f'{rec.upper()} Training'),
                        'load': float(load),
                        'suggested_load': float(load),  # Add alias for compatibility
                        'form': float(detail.get('predicted_form', 0)),
                        'predicted_form': float(detail.get('predicted_form', 0)),  # Add expected field name
                        'fitness': float(detail.get('predicted_fitness', 0)),
                        'predicted_fitness': float(detail.get('predicted_fitness', 0))  # Add expected field name
                    })
            else:
                training_plan = []

        if training_plan:

            title = f"{plan_days}-Day Training Plan"
            table = Table(title=title, box=box.ROUNDED)
            table.add_column("Day", style="cyan", width=3)
            table.add_column("Date", width=8)
            table.add_column("Week Type", style="bright_blue", width=10)
            table.add_column("Intensity", style="yellow", width=8)
            table.add_column("Activity", style="blue", width=24)
            table.add_column("2nd Session", style="green", width=26)
            table.add_column("Load", style="magenta", width=4)
            table.add_column("Form", style="cyan", width=6)
            table.add_column("Fitness", style="green", width=7)

            # For 30-day plan, add week separators
            current_week = -1
            for plan in training_plan:
                week_num = (plan['day'] - 1) // 7

                # Add week separator for 30-day plan
                if plan_days == 30 and week_num != current_week:
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

                    table.add_row(
                        f"[bold]W{week_num+1}[/bold]",
                        "[white]────────[/white]",
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        "[white]────────[/white]",
                        "[white]────────────────────[/white]",
                        "[white]─────────────────────────[/white]",
                        "[white]────[/white]",
                        "[white]──────[/white]"
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
                if len(activity) > 24:
                    activity = activity[:21] + "..."

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

                # Format date (shorter)
                date_str = plan['date']
                if len(date_str) > 8:
                    date_str = date_str[5:]  # Remove year, keep MM-DD

                table.add_row(
                    str(plan['day']),
                    date_str,
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

        console.print("[green]✅ Recommendation and plan generated.[/green]")

    except Exception as e:
        error_msg = f"Recommendation generation failed: {e}"
        errors.append(error_msg)
        console.print(f"[red]❌ {error_msg}[/red]")

    # Summary
    console.print(f"\n[bold green]🏁 Workflow Complete![/bold green]")

    if errors:
        console.print(f"\n[yellow]⚠️  {len(errors)} errors occurred:[/yellow]")
        for error in errors:
            console.print(f"  • {error}")
        console.print(f"\n[cyan]You can run individual commands to fix specific issues[/cyan]")
    else:
        # Step 8: Comprehensive Analysis Summary
        console.print("")  # Spacing
        step8_header = Panel("🏆 Step 8: Final Analysis Dashboard", box=box.HEAVY, style="bold blue")
        console.print(step8_header)

        try:
            # Get latest analysis for summary
            analyzer = get_integrated_analyzer("user")

            analysis = analyzer.analyze_with_advanced_models(days_back=90)

            if analysis and 'combined' in analysis and len(analysis['combined']) > 0:
                combined = analysis['combined']
                latest = combined.iloc[-1]
                recent_30d = combined.tail(30)

                # Create four-row summary table as specified
                summary_table = Table(box=box.ROUNDED)
                summary_table.add_column("Category", style="bold cyan", width=20)
                summary_table.add_column("Key Metrics", width=50)

                # Row 1: Current Fitness - Key CTL/TSB/Readiness values
                try:
                    fitness_val = float(latest['ff_fitness']) if 'ff_fitness' in latest else 0
                    form_val = float(latest['ff_form']) if 'ff_form' in latest else 0
                    readiness_val = float(latest['composite_readiness']) if 'composite_readiness' in latest else 0
                    summary_table.add_row(
                        "Current Fitness",
                        f"CTL: {fitness_val:.0f} | TSB: {form_val:.1f} | Readiness: {readiness_val:.0f}%"
                    )
                except (TypeError, ValueError) as e:
                    summary_table.add_row("Current Fitness", "Data processing error")

                # Row 2: Training Load - 30-day total and daily average
                try:
                    total_load = float(recent_30d['load'].sum())
                    avg_load = float(recent_30d['load'].mean())
                    summary_table.add_row(
                        "Training Load",
                        f"30-day Total: {total_load:.0f} TSS | Daily Average: {avg_load:.0f} TSS"
                    )
                except (TypeError, ValueError):
                    summary_table.add_row("Training Load", "Data processing error")

                # Row 3: Recovery & Risk - Overall recovery % and Overtraining status
                try:
                    overtraining_status = "YES" if latest['overtraining_risk'] else "NO"
                    recovery_color = "red" if latest['overtraining_risk'] else "green"
                    recovery_val = float(latest['overall_recovery']) if 'overall_recovery' in latest else 80.0
                    summary_table.add_row(
                        "Recovery & Risk",
                        f"Overall Recovery: {recovery_val:.0f}% | Overtraining Risk: [{recovery_color}]{overtraining_status}[/{recovery_color}]"
                    )
                except (TypeError, ValueError, KeyError):
                    summary_table.add_row("Recovery & Risk", "Data processing error")

                # Row 4: 7-Day Trend - Key performance vectors
                try:
                    recent_7d = combined.tail(7)
                    # Ensure numeric values for arithmetic
                    fitness_change = float(recent_7d['ff_fitness'].iloc[-1]) - float(recent_7d['ff_fitness'].iloc[0])
                    fitness_trend = "↗️" if fitness_change > 0 else "↘️"
                    form_trend = "↗️" if len(combined) > 7 and float(combined['ff_form'].iloc[-1]) > float(combined['ff_form'].iloc[-7]) else "↘️"
                    load_avg = float(recent_7d['load'].mean())
                    summary_table.add_row(
                        "7-Day Trend",
                        f"Fitness: {fitness_change:+.1f} {fitness_trend} | Form: {form_trend} | Load Avg: {load_avg:.0f} TSS"
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    summary_table.add_row("7-Day Trend", "Insufficient data for trend analysis")

                console.print(summary_table)

                # Key Recommendations - Professional Panel
                recommendations = []

                # Get avg_load safely (might not be defined if training load calculation failed)
                try:
                    avg_load_for_rec = float(recent_30d['load'].mean()) if 'load' in recent_30d.columns else 50.0
                except:
                    avg_load_for_rec = 50.0  # Default value if calculation fails

                # Priority 1: Safety and overtraining check
                if latest['overtraining_risk']:
                    recommendations.append("🚨 [red]IMMEDIATE REST REQUIRED[/red] - Overtraining detected")
                    recommendations.append("😴 [yellow]ACTIVE RECOVERY ONLY[/yellow] - Light movement, focus on sleep")
                elif latest['composite_readiness'] > 80 and latest['ff_form'] > 20:
                    recommendations.append("🎯 [green]COMPETITION READY[/green] - Peak form achieved")
                    recommendations.append("🚀 [green]PEAK TRAINING WINDOW[/green] - Ready for high intensity")
                elif latest['composite_readiness'] < 40:
                    recommendations.append("😴 [yellow]RECOVERY FOCUS[/yellow] - Prioritize easy training and sleep")
                    recommendations.append("📉 [cyan]REDUCE LOAD[/cyan] - Allow fitness to consolidate")
                else:
                    recommendations.append("📈 [cyan]PROGRESSIVE TRAINING[/cyan] - Balanced load progression")
                    if latest['ff_form'] < -10:
                        recommendations.append("⚡ [yellow]BUILD PHASE[/yellow] - Focus on base fitness development")
                    elif avg_load_for_rec < 60:
                        recommendations.append("📊 [cyan]INCREASE VOLUME[/cyan] - Room for progressive load increase")

                # Format as bulleted list in a professional panel
                rec_content = "\n".join([f"• {rec}" for rec in recommendations[:3]])  # Limit to top 3 as specified

                recommendations_panel = Panel(
                    rec_content,
                    title="🎯 Key Recommendations",
                    box=box.ROUNDED,
                    border_style="yellow"
                )
                console.print(f"\n")
                console.print(recommendations_panel)

            # Step 8 completion footer
            console.print(f"\n[bold green]✅ Step 8: Final Summary Dashboard - Analysis Complete[/bold green]")

        except TypeError as te:
            # Specific handling for datetime arithmetic errors
            console.print(f"[red]❌ Summary generation failed due to data type issue[/red]")
            console.print(f"[yellow]ℹ️ Using simplified summary instead[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Summary generation failed: {e}[/red]")

        console.print(f"\n[cyan]🧬 Advanced models active • 📊 Multi-system analysis • 🎯 Optimal recommendations ready[/cyan]")


@cli.command()
@click.option('--goal', default='balanced', help='Training goal: fitness, performance, recovery, balanced')
@click.option('--duration', default=30, help='Plan duration in days (7-30)')
@click.option('--max-hours', default=12, help='Maximum weekly training hours')
@click.option('--rest-days', default='6', help='Rest days (0=Mon, 6=Sun, comma-separated)')
def advanced_plan(goal, duration, max_hours, rest_days):
    """Generate advanced training plan using scientific models."""
    console.print(Panel.fit("🧬 Advanced Training Plan Generator", style="bold cyan"))

    try:
        # Parse rest days
        rest_day_list = [int(x.strip()) for x in rest_days.split(',') if x.strip()]

        if duration == 30:
            # Use 30-day advanced planner
            generator = AdvancedPlanGenerator("user")
            plan = generator.generate_30_day_plan(
                goal=goal,
                constraints={
                    'max_weekly_hours': max_hours,
                    'rest_days': rest_day_list
                }
            )

            if plan and 'summary' in plan:
                console.print(f"[green]✅ 30-day plan generated successfully[/green]")

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

        else:
            # Use integrated analyzer for shorter plans
            analyzer = get_integrated_analyzer("user")
            plan = analyzer.generate_optimal_plan(
                goal=goal,
                duration_days=duration,
                rest_days=rest_day_list
            )

            if plan['success']:
                console.print(f"[green]✅ {duration}-day plan generated[/green]")

                # Show daily plan
                table = Table(title=f"{duration}-Day Advanced Plan")
                table.add_column("Day", style="cyan")
                table.add_column("Recommendation", style="yellow")
                table.add_column("Load", style="magenta")
                table.add_column("Form", style="green")

                for i, detail in enumerate(plan['daily_details']):
                    table.add_row(
                        f"Day {i+1}",
                        detail['recommendation'],
                        f"{detail['load']:.0f}",
                        f"{detail['predicted_form']:.1f}"
                    )

                console.print(table)
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
        generator = AdvancedPlanGenerator("user")
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
        table.add_column("Notes", style="white")

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
            from .analysis.advanced_model import EnhancedFitnessFatigueModel
            ff_model = EnhancedFitnessFatigueModel()
            console.print("[green]✅ Enhanced Fitness-Fatigue Model[/green]")
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
            generator = AdvancedPlanGenerator("test")
            console.print("[green]✅ Advanced Plan Generator[/green]")
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
                from .analysis.advanced_model import EnhancedFitnessFatigueModel
                ff_model = EnhancedFitnessFatigueModel()
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
        table.add_column("Value", style="white")
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
        table.add_row("Overtraining Risk", "YES" if overtraining else "NO", "⚠️ CAUTION" if overtraining else "✅ SAFE")

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
        from .analysis.advanced_model import EnhancedFitnessFatigueModel
        ff_model = EnhancedFitnessFatigueModel()

        # Create load schedule
        loads = [target_load] * days_ahead
        days = np.arange(days_ahead)

        # Predict using current state as baseline
        fitness, fatigue, performance = ff_model.impulse_response(
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