"""Command-line interface for Strava Supercompensation tool."""

import click
from datetime import datetime
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
from .analysis import SupercompensationAnalyzer, RecommendationEngine
from .analysis.multisport_metrics import MultiSportCalculator

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
            table.add_column("Name", style="white")
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
                    table.add_column("Name", style="white")
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
    console.print(Panel.fit(f"📊 Analyzing Training Data ({days} days)", style="bold blue"))

    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        console.print("[red]❌ Not authenticated. Please run 'strava-super auth' first.[/red]")
        return

    try:
        analyzer = SupercompensationAnalyzer()

        with console.status("[cyan]Calculating fitness metrics...[/cyan]"):
            df = analyzer.analyze(days_back=days)

        current_state = analyzer.get_current_state()

        # Display current metrics
        metrics_panel = Panel(
            f"""
[bold]Current Training Status[/bold]

📅 Date: {current_state['date'].strftime('%Y-%m-%d')}

[cyan]🏃 Fitness (CTL):[/cyan] {current_state['fitness']:.1f}
[yellow]😓 Fatigue (ATL):[/yellow] {current_state['fatigue']:.1f}
[green]📈 Form (TSB):[/green] {current_state['form']:.1f}

[dim]Daily Load:[/dim] {current_state['daily_load']:.0f}
            """,
            title="📊 Metrics",
            box=box.ROUNDED,
        )
        console.print(metrics_panel)

        # Show trend
        history = analyzer.get_metrics_history(days=14)
        if history:
            table = Table(title="14-Day Trend", box=box.SIMPLE)
            table.add_column("Date", style="cyan")
            table.add_column("Fitness", style="blue")
            table.add_column("Fatigue", style="yellow")
            table.add_column("Form", style="green")
            table.add_column("Load", style="magenta")

            for h in history[-14:]:  # Show last 14 days
                date = datetime.fromisoformat(h['date']).strftime('%m/%d')
                table.add_row(
                    date,
                    f"{h['fitness']:.1f}",
                    f"{h['fatigue']:.1f}",
                    f"{h['form']:.1f}",
                    f"{h['load']:.0f}",
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
        engine = RecommendationEngine()
        recommendation = engine.get_recommendation()

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

        rec_color = color_map.get(recommendation["type"], "white")

        # Create recommendation panel
        rec_text = f"""
[bold {rec_color}]{recommendation['type']}[/bold {rec_color}]

[bold]Intensity:[/bold] {recommendation['intensity']}
[bold]Duration:[/bold] {recommendation['duration_minutes']} minutes
"""

        # Add sport-specific activity recommendation
        if recommendation.get('activity'):
            rec_text += f"[bold]Recommended Activity:[/bold] {recommendation['activity']}\n"

        if recommendation.get('activity_rationale'):
            rec_text += f"[bold]Why:[/bold] {recommendation['activity_rationale']}\n"

        if recommendation.get('alternative_activities'):
            rec_text += f"[bold]Alternatives:[/bold] {', '.join(recommendation['alternative_activities'])}\n"

        # Add double session information if available
        if recommendation.get('second_activity'):
            rec_text += f"\n[bold cyan]Double Session Opportunity:[/bold cyan]\n"
            rec_text += f"[bold]Second Activity:[/bold] {recommendation['second_activity']}\n"
            rec_text += f"[bold]Timing:[/bold] {recommendation['session_timing']}\n"
            rec_text += f"[bold]Why Double:[/bold] {recommendation['double_rationale']}\n"

        if recommendation.get('suggested_load'):
            rec_text += f"[bold]Suggested Load:[/bold] {recommendation['suggested_load']:.0f}\n"

        if recommendation['notes']:
            rec_text += "\n[bold]Notes:[/bold]\n"
            for note in recommendation['notes']:
                rec_text += f"  • {note}\n"

        if recommendation['metrics']:
            m = recommendation['metrics']
            rec_text += f"""
[dim]Current Metrics:[/dim]
  Fitness: {m['fitness']} | Fatigue: {m['fatigue']} | Form: {m['form']}
"""

        console.print(Panel(rec_text, title="🎯 Today's Recommendation", box=box.DOUBLE))

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
                table.add_column("Date", style="white", width=8)
                table.add_column("Week Type", style="bright_blue", width=10)
                table.add_column("Intensity", style="yellow", width=8)
                table.add_column("Activity", style="blue", width=24)
                table.add_column("2nd Session", style="green", width=29)
                table.add_column("Load", style="magenta", width=4)
                table.add_column("Form", style="cyan", width=6)

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
                            "[dim]────────[/dim]",
                            f"[{week_type_color}]{week_type}[/{week_type_color}]",
                            "[dim]────────[/dim]",
                            "[dim]────────────────────[/dim]",
                            "[dim]─────────────────────────[/dim]",
                            "[dim]────[/dim]",
                            "[dim]──────[/dim]"
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
                        f"[green]{second_session}[/green]" if plan.get('second_activity') else f"[dim]{second_session}[/dim]",
                        f"{plan['suggested_load']:.0f}",
                        f"{plan['predicted_form']:.1f}",
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
                engine = RecommendationEngine()
                recommendation = engine.get_recommendation()
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
                console.print(f"\n[dim]Last sync: {last_activity.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]")

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
                    console.print(f"\n[dim]After {activity_name}:[/dim]")
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
            console.print(f"[dim]Display Name: {result.get('display_name')}[/dim]")
            console.print(f"[dim]User ID: {result.get('user_id')}[/dim]")
            console.print(f"[dim]Email: {result.get('email')}[/dim]")
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
        console.print(f"[dim]Date range: {results['date_range']}[/dim]")

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
        console.print("[dim]You will be prompted for an MFA code if needed[/dim]")

        result = garmin_client.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[dim]Display Name: {result.get('display_name')}[/dim]")
            console.print(f"[dim]User ID: {result.get('user_id')}[/dim]")
            console.print(f"[dim]Email: {result.get('email')}[/dim]")
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
        engine = RecommendationEngine()
        recommendation = engine.get_recommendation()

        if recommendation.get("wellness") and recommendation["wellness"].get("readiness_score"):
            readiness = recommendation["wellness"]["readiness_score"]
            console.print(f"  • Readiness Score: {readiness:.0f}/100")

            if "wellness_modifier" in recommendation.get("metrics", {}):
                modifier = recommendation["metrics"]["wellness_modifier"]
                console.print(f"  • Training Adjustment: {modifier:.1f}x")

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
            console.print(f"[dim]User: {result.get('user', 'Unknown')}[/dim]")
            console.print(f"[dim]Message: {result.get('message')}[/dim]")
        elif result["status"] == "mfa_required":
            console.print("[yellow]🔑 MFA code required[/yellow]")
            console.print("[yellow]Please run: strava-super garmin login --code YOUR_MFA_CODE[/yellow]")
            console.print("[dim]Get the code from your Garmin Connect mobile app or email[/dim]")
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
        console.print("[dim]Note: You may be prompted for an MFA code if authentication is needed[/dim]")

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
        console.print(f"[dim]Date range: {results['date_range']}[/dim]")

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
    console.print(Panel.fit("🚀 Complete Training Analysis Workflow", style="bold green"))

    errors = []

    # Step 1: Sync Strava data
    if not skip_strava:
        console.print("\n[bold cyan]📱 Step 1: Syncing Strava Activities[/bold cyan]")
        try:
            # Call the main CLI sync command (not personal Garmin sync)
            ctx = click.get_current_context()
            # Find the main sync command in the CLI group
            main_sync = None
            for command_name, command in cli.commands.items():
                if command_name == "sync":
                    main_sync = command
                    break

            if main_sync:
                ctx.invoke(main_sync, days=strava_days)
            else:
                raise Exception("Main sync command not found")

            console.print("[green]✅ Strava sync completed[/green]")
        except Exception as e:
            error_msg = f"Strava sync failed: {e}"
            errors.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/red]")
    else:
        console.print("\n[dim]⏭️  Skipping Strava sync[/dim]")

    # Step 2: Sync Garmin data (using personal access only)
    if not skip_garmin:
        console.print(f"\n[bold cyan]⌚ Step 2: Syncing Garmin Wellness Data ({garmin_days} days)[/bold cyan]")
        try:
            # Use personal Garmin access instead of OAuth1
            if get_garmin_client is not None:
                ctx = click.get_current_context()
                ctx.invoke(garmin.commands["sync-mfa"], days=garmin_days)
                console.print("[green]✅ Garmin wellness sync completed[/green]")
            else:
                console.print("[yellow]⚠️  Garmin client not available. Install garth for wellness data.[/yellow]")
        except Exception as e:
            error_msg = f"Garmin wellness sync failed: {e}"
            errors.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/red]")
    else:
        console.print("\n[dim]⏭️  Skipping Garmin sync[/dim]")

    # Step 3: Run analysis
    if not skip_analysis:
        console.print("\n[bold cyan]📊 Step 3: Analyzing Training Data[/bold cyan]")
        try:
            ctx = click.get_current_context()
            ctx.invoke(analyze)
            console.print("[green]✅ Analysis completed[/green]")
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            errors.append(error_msg)
            console.print(f"[red]❌ {error_msg}[/red]")
    else:
        console.print("\n[dim]⏭️  Skipping analysis[/dim]")

    # Step 4: Get recommendations with training plan
    console.print(f"\n[bold cyan]🎯 Step 4: Generating {plan_days}-Day Training Plan[/bold cyan]")
    try:
        # Get the recommendation with auto-generated training plan
        from .analysis.recommendations import RecommendationEngine
        engine = RecommendationEngine()

        # Get today's recommendation
        recommendation = engine.get_recommendation()

        if recommendation:
            # Display today's recommendation
            console.print(Panel.fit("🎯 Today's Recommendation", style="bold blue"))

            rec_colors = {
                "REST": "red",
                "RECOVERY": "yellow",
                "EASY": "green",
                "MODERATE": "cyan",
                "HARD": "magenta",
                "PEAK": "bold magenta"
            }
            rec_color = rec_colors.get(recommendation.get("recommendation", ""), "white")

            console.print(f"\n[{rec_color}]{recommendation.get('recommendation', 'Unknown')}[/{rec_color}]")
            console.print(f"Activity: {recommendation.get('activity', 'N/A')}")
            console.print(f"Why: {recommendation.get('rationale', 'N/A')}")

            if recommendation.get('metrics'):
                metrics = recommendation['metrics']
                console.print(f"\nCurrent Metrics:")
                console.print(f"  Fitness: {metrics.get('fitness', 0):.1f} | Fatigue: {metrics.get('fatigue', 0):.1f} | Form: {metrics.get('form', 0):.1f}")

        # Generate and display training plan
        training_plan = engine.get_training_plan(days=plan_days)

        if training_plan:
            from rich.table import Table
            from rich import box

            title = f"{plan_days}-Day Training Plan"
            table = Table(title=title, box=box.ROUNDED)
            table.add_column("Day", style="cyan", width=3)
            table.add_column("Date", style="white", width=8)
            table.add_column("Week Type", style="bright_blue", width=10)
            table.add_column("Intensity", style="yellow", width=8)
            table.add_column("Activity", style="blue", width=24)
            table.add_column("2nd Session", style="green", width=26)
            table.add_column("Load", style="magenta", width=4)
            table.add_column("Form", style="cyan", width=6)

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
                        "[dim]────────[/dim]",
                        f"[{week_type_color}]{week_type}[/{week_type_color}]",
                        "[dim]────────[/dim]",
                        "[dim]────────────────────[/dim]",
                        "[dim]─────────────────────────[/dim]",
                        "[dim]────[/dim]",
                        "[dim]──────[/dim]"
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
                    f"[green]{second_session}[/green]" if plan.get('second_activity') else f"[dim]{second_session}[/dim]",
                    f"{plan['suggested_load']:.0f}",
                    f"{plan['predicted_form']:.1f}",
                )

            console.print(table)

            # Show summary
            total_load = sum(p['suggested_load'] for p in training_plan)
            rest_days = len([p for p in training_plan if p['recommendation'] == 'REST'])
            hard_days = len([p for p in training_plan if p['recommendation'] in ['HARD', 'PEAK']])

            console.print(Panel(f"Monthly Summary:\n"
                                f"• Total Load: {total_load:.0f} TSS\n"
                                f"• Average Daily Load: {total_load/len(training_plan):.0f} TSS\n"
                                f"• Rest Days: {rest_days}\n"
                                f"• Hard/Peak Days: {hard_days}\n"
                                f"• Periodization: 3:1 (3 weeks build, 1 week recovery)",
                                title="📊 Training Summary", style="bold green"))

        console.print("[green]✅ Training plan generated[/green]")

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
        console.print(f"\n[dim]You can run individual commands to fix specific issues[/dim]")
    else:
        console.print(f"[green]✅ All steps completed successfully![/green]")
        console.print(f"[dim]Data synced • Analysis complete • Training plan ready[/dim]")


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