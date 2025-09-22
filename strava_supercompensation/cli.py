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
from .auth.garmin_oauth import get_garmin_auth, GarminOAuthError
from .api import StravaClient
from .api.garmin_client import get_garmin_client, GarminAPIError
from .analysis import SupercompensationAnalyzer, RecommendationEngine
from .analysis.multisport_metrics import MultiSportCalculator

console = Console()


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
        console.print("[red]❌ Not authenticated. Please run 'strava-super auth' first.[/red]")
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

            for h in history[-7:]:  # Show last 7 days
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

        # Show weekly plan
        if click.confirm("\nWould you like to see the weekly plan?"):
            weekly_plan = engine.get_weekly_plan()
            if weekly_plan:
                table = Table(title="7-Day Training Plan", box=box.ROUNDED)
                table.add_column("Day", style="cyan")
                table.add_column("Date", style="white")
                table.add_column("Recommendation", style="yellow")
                table.add_column("Load", style="green")
                table.add_column("Form", style="magenta")

                for plan in weekly_plan:
                    table.add_row(
                        str(plan['day']),
                        plan['date'],
                        plan['recommendation'],
                        f"{plan['suggested_load']:.0f}",
                        f"{plan['predicted_form']:.1f}",
                    )

                console.print("\n", table)

    except Exception as e:
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
    """Garmin Connect integration commands."""
    pass


@garmin.command()
def auth():
    """Authenticate with Garmin Connect."""
    console.print(Panel.fit("🔐 Garmin Connect Authentication", style="bold blue"))

    try:
        garmin_auth = get_garmin_auth()

        if garmin_auth.is_authenticated():
            console.print("[green]✅ Already authenticated with Garmin![/green]")
            if click.confirm("Do you want to re-authenticate?"):
                garmin_auth.revoke_tokens()
            else:
                return

        console.print("\n[cyan]Starting Garmin OAuth1 flow...[/cyan]")

        # Step 1: Get authorization URL
        auth_url, token_secret = garmin_auth.get_authorization_url()

        console.print(f"\n[bold]1. Open this URL in your browser:[/bold]")
        console.print(f"[link]{auth_url}[/link]")
        console.print("\n[bold]2. Authorize the application[/bold]")
        console.print("[bold]3. Copy the verification code from the page[/bold]")

        # Step 2: Get verification code from user
        oauth_token = auth_url.split("oauth_token=")[1].split("&")[0] if "oauth_token=" in auth_url else None
        if not oauth_token:
            console.print("[red]❌ Failed to extract oauth_token from URL[/red]")
            return

        verifier = click.prompt("\n[bold]Enter verification code", type=str).strip()

        if not verifier:
            console.print("[red]❌ Verification code is required[/red]")
            return

        # Step 3: Exchange for access token
        with console.status("[cyan]Exchanging code for access token...[/cyan]"):
            token_info = garmin_auth.exchange_code_for_token(oauth_token, token_secret, verifier)

        console.print("[green]✅ Successfully authenticated with Garmin Connect![/green]")
        console.print(f"[dim]User ID: {token_info.get('garmin_user_id')}[/dim]")
        console.print(f"[dim]Display Name: {token_info.get('display_name')}[/dim]")

    except GarminOAuthError as e:
        console.print(f"[red]❌ Garmin authentication error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")


@garmin.command()
def test():
    """Test Garmin Connect API connection."""
    console.print(Panel.fit("🧪 Testing Garmin Connection", style="bold blue"))

    try:
        garmin_auth = get_garmin_auth()

        if not garmin_auth.is_authenticated():
            console.print("[red]❌ Not authenticated with Garmin. Run 'garmin auth' first.[/red]")
            return

        with console.status("[cyan]Testing connection...[/cyan]"):
            result = garmin_auth.test_connection()

        if result["status"] == "success":
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[dim]User ID: {result.get('user_id')}[/dim]")
            console.print(f"[dim]Display Name: {result.get('display_name')}[/dim]")
            console.print(f"[dim]Email: {result.get('email')}[/dim]")
        else:
            console.print(f"[red]❌ Connection failed: {result.get('message')}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Error testing connection: {e}[/red]")


@garmin.command()
@click.option("--days", default=30, help="Number of days to sync")
@click.option("--type", "data_type", type=click.Choice(['hrv', 'sleep', 'wellness', 'all']), default='all', help="Type of data to sync")
def sync(days, data_type):
    """Sync wellness data from Garmin Connect."""
    console.print(Panel.fit(f"🔄 Syncing Garmin Data ({data_type}, {days} days)", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        # Check authentication
        if not garmin_client.auth.is_authenticated():
            console.print("[red]❌ Not authenticated with Garmin. Run 'garmin auth' first.[/red]")
            return

        # Sync requested data type
        with console.status(f"[cyan]Syncing {data_type} data from Garmin...[/cyan]"):
            if data_type == "hrv":
                results = [garmin_client.sync_hrv_data(days)]
            elif data_type == "sleep":
                results = [garmin_client.sync_sleep_data(days)]
            elif data_type == "wellness":
                results = [garmin_client.sync_wellness_data(days)]
            else:  # all
                results = garmin_client.sync_all(days)

        # Display results
        table = Table(title="Sync Results", box=box.ROUNDED)
        table.add_column("Data Type", style="cyan")
        table.add_column("New Records", style="green")
        table.add_column("Updated", style="yellow")
        table.add_column("Date Range", style="white")
        table.add_column("Status", style="magenta")

        for result in results:
            if "error" in result:
                table.add_row(
                    result["type"].title(),
                    "N/A",
                    "N/A",
                    "N/A",
                    f"[red]Error: {result['error'][:30]}...[/red]"
                )
            else:
                table.add_row(
                    result["type"].title(),
                    str(result.get("synced", 0)),
                    str(result.get("updated", 0)),
                    result.get("date_range", "N/A"),
                    "[green]Success[/green]"
                )

        console.print(table)

        # Show quick stats
        total_new = sum(r.get("synced", 0) for r in results if "error" not in r)
        total_updated = sum(r.get("updated", 0) for r in results if "error" not in r)

        if total_new > 0 or total_updated > 0:
            console.print(f"\n[green]✅ Sync complete! {total_new} new records, {total_updated} updated[/green]")
        else:
            console.print(f"\n[yellow]ℹ️  No new data found[/yellow]")

    except GarminAPIError as e:
        console.print(f"[red]❌ Garmin API error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error syncing data: {e}[/red]")


@garmin.command()
@click.option("--days", default=7, help="Number of days to analyze")
def wellness():
    """Show wellness trends and insights."""
    console.print(Panel.fit(f"🧘‍♀️ Wellness Analysis ({days} days)", style="bold blue"))

    try:
        garmin_client = get_garmin_client()

        if not garmin_client.auth.is_authenticated():
            console.print("[red]❌ Not authenticated with Garmin. Run 'garmin auth' first.[/red]")
            return

        # Get wellness trend
        trend = garmin_client.get_wellness_trend(days)

        if trend["status"] == "no_data":
            console.print("[yellow]No wellness data found. Run 'garmin sync' first.[/yellow]")
            return

        # Display trend
        avg_stress_str = f"{trend['avg_stress']:.0f}/100" if trend['avg_stress'] else 'N/A'
        avg_sleep_str = f"{trend['avg_sleep_score']:.0f}/100" if trend['avg_sleep_score'] else 'N/A'

        wellness_panel = Panel(
            f"""
[bold]Wellness Overview ({days} days)[/bold]

[bold]🧠 Average Stress:[/bold] {avg_stress_str}
[bold]😴 Average Sleep Score:[/bold] {avg_sleep_str}
[bold]📈 Stress Trend:[/bold] {trend['stress_trend'].title()}

[dim]Records analyzed: {trend['records_count']}[/dim]
            """,
            title="🧘‍♀️ Wellness Insights",
            box=box.ROUNDED,
        )
        console.print(wellness_panel)

        # Get latest scores
        latest_hrv = garmin_client.get_latest_hrv_score()
        latest_sleep = garmin_client.get_latest_sleep_score()

        if latest_hrv or latest_sleep:
            hrv_str = f"{latest_hrv:.0f}/100" if latest_hrv else 'No data'
            sleep_str = f"{latest_sleep:.0f}/100" if latest_sleep else 'No data'

            scores_panel = Panel(
                f"""
[bold]Latest Scores[/bold]

[bold]❤️  HRV Score:[/bold] {hrv_str}
[bold]😴 Sleep Score:[/bold] {sleep_str}
                """,
                title="📊 Current State",
                box=box.ROUNDED,
            )
            console.print(scores_panel)

        # Wellness recommendations
        recommendations = []
        if trend['avg_stress'] and trend['avg_stress'] > 50:
            recommendations.append("Consider stress management techniques")
        if trend['avg_sleep_score'] and trend['avg_sleep_score'] < 70:
            recommendations.append("Focus on improving sleep quality")
        if latest_hrv and latest_hrv < 40:
            recommendations.append("HRV indicates need for recovery")

        if recommendations:
            console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")
            for rec in recommendations:
                console.print(f"  • {rec}")

    except Exception as e:
        console.print(f"[red]❌ Error analyzing wellness data: {e}[/red]")


@garmin.command()
def status():
    """Show Garmin integration status."""
    console.print(Panel.fit("ℹ️  Garmin Status", style="bold blue"))

    try:
        garmin_auth = get_garmin_auth()

        # Check authentication
        if garmin_auth.is_authenticated():
            console.print("[green]✅ Authenticated with Garmin Connect[/green]")

            # Test connection
            with console.status("[cyan]Testing connection...[/cyan]"):
                result = garmin_auth.test_connection()

            if result["status"] == "success":
                console.print(f"[green]✅ API connection working[/green]")
                console.print(f"[dim]User: {result.get('display_name')}[/dim]")
            else:
                console.print(f"[yellow]⚠️  API connection issue: {result.get('message')}[/yellow]")
        else:
            console.print("[yellow]⚠️  Not authenticated with Garmin[/yellow]")

        # Check data availability
        try:
            from .db import get_db
            db = get_db()
            with db.get_session() as session:
                from .db.models import HRVData, SleepData, WellnessData

                hrv_count = session.query(HRVData).count()
                sleep_count = session.query(SleepData).count()
                wellness_count = session.query(WellnessData).count()

                console.print(f"\n[cyan]📊 Wellness Data:[/cyan]")
                console.print(f"  • HRV records: {hrv_count}")
                console.print(f"  • Sleep records: {sleep_count}")
                console.print(f"  • Wellness records: {wellness_count}")

                # Get latest records
                latest_hrv = session.query(HRVData).order_by(HRVData.date.desc()).first()
                latest_sleep = session.query(SleepData).order_by(SleepData.date.desc()).first()

                if latest_hrv:
                    console.print(f"\n[dim]Latest HRV: {latest_hrv.date.strftime('%Y-%m-%d')}[/dim]")
                if latest_sleep:
                    console.print(f"[dim]Latest Sleep: {latest_sleep.date.strftime('%Y-%m-%d')}[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Database error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Error checking status: {e}[/red]")


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