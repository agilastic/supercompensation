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
from .analysis import SupercompensationAnalyzer, RecommendationEngine

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

        # Reset database
        from .db import get_db
        db = get_db()
        db.drop_tables()
        db.create_tables()

        console.print("[green]✅ Application reset successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Error resetting application: {e}[/red]")


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