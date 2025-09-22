"""Strava API client implementation."""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from requests.exceptions import RequestException

from ..config import config
from ..auth import AuthManager
from ..db import get_db
from ..db.models import Activity
from ..analysis.sports_metrics import SportsMetricsCalculator
from ..analysis.multisport_metrics import MultiSportCalculator
import numpy as np


class StravaClient:
    """Client for interacting with Strava API."""

    def __init__(self, user_id: str = "default"):
        self.base_url = config.STRAVA_API_BASE_URL
        self.auth_manager = AuthManager(user_id)
        self.db = get_db()

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        token = self.auth_manager.get_valid_token()
        if not token:
            raise ValueError("No valid authentication token available")
        return {"Authorization": f"Bearer {token}"}

    def get_athlete(self) -> Dict[str, Any]:
        """Get authenticated athlete information."""
        response = requests.get(
            f"{self.base_url}/athlete",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def get_activities(
        self,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        page: int = 1,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Get athlete activities."""
        params = {
            "page": page,
            "per_page": per_page
        }

        if after:
            params["after"] = int(after.timestamp())
        if before:
            params["before"] = int(before.timestamp())

        response = requests.get(
            f"{self.base_url}/athlete/activities",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_activity_details(self, activity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific activity."""
        response = requests.get(
            f"{self.base_url}/activities/{activity_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def sync_activities(self, days_back: int = 30) -> int:
        """Sync activities from Strava to local database."""
        after_date = datetime.utcnow() - timedelta(days=days_back)
        activities = []
        page = 1

        print(f"Fetching activities from the last {days_back} days...")

        while True:
            batch = self.get_activities(after=after_date, page=page)
            if not batch:
                break
            activities.extend(batch)
            page += 1
            if len(batch) < 100:
                break

        # Save to database
        saved_count = 0
        with self.db.get_session() as session:
            for activity_data in activities:
                # Check if activity already exists
                existing = session.query(Activity).filter_by(
                    strava_id=str(activity_data["id"])
                ).first()

                if existing:
                    # Update existing activity
                    self._update_activity(existing, activity_data)
                else:
                    # Create new activity
                    activity = self._create_activity(activity_data)
                    session.add(activity)
                    saved_count += 1

            session.commit()

        print(f"Synced {len(activities)} activities ({saved_count} new)")
        return len(activities)

    def _create_activity(self, data: Dict[str, Any]) -> Activity:
        """Create Activity model from Strava data."""
        return Activity(
            strava_id=str(data["id"]),
            name=data.get("name"),
            type=data.get("type"),
            workout_type=data.get("workout_type"),  # 1=Race, 2=Long Run, 3=Workout
            start_date=datetime.fromisoformat(data["start_date"].replace("Z", "+00:00")),
            distance=data.get("distance"),
            moving_time=data.get("moving_time"),
            elapsed_time=data.get("elapsed_time"),
            total_elevation_gain=data.get("total_elevation_gain"),
            average_heartrate=data.get("average_heartrate"),
            max_heartrate=data.get("max_heartrate"),
            average_speed=data.get("average_speed"),
            max_speed=data.get("max_speed"),
            average_cadence=data.get("average_cadence"),
            average_watts=data.get("average_watts"),
            kilojoules=data.get("kilojoules"),
            suffer_score=data.get("suffer_score"),
            training_load=self._calculate_training_load(data),
            raw_data=json.dumps(data)
        )

    def _update_activity(self, activity: Activity, data: Dict[str, Any]) -> None:
        """Update existing activity with new data."""
        activity.name = data.get("name")
        activity.type = data.get("type")
        activity.workout_type = data.get("workout_type")
        activity.distance = data.get("distance")
        activity.moving_time = data.get("moving_time")
        activity.elapsed_time = data.get("elapsed_time")
        activity.total_elevation_gain = data.get("total_elevation_gain")
        activity.average_heartrate = data.get("average_heartrate")
        activity.max_heartrate = data.get("max_heartrate")
        activity.average_speed = data.get("average_speed")
        activity.max_speed = data.get("max_speed")
        activity.average_cadence = data.get("average_cadence")
        activity.average_watts = data.get("average_watts")
        activity.kilojoules = data.get("kilojoules")
        activity.suffer_score = data.get("suffer_score")
        activity.training_load = self._calculate_training_load(data)
        activity.raw_data = json.dumps(data)

    def _calculate_training_load(self, data: Dict[str, Any]) -> float:
        """Calculate training load using sport-specific metrics."""
        # Priority 1: Use suffer_score if available (Strava's relative effort)
        if data.get("suffer_score"):
            return float(data["suffer_score"])

        # Priority 2: Use multi-sport specific calculation
        multisport_calc = MultiSportCalculator()
        sport_specific_load = multisport_calc.calculate_sport_specific_load(data)

        if sport_specific_load > 0:
            return sport_specific_load

        # Priority 3: Calculate TRIMP if heart rate data is available
        if data.get("average_heartrate") and data.get("moving_time"):
            metrics_calc = SportsMetricsCalculator()
            duration_minutes = data["moving_time"] / 60
            avg_hr = data["average_heartrate"]
            max_hr = data.get("max_heartrate")

            # Calculate TRIMP
            trimp = metrics_calc.calculate_trimp(duration_minutes, avg_hr, max_hr)

            # Scale TRIMP to be comparable to TSS (roughly 100 = 1 hour at threshold)
            # TRIMP of 150 ≈ TSS of 100
            return trimp * 0.67

        # Priority 4: Calculate HRSS if we have heart rate
        if data.get("average_heartrate") and data.get("moving_time"):
            metrics_calc = SportsMetricsCalculator()
            duration_minutes = data["moving_time"] / 60
            avg_hr = data["average_heartrate"]
            return metrics_calc.calculate_hrss(duration_minutes, avg_hr)

        # Fallback: Basic estimation
        duration_hours = (data.get("moving_time", 0) / 3600.0)
        return duration_hours * 60  # Conservative estimate

    def get_recent_activities(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent activities from database."""
        with self.db.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            activities = session.query(Activity).filter(
                Activity.start_date >= cutoff_date
            ).order_by(Activity.start_date.desc()).all()

            # Convert to dictionaries to avoid session issues
            return [
                {
                    "strava_id": a.strava_id,
                    "name": a.name,
                    "type": a.type,
                    "start_date": a.start_date,
                    "distance": a.distance,
                    "moving_time": a.moving_time,
                    "training_load": a.training_load,
                    "average_heartrate": a.average_heartrate,
                    "max_heartrate": a.max_heartrate,
                    "average_speed": a.average_speed,
                    "total_elevation_gain": a.total_elevation_gain,
                    "average_watts": a.average_watts,
                }
                for a in activities
            ]

    def get_activity_streams(self, activity_id: str, stream_types: List[str] = None) -> Dict[str, List]:
        """Get detailed activity streams (HR, power, cadence, etc.).

        Args:
            activity_id: Strava activity ID
            stream_types: List of stream types to fetch (e.g., ['heartrate', 'watts', 'cadence'])
                        If None, fetches all available streams

        Returns:
            Dictionary of stream_type -> data array
        """
        if stream_types is None:
            stream_types = ['time', 'heartrate', 'watts', 'cadence', 'velocity_smooth', 'altitude']

        keys = ','.join(stream_types)

        try:
            response = requests.get(
                f"{self.base_url}/activities/{activity_id}/streams",
                headers=self._get_headers(),
                params={'keys': keys, 'key_by_type': True}
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Failed to fetch activity streams: {e}")
            return {}

    def calculate_advanced_metrics(self, activity_id: str) -> Dict[str, float]:
        """Calculate advanced metrics for a specific activity.

        Args:
            activity_id: Strava activity ID

        Returns:
            Dictionary of metric_name -> value
        """
        # Get activity details
        activity = self.get_activity_details(activity_id)
        if not activity:
            return {}

        # Get activity streams
        streams = self.get_activity_streams(activity_id)

        metrics_calc = SportsMetricsCalculator()
        advanced_metrics = {}

        # Calculate zone distribution if HR stream available
        if 'heartrate' in streams and streams['heartrate']:
            hr_data = streams['heartrate']['data']
            zone_dist = metrics_calc.calculate_zone_distribution(hr_data)

            # Convert enum keys to strings for JSON serialization
            for zone, percentage in zone_dist.items():
                advanced_metrics[f"zone_{zone.name.lower()}_pct"] = round(percentage, 1)

        # Calculate cardiac drift if we have HR and velocity
        if 'heartrate' in streams and 'velocity_smooth' in streams:
            hr_data = streams['heartrate']['data']
            velocity_data = streams['velocity_smooth']['data']

            if len(hr_data) > 100 and len(velocity_data) > 100:
                mid_point = len(hr_data) // 2

                first_half_hr = np.mean(hr_data[:mid_point])
                second_half_hr = np.mean(hr_data[mid_point:])
                first_half_pace = np.mean(velocity_data[:mid_point])
                second_half_pace = np.mean(velocity_data[mid_point:])

                cardiac_drift = metrics_calc.calculate_cardiac_drift(
                    first_half_hr, second_half_hr,
                    first_half_pace, second_half_pace
                )
                advanced_metrics['cardiac_drift_pct'] = round(cardiac_drift, 2)

        # Calculate efficiency factor for cycling
        if activity.get('type') in ['Ride', 'VirtualRide'] and activity.get('average_watts'):
            if activity.get('average_heartrate'):
                ef = metrics_calc.calculate_efficiency_factor(
                    activity['average_watts'],
                    activity['average_heartrate']
                )
                advanced_metrics['efficiency_factor'] = round(ef, 2)

        # Estimate VO2max for runs
        if activity.get('type') == 'Run' and activity.get('average_speed'):
            if activity.get('moving_time'):
                vo2max = metrics_calc.estimate_vo2max(
                    activity['average_speed'],
                    activity['moving_time'] / 60
                )
                advanced_metrics['estimated_vo2max'] = round(vo2max, 1)

        return advanced_metrics