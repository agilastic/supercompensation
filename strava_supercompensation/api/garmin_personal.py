"""Personal Garmin Connect data access using python-garminconnect library."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectTooManyRequestsError, GarminConnectConnectionError

from ..db import get_db
from ..db.models import HRVData, SleepData, WellnessData


class GarminPersonalError(Exception):
    """Garmin Personal client specific errors."""
    pass


class GarminPersonalClient:
    """Minimal Garmin Connect client for essential HRV and sleep data."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.client = None

        # Get credentials from environment
        self.email = os.getenv("GARMIN_EMAIL")
        self.password = os.getenv("GARMIN_PASSWORD")

        if not self.email or not self.password:
            raise GarminPersonalError(
                "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD environment variables."
            )

    def login(self, mfa_code: str = None) -> bool:
        """Login to Garmin Connect with optional MFA support."""
        try:
            self.client = Garmin(self.email, self.password)

            if mfa_code:
                # Try login with MFA code
                self.client.login()
                return True
            else:
                # Try normal login first
                self.client.login()
                return True

        except GarminConnectAuthenticationError as e:
            error_msg = str(e).lower()
            if "mfa" in error_msg or "verification" in error_msg or "code" in error_msg:
                # MFA required - return special error for CLI to handle
                raise GarminPersonalError("MFA_REQUIRED")
            raise GarminPersonalError(f"Authentication failed: {str(e)}")
        except Exception as e:
            error_msg = str(e).lower()
            if "mfa" in error_msg or "verification" in error_msg or "code" in error_msg:
                raise GarminPersonalError("MFA_REQUIRED")
            raise GarminPersonalError(f"Login failed: {str(e)}")

    def _get_existing_data_dates(self, start_date: datetime.date, end_date: datetime.date, data_type: str) -> set:
        """Get dates that already have data in the database for the given range.

        Args:
            start_date: Start date of the range
            end_date: End date of the range
            data_type: Type of data ('hrv', 'sleep', 'wellness')

        Returns:
            Set of dates that already have data
        """
        with self.db.get_session() as session:
            if data_type == 'hrv':
                model = HRVData
            elif data_type == 'sleep':
                model = SleepData
            elif data_type == 'wellness':
                model = WellnessData
            else:
                return set()

            existing_records = session.query(model.date).filter(
                model.user_id == self.user_id,
                model.date >= datetime.combine(start_date, datetime.min.time()),
                model.date <= datetime.combine(end_date, datetime.min.time())
            ).all()

            # Convert to set of dates
            return {record.date.date() for record in existing_records}

    def sync_essential_data(self, days: int = 7) -> Dict[str, int]:
        """Sync only essential HRV and sleep data for specified days.

        Args:
            days: Number of days to sync (default 7)

        Returns:
            Dictionary with sync results
        """
        if not self.client:
            self.login()

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        # Check for existing data to skip API calls
        existing_hrv_dates = self._get_existing_data_dates(start_date, end_date, 'hrv')
        existing_sleep_dates = self._get_existing_data_dates(start_date, end_date, 'sleep')
        existing_wellness_dates = self._get_existing_data_dates(start_date, end_date, 'wellness')

        results = {
            "hrv_synced": 0,
            "sleep_synced": 0,
            "wellness_synced": 0,
            "hrv_skipped": len(existing_hrv_dates),
            "sleep_skipped": len(existing_sleep_dates),
            "wellness_skipped": len(existing_wellness_dates),
            "errors": 0,
            "date_range": f"{start_date} to {end_date}"
        }

        # Sync day by day to avoid rate limits
        current_date = start_date
        while current_date <= end_date:
            try:
                date_str = current_date.strftime('%Y-%m-%d')

                # Get HRV data (skip if already exists)
                if current_date not in existing_hrv_dates:
                    if self._sync_hrv_for_date(current_date):
                        results["hrv_synced"] += 1

                # Get sleep data (skip if already exists)
                if current_date not in existing_sleep_dates:
                    if self._sync_sleep_for_date(current_date):
                        results["sleep_synced"] += 1

                # Get basic wellness data (skip if already exists)
                if current_date not in existing_wellness_dates:
                    if self._sync_wellness_for_date(current_date):
                        results["wellness_synced"] += 1

            except GarminConnectTooManyRequestsError:
                print(f"Rate limited at {current_date}. Stopping sync.")
                break
            except Exception as e:
                print(f"Error syncing {current_date}: {e}")
                results["errors"] += 1

            current_date += timedelta(days=1)

        return results

    def _sync_hrv_for_date(self, date: datetime.date) -> bool:
        """Sync HRV data for a specific date."""
        try:
            date_str = date.strftime('%Y-%m-%d')

            # Get HRV data
            hrv_data = self.client.get_hrv_data(date_str)

            if not hrv_data or not hrv_data.get('hrvSummary'):
                return False

            hrv_summary = hrv_data['hrvSummary']

            # Extract essential HRV metrics
            hrv_score = hrv_summary.get('hrvScore')
            hrv_rmssd = hrv_summary.get('rmssd')
            hrv_status = hrv_summary.get('hrvStatus', 'unknown')

            if hrv_score is None and hrv_rmssd is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                # Check if record exists
                existing = session.query(HRVData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                if existing:
                    # Update existing
                    existing.hrv_score = hrv_score
                    existing.hrv_rmssd = hrv_rmssd
                    existing.hrv_status = hrv_status
                    existing.raw_data = json.dumps(hrv_data)
                else:
                    # Create new
                    hrv_record = HRVData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        hrv_score=hrv_score,
                        hrv_rmssd=hrv_rmssd,
                        hrv_status=hrv_status,
                        raw_data=json.dumps(hrv_data)
                    )
                    session.add(hrv_record)

                session.commit()
                return True

        except Exception as e:
            print(f"HRV sync error for {date}: {e}")
            return False

    def _sync_sleep_for_date(self, date: datetime.date) -> bool:
        """Sync sleep data for a specific date."""
        try:
            date_str = date.strftime('%Y-%m-%d')

            # Get sleep data
            sleep_data = self.client.get_sleep_data(date_str)

            if not sleep_data or not sleep_data.get('dailySleepDTO'):
                return False

            sleep_summary = sleep_data['dailySleepDTO']

            # Extract essential sleep metrics
            sleep_score = sleep_summary.get('sleepScores', {}).get('overall', {}).get('value')
            total_sleep_seconds = sleep_summary.get('sleepTimeSeconds')
            deep_sleep_seconds = sleep_summary.get('deepSleepSeconds')
            light_sleep_seconds = sleep_summary.get('lightSleepSeconds')
            rem_sleep_seconds = sleep_summary.get('remSleepSeconds')
            awake_seconds = sleep_summary.get('awakeSleepSeconds')

            if sleep_score is None and total_sleep_seconds is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                # Check if record exists
                existing = session.query(SleepData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                if existing:
                    # Update existing
                    existing.sleep_score = sleep_score
                    existing.total_sleep_time = total_sleep_seconds
                    existing.deep_sleep_time = deep_sleep_seconds
                    existing.light_sleep_time = light_sleep_seconds
                    existing.rem_sleep_time = rem_sleep_seconds
                    existing.awake_time = awake_seconds
                    existing.raw_data = json.dumps(sleep_data)
                else:
                    # Create new
                    sleep_record = SleepData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        sleep_score=sleep_score,
                        total_sleep_time=total_sleep_seconds,
                        deep_sleep_time=deep_sleep_seconds,
                        light_sleep_time=light_sleep_seconds,
                        rem_sleep_time=rem_sleep_seconds,
                        awake_time=awake_seconds,
                        raw_data=json.dumps(sleep_data)
                    )
                    session.add(sleep_record)

                session.commit()
                return True

        except Exception as e:
            print(f"Sleep sync error for {date}: {e}")
            return False

    def _sync_wellness_for_date(self, date: datetime.date) -> bool:
        """Sync basic wellness data for a specific date."""
        try:
            date_str = date.strftime('%Y-%m-%d')

            # Get stress data
            stress_data = self.client.get_stress_data(date_str)

            # Get daily summary for steps, etc.
            daily_summary = self.client.get_daily_summary(date_str)

            # Extract essential wellness metrics
            stress_avg = None
            if stress_data and stress_data.get('stressValuesArray'):
                stress_values = [s.get('value') for s in stress_data['stressValuesArray'] if s.get('value') is not None]
                if stress_values:
                    stress_avg = sum(stress_values) / len(stress_values)

            steps = None
            if daily_summary and daily_summary.get('totalSteps'):
                steps = daily_summary['totalSteps']

            if stress_avg is None and steps is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                # Check if record exists
                existing = session.query(WellnessData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                wellness_raw = {
                    'stress_data': stress_data,
                    'daily_summary': daily_summary
                }

                if existing:
                    # Update existing
                    existing.stress_avg = stress_avg
                    existing.total_steps = steps
                    existing.raw_data = json.dumps(wellness_raw)
                else:
                    # Create new
                    wellness_record = WellnessData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        stress_avg=stress_avg,
                        total_steps=steps,
                        raw_data=json.dumps(wellness_raw)
                    )
                    session.add(wellness_record)

                session.commit()
                return True

        except Exception as e:
            print(f"Wellness sync error for {date}: {e}")
            return False

    def test_connection(self) -> Dict[str, str]:
        """Test connection to Garmin Connect."""
        try:
            if not self.client:
                self.login()

            # Get basic user info
            user_summary = self.client.get_user_summary()

            return {
                "status": "success",
                "display_name": user_summary.get('displayName', 'Unknown'),
                "user_id": str(user_summary.get('userProfileId', 'Unknown')),
                "email": self.email
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def get_latest_scores(self) -> Dict[str, Optional[float]]:
        """Get latest HRV and sleep scores for quick check."""
        with self.db.get_session() as session:
            # Get latest HRV
            latest_hrv = session.query(HRVData).filter_by(
                user_id=self.user_id
            ).order_by(HRVData.date.desc()).first()

            # Get latest sleep
            latest_sleep = session.query(SleepData).filter_by(
                user_id=self.user_id
            ).order_by(SleepData.date.desc()).first()

            return {
                "hrv_score": latest_hrv.hrv_score if latest_hrv else None,
                "hrv_date": latest_hrv.date.strftime('%Y-%m-%d') if latest_hrv else None,
                "sleep_score": latest_sleep.sleep_score if latest_sleep else None,
                "sleep_date": latest_sleep.date.strftime('%Y-%m-%d') if latest_sleep else None
            }


def get_garmin_personal_client(user_id: str = "default") -> GarminPersonalClient:
    """Get Garmin personal client for user."""
    return GarminPersonalClient(user_id)