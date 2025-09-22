"""Garmin Connect client with MFA support using garth library."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dotenv import load_dotenv
import garth

load_dotenv()

from ..db import get_db
from ..db.models import HRVData, SleepData, WellnessData


class GarminMFAError(Exception):
    """Garmin MFA specific errors."""
    pass


class GarminMFAClient:
    """Garmin Connect client with manual MFA code support."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

        # Get credentials from environment
        self.email = os.getenv("GARMIN_EMAIL")
        self.password = os.getenv("GARMIN_PASSWORD")

        if not self.email or not self.password:
            raise GarminMFAError(
                "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD environment variables."
            )

    def login_with_mfa(self, mfa_code: str = None) -> Dict[str, str]:
        """Login to Garmin Connect with MFA code support."""
        try:
            token_path = os.path.expanduser("~/.garth")

            # Configure garth domain
            garth.configure(domain="garmin.com")

            # Try to use saved tokens first
            if os.path.exists(token_path):
                try:
                    garth.resume(token_path)

                    # Test if tokens work by getting profile
                    profile = garth.client.profile
                    if profile:
                        user_name = profile.get("displayName", profile.get("userProfileId", "Unknown"))
                        print("Using saved authentication tokens")
                        return {
                            "status": "success",
                            "message": "Logged in with saved tokens",
                            "user": user_name
                        }
                except Exception as e:
                    print(f"Saved tokens invalid: {e}")
                    # Continue with fresh login

            # Fresh login attempt with MFA support
            def mfa_handler():
                if mfa_code:
                    print(f"Using provided MFA code: {mfa_code}")
                    return mfa_code
                else:
                    # Interactive prompt - wait for user input
                    while True:
                        try:
                            code = input("Enter MFA code from Garmin Connect: ").strip()
                            if code and len(code) >= 6:  # Basic validation
                                return code
                            else:
                                print("Please enter a valid 6-digit MFA code")
                        except (EOFError, KeyboardInterrupt):
                            raise GarminMFAError("MFA authentication cancelled by user")

            print("Attempting fresh login with MFA support...")
            garth.login(self.email, self.password, prompt_mfa=mfa_handler)

            # Save tokens for future use
            os.makedirs(token_path, exist_ok=True)
            garth.save(token_path)
            print("Authentication tokens saved for future use")

            # Get user info using garth client
            profile = garth.client.profile
            user_name = profile.get("displayName", profile.get("userProfileId", "Unknown"))
            return {
                "status": "success",
                "message": "Successfully authenticated with MFA",
                "user": user_name
            }

        except Exception as e:
            error_str = str(e).lower()
            if "mfa" in error_str or "verification" in error_str:
                return {
                    "status": "mfa_required",
                    "message": "MFA code required for authentication"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Login failed: {str(e)}"
                }

    def test_connection(self) -> Dict[str, str]:
        """Test connection to Garmin Connect."""
        try:
            # Try to login first
            login_result = self.login_with_mfa()

            if login_result["status"] != "success":
                return login_result

            # Test API call using garth client
            profile = garth.client.profile
            user_name = profile.get("displayName", profile.get("userProfileId", "Unknown"))
            user_id = str(profile.get("userProfileId", "Unknown"))

            return {
                "status": "success",
                "display_name": user_name,
                "user_id": user_id,
                "email": self.email
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

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
        """Sync essential HRV and sleep data."""
        # Ensure we're authenticated
        login_result = self.login_with_mfa()
        if login_result["status"] != "success":
            raise GarminMFAError(f"Login failed: {login_result['message']}")

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

        print(f"Syncing Garmin data from {start_date} to {end_date}")
        total_days = (end_date - start_date).days + 1
        days_to_sync = total_days - len(existing_hrv_dates | existing_sleep_dates | existing_wellness_dates)
        if days_to_sync < total_days:
            print(f"Found existing data, will skip {total_days - days_to_sync} days with complete data")

        # Sync day by day
        current_date = start_date
        while current_date <= end_date:
            try:
                # Skip completely if all data types exist for this date
                if (current_date in existing_hrv_dates and
                    current_date in existing_sleep_dates and
                    current_date in existing_wellness_dates):
                    current_date += timedelta(days=1)
                    continue

                print(f"Syncing data for {current_date}")

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

            except Exception as e:
                print(f"Error syncing {current_date}: {e}")
                results["errors"] += 1

            current_date += timedelta(days=1)

        return results

    def _sync_hrv_for_date(self, date: datetime.date) -> bool:
        """Sync HRV data for a specific date."""
        try:
            # Use garth built-in method for HRV data
            hrv_records = garth.DailyHRV.list()

            # Find record for the specific date
            hrv_record = None
            for record in hrv_records:
                if record.calendar_date == date:
                    hrv_record = record
                    break

            if not hrv_record:
                return False

            # Extract HRV metrics from garth record
            hrv_score = hrv_record.weekly_avg  # Weekly average HRV
            hrv_rmssd = hrv_record.last_night_avg  # Last night average
            hrv_status = hrv_record.status.lower() if hrv_record.status else 'unknown'

            if hrv_score is None and hrv_rmssd is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                existing = session.query(HRVData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                if existing:
                    existing.hrv_score = hrv_score
                    existing.hrv_rmssd = hrv_rmssd
                    existing.hrv_status = hrv_status
                    existing.raw_data = json.dumps({
                        'weekly_avg': hrv_record.weekly_avg,
                        'last_night_avg': hrv_record.last_night_avg,
                        'last_night_5_min_high': hrv_record.last_night_5_min_high,
                        'status': hrv_record.status,
                        'feedback_phrase': hrv_record.feedback_phrase
                    })
                else:
                    hrv_db_record = HRVData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        hrv_score=hrv_score,
                        hrv_rmssd=hrv_rmssd,
                        hrv_status=hrv_status,
                        raw_data=json.dumps({
                            'weekly_avg': hrv_record.weekly_avg,
                            'last_night_avg': hrv_record.last_night_avg,
                            'last_night_5_min_high': hrv_record.last_night_5_min_high,
                            'status': hrv_record.status,
                            'feedback_phrase': hrv_record.feedback_phrase
                        })
                    )
                    session.add(hrv_db_record)

                session.commit()
                return True

        except Exception as e:
            print(f"HRV sync error for {date}: {e}")
            return False

    def _sync_sleep_for_date(self, date: datetime.date) -> bool:
        """Sync sleep data for a specific date."""
        try:
            # Use garth built-in method for sleep data
            sleep_records = garth.DailySleep.list()

            # Find record for the specific date
            sleep_record = None
            for record in sleep_records:
                if record.calendar_date == date:
                    sleep_record = record
                    break

            if not sleep_record:
                return False

            # Extract sleep metrics from garth record
            sleep_score = sleep_record.value  # Sleep score (0-100)
            total_sleep_seconds = None  # Not available in basic sleep data

            if sleep_score is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                existing = session.query(SleepData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                if existing:
                    existing.sleep_score = sleep_score
                    existing.total_sleep_time = total_sleep_seconds
                    existing.raw_data = json.dumps({
                        'sleep_score': sleep_record.value,
                        'calendar_date': sleep_record.calendar_date.isoformat()
                    })
                else:
                    sleep_db_record = SleepData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        sleep_score=sleep_score,
                        total_sleep_time=total_sleep_seconds,
                        raw_data=json.dumps({
                            'sleep_score': sleep_record.value,
                            'calendar_date': sleep_record.calendar_date.isoformat()
                        })
                    )
                    session.add(sleep_db_record)

                session.commit()
                return True

        except Exception as e:
            print(f"Sleep sync error for {date}: {e}")
            return False

    def _sync_wellness_for_date(self, date: datetime.date) -> bool:
        """Sync basic wellness data for a specific date."""
        try:
            # Get stress data using garth built-in method
            stress_records = garth.DailyStress.list()
            steps_records = garth.DailySteps.list()

            # Find records for the specific date
            stress_record = None
            steps_record = None

            for record in stress_records:
                if record.calendar_date == date:
                    stress_record = record
                    break

            for record in steps_records:
                if record.calendar_date == date:
                    steps_record = record
                    break

            stress_avg = stress_record.overall_stress_level if stress_record else None
            steps = steps_record.total_steps if steps_record else None

            if stress_avg is None and steps is None:
                return False

            # Store in database
            with self.db.get_session() as session:
                existing = session.query(WellnessData).filter_by(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time())
                ).first()

                wellness_raw = {
                    'stress_level': stress_avg,
                    'total_steps': steps,
                    'stress_durations': {
                        'rest': stress_record.rest_stress_duration if stress_record else None,
                        'low': stress_record.low_stress_duration if stress_record else None,
                        'medium': stress_record.medium_stress_duration if stress_record else None,
                        'high': stress_record.high_stress_duration if stress_record else None
                    } if stress_record else None,
                    'steps_distance': steps_record.total_distance if steps_record else None,
                    'step_goal': steps_record.step_goal if steps_record else None
                }

                if existing:
                    existing.stress_avg = stress_avg
                    existing.total_steps = steps
                    existing.raw_data = json.dumps(wellness_raw)
                else:
                    wellness_db_record = WellnessData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        stress_avg=stress_avg,
                        total_steps=steps,
                        raw_data=json.dumps(wellness_raw)
                    )
                    session.add(wellness_db_record)

                session.commit()
                return True

        except Exception as e:
            print(f"Wellness sync error for {date}: {e}")
            return False

    def get_latest_scores(self) -> Dict[str, Optional[float]]:
        """Get latest HRV and sleep scores."""
        with self.db.get_session() as session:
            latest_hrv = session.query(HRVData).filter_by(
                user_id=self.user_id
            ).order_by(HRVData.date.desc()).first()

            latest_sleep = session.query(SleepData).filter_by(
                user_id=self.user_id
            ).order_by(SleepData.date.desc()).first()

            return {
                "hrv_score": latest_hrv.hrv_score if latest_hrv else None,
                "hrv_date": latest_hrv.date.strftime('%Y-%m-%d') if latest_hrv else None,
                "sleep_score": latest_sleep.sleep_score if latest_sleep else None,
                "sleep_date": latest_sleep.date.strftime('%Y-%m-%d') if latest_sleep else None
            }


def get_garmin_mfa_client(user_id: str = "default") -> GarminMFAClient:
    """Get Garmin MFA client for user."""
    return GarminMFAClient(user_id)