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


class GarminError(Exception):
    """Garmin specific errors."""
    pass


class GarminClient:
    """Garmin Connect client with manual MFA code support."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

        # Get credentials from environment
        self.email = os.getenv("GARMIN_EMAIL")
        self.password = os.getenv("GARMIN_PASSWORD")

        if not self.email or not self.password:
            raise GarminError(
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
                            raise GarminError("MFA authentication cancelled by user")

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
            raise GarminError(f"Login failed: {login_result['message']}")

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

        # Count days that have ALL three data types (truly complete)
        complete_days = existing_hrv_dates & existing_sleep_dates & existing_wellness_dates
        days_with_complete_data = len(complete_days)

        # Calculate days that need syncing
        days_to_process = []
        current_date = start_date
        while current_date <= end_date:
            needs_hrv = current_date not in existing_hrv_dates
            needs_sleep = current_date not in existing_sleep_dates
            needs_wellness = current_date not in existing_wellness_dates

            if needs_hrv or needs_sleep or needs_wellness:
                days_to_process.append((current_date, needs_hrv, needs_sleep, needs_wellness))
            current_date += timedelta(days=1)

        # Print summary of what will be synced
        if len(days_to_process) == 0:
            print(f"All {total_days} days already have complete data. Nothing to sync.")
            return results

        # Show initial status
        print(f"Syncing {len(days_to_process)} days of Garmin data...")
        print(f"📅 Date range: {start_date} to {end_date} ({total_days} days total)")
        print(f"📊 Already synced: HRV={len(existing_hrv_dates)}, Sleep={len(existing_sleep_dates)}, Wellness={len(existing_wellness_dates)} days")

        # Sync day by day
        for idx, (sync_date, needs_hrv, needs_sleep, needs_wellness) in enumerate(days_to_process, 1):
            try:
                # Build list of data types needed
                data_types_needed = []
                if needs_hrv:
                    data_types_needed.append("HRV")
                if needs_sleep:
                    data_types_needed.append("Sleep")
                if needs_wellness:
                    data_types_needed.append("Wellness")

                # Show compact progress (overwrite same line)
                print(f"\rSyncing Garmin data: {idx}/{len(days_to_process)} days completed", end='', flush=True)

                # Get HRV data (skip if already exists)
                if needs_hrv:
                    if self._sync_hrv_for_date(sync_date):
                        results["hrv_synced"] += 1

                # Get sleep data (skip if already exists)
                if needs_sleep:
                    if self._sync_sleep_for_date(sync_date):
                        results["sleep_synced"] += 1

                # Get basic wellness data (skip if already exists)
                if needs_wellness:
                    if self._sync_wellness_for_date(sync_date):
                        results["wellness_synced"] += 1

            except Exception as e:
                print(f"\nError syncing {sync_date}: {e}")
                results["errors"] += 1

        # Complete the progress line
        print(f"\rSyncing Garmin data: {len(days_to_process)}/{len(days_to_process)} days completed ✓")

        return results

    def _sync_hrv_for_date(self, date: datetime.date) -> bool:
        """Sync HRV data for a specific date."""
        try:
            # Use garth built-in method for HRV data with date range
            # Try to get HRV data for a range around the target date to ensure we catch it
            start_date = date - timedelta(days=1)  # Day before
            hrv_records = garth.DailyHRV.list(start_date, 3)  # 3 days range

            # Find record for the specific date
            hrv_record = None
            for record in hrv_records:
                if hasattr(record, 'calendar_date') and record.calendar_date == date:
                    hrv_record = record
                    break

            if not hrv_record:
                # Also try without date range as fallback (in case the API changed)
                try:
                    all_hrv_records = garth.DailyHRV.list()
                    for record in all_hrv_records:
                        if hasattr(record, 'calendar_date') and record.calendar_date == date:
                            hrv_record = record
                            break
                except:
                    pass

            if not hrv_record:
                return False

            # Get heart rate data from Garmin using multiple methods
            resting_hr = None
            max_hr = None
            min_hr = None

            # First, check if RHR is directly available in the HRV record itself
            if hasattr(hrv_record, 'resting_heart_rate'):
                resting_hr = hrv_record.resting_heart_rate
            elif hasattr(hrv_record, 'restingHeartRate'):
                resting_hr = hrv_record.restingHeartRate
            elif hasattr(hrv_record, 'rhr'):
                resting_hr = hrv_record.rhr
            elif hasattr(hrv_record, 'baseline_low_upper'):
                # Sometimes RHR is in the HRV baseline data
                resting_hr = hrv_record.baseline_low_upper

            # Alternative: Try to get RHR from DailyHeartRate if available
            if not resting_hr:
                try:
                    daily_hr = garth.DailyHeartRate.get(date.strftime('%Y-%m-%d'))
                    if daily_hr:
                        resting_hr = (getattr(daily_hr, 'restingHeartRate', None) or
                                     getattr(daily_hr, 'resting_heart_rate', None) or
                                     getattr(daily_hr, 'rhr', None))
                except:
                    pass

            # Alternative: Try Activities for that day (activities often contain RHR)
            if not resting_hr:
                try:
                    activities = garth.Activities.list(date, 1)  # Get activities for the date
                    if activities:
                        for activity in activities:
                            if hasattr(activity, 'averageHR') or hasattr(activity, 'minHR'):
                                # Use minimum HR from activities as an approximation
                                min_hr = getattr(activity, 'minHR', None)
                                if min_hr and min_hr < 100:  # Reasonable RHR range
                                    resting_hr = min_hr
                                    break
                except:
                    pass

            # Get heart rate data using available garth classes
            # Write debug to file so we can see what happens
            debug_log_path = "/Users/alex/workspace/supercompensation/rhr_debug.log"
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"\n{'='*50}\n")
                debug_file.write(f"🔍 RHR Debug for {date} at {datetime.now()}\n")

                # Debug: Show what's in the HRV record
                if hrv_record:
                    hrv_attrs = [attr for attr in dir(hrv_record) if not attr.startswith('_')]
                    debug_file.write(f"📊 HRV Record attributes: {hrv_attrs}\n")

                    # Check specific fields that might contain heart rate data
                    hr_related_attrs = [attr for attr in hrv_attrs if any(keyword in attr.lower()
                                      for keyword in ['heart', 'hr', 'rate', 'rhr', 'baseline', 'pulse'])]
                    if hr_related_attrs:
                        debug_file.write(f"💓 HR-related attributes: {hr_related_attrs}\n")
                        for attr in hr_related_attrs:
                            try:
                                value = getattr(hrv_record, attr)
                                debug_file.write(f"  • {attr}: {value}\n")
                            except:
                                debug_file.write(f"  • {attr}: <unable to access>\n")

                    if resting_hr:
                        debug_file.write(f"✅ Found RHR {resting_hr} directly from HRV record\n")
                    else:
                        debug_file.write(f"⚠️  No RHR found in HRV record, trying other methods...\n")

                # Test the new methods we added
                if not resting_hr:
                    debug_file.write(f"🫀 Testing DailyHeartRate class...\n")
                    try:
                        daily_hr = garth.DailyHeartRate.get(date.strftime('%Y-%m-%d'))
                        if daily_hr:
                            hr_attrs = [attr for attr in dir(daily_hr) if not attr.startswith('_')]
                            debug_file.write(f"📊 DailyHeartRate attributes: {hr_attrs}\n")
                    except Exception as e:
                        debug_file.write(f"⚠️  DailyHeartRate failed: {e}\n")

                    debug_file.write(f"🏃 Testing Activities for RHR approximation...\n")
                    try:
                        activities = garth.Activities.list(date, 1)
                        if activities:
                            debug_file.write(f"📊 Found {len(activities)} activities for {date}\n")
                            for i, activity in enumerate(activities[:3]):  # Check first 3
                                activity_attrs = [attr for attr in dir(activity) if not attr.startswith('_') and 'hr' in attr.lower()]
                                if activity_attrs:
                                    debug_file.write(f"  Activity {i} HR attrs: {activity_attrs}\n")
                        else:
                            debug_file.write(f"📊 No activities found for {date}\n")
                    except Exception as e:
                        debug_file.write(f"⚠️  Activities failed: {e}\n")

                # Method 1: Try multiple garth classes for heart rate data
                debug_file.write(f"🔍 Trying available garth classes for HR data...\n")

                # Try UserProfile (might have baseline RHR)
                try:
                    debug_file.write(f"👤 Trying UserProfile for baseline RHR...\n")
                    profile = garth.UserProfile.get()
                    if profile:
                        resting_hr = (getattr(profile, 'restingHeartRate', None) or
                                     getattr(profile, 'resting_heart_rate', None))
                        if resting_hr:
                            debug_file.write(f"✅ Found baseline RHR {resting_hr} from UserProfile\n")
                        else:
                            profile_attrs = [attr for attr in dir(profile) if not attr.startswith('_') and 'heart' in attr.lower()]
                            debug_file.write(f"🔍 UserProfile HR attrs: {profile_attrs}\n")
                except Exception as e:
                    debug_file.write(f"⚠️  UserProfile failed: {e}\n")

                # Try HRVData class (might have embedded HR)
                if not resting_hr:
                    try:
                        debug_file.write(f"💓 Trying HRVData class for embedded HR...\n")
                        hrv_data = garth.HRVData.get(date.strftime('%Y-%m-%d'))
                        if hrv_data:
                            resting_hr = (getattr(hrv_data, 'restingHeartRate', None) or
                                         getattr(hrv_data, 'resting_heart_rate', None))
                            if resting_hr:
                                debug_file.write(f"✅ Found RHR {resting_hr} from HRVData\n")
                            else:
                                hrv_attrs = [attr for attr in dir(hrv_data) if not attr.startswith('_') and 'heart' in attr.lower()]
                                debug_file.write(f"🔍 HRVData HR attrs: {hrv_attrs}\n")
                    except Exception as e:
                        debug_file.write(f"⚠️  HRVData failed: {e}\n")

                # Try DailyBodyBatteryStress
                if not resting_hr:
                    try:
                        debug_file.write(f"🔋 Trying DailyBodyBatteryStress for {date}...\n")
                        bb_data = garth.DailyBodyBatteryStress.get(date.strftime('%Y-%m-%d'))
                        if bb_data:
                            # Try accessing body battery raw data
                            if hasattr(bb_data, 'body_battery_readings') and bb_data.body_battery_readings:
                                debug_file.write(f"📊 Found body_battery_readings data\n")
                                # Sometimes RHR is embedded in the readings
                                readings = bb_data.body_battery_readings
                                if readings and len(readings) > 0:
                                    first_reading = readings[0]
                                    if hasattr(first_reading, 'restingHeartRate'):
                                        resting_hr = first_reading.restingHeartRate
                                        debug_file.write(f"✅ Found RHR {resting_hr} in body battery readings\n")

                    except Exception as e:
                        debug_file.write(f"⚠️  DailyBodyBatteryStress enhanced failed: {e}\n")

                # Method 2: Try heart rate specific endpoints
                if not resting_hr:
                    debug_file.write(f"🔄 Trying heart rate specific endpoints...\n")

                    # Try heart rate specific endpoints that might work
                    hr_endpoints = [
                        f"/wellness-service/wellness/heartRate/daily/{date.strftime('%Y-%m-%d')}",
                        f"/metrics-service/metrics/heartRate/{date.strftime('%Y-%m-%d')}",
                        f"/userstats-service/heartRate/daily/{date.strftime('%Y-%m-%d')}",
                        f"/summary-service/stats/heartRate/{date.strftime('%Y-%m-%d')}",
                        f"/wellness-service/wellness/daily/{date.strftime('%Y-%m-%d')}"
                    ]

                    for endpoint in hr_endpoints:
                        try:
                            debug_file.write(f"🔧 Trying HR endpoint: {endpoint}\n")
                            data = garth.connectapi(endpoint)

                            if data and isinstance(data, dict):
                                debug_file.write(f"📊 HR API returned {type(data)} with {len(data)} keys\n")

                                # Check for HR data in various forms
                                resting_hr = (data.get("restingHeartRate") or
                                             data.get("resting_heart_rate") or
                                             data.get("restingHR") or
                                             data.get("restingHr") or
                                             data.get("lowestHeartRate") or
                                             data.get("minHeartRate") or
                                             data.get("dailyRestingHeartRate"))

                                if resting_hr:
                                    debug_file.write(f"✅ Found RHR {resting_hr} via {endpoint}\n")
                                    max_hr = (data.get("maxHeartRate") or data.get("maximumHeartRate"))
                                    min_hr = (data.get("minHeartRate") or data.get("minimumHeartRate"))
                                    break
                                else:
                                    # Show available keys for debugging
                                    api_keys = list(data.keys())[:15]
                                    debug_file.write(f"🔍 HR API keys: {api_keys}\n")

                            elif data and isinstance(data, list) and len(data) > 0:
                                debug_file.write(f"📊 HR API returned list with {len(data)} items\n")
                                # Sometimes HR data is in a list format
                                for item in data[:3]:  # Check first 3 items
                                    if isinstance(item, dict):
                                        resting_hr = (item.get("restingHeartRate") or
                                                     item.get("resting_heart_rate") or
                                                     item.get("restingHR") or
                                                     item.get("value"))
                                        if resting_hr:
                                            debug_file.write(f"✅ Found RHR {resting_hr} in list via {endpoint}\n")
                                            break
                                if resting_hr:
                                    break

                        except Exception as api_e:
                            debug_file.write(f"⚠️  HR API {endpoint} failed: {api_e}\n")
                            continue

                # Final result check
                if resting_hr or max_hr or min_hr:
                    debug_file.write(f"✅ Final HR data for {date}: RHR={resting_hr}, Max={max_hr}, Min={min_hr}\n")
                else:
                    debug_file.write(f"❌ No heart rate data found for {date} after trying all methods\n")

            # Fallback: Check if heart rate data is embedded in HRV record
            if not resting_hr and hrv_record:
                # Some Garmin devices include HR data with HRV measurements
                resting_hr = getattr(hrv_record, 'resting_heart_rate', None)
                max_hr = getattr(hrv_record, 'max_heart_rate', None)
                min_hr = getattr(hrv_record, 'min_heart_rate', None)
                if resting_hr:
                    print(f"✅ Found RHR {resting_hr} embedded in HRV record for {date}")

            # Debug: show what heart rate data we found
            if resting_hr or max_hr or min_hr:
                print(f"✅ Heart rate data for {date}: RHR={resting_hr}, Max={max_hr}, Min={min_hr}")
            else:
                print(f"⚠️  No heart rate data found for {date}")

            # Extract HRV metrics from garth record
            # Use nightly RMSSD as primary metric (scientifically correct)
            hrv_rmssd = hrv_record.last_night_avg  # Last night average RMSSD
            # Store weekly_avg separately for reference but don't use as primary score
            weekly_avg_rmssd = hrv_record.weekly_avg  # Weekly average for context
            hrv_status = hrv_record.status.lower() if hrv_record.status else 'unknown'

            # Use daily RMSSD as the primary HRV score (not weekly average)
            hrv_score = hrv_rmssd if hrv_rmssd is not None else weekly_avg_rmssd

            if hrv_score is None and hrv_rmssd is None:
                return False

            # Note: RHR will only be populated from OMRON blood pressure data, not estimated
            # Garmin API RHR fetching remains for when OAuth issues are resolved

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
                    existing.resting_heart_rate = resting_hr
                    existing.max_heart_rate = max_hr
                    existing.min_heart_rate = min_hr
                    existing.raw_data = json.dumps({
                        'daily_rmssd': hrv_record.last_night_avg,  # Primary daily metric
                        'weekly_avg_rmssd': hrv_record.weekly_avg,  # Context only
                        'last_night_5_min_high': hrv_record.last_night_5_min_high,
                        'status': hrv_record.status,
                        'feedback_phrase': hrv_record.feedback_phrase,
                        'note': 'hrv_score uses daily_rmssd for baseline analysis',
                        'heart_rate': {
                            'resting': resting_hr,
                            'max': max_hr,
                            'min': min_hr
                        }
                    })
                else:
                    hrv_db_record = HRVData(
                        user_id=self.user_id,
                        date=datetime.combine(date, datetime.min.time()),
                        hrv_score=hrv_score,
                        hrv_rmssd=hrv_rmssd,
                        hrv_status=hrv_status,
                        resting_heart_rate=resting_hr,
                        max_heart_rate=max_hr,
                        min_heart_rate=min_hr,
                        raw_data=json.dumps({
                            'daily_rmssd': hrv_record.last_night_avg,  # Primary daily metric
                            'weekly_avg_rmssd': hrv_record.weekly_avg,  # Context only
                            'last_night_5_min_high': hrv_record.last_night_5_min_high,
                            'status': hrv_record.status,
                            'feedback_phrase': hrv_record.feedback_phrase,
                            'note': 'hrv_score uses daily_rmssd for baseline analysis',
                            'heart_rate': {
                                'resting': resting_hr,
                                'max': max_hr,
                                'min': min_hr
                            }
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
            # Multiple approaches to get historical sleep data
            sleep_record = None

            # Method 1: Try with larger date range (sometimes needs more days)
            try:
                start_date = date - timedelta(days=2)  # 2 days before
                sleep_records = garth.DailySleep.list(start_date, 5)  # 5 days range
                for record in sleep_records:
                    if hasattr(record, 'calendar_date') and record.calendar_date == date:
                        sleep_record = record
                        break
            except Exception as e:
                pass

            # Method 2: Try specific date format approach
            if not sleep_record:
                try:
                    # Some garth methods need specific date string format
                    date_str = date.strftime('%Y-%m-%d')
                    sleep_record = garth.DailySleep.get(date_str)
                except Exception as e:
                    pass

            # Method 3: Try with different range parameters
            if not sleep_record:
                try:
                    # Try getting sleep data for exact date with 1-day range
                    sleep_records = garth.DailySleep.list(date, 1)
                    if sleep_records:
                        sleep_record = sleep_records[0] if len(sleep_records) > 0 else None
                except Exception as e:
                    pass

            # Method 4: Original approach as final fallback
            if not sleep_record:
                try:
                    all_sleep_records = garth.DailySleep.list()
                    for record in all_sleep_records:
                        if hasattr(record, 'calendar_date') and record.calendar_date == date:
                            sleep_record = record
                            break
                except:
                    pass

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
            # Get stress and steps data using garth built-in method with date range
            start_date = date - timedelta(days=1)  # Day before
            stress_records = garth.DailyStress.list(start_date, 3)  # 3 days range
            steps_records = garth.DailySteps.list(start_date, 3)  # 3 days range

            # Heart rate data is now handled in HRV sync

            # Find records for the specific date
            stress_record = None
            steps_record = None

            for record in stress_records:
                if hasattr(record, 'calendar_date') and record.calendar_date == date:
                    stress_record = record
                    break

            for record in steps_records:
                if hasattr(record, 'calendar_date') and record.calendar_date == date:
                    steps_record = record
                    break

            # Fallback to getting all records if date range didn't work
            if not stress_record or not steps_record:
                try:
                    if not stress_record:
                        all_stress_records = garth.DailyStress.list()
                        for record in all_stress_records:
                            if hasattr(record, 'calendar_date') and record.calendar_date == date:
                                stress_record = record
                                break

                    if not steps_record:
                        all_steps_records = garth.DailySteps.list()
                        for record in all_steps_records:
                            if hasattr(record, 'calendar_date') and record.calendar_date == date:
                                steps_record = record
                                break
                except:
                    pass

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

    def get_latest_hrv_score(self) -> Optional[float]:
        """Get the latest HRV score from the database."""
        with self.db.get_session() as session:
            latest_hrv = session.query(HRVData).filter_by(
                user_id=self.user_id
            ).order_by(HRVData.date.desc()).first()

            return latest_hrv.hrv_score if latest_hrv else None

    def get_latest_sleep_score(self) -> Optional[float]:
        """Get the latest sleep score from the database."""
        with self.db.get_session() as session:
            latest_sleep = session.query(SleepData).filter_by(
                user_id=self.user_id
            ).order_by(SleepData.date.desc()).first()

            return latest_sleep.sleep_score if latest_sleep else None

    def get_wellness_trend(self, days: int = 7) -> Dict:
        """Get wellness trend over the specified number of days."""
        with self.db.get_session() as session:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            # Get HRV data
            hrv_data = session.query(HRVData).filter(
                HRVData.user_id == self.user_id,
                HRVData.date >= datetime.combine(start_date, datetime.min.time()),
                HRVData.date <= datetime.combine(end_date, datetime.min.time())
            ).order_by(HRVData.date).all()

            # Get Sleep data
            sleep_data = session.query(SleepData).filter(
                SleepData.user_id == self.user_id,
                SleepData.date >= datetime.combine(start_date, datetime.min.time()),
                SleepData.date <= datetime.combine(end_date, datetime.min.time())
            ).order_by(SleepData.date).all()

            # Get Wellness data
            wellness_data = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= datetime.combine(start_date, datetime.min.time()),
                WellnessData.date <= datetime.combine(end_date, datetime.min.time())
            ).order_by(WellnessData.date).all()

            return {
                "hrv_trend": [{"date": d.date.strftime('%Y-%m-%d'), "score": d.hrv_score} for d in hrv_data if d.hrv_score],
                "sleep_trend": [{"date": d.date.strftime('%Y-%m-%d'), "score": d.sleep_score} for d in sleep_data if d.sleep_score],
                "stress_trend": [{"date": d.date.strftime('%Y-%m-%d'), "stress": d.stress_avg_value} for d in wellness_data if d.stress_avg_value],
                "body_battery_trend": [{"date": d.date.strftime('%Y-%m-%d'), "battery": d.body_battery_charged_value} for d in wellness_data if d.body_battery_charged_value],
                "date_range": f"{start_date} to {end_date}"
            }


def get_garmin_client(user_id: str = "default") -> GarminClient:
    """Get Garmin client for user."""
    return GarminClient(user_id)