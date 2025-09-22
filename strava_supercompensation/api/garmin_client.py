"""Garmin Connect API client for fetching wellness data."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..auth.garmin_oauth import get_garmin_auth, GarminOAuthError
from ..db import get_db
from ..db.models import HRVData, SleepData, WellnessData


class GarminAPIError(Exception):
    """Garmin API specific errors."""
    pass


class GarminClient:
    """Garmin Connect API client for wellness data."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.auth = get_garmin_auth(user_id)
        self.db = get_db()

        # Garmin Connect API endpoints
        self.base_url = "https://connectapi.garmin.com"

    def sync_hrv_data(self, days: int = 30) -> Dict:
        """Sync HRV data from Garmin Connect.

        Args:
            days: Number of days to sync (default 30)

        Returns:
            Dictionary with sync results
        """
        if not self.auth.is_authenticated():
            raise GarminAPIError("Not authenticated with Garmin. Run auth first.")

        try:
            oauth_session = self.auth.get_authenticated_session()
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)

            synced_count = 0
            updated_count = 0

            # Fetch HRV data day by day (Garmin API limitation)
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')

                # Get HRV data for the day
                hrv_url = f"{self.base_url}/hrv-service/hrv/{date_str}"
                response = oauth_session.get(hrv_url)

                if response.status_code == 200:
                    hrv_data = response.json()
                    if hrv_data and self._has_valid_hrv_data(hrv_data):
                        if self._store_hrv_data(current_date, hrv_data):
                            synced_count += 1
                        else:
                            updated_count += 1
                elif response.status_code == 404:
                    # No HRV data for this day - normal
                    pass
                else:
                    print(f"Warning: Failed to get HRV data for {date_str}: {response.status_code}")

                current_date += timedelta(days=1)

            return {
                "type": "hrv",
                "synced": synced_count,
                "updated": updated_count,
                "date_range": f"{start_date} to {end_date}"
            }

        except Exception as e:
            raise GarminAPIError(f"Failed to sync HRV data: {str(e)}")

    def sync_sleep_data(self, days: int = 30) -> Dict:
        """Sync sleep data from Garmin Connect.

        Args:
            days: Number of days to sync (default 30)

        Returns:
            Dictionary with sync results
        """
        if not self.auth.is_authenticated():
            raise GarminAPIError("Not authenticated with Garmin. Run auth first.")

        try:
            oauth_session = self.auth.get_authenticated_session()
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)

            synced_count = 0
            updated_count = 0

            # Fetch sleep data day by day
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')

                # Get sleep data for the day
                sleep_url = f"{self.base_url}/wellness-service/wellness/dailySleepData/{self.user_id}"
                params = {"date": date_str}
                response = oauth_session.get(sleep_url, params=params)

                if response.status_code == 200:
                    sleep_data = response.json()
                    if sleep_data and self._has_valid_sleep_data(sleep_data):
                        if self._store_sleep_data(current_date, sleep_data):
                            synced_count += 1
                        else:
                            updated_count += 1
                elif response.status_code == 404:
                    # No sleep data for this day - normal
                    pass
                else:
                    print(f"Warning: Failed to get sleep data for {date_str}: {response.status_code}")

                current_date += timedelta(days=1)

            return {
                "type": "sleep",
                "synced": synced_count,
                "updated": updated_count,
                "date_range": f"{start_date} to {end_date}"
            }

        except Exception as e:
            raise GarminAPIError(f"Failed to sync sleep data: {str(e)}")

    def sync_wellness_data(self, days: int = 30) -> Dict:
        """Sync wellness data (stress, body battery, etc.) from Garmin Connect.

        Args:
            days: Number of days to sync (default 30)

        Returns:
            Dictionary with sync results
        """
        if not self.auth.is_authenticated():
            raise GarminAPIError("Not authenticated with Garmin. Run auth first.")

        try:
            oauth_session = self.auth.get_authenticated_session()
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)

            synced_count = 0
            updated_count = 0

            # Fetch wellness data day by day
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')

                # Get wellness summary for the day
                wellness_url = f"{self.base_url}/wellness-service/wellness/dailySummaryChart/{self.user_id}"
                params = {"date": date_str}
                response = oauth_session.get(wellness_url, params=params)

                if response.status_code == 200:
                    wellness_data = response.json()
                    if wellness_data and self._has_valid_wellness_data(wellness_data):
                        if self._store_wellness_data(current_date, wellness_data):
                            synced_count += 1
                        else:
                            updated_count += 1
                elif response.status_code == 404:
                    # No wellness data for this day - normal
                    pass
                else:
                    print(f"Warning: Failed to get wellness data for {date_str}: {response.status_code}")

                current_date += timedelta(days=1)

            return {
                "type": "wellness",
                "synced": synced_count,
                "updated": updated_count,
                "date_range": f"{start_date} to {end_date}"
            }

        except Exception as e:
            raise GarminAPIError(f"Failed to sync wellness data: {str(e)}")

    def sync_all(self, days: int = 30) -> List[Dict]:
        """Sync all wellness data types.

        Args:
            days: Number of days to sync (default 30)

        Returns:
            List of sync results for each data type
        """
        results = []

        try:
            results.append(self.sync_hrv_data(days))
        except Exception as e:
            results.append({"type": "hrv", "error": str(e)})

        try:
            results.append(self.sync_sleep_data(days))
        except Exception as e:
            results.append({"type": "sleep", "error": str(e)})

        try:
            results.append(self.sync_wellness_data(days))
        except Exception as e:
            results.append({"type": "wellness", "error": str(e)})

        return results

    def _has_valid_hrv_data(self, data: Dict) -> bool:
        """Check if HRV data contains valid measurements."""
        return bool(
            data.get("hrvRmssd") or
            data.get("hrvScore") or
            data.get("hrvStatus")
        )

    def _has_valid_sleep_data(self, data: Dict) -> bool:
        """Check if sleep data contains valid measurements."""
        return bool(
            data.get("totalSleepTimeSeconds") or
            data.get("sleepScore") or
            data.get("sleepStartTimestampGMT")
        )

    def _has_valid_wellness_data(self, data: Dict) -> bool:
        """Check if wellness data contains valid measurements."""
        return bool(
            data.get("stressAvgValue") or
            data.get("bodyBatteryChargedValue") or
            data.get("totalSteps")
        )

    def _store_hrv_data(self, date: datetime.date, data: Dict) -> bool:
        """Store HRV data in database.

        Returns:
            True if new record created, False if updated existing
        """
        with self.db.get_session() as session:
            # Check if record already exists
            existing = session.query(HRVData).filter_by(
                user_id=self.user_id,
                date=datetime.combine(date, datetime.min.time())
            ).first()

            if existing:
                # Update existing record
                self._update_hrv_record(existing, data)
                session.commit()
                return False
            else:
                # Create new record
                hrv_record = HRVData(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time()),
                    hrv_rmssd=data.get("hrvRmssd"),
                    hrv_score=data.get("hrvScore"),
                    hrv_status=data.get("hrvStatus"),
                    stress_level=data.get("stressLevel"),
                    stress_qualifier=data.get("stressQualifier"),
                    recovery_advisor=data.get("recoveryAdvisor"),
                    baseline_low_upper=data.get("baselineLowUpper"),
                    baseline_balanced_lower=data.get("baselineBalancedLower"),
                    baseline_balanced_upper=data.get("baselineBalancedUpper"),
                    measurement_timestamp=self._parse_timestamp(data.get("measurementTimestampGMT")),
                    raw_data=json.dumps(data)
                )
                session.add(hrv_record)
                session.commit()
                return True

    def _store_sleep_data(self, date: datetime.date, data: Dict) -> bool:
        """Store sleep data in database.

        Returns:
            True if new record created, False if updated existing
        """
        with self.db.get_session() as session:
            # Check if record already exists
            existing = session.query(SleepData).filter_by(
                user_id=self.user_id,
                date=datetime.combine(date, datetime.min.time())
            ).first()

            if existing:
                # Update existing record
                self._update_sleep_record(existing, data)
                session.commit()
                return False
            else:
                # Create new record
                sleep_record = SleepData(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time()),
                    total_sleep_time=data.get("totalSleepTimeSeconds"),
                    deep_sleep_time=data.get("deepSleepSeconds"),
                    light_sleep_time=data.get("lightSleepSeconds"),
                    rem_sleep_time=data.get("remSleepSeconds"),
                    awake_time=data.get("awakeTimeSeconds"),
                    sleep_score=data.get("sleepScore"),
                    sleep_efficiency=data.get("sleepEfficiency"),
                    sleep_latency=data.get("sleepLatency"),
                    bedtime=self._parse_timestamp(data.get("bedTimeTimestampGMT")),
                    sleep_start_time=self._parse_timestamp(data.get("sleepStartTimestampGMT")),
                    sleep_end_time=self._parse_timestamp(data.get("sleepEndTimestampGMT")),
                    wake_time=self._parse_timestamp(data.get("wakeTimestampGMT")),
                    restlessness=data.get("restlessness"),
                    interruptions=data.get("interruptions"),
                    average_heart_rate=data.get("averageHeartRate"),
                    lowest_heart_rate=data.get("lowestHeartRate"),
                    deep_sleep_pct=data.get("deepSleepPercentage"),
                    light_sleep_pct=data.get("lightSleepPercentage"),
                    rem_sleep_pct=data.get("remSleepPercentage"),
                    awake_pct=data.get("awakePercentage"),
                    overnight_hrv=data.get("overnightHrv"),
                    sleep_stress=data.get("sleepStress"),
                    recovery_score=data.get("recoveryScore"),
                    raw_data=json.dumps(data)
                )
                session.add(sleep_record)
                session.commit()
                return True

    def _store_wellness_data(self, date: datetime.date, data: Dict) -> bool:
        """Store wellness data in database.

        Returns:
            True if new record created, False if updated existing
        """
        with self.db.get_session() as session:
            # Check if record already exists
            existing = session.query(WellnessData).filter_by(
                user_id=self.user_id,
                date=datetime.combine(date, datetime.min.time())
            ).first()

            if existing:
                # Update existing record
                self._update_wellness_record(existing, data)
                session.commit()
                return False
            else:
                # Create new record
                wellness_record = WellnessData(
                    user_id=self.user_id,
                    date=datetime.combine(date, datetime.min.time()),
                    stress_avg=data.get("stressAvgValue"),
                    stress_max=data.get("stressMaxValue"),
                    stress_qualifier=data.get("stressQualifier"),
                    rest_stress_duration=data.get("restStressDuration"),
                    low_stress_duration=data.get("lowStressDuration"),
                    medium_stress_duration=data.get("mediumStressDuration"),
                    high_stress_duration=data.get("highStressDuration"),
                    stress_duration=data.get("stressDuration"),
                    body_battery_charged=data.get("bodyBatteryChargedValue"),
                    body_battery_drained=data.get("bodyBatteryDrainedValue"),
                    body_battery_highest=data.get("bodyBatteryHighestValue"),
                    body_battery_lowest=data.get("bodyBatteryLowestValue"),
                    avg_respiration=data.get("avgRespirationValue"),
                    max_respiration=data.get("maxRespirationValue"),
                    min_respiration=data.get("minRespirationValue"),
                    avg_spo2=data.get("avgSpo2Value"),
                    lowest_spo2=data.get("lowestSpo2Value"),
                    total_steps=data.get("totalSteps"),
                    goal_steps=data.get("goalSteps"),
                    calories_goal=data.get("caloriesGoal"),
                    calories_bmr=data.get("caloriesBMR"),
                    calories_active=data.get("caloriesActive"),
                    calories_consumed=data.get("caloriesConsumed"),
                    total_distance=data.get("totalDistanceMeters"),
                    floors_ascended=data.get("floorsAscended"),
                    floors_descended=data.get("floorsDescended"),
                    moderate_intensity_minutes=data.get("moderateIntensityMinutes"),
                    vigorous_intensity_minutes=data.get("vigorousIntensityMinutes"),
                    raw_data=json.dumps(data)
                )
                session.add(wellness_record)
                session.commit()
                return True

    def _update_hrv_record(self, record: HRVData, data: Dict):
        """Update existing HRV record with new data."""
        record.hrv_rmssd = data.get("hrvRmssd") or record.hrv_rmssd
        record.hrv_score = data.get("hrvScore") or record.hrv_score
        record.hrv_status = data.get("hrvStatus") or record.hrv_status
        record.stress_level = data.get("stressLevel") or record.stress_level
        record.stress_qualifier = data.get("stressQualifier") or record.stress_qualifier
        record.recovery_advisor = data.get("recoveryAdvisor") or record.recovery_advisor
        record.baseline_low_upper = data.get("baselineLowUpper") or record.baseline_low_upper
        record.baseline_balanced_lower = data.get("baselineBalancedLower") or record.baseline_balanced_lower
        record.baseline_balanced_upper = data.get("baselineBalancedUpper") or record.baseline_balanced_upper
        record.measurement_timestamp = self._parse_timestamp(data.get("measurementTimestampGMT")) or record.measurement_timestamp
        record.raw_data = json.dumps(data)

    def _update_sleep_record(self, record: SleepData, data: Dict):
        """Update existing sleep record with new data."""
        record.total_sleep_time = data.get("totalSleepTimeSeconds") or record.total_sleep_time
        record.deep_sleep_time = data.get("deepSleepSeconds") or record.deep_sleep_time
        record.light_sleep_time = data.get("lightSleepSeconds") or record.light_sleep_time
        record.rem_sleep_time = data.get("remSleepSeconds") or record.rem_sleep_time
        record.awake_time = data.get("awakeTimeSeconds") or record.awake_time
        record.sleep_score = data.get("sleepScore") or record.sleep_score
        record.sleep_efficiency = data.get("sleepEfficiency") or record.sleep_efficiency
        record.sleep_latency = data.get("sleepLatency") or record.sleep_latency
        record.bedtime = self._parse_timestamp(data.get("bedTimeTimestampGMT")) or record.bedtime
        record.sleep_start_time = self._parse_timestamp(data.get("sleepStartTimestampGMT")) or record.sleep_start_time
        record.sleep_end_time = self._parse_timestamp(data.get("sleepEndTimestampGMT")) or record.sleep_end_time
        record.wake_time = self._parse_timestamp(data.get("wakeTimestampGMT")) or record.wake_time
        record.restlessness = data.get("restlessness") or record.restlessness
        record.interruptions = data.get("interruptions") or record.interruptions
        record.average_heart_rate = data.get("averageHeartRate") or record.average_heart_rate
        record.lowest_heart_rate = data.get("lowestHeartRate") or record.lowest_heart_rate
        record.deep_sleep_pct = data.get("deepSleepPercentage") or record.deep_sleep_pct
        record.light_sleep_pct = data.get("lightSleepPercentage") or record.light_sleep_pct
        record.rem_sleep_pct = data.get("remSleepPercentage") or record.rem_sleep_pct
        record.awake_pct = data.get("awakePercentage") or record.awake_pct
        record.overnight_hrv = data.get("overnightHrv") or record.overnight_hrv
        record.sleep_stress = data.get("sleepStress") or record.sleep_stress
        record.recovery_score = data.get("recoveryScore") or record.recovery_score
        record.raw_data = json.dumps(data)

    def _update_wellness_record(self, record: WellnessData, data: Dict):
        """Update existing wellness record with new data."""
        record.stress_avg = data.get("stressAvgValue") or record.stress_avg
        record.stress_max = data.get("stressMaxValue") or record.stress_max
        record.stress_qualifier = data.get("stressQualifier") or record.stress_qualifier
        record.rest_stress_duration = data.get("restStressDuration") or record.rest_stress_duration
        record.low_stress_duration = data.get("lowStressDuration") or record.low_stress_duration
        record.medium_stress_duration = data.get("mediumStressDuration") or record.medium_stress_duration
        record.high_stress_duration = data.get("highStressDuration") or record.high_stress_duration
        record.stress_duration = data.get("stressDuration") or record.stress_duration
        record.body_battery_charged = data.get("bodyBatteryChargedValue") or record.body_battery_charged
        record.body_battery_drained = data.get("bodyBatteryDrainedValue") or record.body_battery_drained
        record.body_battery_highest = data.get("bodyBatteryHighestValue") or record.body_battery_highest
        record.body_battery_lowest = data.get("bodyBatteryLowestValue") or record.body_battery_lowest
        record.avg_respiration = data.get("avgRespirationValue") or record.avg_respiration
        record.max_respiration = data.get("maxRespirationValue") or record.max_respiration
        record.min_respiration = data.get("minRespirationValue") or record.min_respiration
        record.avg_spo2 = data.get("avgSpo2Value") or record.avg_spo2
        record.lowest_spo2 = data.get("lowestSpo2Value") or record.lowest_spo2
        record.total_steps = data.get("totalSteps") or record.total_steps
        record.goal_steps = data.get("goalSteps") or record.goal_steps
        record.calories_goal = data.get("caloriesGoal") or record.calories_goal
        record.calories_bmr = data.get("caloriesBMR") or record.calories_bmr
        record.calories_active = data.get("caloriesActive") or record.calories_active
        record.calories_consumed = data.get("caloriesConsumed") or record.calories_consumed
        record.total_distance = data.get("totalDistanceMeters") or record.total_distance
        record.floors_ascended = data.get("floorsAscended") or record.floors_ascended
        record.floors_descended = data.get("floorsDescended") or record.floors_descended
        record.moderate_intensity_minutes = data.get("moderateIntensityMinutes") or record.moderate_intensity_minutes
        record.vigorous_intensity_minutes = data.get("vigorousIntensityMinutes") or record.vigorous_intensity_minutes
        record.raw_data = json.dumps(data)

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse Garmin timestamp string to datetime."""
        if not timestamp_str:
            return None

        try:
            # Garmin timestamps are typically in milliseconds
            if isinstance(timestamp_str, (int, float)):
                return datetime.fromtimestamp(timestamp_str / 1000)
            elif isinstance(timestamp_str, str) and timestamp_str.isdigit():
                return datetime.fromtimestamp(int(timestamp_str) / 1000)
            else:
                # Try parsing as ISO format
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None

    def get_latest_hrv_score(self) -> Optional[float]:
        """Get the latest HRV score for quick analysis."""
        with self.db.get_session() as session:
            latest_hrv = session.query(HRVData).filter_by(
                user_id=self.user_id
            ).order_by(HRVData.date.desc()).first()

            return latest_hrv.hrv_score if latest_hrv else None

    def get_latest_sleep_score(self) -> Optional[float]:
        """Get the latest sleep score for quick analysis."""
        with self.db.get_session() as session:
            latest_sleep = session.query(SleepData).filter_by(
                user_id=self.user_id
            ).order_by(SleepData.date.desc()).first()

            return latest_sleep.sleep_score if latest_sleep else None

    def get_wellness_trend(self, days: int = 7) -> Dict:
        """Get wellness trend over specified days."""
        with self.db.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Get recent wellness data
            wellness_data = session.query(WellnessData).filter(
                WellnessData.user_id == self.user_id,
                WellnessData.date >= cutoff_date
            ).order_by(WellnessData.date.desc()).all()

            if not wellness_data:
                return {"status": "no_data"}

            # Calculate averages
            stress_values = [w.stress_avg for w in wellness_data if w.stress_avg]
            sleep_scores = []

            # Get corresponding sleep scores
            for w in wellness_data:
                sleep_record = session.query(SleepData).filter(
                    SleepData.user_id == self.user_id,
                    SleepData.date == w.date
                ).first()
                if sleep_record and sleep_record.sleep_score:
                    sleep_scores.append(sleep_record.sleep_score)

            return {
                "status": "success",
                "days": days,
                "avg_stress": sum(stress_values) / len(stress_values) if stress_values else None,
                "avg_sleep_score": sum(sleep_scores) / len(sleep_scores) if sleep_scores else None,
                "stress_trend": "improving" if len(stress_values) > 1 and stress_values[0] < stress_values[-1] else "stable",
                "records_count": len(wellness_data)
            }


def get_garmin_client(user_id: str = "default") -> GarminClient:
    """Get Garmin client for user."""
    return GarminClient(user_id)