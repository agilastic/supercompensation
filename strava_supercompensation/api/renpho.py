"""
RENPHO cloud API client for body composition data.
This client implements a resilient authentication strategy with an automatic fallback
to handle potential instability in the unofficial RENPHO API.
Based on the working hass_renpho v3.0.1.1 component.
"""

import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import requests

from ..db import get_db
from ..db.models import BodyComposition

class RenphoError(Exception):
    """Custom exception for RENPHO-related errors."""
    pass

class RenphoClient:
    """A self-contained and resilient client for the RENPHO cloud API."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()
        self.email = os.getenv("RENPHO_EMAIL")
        self.password = os.getenv("RENPHO_PASSWORD")

        if not self.email or not self.password:
            raise RenphoError("RENPHO_EMAIL and RENPHO_PASSWORD must be set in your .env file.")

        # Remove quotes if present in environment variables
        self.email = self.email.strip('"')
        self.password = self.password.strip('"')

        self.base_url = "https://renpho.qnclouds.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Renpho/2.2.3 (iPhone; iOS 14.4; Scale/2.2.3)",
            "Accept-Language": "en-US",
            "app_id": "Renpho",
            "Content-Type": "application/json",
        })
        self.auth_data = {}
        self.api_version_used = None

    def _md5_hash(self, text: str) -> str:
        """Create an MD5 hash as required by the RENPHO API."""
        return hashlib.md5(text.encode()).hexdigest()

    def connect(self):
        """Authenticate with the RENPHO API, with fallback logic."""
        # --- Attempt 1: Modern v3 Endpoint ---
        try:
            secure_hash = self._md5_hash(self.password)
            password_md5 = self._md5_hash(f"da2156a3456487a2a0935b3783474415{secure_hash}renpho")
            payload_v3 = {
                "secure_flag": "1",
                "email": self.email,
                "password": password_md5,
                "app_id": "Renpho",
                "client_id": "Renpho",
                "client_secret": "206206d206f6424e93d3032e3a7f342d",
            }
            response = self.session.post(f"{self.base_url}/api/v3/users/sign_in.json", json=payload_v3, timeout=10)

            if 500 <= response.status_code < 600:
                print("RENPHO v3 API returned a server error. Attempting fallback to v1 API...")
            else:
                response.raise_for_status()
                data = response.json()

                # Check for authentication errors
                if data.get("status_code") == "50000":
                    error_msg = data.get("status_message", "Authentication failed")
                    if "not registered" in error_msg or "does not exist" in error_msg:
                        raise RenphoError(f"Account not registered: {error_msg}")

                if "terminal_user_session_token" in data:
                    self.auth_data = data
                    self.session.headers.update({"terminal-user-session-token": data["terminal_user_session_token"]})
                    self.api_version_used = "v3"
                    print("✅ Successfully connected to RENPHO v3 API")
                    return
        except RenphoError:
            raise  # Re-raise account not registered errors
        except requests.RequestException:
            pass  # Silently fail to allow fallback

        # --- Attempt 2: Fallback v1 Endpoint (CORRECTED - no /v1/ in path) ---
        print("Trying RENPHO v1 API fallback...")
        try:
            password_md5_v1 = self._md5_hash(self.password)
            # CORRECTED: Using /api/users/sign_in.json instead of /api/v1/users/sign_in.json
            payload_v1 = {
                "email": self.email,
                "password": password_md5_v1,
                "app_id": "Renpho"
            }
            response = self.session.post(f"{self.base_url}/api/users/sign_in.json", json=payload_v1, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for authentication errors
            if data.get("status_code") == "50000":
                error_msg = data.get("status_message", "Authentication failed")
                raise RenphoError(f"Login failed: {error_msg}")

            if "terminal_user_session_token" in data:
                self.auth_data = data
                self.session.headers.update({"terminal-user-session-token": data["terminal_user_session_token"]})
                self.api_version_used = "v1"
                print("✅ Successfully connected to RENPHO v1 API")
                return
        except requests.RequestException as e:
            print(f"v1 API fallback failed: {e}")

        raise RenphoError(
            "Authentication failed on all endpoints. Please check:\n"
            "1. Your RENPHO account is registered (use the mobile app first)\n"
            "2. Your email and password in .env are correct\n"
            "3. The RENPHO API may be temporarily unavailable"
        )

    def get_measurements(self) -> List[dict]:
        """Fetch all available body composition measurements."""
        if not self.auth_data:
            self.connect()

        user_id = self.auth_data.get("id")
        if not user_id:
            raise RenphoError("Could not determine user ID after login.")

        try:
            # Use the appropriate endpoint based on API version
            if self.api_version_used == "v3":
                endpoint = f"{self.base_url}/api/v2/measurements/list.json?user_id={user_id}"
                response = self.session.get(endpoint, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get("measurements", [])
            elif self.api_version_used == "v1":
                # Try multiple possible v1 endpoints
                endpoints = [
                    f"{self.base_url}/api/v1/scale_users/list_product_funk.json?user_id={user_id}",
                    f"{self.base_url}/api/measurements/list.json?user_id={user_id}"
                ]

                for endpoint in endpoints:
                    try:
                        response = self.session.get(endpoint, timeout=10)
                        if response.status_code == 404:
                            continue
                        response.raise_for_status()
                        data = response.json()

                        # Handle different response formats
                        if "scale_users" in data and len(data["scale_users"]) > 0:
                            return data["scale_users"][0].get("last_ary", [])
                        elif "measurements" in data:
                            return data["measurements"]
                        elif "data" in data:
                            return data["data"]
                    except:
                        continue

            return []
        except requests.RequestException as e:
            raise RenphoError(f"Failed to fetch measurements: {e}")

    def sync_measurements(self, days_back: int = 30) -> Dict[str, int]:
        """
        Fetch and sync body composition measurements to database.

        Args:
            days_back: Number of days to sync (default: 30)

        Returns:
            Dictionary with sync results
        """
        results = {
            "new_measurements": 0,
            "updated_measurements": 0,
            "total_fetched": 0,
            "errors": 0
        }

        try:
            measurements = self.get_measurements()
            results["total_fetched"] = len(measurements)

            if not measurements:
                print("No measurements found in RENPHO account")
                return results

            # Filter measurements to specified time range
            cutoff_date = datetime.now() - timedelta(days=days_back)

            with self.db.get_session() as session:
                for measurement_data in measurements:
                    try:
                        # Handle different timestamp formats
                        timestamp_ms = measurement_data.get("created_at_timestamp")
                        timestamp_s = measurement_data.get("measure_time_stamp") or measurement_data.get("time_stamp")
                        timestamp_str = measurement_data.get("created_at") or measurement_data.get("measure_time")

                        if timestamp_ms:
                            measurement_time = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
                        elif timestamp_s:
                            measurement_time = datetime.fromtimestamp(int(timestamp_s), tz=timezone.utc)
                        elif timestamp_str:
                            # Try parsing ISO format
                            try:
                                measurement_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            except:
                                continue
                        else:
                            continue

                        # Convert to naive datetime for database storage
                        measurement_time = measurement_time.replace(tzinfo=None)

                        # Skip measurements older than cutoff
                        if measurement_time < cutoff_date:
                            continue

                        # Check if measurement already exists
                        existing = session.query(BodyComposition).filter_by(
                            date=measurement_time,
                            user_id=self.user_id
                        ).first()

                        if existing:
                            # Update existing record with new data
                            self._update_body_composition(existing, measurement_data)
                            results["updated_measurements"] += 1
                        else:
                            # Create new record
                            new_record = self._create_body_composition(measurement_data, measurement_time)
                            session.add(new_record)
                            results["new_measurements"] += 1

                    except Exception as e:
                        results["errors"] += 1
                        print(f"Error processing measurement: {e}")

                session.commit()

        except RenphoError:
            raise
        except Exception as e:
            raise RenphoError(f"Database sync failed: {e}")

        return results

    def _create_body_composition(self, measurement_data: Dict, measurement_time: datetime) -> BodyComposition:
        """Create a BodyComposition record from RENPHO measurement data."""
        return BodyComposition(
            user_id=self.user_id,
            date=measurement_time,
            weight_kg=measurement_data.get('weight') or measurement_data.get('weight_kg'),
            bmi=measurement_data.get('bmi'),
            muscle_mass_percent=measurement_data.get('muscle_mass') or measurement_data.get('muscle'),
            bone_mass_percent=measurement_data.get('bone_mass') or measurement_data.get('bone'),
            body_fat_percent=measurement_data.get('body_fat') or measurement_data.get('bodyfat'),
            water_percent=measurement_data.get('body_water') or measurement_data.get('water'),
            basal_metabolism_kcal=measurement_data.get('bmr'),
            visceral_fat=measurement_data.get('visceral_fat') or measurement_data.get('vfat'),
            protein_percent=measurement_data.get('protein'),
            metabolic_age=measurement_data.get('metabolic_age') or measurement_data.get('bodyage'),
            skeletal_muscle_percent=measurement_data.get('skeletal_muscle') or measurement_data.get('muscle'),
            raw_data=json.dumps(measurement_data, default=str)
        )

    def _update_body_composition(self, record: BodyComposition, measurement_data: Dict) -> None:
        """Update an existing BodyComposition record with new measurement data."""
        record.weight_kg = measurement_data.get('weight') or measurement_data.get('weight_kg') or record.weight_kg
        record.bmi = measurement_data.get('bmi') or record.bmi
        record.muscle_mass_percent = measurement_data.get('muscle_mass') or measurement_data.get('muscle') or record.muscle_mass_percent
        record.bone_mass_percent = measurement_data.get('bone_mass') or measurement_data.get('bone') or record.bone_mass_percent
        record.body_fat_percent = measurement_data.get('body_fat') or measurement_data.get('bodyfat') or record.body_fat_percent
        record.water_percent = measurement_data.get('body_water') or measurement_data.get('water') or record.water_percent
        record.basal_metabolism_kcal = measurement_data.get('bmr') or record.basal_metabolism_kcal
        record.visceral_fat = measurement_data.get('visceral_fat') or measurement_data.get('vfat') or record.visceral_fat
        record.protein_percent = measurement_data.get('protein') or record.protein_percent
        record.metabolic_age = measurement_data.get('metabolic_age') or measurement_data.get('bodyage') or record.metabolic_age
        record.skeletal_muscle_percent = measurement_data.get('skeletal_muscle') or measurement_data.get('muscle') or record.skeletal_muscle_percent
        record.raw_data = json.dumps(measurement_data, default=str)
        record.updated_at = datetime.now(timezone.utc)

    def get_latest_measurement(self) -> Optional[BodyComposition]:
        """Get the most recent body composition measurement."""
        with self.db.get_session() as session:
            return session.query(BodyComposition)\
                .filter_by(user_id=self.user_id)\
                .order_by(BodyComposition.date.desc())\
                .first()

    def get_trend_analysis(self, days: int = 30) -> Dict:
        """
        Analyze body composition trends over specified period.

        Returns metrics critical for training periodization:
        - Weight stability (hydration/recovery indicator)
        - Muscle mass trend (adaptation indicator)
        - Body fat trend (performance indicator)
        - Metabolic efficiency (BMR changes)
        """
        with self.db.get_session() as session:
            measurements = session.query(BodyComposition)\
                .filter_by(user_id=self.user_id)\
                .filter(BodyComposition.date >= datetime.now() - timedelta(days=days))\
                .order_by(BodyComposition.date)\
                .all()

            if len(measurements) < 2:
                return {"status": "insufficient_data", "measurements_count": len(measurements)}

            # Calculate trends
            first = measurements[0]
            latest = measurements[-1]

            return {
                "status": "success",
                "period_days": days,
                "measurements_count": len(measurements),
                "weight_change_kg": latest.weight_kg - first.weight_kg if (latest.weight_kg and first.weight_kg) else None,
                "muscle_mass_change": latest.muscle_mass_percent - first.muscle_mass_percent if (latest.muscle_mass_percent and first.muscle_mass_percent) else None,
                "body_fat_change": latest.body_fat_percent - first.body_fat_percent if (latest.body_fat_percent and first.body_fat_percent) else None,
                "bmr_change_kcal": latest.basal_metabolism_kcal - first.basal_metabolism_kcal if (latest.basal_metabolism_kcal and first.basal_metabolism_kcal) else None,
                "latest_measurement": {
                    "date": latest.date,
                    "weight_kg": latest.weight_kg,
                    "body_fat_percent": latest.body_fat_percent,
                    "muscle_mass_percent": latest.muscle_mass_percent,
                    "metabolic_age": latest.metabolic_age
                }
            }