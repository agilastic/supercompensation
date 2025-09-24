"""
RENPHO client using session token authentication.
This is an alternative client that uses a captured session token
instead of email/password authentication.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

from ..db import get_db
from ..db.models import BodyComposition


class RenphoTokenClient:
    """RENPHO client using direct session token authentication."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

        # Get token from environment variable
        self.session_token = os.getenv("RENPHO_SESSION_TOKEN")

        if not self.session_token:
            raise ValueError(
                "RENPHO_SESSION_TOKEN must be set in your .env file.\n"
                "To get your token:\n"
                "1. Install mitmproxy: brew install mitmproxy\n"
                "2. Run: mitmproxy\n"
                "3. Configure your phone's WiFi proxy to your Mac's IP:8080\n"
                "4. Open RENPHO app and sync\n"
                "5. Look for 'terminal-user-session-token' in the requests\n"
                "6. Add to .env: RENPHO_SESSION_TOKEN=your_token_here"
            )

        self.base_url = "https://renpho.qnclouds.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Renpho/3.0.0 (iPhone; iOS 14.4; Scale/3.0.0)",
            "terminal-user-session-token": self.session_token.strip('"'),
            "Content-Type": "application/json",
        })
        self.user_info = None

    def get_user_info(self) -> Dict:
        """Get user information to verify token works."""
        try:
            # Try to get user info to verify token
            response = self.session.get(f"{self.base_url}/api/v3/users/info.json", timeout=10)
            response.raise_for_status()
            self.user_info = response.json()
            return self.user_info
        except Exception as e:
            raise Exception(f"Token validation failed: {e}")

    def get_measurements(self) -> List[Dict]:
        """Fetch body composition measurements using session token."""
        try:
            # Get user ID from token if not already fetched
            if not self.user_info:
                self.get_user_info()

            user_id = self.user_info.get("id")
            if not user_id:
                # Try without user_id parameter
                response = self.session.get(f"{self.base_url}/api/v2/measurements/list.json", timeout=10)
            else:
                response = self.session.get(f"{self.base_url}/api/v2/measurements/list.json?user_id={user_id}", timeout=10)

            response.raise_for_status()
            data = response.json()

            # Handle different response formats
            if "measurements" in data:
                return data["measurements"]
            elif "last_ary" in data:
                return data["last_ary"]
            elif isinstance(data, list):
                return data
            else:
                return []

        except Exception as e:
            raise Exception(f"Failed to fetch measurements: {e}")

    def sync_measurements(self, days_back: int = 30) -> Dict[str, int]:
        """Sync measurements to database."""
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
                print("No measurements found")
                return results

            cutoff_date = datetime.now() - timedelta(days=days_back)

            with self.db.get_session() as session:
                for measurement_data in measurements:
                    try:
                        # Parse timestamp
                        timestamp_ms = measurement_data.get("time_stamp") or measurement_data.get("created_at_timestamp")
                        timestamp_str = measurement_data.get("created_at") or measurement_data.get("measure_time")

                        if timestamp_ms:
                            measurement_time = datetime.fromtimestamp(int(timestamp_ms) / 1000.0)
                        elif timestamp_str:
                            measurement_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            continue

                        if measurement_time < cutoff_date:
                            continue

                        # Check if exists
                        existing = session.query(BodyComposition).filter_by(
                            date=measurement_time,
                            user_id=self.user_id
                        ).first()

                        if not existing:
                            new_record = BodyComposition(
                                user_id=self.user_id,
                                date=measurement_time,
                                weight_kg=measurement_data.get('weight'),
                                bmi=measurement_data.get('bmi'),
                                muscle_mass_percent=measurement_data.get('muscle') or measurement_data.get('muscle_mass'),
                                bone_mass_percent=measurement_data.get('bone') or measurement_data.get('bone_mass'),
                                body_fat_percent=measurement_data.get('bodyfat') or measurement_data.get('body_fat'),
                                water_percent=measurement_data.get('water') or measurement_data.get('body_water'),
                                basal_metabolism_kcal=measurement_data.get('bmr'),
                                visceral_fat=measurement_data.get('vfat') or measurement_data.get('visceral_fat'),
                                protein_percent=measurement_data.get('protein'),
                                metabolic_age=measurement_data.get('bodyage') or measurement_data.get('metabolic_age'),
                                skeletal_muscle_percent=measurement_data.get('muscle'),
                                raw_data=json.dumps(measurement_data, default=str)
                            )
                            session.add(new_record)
                            results["new_measurements"] += 1
                        else:
                            results["updated_measurements"] += 1

                    except Exception as e:
                        results["errors"] += 1
                        print(f"Error processing measurement: {e}")

                session.commit()

        except Exception as e:
            raise Exception(f"Sync failed: {e}")

        return results


def test_token_auth():
    """Test function to verify token authentication works."""
    try:
        client = RenphoTokenClient()
        print("✅ Token client initialized")

        user_info = client.get_user_info()
        print(f"✅ User verified: {user_info.get('email', 'Unknown')}")

        measurements = client.get_measurements()
        print(f"✅ Found {len(measurements)} measurements")

        if measurements:
            latest = measurements[0]
            print(f"   Latest: {latest.get('weight', 'N/A')}kg on {latest.get('created_at', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Token auth failed: {e}")
        return False