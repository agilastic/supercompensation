"""
RENPHO CSV data import and analysis for athletic training.
Focus on sports science metrics critical for performance optimization.
"""

import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from ..db import get_db
from ..db.models import BodyComposition


class RenphoCsvImporter:
    """Import and analyze RENPHO data from CSV export for athletic training."""

    def __init__(self, csv_path: str = None, user_id: str = "default"):
        self.csv_path = csv_path
        self.user_id = user_id
        self.db = get_db()

    @staticmethod
    def discover_renpho_csvs(directory: str = ".") -> List[str]:
        """Discover RENPHO CSV files in the given directory by checking column structure."""
        renpho_files = []

        # Expected RENPHO columns (German and English variations)
        required_columns = {
            'german': ['Datum', 'Zeitraum', 'Gewicht(kg)', 'BMI', 'Körperfett(%)'],
            'english': ['Date', 'Time', 'Weight(kg)', 'BMI', 'Body Fat(%)']
        }

        # Find all CSV files
        csv_files = glob.glob(os.path.join(directory, "*.csv"))

        for csv_file in csv_files:
            try:
                # Read first few rows to check columns
                df = pd.read_csv(csv_file, nrows=1, encoding='utf-8')
                columns = [col.strip() for col in df.columns]

                # Check if it matches RENPHO structure
                is_german_renpho = all(col in columns for col in required_columns['german'])
                is_english_renpho = all(col in columns for col in required_columns['english'])

                if is_german_renpho or is_english_renpho:
                    renpho_files.append(csv_file)

            except Exception:
                continue

        return renpho_files

    @staticmethod
    def auto_import_all(directory: str = ".", user_id: str = "default") -> Dict[str, Dict]:
        """Auto-discover and import all RENPHO CSV files in directory."""
        discovered_files = RenphoCsvImporter.discover_renpho_csvs(directory)
        results = {}

        if not discovered_files:
            return {"status": "no_files_found", "message": "No RENPHO CSV files found in directory"}

        for csv_file in discovered_files:
            try:
                importer = RenphoCsvImporter(csv_file, user_id)
                file_results = importer.import_csv_data()
                results[os.path.basename(csv_file)] = {
                    "status": "success",
                    "results": file_results
                }
            except Exception as e:
                results[os.path.basename(csv_file)] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def import_csv_data(self) -> Dict[str, int]:
        """
        Import RENPHO CSV data with focus on athletic training metrics.

        Returns:
            Dictionary with import results
        """
        results = {
            "new_measurements": 0,
            "updated_measurements": 0,
            "total_processed": 0,
            "errors": 0
        }

        try:
            # Read German CSV format with proper handling of spaces
            df = pd.read_csv(self.csv_path, encoding='utf-8')

            # Clean column names by stripping spaces
            df.columns = df.columns.str.strip()
            results["total_processed"] = len(df)

            with self.db.get_session() as session:
                for _, row in df.iterrows():
                    try:
                        # Parse German date format: 24.09.25,07:05:48
                        date_str = f"{row['Datum']},{row['Zeitraum']}"
                        measurement_time = datetime.strptime(date_str, "%d.%m.%y,%H:%M:%S")

                        # Check if measurement already exists
                        existing = session.query(BodyComposition).filter_by(
                            date=measurement_time,
                            user_id=self.user_id
                        ).first()

                        # Extract values, handle "--" as None
                        def safe_float(value):
                            if value == "--" or pd.isna(value):
                                return None
                            return float(str(value).replace(',', '.'))

                        if existing:
                            # Update existing record
                            existing.weight_kg = safe_float(row['Gewicht(kg)'])
                            existing.bmi = safe_float(row['BMI'])
                            existing.body_fat_percent = safe_float(row['Körperfett(%)'])
                            existing.skeletal_muscle_percent = safe_float(row['Skelettmuskel(%)'])
                            existing.water_percent = safe_float(row['Körperwasser(%)'])
                            existing.muscle_mass_percent = self._calculate_muscle_percent(
                                safe_float(row['Muskelmasse(kg)']),
                                safe_float(row['Gewicht(kg)'])
                            )
                            existing.basal_metabolism_kcal = safe_float(row['Grundumsatz(kcal)'])
                            existing.visceral_fat = safe_float(row['Viszeralfett'])
                            existing.protein_percent = safe_float(row['Eiweiß (%)'])
                            existing.metabolic_age = safe_float(row['Stoffwechselalter'])
                            existing.bone_mass_percent = self._calculate_bone_percent(
                                safe_float(row['Knochenmasse(kg)']),
                                safe_float(row['Gewicht(kg)'])
                            )
                            existing.raw_data = json.dumps(dict(row), default=str)
                            existing.updated_at = datetime.utcnow()
                            results["updated_measurements"] += 1
                        else:
                            # Create new record
                            new_record = BodyComposition(
                                user_id=self.user_id,
                                date=measurement_time,
                                weight_kg=safe_float(row['Gewicht(kg)']),
                                bmi=safe_float(row['BMI']),
                                body_fat_percent=safe_float(row['Körperfett(%)']),
                                skeletal_muscle_percent=safe_float(row['Skelettmuskel(%)']),
                                water_percent=safe_float(row['Körperwasser(%)']),
                                muscle_mass_percent=self._calculate_muscle_percent(
                                    safe_float(row['Muskelmasse(kg)']),
                                    safe_float(row['Gewicht(kg)'])
                                ),
                                basal_metabolism_kcal=safe_float(row['Grundumsatz(kcal)']),
                                visceral_fat=safe_float(row['Viszeralfett']),
                                protein_percent=safe_float(row['Eiweiß (%)']),
                                metabolic_age=safe_float(row['Stoffwechselalter']),
                                bone_mass_percent=self._calculate_bone_percent(
                                    safe_float(row['Knochenmasse(kg)']),
                                    safe_float(row['Gewicht(kg)'])
                                ),
                                raw_data=json.dumps(dict(row), default=str)
                            )
                            session.add(new_record)
                            results["new_measurements"] += 1

                    except Exception as e:
                        results["errors"] += 1
                        print(f"Error processing row: {e}")

                session.commit()

        except Exception as e:
            raise Exception(f"CSV import failed: {e}")

        return results

    def _calculate_muscle_percent(self, muscle_mass_kg: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
        """Calculate muscle mass percentage from absolute values."""
        if muscle_mass_kg and weight_kg and weight_kg > 0:
            return round((muscle_mass_kg / weight_kg) * 100, 1)
        return None

    def _calculate_bone_percent(self, bone_mass_kg: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
        """Calculate bone mass percentage from absolute values."""
        if bone_mass_kg and weight_kg and weight_kg > 0:
            return round((bone_mass_kg / weight_kg) * 100, 1)
        return None

    def get_athletic_trends(self, days: int = 60) -> Dict:
        """
        Analyze trends critical for athletic performance over specified period.

        Focus on key metrics:
        - Weight stability (hydration/recovery indicator)
        - Power-to-weight potential (lean body mass trends)
        - Recovery markers (water %, metabolic age)
        - Training adaptation (muscle mass changes)
        """
        with self.db.get_session() as session:
            measurements = session.query(BodyComposition)\
                .filter_by(user_id=self.user_id)\
                .filter(BodyComposition.date >= datetime.now() - timedelta(days=days))\
                .order_by(BodyComposition.date)\
                .all()

            if len(measurements) < 2:
                return {"status": "insufficient_data", "count": len(measurements)}

            # Calculate athletic performance indicators
            first = measurements[0]
            latest = measurements[-1]

            # Power-to-weight calculations
            lean_body_mass_first = self._calculate_lean_mass(first)
            lean_body_mass_latest = self._calculate_lean_mass(latest)

            # Hydration stability (critical for endurance)
            water_stability = self._calculate_stability([m.water_percent for m in measurements if m.water_percent])

            # Weight stability (recovery indicator)
            weight_stability = self._calculate_stability([m.weight_kg for m in measurements if m.weight_kg])

            return {
                "status": "success",
                "period_days": days,
                "measurements_count": len(measurements),

                # Athletic Performance Indicators
                "weight_change_kg": latest.weight_kg - first.weight_kg if (latest.weight_kg and first.weight_kg) else None,
                "lean_mass_change_kg": lean_body_mass_latest - lean_body_mass_first if (lean_body_mass_latest and lean_body_mass_first) else None,
                "muscle_mass_change": latest.skeletal_muscle_percent - first.skeletal_muscle_percent if (latest.skeletal_muscle_percent and first.skeletal_muscle_percent) else None,
                "body_fat_change": latest.body_fat_percent - first.body_fat_percent if (latest.body_fat_percent and first.body_fat_percent) else None,

                # Recovery & Adaptation Markers
                "water_stability_score": water_stability,
                "weight_stability_score": weight_stability,
                "metabolic_age_change": latest.metabolic_age - first.metabolic_age if (latest.metabolic_age and first.metabolic_age) else None,
                "bmr_change_kcal": latest.basal_metabolism_kcal - first.basal_metabolism_kcal if (latest.basal_metabolism_kcal and first.basal_metabolism_kcal) else None,

                # Current State
                "latest_measurement": {
                    "date": latest.date,
                    "weight_kg": latest.weight_kg,
                    "lean_body_mass_kg": lean_body_mass_latest,
                    "body_fat_percent": latest.body_fat_percent,
                    "skeletal_muscle_percent": latest.skeletal_muscle_percent,
                    "water_percent": latest.water_percent,
                    "metabolic_age": latest.metabolic_age,
                    "power_to_weight_potential": self._calculate_power_to_weight_potential(latest)
                }
            }

    def _calculate_lean_mass(self, measurement: BodyComposition) -> Optional[float]:
        """Calculate lean body mass for power-to-weight analysis."""
        if measurement.weight_kg and measurement.body_fat_percent:
            return round(measurement.weight_kg * (1 - measurement.body_fat_percent / 100), 1)
        return None

    def _calculate_stability(self, values: List[float]) -> Optional[float]:
        """Calculate stability score (lower coefficient of variation = more stable)."""
        if len(values) < 2:
            return None
        import numpy as np
        cv = np.std(values) / np.mean(values) * 100
        # Return stability score: 100 - CV (higher = more stable)
        return max(0, round(100 - cv, 1))

    def _calculate_power_to_weight_potential(self, measurement: BodyComposition) -> Optional[float]:
        """
        Estimate power-to-weight potential based on lean body mass.
        Higher lean mass % = better power-to-weight potential for cycling/climbing.
        """
        if measurement.weight_kg and measurement.body_fat_percent:
            lean_percent = 100 - measurement.body_fat_percent
            # Normalize to 0-100 scale (85% lean mass = excellent for endurance athletes)
            potential = min(100, (lean_percent / 85) * 100)
            return round(potential, 1)
        return None

    def get_training_log_metrics(self, date: datetime) -> Dict:
        """
        Get key metrics for display in 60-day training log.
        Focus on most critical data points for daily training decisions.
        """
        with self.db.get_session() as session:
            # Get measurement for specific date (within same day)
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)

            measurement = session.query(BodyComposition).filter(
                BodyComposition.user_id == self.user_id,
                BodyComposition.date >= start_date,
                BodyComposition.date < end_date
            ).first()

            if not measurement:
                return {}

            # Return only the most critical metrics for daily training log
            return {
                "weight_kg": measurement.weight_kg,
                "body_fat_percent": measurement.body_fat_percent,
                "water_percent": measurement.water_percent,
                "lean_mass_kg": self._calculate_lean_mass(measurement),
                "recovery_score": self._calculate_recovery_score(measurement)
            }

    def _calculate_recovery_score(self, measurement: BodyComposition) -> Optional[float]:
        """
        Calculate recovery score based on hydration and metabolic markers.
        Higher score = better recovery state.
        """
        if not (measurement.water_percent and measurement.metabolic_age):
            return None

        # Ideal water % for athletes: 60-65%
        water_score = min(100, max(0, (measurement.water_percent - 55) * 10))  # 60% = 50 points, 65% = 100 points

        # Metabolic age score (lower age = better)
        # Assume chronological age ~30-45 for calculation
        metabolic_score = max(0, 100 - max(0, measurement.metabolic_age - 25) * 2)  # Age 25 = 100, each year older = -2 points

        # Combined recovery score
        recovery_score = (water_score * 0.6) + (metabolic_score * 0.4)  # Weight hydration higher
        return round(recovery_score, 1)