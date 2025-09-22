#!/usr/bin/env python
"""Database migration script to add workout_type column."""

import sqlite3
import sys
from pathlib import Path

def migrate_database(db_path="strava_supercompensation.db"):
    """Add workout_type column to activities table."""

    if not Path(db_path).exists():
        print(f"Database {db_path} not found!")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(activities)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'workout_type' in columns:
            print("Column 'workout_type' already exists in activities table.")
            return True

        # Add the new column
        print("Adding 'workout_type' column to activities table...")
        cursor.execute("ALTER TABLE activities ADD COLUMN workout_type INTEGER")

        conn.commit()
        print("✅ Migration successful! Column 'workout_type' has been added.")

        # Show table structure
        cursor.execute("PRAGMA table_info(activities)")
        print("\nUpdated activities table structure:")
        for column in cursor.fetchall():
            print(f"  - {column[1]}: {column[2]}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    # Allow custom database path as argument
    db_path = sys.argv[1] if len(sys.argv) > 1 else "strava_supercompensation.db"

    success = migrate_database(db_path)
    sys.exit(0 if success else 1)