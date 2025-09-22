#!/usr/bin/env python
"""Database migration script to add adaptive model and performance tracking tables."""

import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from strava_supercompensation.db.models import Base, PerformanceOutcome, AdaptiveModelParameters
from strava_supercompensation.config import config


def migrate_database(db_url=None):
    """Add new tables for adaptive model and performance tracking."""

    if db_url is None:
        db_url = config.DATABASE_URL

    print(f"Migrating database: {db_url}")

    try:
        # Create engine
        engine = create_engine(db_url)

        # Create new tables (won't affect existing tables)
        print("Creating new tables for adaptive model and performance tracking...")
        Base.metadata.create_all(
            engine,
            tables=[
                PerformanceOutcome.__table__,
                AdaptiveModelParameters.__table__,
            ]
        )

        print("✅ Migration successful! New tables have been added:")
        print("  - performance_outcomes: Track training outcomes for model adaptation")
        print("  - adaptive_model_parameters: Store personalized model parameters")

        # Verify tables were created
        Session = sessionmaker(bind=engine)
        session = Session()

        # Check if we can query the new tables
        try:
            session.query(PerformanceOutcome).first()
            session.query(AdaptiveModelParameters).first()
            print("\n✅ Tables verified successfully!")
        except Exception as e:
            print(f"\n⚠️  Table verification failed: {e}")
        finally:
            session.close()

        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


if __name__ == "__main__":
    # Allow custom database URL as argument
    db_url = sys.argv[1] if len(sys.argv) > 1 else None

    success = migrate_database(db_url)
    sys.exit(0 if success else 1)