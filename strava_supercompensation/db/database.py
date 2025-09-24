"""Database connection and session management."""

from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import config
from .models import Base


class Database:
    """Database connection manager."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        self.database_url = database_url or config.DATABASE_URL

        # Special handling for SQLite to avoid threading issues
        if "sqlite" in self.database_url:
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            self.engine = create_engine(self.database_url, echo=False)

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def migrate_schema(self):
        """Apply schema migrations for new columns."""
        try:
            # Check if resting_heart_rate column exists in hrv_data
            result = self.engine.execute("PRAGMA table_info(hrv_data)")
            columns = [row[1] for row in result]

            # Add missing heart rate columns to hrv_data if they don't exist
            if 'resting_heart_rate' not in columns:
                self.engine.execute("ALTER TABLE hrv_data ADD COLUMN resting_heart_rate INTEGER")
            if 'max_heart_rate' not in columns:
                self.engine.execute("ALTER TABLE hrv_data ADD COLUMN max_heart_rate INTEGER")
            if 'min_heart_rate' not in columns:
                self.engine.execute("ALTER TABLE hrv_data ADD COLUMN min_heart_rate INTEGER")

            # Remove heart rate columns from wellness_data if they exist (cleanup)
            wellness_result = self.engine.execute("PRAGMA table_info(wellness_data)")
            wellness_columns = [row[1] for row in wellness_result]

            # SQLite doesn't support DROP COLUMN, so we note the cleanup needed
            if any(col in wellness_columns for col in ['resting_heart_rate', 'max_heart_rate', 'min_heart_rate']):
                print("Note: Heart rate columns found in wellness_data table - consider manual cleanup")

        except Exception as e:
            # Migration failed, but don't crash the app
            print(f"Schema migration warning: {e}")

    def close(self):
        """Close database connection."""
        self.engine.dispose()


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
        _db.migrate_schema()  # Apply any schema migrations
    return _db


def close_db():
    """Close the global database connection."""
    global _db
    if _db is not None:
        _db.close()
        _db = None