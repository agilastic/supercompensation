"""Authentication module for Strava API."""

from .oauth import StravaOAuth, AuthManager

__all__ = ["StravaOAuth", "AuthManager"]