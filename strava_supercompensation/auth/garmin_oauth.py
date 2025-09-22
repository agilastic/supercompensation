"""Garmin Connect OAuth1 authentication handler."""

import os
import json
import time
import hashlib
import hmac
import base64
import urllib.parse
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests
from requests_oauthlib import OAuth1Session

from ..config import config
from ..db import get_db
from ..db.models import GarminAuthToken


class GarminOAuthError(Exception):
    """Garmin OAuth specific errors."""
    pass


class GarminOAuth:
    """Handle Garmin Connect OAuth1 authentication flow."""

    # Garmin Connect OAuth1 endpoints
    REQUEST_TOKEN_URL = "https://connectapi.garmin.com/oauth-service/oauth/request_token"
    AUTHORIZE_URL = "https://connect.garmin.com/oauthConfirm"
    ACCESS_TOKEN_URL = "https://connectapi.garmin.com/oauth-service/oauth/access_token"

    # Garmin Connect API base
    API_BASE = "https://connectapi.garmin.com"

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db = get_db()

        # Get OAuth1 credentials from environment
        self.consumer_key = os.getenv("GARMIN_CONSUMER_KEY")
        self.consumer_secret = os.getenv("GARMIN_CONSUMER_SECRET")

        if not self.consumer_key or not self.consumer_secret:
            raise GarminOAuthError(
                "Garmin OAuth1 credentials not found. Set GARMIN_CONSUMER_KEY and GARMIN_CONSUMER_SECRET environment variables."
            )

    def get_authorization_url(self) -> Tuple[str, str]:
        """Get authorization URL for OAuth1 flow.

        Returns:
            Tuple of (authorization_url, oauth_token_secret) for verification
        """
        try:
            # Create OAuth1 session
            oauth = OAuth1Session(
                client_key=self.consumer_key,
                client_secret=self.consumer_secret,
                callback_uri="oob"  # Out-of-band for CLI apps
            )

            # Get request token
            response = oauth.fetch_request_token(self.REQUEST_TOKEN_URL)
            oauth_token = response.get("oauth_token")
            oauth_token_secret = response.get("oauth_token_secret")

            if not oauth_token or not oauth_token_secret:
                raise GarminOAuthError("Failed to get request token from Garmin")

            # Generate authorization URL
            auth_url = oauth.authorization_url(self.AUTHORIZE_URL)

            return auth_url, oauth_token_secret

        except Exception as e:
            raise GarminOAuthError(f"Failed to get authorization URL: {str(e)}")

    def exchange_code_for_token(self, oauth_token: str, oauth_token_secret: str, oauth_verifier: str) -> Dict:
        """Exchange authorization code for access token.

        Args:
            oauth_token: OAuth token from authorization URL
            oauth_token_secret: OAuth token secret from get_authorization_url
            oauth_verifier: Verification code from user authorization

        Returns:
            Dictionary with token information
        """
        try:
            # Create OAuth1 session with request token
            oauth = OAuth1Session(
                client_key=self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=oauth_token,
                resource_owner_secret=oauth_token_secret,
                verifier=oauth_verifier
            )

            # Exchange for access token
            response = oauth.fetch_access_token(self.ACCESS_TOKEN_URL)

            access_token = response.get("oauth_token")
            access_token_secret = response.get("oauth_token_secret")

            if not access_token or not access_token_secret:
                raise GarminOAuthError("Failed to get access token from Garmin")

            # Get user info to verify connection
            user_info = self._get_user_info(access_token, access_token_secret)

            # Store tokens in database
            self._store_tokens(
                access_token=access_token,
                access_token_secret=access_token_secret,
                user_info=user_info
            )

            return {
                "access_token": access_token,
                "access_token_secret": access_token_secret,
                "garmin_user_id": user_info.get("userId"),
                "display_name": user_info.get("displayName"),
            }

        except Exception as e:
            raise GarminOAuthError(f"Failed to exchange code for token: {str(e)}")

    def _get_user_info(self, access_token: str, access_token_secret: str) -> Dict:
        """Get user information to verify token validity."""
        try:
            oauth = OAuth1Session(
                client_key=self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=access_token,
                resource_owner_secret=access_token_secret
            )

            # Get user profile
            response = oauth.get(f"{self.API_BASE}/userprofile-service/userprofile")

            if response.status_code != 200:
                raise GarminOAuthError(f"Failed to get user info: {response.status_code}")

            return response.json()

        except Exception as e:
            raise GarminOAuthError(f"Failed to get user info: {str(e)}")

    def _store_tokens(self, access_token: str, access_token_secret: str, user_info: Dict):
        """Store OAuth1 tokens in database."""
        with self.db.get_session() as session:
            # Check if token already exists for this user
            existing_token = session.query(GarminAuthToken).filter_by(user_id=self.user_id).first()

            if existing_token:
                # Update existing token
                existing_token.access_token = access_token
                existing_token.access_token_secret = access_token_secret
                existing_token.consumer_key = self.consumer_key
                existing_token.consumer_secret = self.consumer_secret
                existing_token.garmin_user_id = str(user_info.get("userId", ""))
                existing_token.updated_at = datetime.utcnow()
            else:
                # Create new token
                token = GarminAuthToken(
                    user_id=self.user_id,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    garmin_user_id=str(user_info.get("userId", ""))
                )
                session.add(token)

            session.commit()

    def get_stored_tokens(self) -> Optional[GarminAuthToken]:
        """Get stored OAuth1 tokens for the user."""
        with self.db.get_session() as session:
            return session.query(GarminAuthToken).filter_by(user_id=self.user_id).first()

    def is_authenticated(self) -> bool:
        """Check if user has valid Garmin authentication."""
        tokens = self.get_stored_tokens()
        return tokens is not None

    def get_authenticated_session(self) -> OAuth1Session:
        """Get authenticated OAuth1 session for API calls."""
        tokens = self.get_stored_tokens()
        if not tokens:
            raise GarminOAuthError("No stored Garmin tokens found. Please authenticate first.")

        return OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=tokens.access_token,
            resource_owner_secret=tokens.access_token_secret
        )

    def revoke_tokens(self):
        """Revoke stored tokens."""
        with self.db.get_session() as session:
            token = session.query(GarminAuthToken).filter_by(user_id=self.user_id).first()
            if token:
                session.delete(token)
                session.commit()

    def test_connection(self) -> Dict:
        """Test the Garmin Connect API connection."""
        try:
            oauth_session = self.get_authenticated_session()

            # Test with user profile endpoint
            response = oauth_session.get(f"{self.API_BASE}/userprofile-service/userprofile")

            if response.status_code == 200:
                user_data = response.json()
                return {
                    "status": "success",
                    "user_id": user_data.get("userId"),
                    "display_name": user_data.get("displayName"),
                    "email": user_data.get("email"),
                }
            else:
                return {
                    "status": "error",
                    "message": f"API call failed with status {response.status_code}",
                    "response": response.text[:200]
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


def get_garmin_auth(user_id: str = "default") -> GarminOAuth:
    """Get Garmin OAuth handler for user."""
    return GarminOAuth(user_id)