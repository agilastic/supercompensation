"""Strava OAuth2 implementation."""

import time
import webbrowser
from typing import Optional, Dict, Any
from urllib.parse import urlencode
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import json

from ..config import config
from ..db import get_db
from ..db.models import AuthToken


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""

    def do_GET(self):
        """Handle GET request with authorization code."""
        query = urlparse(self.path).query
        params = parse_qs(query)

        if "code" in params:
            self.server.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Authorization successful!</h1>"
                b"<p>You can close this window and return to the terminal.</p></body></html>"
            )
        else:
            self.server.auth_code = None
            error = params.get("error", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h1>Authorization failed!</h1><p>Error: {error}</p></body></html>".encode()
            )

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


class StravaOAuth:
    """Handle Strava OAuth2 flow."""

    def __init__(self):
        self.client_id = config.STRAVA_CLIENT_ID
        self.client_secret = config.STRAVA_CLIENT_SECRET
        self.redirect_uri = config.STRAVA_REDIRECT_URI
        self.auth_base_url = config.STRAVA_AUTH_BASE_URL
        self.token_url = f"{self.auth_base_url}/token"

    def get_authorization_url(self, scope: str = "activity:read_all") -> str:
        """Generate the authorization URL."""
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "approval_prompt": "force",
            "scope": scope,
        }
        return f"{self.auth_base_url}/authorize?{urlencode(params)}"

    def authorize_browser(self, scope: str = "activity:read_all") -> Optional[str]:
        """Open browser for authorization and capture the code."""
        auth_url = self.get_authorization_url(scope)

        # Parse port from redirect URI
        parsed_uri = urlparse(self.redirect_uri)
        port = parsed_uri.port or 8000

        # Start local server to capture callback
        server = HTTPServer(("localhost", port), OAuthCallbackHandler)
        server.auth_code = None

        # Open browser
        print(f"Opening browser for authorization...")
        print(f"If browser doesn't open, visit: {auth_url}")
        webbrowser.open(auth_url)

        # Handle one request
        server.handle_request()

        return server.auth_code

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an expired access token."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()
        return response.json()


class AuthManager:
    """Manage authentication tokens and sessions."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.oauth = StravaOAuth()
        self.db = get_db()

    def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        with self.db.get_session() as session:
            token_record = session.query(AuthToken).filter_by(user_id=self.user_id).first()

            if not token_record:
                return None

            # Check if token is expired
            if token_record.is_expired():
                # Refresh the token
                try:
                    new_token_data = self.oauth.refresh_access_token(token_record.refresh_token)
                    self._save_token(new_token_data)
                    return new_token_data["access_token"]
                except Exception as e:
                    print(f"Failed to refresh token: {e}")
                    return None

            return token_record.access_token

    def authenticate(self, scope: str = "activity:read_all") -> bool:
        """Perform full authentication flow."""
        code = self.oauth.authorize_browser(scope)

        if not code:
            print("Authorization failed or was cancelled.")
            return False

        try:
            token_data = self.oauth.exchange_code_for_token(code)
            self._save_token(token_data)
            print(f"Successfully authenticated as {token_data.get('athlete', {}).get('firstname', 'Unknown')}")
            return True
        except Exception as e:
            print(f"Failed to exchange code for token: {e}")
            return False

    def _save_token(self, token_data: Dict[str, Any]) -> None:
        """Save token data to database."""
        with self.db.get_session() as session:
            token_record = session.query(AuthToken).filter_by(user_id=self.user_id).first()

            if not token_record:
                token_record = AuthToken(user_id=self.user_id)
                session.add(token_record)

            token_record.access_token = token_data["access_token"]
            token_record.refresh_token = token_data["refresh_token"]
            token_record.expires_at = token_data["expires_at"]

            if "athlete" in token_data:
                athlete = token_data["athlete"]
                token_record.athlete_id = str(athlete.get("id", ""))
                token_record.athlete_name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()

            session.commit()

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.get_valid_token() is not None

    def logout(self) -> None:
        """Remove stored authentication tokens."""
        with self.db.get_session() as session:
            token_record = session.query(AuthToken).filter_by(user_id=self.user_id).first()
            if token_record:
                session.delete(token_record)
                session.commit()
                print("Successfully logged out.")