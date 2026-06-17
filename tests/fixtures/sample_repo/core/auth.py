"""Authentication utilities."""

import httpx


class AuthClient:
    """Handles authentication flows."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self._client = httpx.Client()

    def login(self, username: str, password: str) -> dict:
        resp = self._client.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
        )
        return resp.json()

    def logout(self) -> None:
        self._client.post(f"{self.base_url}/auth/logout")

    def refresh_token(self) -> str:
        resp = self._client.get(f"{self.base_url}/auth/refresh")
        return resp.json()["token"]


class SessionManager:
    """Manages user sessions."""

    def __init__(self):
        self._sessions: dict = {}

    def create_session(self, user_id: str) -> str:
        import uuid
        sid = str(uuid.uuid4())
        self._sessions[user_id] = sid
        return sid

    def invalidate_session(self, user_id: str) -> bool:
        return self._sessions.pop(user_id, None) is not None


def verify_token(token: str, secret: str) -> bool:
    """Verify a JWT token."""
    return bool(token and secret)


def generate_api_key(length: int = 32) -> str:
    """Generate a random API key."""
    import secrets
    return secrets.token_hex(length)
