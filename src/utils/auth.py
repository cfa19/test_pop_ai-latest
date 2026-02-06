"""
Simple authentication functions to be called inside endpoints

Use these to verify tokens manually without FastAPI dependencies
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import jwt
from dotenv import load_dotenv

load_dotenv()

# Supabase JWT configuration
# Supports two modes:
#   1. JWKS (production/Supabase hosted) — set SUPABASE_JWKS_URL, uses ES256
#   2. Shared secret (local dev) — set SUPABASE_JWT_SECRET, uses HS256
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL", "http://localhost:54321/auth/v1/.well-known/jwks.json")
jwks_client = None
if not SUPABASE_JWT_SECRET:
    from jwt import PyJWKClient

    jwks_client = PyJWKClient(SUPABASE_JWKS_URL)


class AuthenticationError(Exception):
    """Raised when authentication fails"""

    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify Supabase JWT token and return user information

    Args:
        token: JWT access token from Supabase (without "Bearer " prefix)

    Returns:
        Dict containing user information:
        {
            "user_id": str,
            "email": Optional[str],
            "role": Optional[str],
            "exp": datetime,
            "iat": datetime
        }

    Raises:
        AuthenticationError: If token is invalid, expired, or malformed

    Example:
        try:
            user_info = verify_token(token)
            print(f"User ID: {user_info['user_id']}")
        except AuthenticationError as e:
            print(f"Auth failed: {e.message}")
    """
    if not token:
        raise AuthenticationError("No token provided", status_code=401)

    try:
        # Decode and verify JWT token
        if not SUPABASE_JWT_SECRET:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",  # important
                options={
                    "verify_exp": True,
                    "verify_aud": True,
                },
            )
        else:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_signature": True, "verify_exp": True, "verify_iat": True, "verify_aud": True},
            )

        # Extract user information
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role"),
            "exp": datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None,
            "iat": datetime.fromtimestamp(payload.get("iat")) if payload.get("iat") else None,
        }

    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired", status_code=401)
    except jwt.InvalidAudienceError:
        raise AuthenticationError("Invalid token audience", status_code=401)
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}", status_code=401)
    except Exception as e:
        raise AuthenticationError(f"Token verification failed: {str(e)}", status_code=401)


def extract_token_from_header(authorization_header: Optional[str]) -> str:
    """
    Extract JWT token from Authorization header

    Args:
        authorization_header: The Authorization header value (e.g., "Bearer xxx")

    Returns:
        The JWT token string

    Raises:
        AuthenticationError: If header is missing or malformed

    Example:
        auth_header = request.headers.get("Authorization")
        token = extract_token_from_header(auth_header)
    """
    if not authorization_header:
        raise AuthenticationError("Authorization header missing", status_code=401)

    parts = authorization_header.split()

    if len(parts) != 2:
        raise AuthenticationError("Invalid authorization header format", status_code=401)

    if parts[0].lower() != "bearer":
        raise AuthenticationError("Authorization header must start with Bearer", status_code=401)

    return parts[1]


def authenticate_request(authorization_header: Optional[str]) -> Dict[str, Any]:
    """
    Complete authentication flow: extract token from header and verify it

    This is a convenience function that combines extract_token_from_header and verify_token

    Args:
        authorization_header: The Authorization header value (e.g., "Bearer xxx")

    Returns:
        Dict containing user information (see verify_token for structure)

    Raises:
        AuthenticationError: If authentication fails at any step

    Example:
        try:
            user_info = authenticate_request(request.headers.get("Authorization"))
            user_id = user_info["user_id"]
        except AuthenticationError as e:
            return {"error": e.message}, e.status_code
    """
    token = extract_token_from_header(authorization_header)
    return verify_token(token)
