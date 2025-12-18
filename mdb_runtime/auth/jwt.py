"""
JWT Token Utilities

Provides JWT encoding and decoding utilities with automatic format handling.

This module is part of MDB_RUNTIME - MongoDB Multi-Tenant Runtime Engine.
"""

import jwt
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def decode_jwt_token(token: Any, secret_key: str) -> Dict[str, Any]:
    """
    Helper function to decode JWT tokens with automatic fallback to bytes format.
    
    Handles cases where PyJWT might expect bytes instead of strings (version-specific behavior).
    
    Args:
        token: JWT token (can be str, bytes, or other)
        secret_key: Secret key for decoding (str)
    
    Returns:
        Decoded JWT payload as dict
    
    Raises:
        jwt.ExpiredSignatureError: If token has expired
        jwt.InvalidTokenError: If token is invalid
    """
    # Normalize token to string first
    if isinstance(token, bytes):
        token_str = token.decode('utf-8')
    elif isinstance(token, str):
        token_str = token
    else:
        token_str = str(token)
    
    # Normalize secret_key to string
    if isinstance(secret_key, bytes):
        secret_key_str = secret_key.decode('utf-8')
    elif isinstance(secret_key, str):
        secret_key_str = secret_key
    else:
        secret_key_str = str(secret_key)
    
    # Try decoding with string format first (standard PyJWT behavior)
    try:
        return jwt.decode(token_str, secret_key_str, algorithms=["HS256"])
    except jwt.InvalidTokenError as e:
        # If string format fails with "must be bytes" error, try bytes format
        error_msg = str(e)
        if "must be a <class 'bytes'>" in error_msg or ("bytes" in error_msg.lower() and "token" in error_msg.lower()):
            logger.debug(f"JWT decode: Retrying with bytes format (error: {e})")
            # Convert to bytes and try again
            token_bytes = token_str.encode('utf-8')
            secret_key_bytes = secret_key_str.encode('utf-8')
            return jwt.decode(token_bytes, secret_key_bytes, algorithms=["HS256"])
        else:
            # Re-raise if it's a different error
            raise

