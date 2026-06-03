import os
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS512"

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET not set in .env")

security = HTTPBearer(auto_error=False)


def decode_token(token: str) -> dict:
    """Decode JWT and return claims. Raise HTTPException on any failure."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET.encode("utf-8"),
            algorithms=[JWT_ALGORITHM],
            options={"verify_aud": False},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    """
    FastAPI dependency: extract and validate user_id from JWT.
    Use it like:  def endpoint(user_id: int = Depends(get_current_user_id)): ...
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    payload = decode_token(credentials.credentials)
    user_id = payload.get("userId")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Token missing userId claim")

    return int(user_id)
