from fastapi import Request, HTTPException, status
import httpx
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

async def get_current_user(request: Request) -> dict:
    """
    Extracts JWT from the Authorization header and verifies it with Supabase.
    Returns the user object on success, or raises HTTPException on failure.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing or invalid Authorization header")

    jwt_token = auth_header.removeprefix("Bearer ").strip()

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "apikey": SUPABASE_SERVICE_KEY
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="JWT validation failed")

        return response.json()
