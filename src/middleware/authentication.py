from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from typing import Optional
import jwt
from datetime import datetime, timedelta
from src.config import Settings

settings = Settings()

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthHandler:
    """Handle authentication and authorization"""
    
    def __init__(self):
        self.secret = settings.secret_key
        self.algorithm = settings.algorithm
        
    def encode_token(self, user_id: str) -> str:
        """Generate JWT token"""
        payload = {
            'exp': datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes),
            'iat': datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> str:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        if not settings.api_key_enabled:
            return True
        
        if not settings.api_keys:
            return False
        
        return api_key in settings.api_keys

auth_handler = AuthHandler()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Optional[str]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    
    token = credentials.credentials
    user_id = auth_handler.decode_token(token)
    return user_id

async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """Require authentication via JWT or API key"""
    
    # Check API key first
    if api_key and auth_handler.verify_api_key(api_key):
        return "api_key_user"
    
    # Check JWT token
    if credentials:
        token = credentials.credentials
        return auth_handler.decode_token(token)
    
    raise HTTPException(status_code=401, detail="Authentication required")

async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[str]:
    """Optional authentication - returns user if authenticated, None otherwise"""
    
    try:
        return await require_auth(credentials, api_key)
    except HTTPException:
        return None
