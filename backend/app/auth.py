import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, constr
from sqlalchemy import text

from .db import engine


SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash the plain password using Argon2."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify the plain password against the stored hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        return False

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")
router = APIRouter(prefix="/api", tags=["auth"])

# --------------------------
# Pydantic schemas
# --------------------------
class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=32)
    email: EmailStr
    password: constr(min_length=8, max_length=512) 
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# --------------------------
# DB helper functions
# --------------------------
def get_user_by_username(username: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, email, hashed_password, reputation_score FROM users WHERE username = :u"),
            {"u": username}
        ).fetchone()
        return row

def get_user_by_id(user_id: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, email, reputation_score FROM users WHERE id = :id"),
            {"id": user_id}
        ).fetchone()
        return row

# --------------------------
# JWT helpers
# --------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.utcnow()
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --------------------------
# Endpoints
# --------------------------
@router.post("/register", response_model=dict)
def register(user: UserCreate):
    """
    Register a new user. Returns basic user info (id, username, email).
    """
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM users WHERE username = :u OR email = :e"),
            {"u": user.username, "e": user.email}
        ).fetchone()
        if exists:
            raise HTTPException(status_code=400, detail="Username or email already registered")

        hashed = get_password_hash(user.password) 
        
        result = conn.execute(text("""
            INSERT INTO users (username, email, first_name, last_name, hashed_password)
            VALUES (:username, :email, :first_name, :last_name, :hashed_password)
            RETURNING id, username, email
        """), {
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "hashed_password": hashed
        })
        created = result.fetchone()
        return {"id": str(created.id), "username": created.username, "email": created.email}

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login (OAuth2 password flow). Returns a JWT access token.
    Also performs transparent re-hash of password if the stored hash needs update.
    """
    user_row = get_user_by_username(form_data.username)
    if not user_row:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    stored_hash = user_row.hashed_password
    if not verify_password(form_data.password, stored_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    try:
        if pwd_context.needs_update(stored_hash):
            new_hashed = get_password_hash(form_data.password)
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE users SET hashed_password = :h WHERE id = :id"),
                    {"h": new_hashed, "id": user_row.id}
                )
    except Exception:
        pass

    access_token = create_access_token(data={"sub": str(user_row.id)})
    return {"access_token": access_token, "token_type": "bearer"}

# --------------------------
# Dependency
# --------------------------
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception

    user_row = get_user_by_id(token_data.user_id)
    if user_row is None:
        raise credentials_exception

    return {
        "id": str(user_row.id),
        "username": user_row.username,
        "email": user_row.email,
        "reputation_score": float(user_row.reputation_score) if user_row.reputation_score is not None else 0.5
    }