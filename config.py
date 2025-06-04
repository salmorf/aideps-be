from datetime import datetime, timedelta

import jwt
from decouple import config

SECRET_KEY = config("SECRET_KEY")
DATABASE_URL = config("DATABASE_URL", default="mongodb://localhost:27017")
DATABASE_NAME = config("DATABASE_NAME", default="projectm")
ACCESS_TOKEN_EXPIRE_MINUTES = config(
    "ACCESS_TOKEN_EXPIRE_MINUTES", cast=int, default=30
)
ALGORITHM = config("ALGORITHM", cast=str, default="HS256")
PORT = config("PORT")
SERVER_PASSWORD = config("SERVER_PASSWORD")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
