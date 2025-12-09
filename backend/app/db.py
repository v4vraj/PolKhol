# backend/app/db.py
from sqlalchemy import create_engine
import os

DB_USER = os.getenv("DB_USER", "citysense")
DB_PASS = os.getenv("DB_PASS", "citysense")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "citysense")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
