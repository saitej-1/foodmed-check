import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/dfi_db")

# Parse DATABASE_URL for raw connection params to create database if missing
def create_db_if_not_exists():
    # Remove prefix
    db_uri = DATABASE_URL
    if db_uri.startswith("postgresql://"):
        db_uri = db_uri[13:]
    
    # Split credentials and db name
    parts = db_uri.split("/")
    db_name = parts[-1]
    
    # Rest of connection (host, user, etc.)
    host_parts = "/".join(parts[:-1])
    
    connection_string = f"postgresql://{host_parts}/postgres"
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
    except psycopg2.OperationalError:
        # DB doesn't exist, try connecting to 'postgres' default database
        try:
            print(f"Database '{db_name}' not found. Attempting to create it...")
            conn = psycopg2.connect(
                dbname='postgres',
                user=os.getenv("PGUSER", "postgres"),
                host='localhost',
                port=5432
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE {db_name};")
            cur.close()
            conn.close()
            print(f"Successfully created database '{db_name}'!")
        except Exception as e:
            print(f"Could not automatically create database: {e}")

# Create database if it doesn't exist
create_db_if_not_exists()

# Initialize SQLAlchemy
engine = create_engine(
    DATABASE_URL, 
    pool_size=10, 
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
