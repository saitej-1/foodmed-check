import os
from pathlib import Path
from dotenv import load_dotenv

# Load root .env
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dfi_db")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

MODELS_DIR = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "models")))
MODELS_DIR.mkdir(exist_ok=True)
