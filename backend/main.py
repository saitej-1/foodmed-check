import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

import sys
from pathlib import Path

# Add root project path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_config import get_db
from backend.api import search, predict, explain, admin

app = FastAPI(
    title="AI Drug-Food Interaction Prediction Platform API",
    description="Backend API exposing search autocompletes, RDKit molecular feature prediction, and SHAP explainability insights.",
    version="1.0.0"
)

# CORS Policy configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(search.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Drug-Food Interaction Platform API. Access Swagger docs at /docs"}

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check validating backend execution and database connectivity."""
    try:
        db.execute(text("SELECT 1")).scalar()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": f"disconnected: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
