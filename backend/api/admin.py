from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
import os
import subprocess
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from database.db_config import get_db
from backend.config import MODELS_DIR

router = APIRouter(prefix="/admin", tags=["admin"])

# Background retraining runner
def run_retrain_script():
    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "train_pipeline.py"
    venv_python = Path(__file__).resolve().parent.parent.parent / "venv" / "bin" / "python"
    
    # Run retraining
    try:
        log_file = Path(__file__).resolve().parent.parent.parent / "retrain_run.log"
        with open(log_file, "w") as f:
            subprocess.run([str(venv_python), str(script_path)], stdout=f, stderr=f, check=True)
        print("🚀 Retraining completed successfully!")
    except Exception as e:
        print(f"❌ Retraining task failed: {e}")

@router.post("/retrain")
def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger the XGBoost model retraining pipeline in the background."""
    background_tasks.add_task(run_retrain_script)
    return {"status": "retraining_triggered", "message": "Model feature selection and XGBoost retraining started in the background. Check logs in a few minutes."}

@router.get("/retrain/status")
def check_retrain_status():
    """Read the retraining script run log file."""
    log_file = Path(__file__).resolve().parent.parent.parent / "retrain_run.log"
    if not log_file.exists():
        return {"status": "idle", "log": "No retraining runs have been triggered yet."}
    
    with open(log_file, "r") as f:
        log_content = f.read()
        
    status = "running"
    if "DFI ML Training complete!" in log_content:
        status = "completed"
    elif "Error" in log_content or "Traceback" in log_content:
        status = "failed"
        
    return {
        "status": status,
        "log": log_content[-3000:] # Last 3000 chars
    }

@router.get("/stats")
def get_platform_statistics(db: Session = Depends(get_db)):
    """Fetch analytics statistics for drugs, foods, interactions, and logs."""
    try:
        drugs_cnt = db.execute(text("SELECT COUNT(*) FROM drugs")).scalar()
        foods_cnt = db.execute(text("SELECT COUNT(*) FROM foods")).scalar()
        labels_cnt = db.execute(text("SELECT COUNT(*) FROM dfi_labels")).scalar()
        predictions_cnt = db.execute(text("SELECT COUNT(*) FROM prediction_history")).scalar()
        feedback_cnt = db.execute(text("SELECT COUNT(*) FROM feedback")).scalar()
        users_cnt = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
        
        # Breakdown of severities
        severity_breakdown = db.execute(
            text("SELECT severity, COUNT(*) FROM dfi_labels GROUP BY severity")
        ).fetchall()
        
        # Recent prediction logs
        recent_preds = db.execute(
            text("SELECT id, drug_name, food_name, predicted_class, confidence, created_at "
                 "FROM prediction_history ORDER BY created_at DESC LIMIT 5")
        ).fetchall()
        
        return {
            "statistics": {
                "drugs": drugs_cnt,
                "foods": foods_cnt,
                "interactions_labeled": labels_cnt,
                "predictions_run": predictions_cnt,
                "feedback_records": feedback_cnt,
                "users": users_cnt,
                "severity_counts": {r[0]: r[1] for r in severity_breakdown}
            },
            "recent_predictions": [
                {
                    "id": p[0],
                    "drug_name": p[1],
                    "food_name": p[2],
                    "predicted_class": p[3],
                    "confidence": p[4],
                    "created_at": p[5]
                }
                for p in recent_preds
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@router.get("/model-info")
def get_model_info(db: Session = Depends(get_db)):
    """Get the current active model card metadata and performance specs."""
    try:
        active_model = db.execute(
            text("SELECT version, accuracy, f1_score, best_params, trained_at "
                 "FROM model_metadata WHERE active = TRUE LIMIT 1")
        ).fetchone()
        
        features_path = MODELS_DIR / "feature_names.json"
        features = []
        if features_path.exists():
            with open(features_path, "r") as f:
                features = json.load(f)
                
        if not active_model:
            return {
                "model_active": False,
                "info": "No machine learning model is active. Please trigger retraining in the Admin Console.",
                "features": features
            }
            
        return {
            "model_active": True,
            "version": active_model[0],
            "accuracy": active_model[1],
            "f1_score": active_model[2],
            "best_params": json.loads(active_model[3]) if isinstance(active_model[3], str) else active_model[3],
            "trained_at": active_model[4],
            "features": features
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error loading model metadata: {str(e)}")
