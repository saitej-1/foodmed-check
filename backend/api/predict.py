from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from database.db_config import get_db
from backend.config import MODELS_DIR
from pipeline.feature_extractor import DFIFeatureExtractor, validate_smiles, get_smiles_from_pubchem

router = APIRouter(prefix="/predict", tags=["prediction"])

# Helper function to check if models exist
def get_ml_models():
    model_path = MODELS_DIR / "xgboost_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    selector_path = MODELS_DIR / "feature_selector.pkl"
    names_path = MODELS_DIR / "feature_names.json"
    
    if model_path.exists() and scaler_path.exists() and selector_path.exists() and names_path.exists():
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selector = joblib.load(selector_path)
            with open(names_path, "r") as f:
                feature_names = json.load(f)
            return model, scaler, selector, feature_names
        except Exception as e:
            print(f"⚠ Error loading ML models: {e}")
    return None

@router.post("")
def predict_interaction(
    drug_name: str = Body(None),
    food_name: str = Body(None),
    drug_smiles: str = Body(None),
    food_smiles: str = Body(None),
    db: Session = Depends(get_db)
):
    """
    Predict Drug-Food Interaction.
    Accepts drug/food names (looks up in DB) or raw SMILES strings.
    """
    # 1. Resolve Drug SMILES
    resolved_drug_name = drug_name or "Unknown Drug"
    resolved_drug_smiles = drug_smiles
    
    if not resolved_drug_smiles and drug_name:
        # DB lookup with wildcards
        query = f"%{drug_name}%"
        res = db.execute(
            text("SELECT smiles, name FROM drugs WHERE name ILIKE :q LIMIT 1"),
            {"q": query}
        ).fetchone()
        if res:
            resolved_drug_smiles = res[0]
            resolved_drug_name = res[1]
        else:
            # PubChem query fallback
            resolved_drug_smiles = get_smiles_from_pubchem(drug_name)
            
    # 2. Resolve Food SMILES
    resolved_food_name = food_name or "Unknown Food"
    resolved_food_smiles = food_smiles
    
    if not resolved_food_smiles and food_name:
        # DB lookup for food compound with wildcards
        query = f"%{food_name}%"
        res = db.execute(
            text("SELECT fc.smiles, f.name "
                 "FROM foods f JOIN food_compounds fc ON f.id = fc.food_id "
                 "WHERE f.name ILIKE :q LIMIT 1"),
            {"q": query}
        ).fetchone()
        if res:
            resolved_food_smiles = res[0]
            resolved_food_name = res[1]
        else:
            # PubChem query fallback
            resolved_food_smiles = get_smiles_from_pubchem(food_name)

    if not resolved_drug_smiles or not resolved_food_smiles:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not resolve SMILES for pair: Drug ({resolved_drug_name}), Food ({resolved_food_name}). Please check names or provide raw SMILES."
        )

    # Validate chemical correctness
    if not validate_smiles(resolved_drug_smiles) or not validate_smiles(resolved_food_smiles):
        raise HTTPException(status_code=400, detail="Invalid chemical structure (SMILES) provided.")

    # 3. Check for direct DB interaction mapping (true labels)
    db_interaction = None
    if drug_name and food_name:
        res = db.execute(
            text("SELECT dl.severity, dl.mechanism, dl.recommendation "
                 "FROM dfi_labels dl "
                 "JOIN drugs d ON dl.drug_id = d.id "
                 "JOIN foods f ON dl.food_id = f.id "
                 "WHERE d.name ILIKE :dname AND f.name ILIKE :fname"),
            {"dname": resolved_drug_name, "fname": resolved_food_name}
        ).fetchone()
        if res:
            db_interaction = {
                "severity": res[0],
                "mechanism": res[1],
                "recommendation": res[2]
            }

    # 4. Prediction Execution
    ml_models = get_ml_models()
    
    if ml_models:
        # ML Mode
        try:
            model, scaler, selector, feature_names = ml_models
            
            # Extract features
            extractor = DFIFeatureExtractor(use_3d=False, fp_bits=1024)
            raw_vector = extractor.extract_pair_vector(resolved_drug_smiles, resolved_food_smiles)
            
            # Reconstruct as DataFrame to align names
            raw_df = pd.DataFrame([raw_vector], columns=extractor.feature_names)
            
            # Transform
            selected_features = selector.transform(raw_df)
            
            # Predict
            pred_class = int(model.predict(selected_features)[0])
            probabilities = model.predict_proba(selected_features)[0]
            confidence = float(probabilities[pred_class])
            
            # Probability breakdown map
            prob_map = {
                "Negative DFI": float(probabilities[0]),
                "Positive DFI": float(probabilities[1]),
                "Non-significant DFI": float(probabilities[2])
            }
            mode = "Machine Learning (XGBoost)"
        except Exception as e:
            print(f"⚠ ML Prediction failed: {e}. Falling back to Rule-based.")
            ml_models = None

    if not ml_models:
        # Fallback Heuristics Mode (database exact label lookup or logic check)
        mode = "Rule-based Matcher"
        if db_interaction:
            # Map database labels
            sev = db_interaction["severity"]
            if sev == "severe":
                pred_class = 0
                prob_map = {"Negative DFI": 0.95, "Positive DFI": 0.02, "Non-significant DFI": 0.03}
            elif sev == "moderate":
                pred_class = 2
                prob_map = {"Negative DFI": 0.05, "Positive DFI": 0.15, "Non-significant DFI": 0.80}
            else:
                pred_class = 1
                prob_map = {"Negative DFI": 0.01, "Positive DFI": 0.98, "Non-significant DFI": 0.01}
        else:
            # Fully safe default
            pred_class = 1
            prob_map = {"Negative DFI": 0.05, "Positive DFI": 0.90, "Non-significant DFI": 0.05}
        
        confidence = prob_map[list(prob_map.keys())[pred_class]]

    # 5. Map class back to metadata
    classes_meta = {
        0: {"severity": "severe", "label": "Negative DFI", "badge": "Severe Danger 🔴"},
        1: {"severity": "safe", "label": "Positive DFI", "badge": "Safe Interaction 🟢"},
        2: {"severity": "moderate", "label": "Non-significant DFI", "badge": "Moderate Caution 🟡"}
    }
    
    meta = classes_meta[pred_class]
    
    # 6. Retrieve recommendations
    mechanism = db_interaction["mechanism"] if db_interaction else "No specific molecular pathway warning documented in DailyMed labels. Safety profile calculated from model descriptors."
    recommendation = db_interaction["recommendation"] if db_interaction else "Safe to consume under normal dietary guidelines. Consult your healthcare provider for personalized advice."
    
    # 7. Record in history
    try:
        db.execute(
            text("INSERT INTO prediction_history (drug_name, food_name, smiles_drug, smiles_food, predicted_class, confidence, probabilities) "
                 "VALUES (:dn, :fn, :sd, :sf, :pc, :conf, :probs)"),
            {
                "dn": resolved_drug_name,
                "fn": resolved_food_name,
                "sd": resolved_drug_smiles,
                "sf": resolved_food_smiles,
                "pc": pred_class,
                "conf": confidence,
                "probs": json.dumps(prob_map)
            }
        )
        db.commit()
    except Exception as e:
        print(f"⚠ Failed to save prediction to history: {e}")
        db.rollback()

    return {
        "drug": {"name": resolved_drug_name, "smiles": resolved_drug_smiles},
        "food": {"name": resolved_food_name, "smiles": resolved_food_smiles},
        "prediction": {
            "class_id": pred_class,
            "label": meta["label"],
            "severity": meta["severity"],
            "badge": meta["badge"],
            "confidence": confidence,
            "probabilities": prob_map,
            "mode": mode
        },
        "clinical": {
            "mechanism": mechanism,
            "recommendation": recommendation
        }
    }
