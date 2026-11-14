from fastapi import APIRouter, Depends, HTTPException, Query
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
from pipeline.feature_extractor import DFIFeatureExtractor, validate_smiles
from pipeline.shap_analyzer import DFISHAPAnalyzer

router = APIRouter(prefix="/explain", tags=["explainability"])

# Features translations to make SHAP explanation extremely user-friendly
FEATURE_DESCRIPTIONS = {
    # Molecular Weight / Sizes
    "D_MolWt": "Drug molecular weight",
    "F_MolWt": "Food compound molecular weight",
    "DIFF_MolWt": "Difference in molecular weight between drug and food",
    
    # Solubility / Lipophilicity
    "D_MolLogP": "Drug lipophilicity (octanol-water partition coefficient LogP)",
    "F_MolLogP": "Food compound lipophilicity (LogP)",
    "DIFF_MolLogP": "Difference in lipophilicity (LogP)",
    
    # Polar Surface Area (absorption indicator)
    "D_TPSA": "Drug Topological Polar Surface Area (TPSA)",
    "F_TPSA": "Food Topological Polar Surface Area (TPSA)",
    "DIFF_TPSA": "Difference in TPSA",
    
    # Hydrogen Bonding
    "D_NumHBA": "Number of hydrogen bond acceptors in the drug",
    "D_NumHBD": "Number of hydrogen bond donors in the drug",
    "F_NumHBA": "Number of hydrogen bond acceptors in the food compound",
    "F_NumHBD": "Number of hydrogen bond donors in the food compound",
    
    # Connectivity / Flexibilities
    "D_NumRotatableBonds": "Number of rotatable bonds in the drug (flexibility indicator)",
    "F_NumRotatableBonds": "Number of rotatable bonds in the food compound",
    
    # Pharmacophore / Substructures
    "D_has_cyp3a4_inhibitor": "Drug contains CYP3A4 inhibitor-like pharmacophore",
    "F_has_cyp3a4_inhibitor": "Food compound contains CYP3A4 inhibitor-like pharmacophore (e.g. furanocoumarin)",
    "D_has_vitamin_k_like": "Drug structurally resembles Vitamin K",
    "F_has_vitamin_k_like": "Food compound contains a quinone/Vitamin K-like ring",
    "D_has_tyramine_core": "Drug contains a phenethylamine/tyramine-like core",
    "F_has_tyramine_core": "Food compound contains tyramine structure",
    "D_has_coumarin_core": "Drug has a coumarin core",
    "F_has_coumarin_core": "Food compound has a coumarin structure",
    "D_has_methylxanthine": "Drug contains a methylxanthine derivative",
    "F_has_methylxanthine": "Food compound contains a methylxanthine derivative (e.g. caffeine)",
    "D_has_cyp1a2_inducer": "Drug has an indole/CYP1A2 inducer-like core",
    "F_has_cyp1a2_inducer": "Food compound contains a CYP1A2 inducer core (e.g. cruciferous indole)",
}

def clean_feature_name(name: str) -> str:
    # Translate technical features into human-readable text
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    
    # Regex tidyups
    clean = name.replace("D_", "Drug ").replace("F_", "Food ").replace("DIFF_", "Difference in ")
    clean = clean.replace("Num", "Number of ").replace("Aromatic", " Aromatic ").replace("Rings", " Rings")
    clean = clean.replace("LogP", " Lipophilicity (LogP)").replace("Wt", " Weight")
    return clean

def generate_natural_explanation(top_features: list, pred_class: int) -> str:
    """Generate a clean, professional clinical explanation from SHAP values."""
    if not top_features:
        return "No significant molecular predictors could be determined. The model predicts baseline safe interactions based on the absent warning tags in current FDA drug labels."

    class_statements = {
        0: "Warning: High risk of adverse interaction. The machine learning model identifies highly matching biochemical indicators that trigger mutual metabolic conflict:",
        1: "Safe Interaction: The model shows high stability between the molecular properties. No structural or chemical descriptors trigger warnings:",
        2: "Caution: Mild to moderate interaction risk. Descriptors suggest potential absorption competition or minor enzyme clearance shifts:"
    }
    
    points = []
    for f in top_features[:4]:
        feat_name = f["name"]
        shap_val = f["shap_value"]
        
        # Determine direction of effect
        if pred_class == 0: # Negative DFI (severe risk)
            direction = "increases interaction risk" if shap_val > 0 else "reduces interaction risk"
        elif pred_class == 1: # Positive DFI (safe)
            direction = "supports safe profile" if shap_val > 0 else "compromises safety"
        else: # Moderate/Caution
            direction = "suggests caution is advised" if shap_val > 0 else "indicates lower risk"
            
        points.append(f"• **{feat_name}**: The specific value ({f['value']:.2f}) {direction}.")
        
    intro = class_statements.get(pred_class, "Prediction explanation details:")
    return f"{intro}\n\n" + "\n".join(points)

@router.get("")
def explain_prediction(
    drug_smiles: str = Query(...),
    food_smiles: str = Query(...),
    db: Session = Depends(get_db)
):
    """Generate SHAP feature contribution charts and natural language explanations."""
    if not validate_smiles(drug_smiles) or not validate_smiles(food_smiles):
        raise HTTPException(status_code=400, detail="Invalid chemical structure (SMILES) provided.")

    # 1. Load ML models
    model_path = MODELS_DIR / "xgboost_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    selector_path = MODELS_DIR / "feature_selector.pkl"
    names_path = MODELS_DIR / "feature_names.json"
    
    if not (model_path.exists() and scaler_path.exists() and selector_path.exists() and names_path.exists()):
        return {
            "shap_available": False,
            "explanation": "Explainable AI (SHAP) details will become available once the machine learning model has been fully trained in the Admin Dashboard.",
            "features": []
        }

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)
        with open(names_path, "r") as f:
            feature_names = json.load(f)
            
        # 2. Build single query pair feature vector
        extractor = DFIFeatureExtractor(use_3d=False, fp_bits=1024)
        x_raw = extractor.extract_pair_vector(drug_smiles, food_smiles)
        raw_df = pd.DataFrame([x_raw], columns=extractor.feature_names)
        
        # Transform
        x_selected = selector.transform(raw_df)
        pred_class = int(model.predict(x_selected)[0])
        probabilities = model.predict_proba(x_selected)[0]
        
        # 3. Create a background dataset on-the-fly for SHAP
        # Query 15 random pairs from database
        bg_drugs = db.execute(text("SELECT name, smiles FROM drugs WHERE smiles IS NOT NULL LIMIT 10")).fetchall()
        bg_foods = db.execute(text("SELECT name, smiles FROM food_compounds WHERE smiles IS NOT NULL LIMIT 10")).fetchall()
        
        bg_features_list = []
        for d in bg_drugs:
            for f in bg_foods:
                try:
                    vec = extractor.extract_pair_vector(d[1], f[1])
                    bg_features_list.append(vec)
                except Exception:
                    pass
                if len(bg_features_list) >= 20:
                    break
            if len(bg_features_list) >= 20:
                break
                
        # If DB is not populated, use dummy background for shape compatibility
        if not bg_features_list:
            bg_features_list = [x_raw] * 5
            
        bg_df = pd.DataFrame(bg_features_list, columns=extractor.feature_names)
        bg_selected = selector.transform(bg_df)
        
        # 4. Initialize SHAP TreeExplainer
        analyzer = DFISHAPAnalyzer(model.model_)
        analyzer.fit(bg_selected)
        
        # Explain single prediction
        exp = analyzer.explain_prediction(x_selected[0], feature_names, top_n=10)
        
        # Enhance explanation list with clean human-readable names
        enhanced_features = []
        for f in exp["top_contributing_features"]:
            enhanced_features.append({
                "feature_id": f["feature"],
                "name": clean_feature_name(f["feature"]),
                "shap_value": f["shap_value"],
                "direction": f["direction"],
                "value": f["feature_value"]
            })
            
        natural_exp = generate_natural_explanation(enhanced_features, pred_class)
        
        return {
            "shap_available": True,
            "predicted_class": pred_class,
            "predicted_label": exp["predicted_label"],
            "confidence": exp["confidence"],
            "base_value": exp["base_value"],
            "features": enhanced_features,
            "explanation": natural_exp
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ SHAP analysis failed: {e}")
        return {
            "shap_available": False,
            "explanation": f"Explainable AI (SHAP) calculation encountered an error: {str(e)}",
            "features": []
        }
