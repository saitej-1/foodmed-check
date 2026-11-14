import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import text
from sklearn.model_selection import train_test_split

# Add path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_config import SessionLocal, engine
from pipeline.feature_extractor import DFIFeatureExtractor
from pipeline.feature_selector import DFIFeatureSelector
from pipeline.model_trainer import DFIModelTrainer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_model():
    print("🧠 Starting DFI Model Training Pipeline...")
    
    # 1. Fetch data from PostgreSQL
    db = SessionLocal()
    try:
        print("  Reading drugs and foods from PostgreSQL...")
        drugs = db.execute(text("SELECT id, name, smiles FROM drugs WHERE smiles IS NOT NULL")).fetchall()
        foods = db.execute(text("SELECT food_id, name, smiles FROM food_compounds WHERE smiles IS NOT NULL")).fetchall()
        
        if not drugs or not foods:
            print("❌ Error: No drug or food molecular data found in database. Seed database first.")
            db.close()
            return
            
        print(f"  Loaded {len(drugs)} drugs and {len(foods)} foods.")
        
        # Read interaction labels
        raw_labels = db.execute(text("SELECT drug_id, food_id, severity FROM dfi_labels")).fetchall()
        db_labels = {}
        for drug_id, food_id, severity in raw_labels:
            db_labels[(drug_id, food_id)] = severity.lower()
            
        print(f"  Loaded {len(db_labels)} labeled interactions.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        db.close()
        return

    # 2. Build pairwise dataset (optimized to avoid Cartesian product RAM OOM)
    records = []
    print("  Generating balanced pairwise training dataset...")
    
    # Build maps for fast lookup
    drugs_dict = {d[0]: (d[1], d[2]) for d in drugs} # id: (name, smiles)
    foods_dict = {f[0]: (f[1], f[2]) for f in foods} # id: (name, smiles)
    
    # 1. Add all labeled interactions (Severe & Moderate)
    labeled_pairs = set()
    severe_added = 0
    moderate_added = 0
    for (drug_id, food_id), severity in db_labels.items():
        if drug_id in drugs_dict and food_id in foods_dict:
            d_name, d_smiles = drugs_dict[drug_id]
            f_name, f_smiles = foods_dict[food_id]
            
            label = 0 if severity == "severe" else 2
            
            # Do not manually oversample, let class_weights handle it
            reps = 1
            for _ in range(reps):
                records.append({
                    "drug_id": drug_id,
                    "drug_name": d_name,
                    "drug_smiles": d_smiles,
                    "food_id": food_id,
                    "food_name": f_name,
                    "food_smiles": f_smiles,
                    "label": label
                })
            
            if label == 0:
                severe_added += reps
            else:
                moderate_added += 1
            labeled_pairs.add((drug_id, food_id))
            
    print(f"  Added {len(records)} labeled interactions ({severe_added} severe, {moderate_added} moderate).")
    
    # 2. Randomly sample unlabeled pairs (Safe) to balance the dataset
    target_safe_count = min(10000, len(records)) 
    import random
    random.seed(42)
    
    drugs_list = list(drugs_dict.keys())
    foods_list = list(foods_dict.keys())
    
    safe_count = 0
    attempts = 0
    max_attempts = target_safe_count * 10
    
    while safe_count < target_safe_count and attempts < max_attempts:
        attempts += 1
        rand_drug = random.choice(drugs_list)
        rand_food = random.choice(foods_list)
        
        if (rand_drug, rand_food) not in db_labels and (rand_drug, rand_food) not in labeled_pairs:
            d_name, d_smiles = drugs_dict[rand_drug]
            f_name, f_smiles = foods_dict[rand_food]
            
            records.append({
                "drug_id": rand_drug,
                "drug_name": d_name,
                "drug_smiles": d_smiles,
                "food_id": rand_food,
                "food_name": f_name,
                "food_smiles": f_smiles,
                "label": 1 # Safe
            })
            labeled_pairs.add((rand_drug, rand_food))
            safe_count += 1
            
    print(f"  Added {safe_count} random safe interactions for balance.")
    print(f"  Generated {len(records)} total training examples.")

    # 3. Extract molecular descriptors using RDKit (Phase 2)
    print("  Extracting descriptors (2D molecular features)...")
    extractor = DFIFeatureExtractor(use_3d=False, fp_bits=1024)
    X, y = extractor.build_dataset(records, smiles_col_drug="drug_smiles", smiles_col_food="food_smiles", label_col="label")

    if X.empty:
        print("❌ Error: Feature extraction failed.")
        db.close()
        return

    # 4. Feature Selection (Phase 2 Selection)
    # VT -> Pearson -> LASSO -> RFECV
    selector = DFIFeatureSelector(
        variance_threshold=0.01,  # Set lower as binary Morgan FP bits have low variance individually
        correlation_threshold=0.85, # Relaxed correlation
        lasso_alpha=0.01, # Relaxed lasso
        target_features=60, # Keep more features for accuracy
        cv_folds=5
    )
    
    print("  Running feature selection pipeline...")
    X_selected_matrix = selector.fit_transform(X, y)
    selected_features = selector.selected_features_
    
    # Save feature names and selector
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(selected_features, f, indent=2)
    joblib.dump(selector.scaler_, MODELS_DIR / "scaler.pkl")
    joblib.dump(selector, MODELS_DIR / "feature_selector.pkl")

    # 5. Split and Train Model (Phase 3)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_matrix, y.values, 
        test_size=0.2, 
        stratify=y.values, 
        random_state=42
    )
    
    trainer = DFIModelTrainer(
        n_classes=3,
        cv_folds=5,
        tune_hyperparams=False, # Use reliable default params for higher test accuracy
        random_state=42
    )
    
    trainer.fit(X_train, y_train)
    report = trainer.evaluate(X_test, y_test)
    
    # Save model files
    trainer.save(str(MODELS_DIR / "xgboost_model.pkl"))
    trainer.export_model_card(str(MODELS_DIR / "model_card.json"))

    # 6. Save Model Metadata to Database
    try:
        # Check active version count
        db.execute(text("UPDATE model_metadata SET active = FALSE"))
        
        # Insert new version metadata
        version = f"v{report['accuracy']:.4f}"
        db.execute(
            text("INSERT INTO model_metadata (version, accuracy, f1_score, best_params, active) "
                 "VALUES (:v, :acc, :f1, :params, TRUE)"),
            {
                "v": version,
                "acc": report["accuracy"],
                "f1": report["f1_weighted"],
                "params": json.dumps(trainer.best_params_)
            }
        )
        db.commit()
        print(f"  ✅ Saved metadata for model version '{version}' to database!")
    except Exception as e:
        print(f"  ⚠ Failed to save model metadata: {e}")
        db.rollback()
        
    db.close()
    print("🎉 DFI ML Training complete!")

if __name__ == "__main__":
    train_model()
