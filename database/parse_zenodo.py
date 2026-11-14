import re
import os

from pathlib import Path

# Path to the FooDrugs SQL dump. Override with the ZENODO_SQL_PATH environment variable.
ZENODO_SQL_PATH = os.getenv(
    "ZENODO_SQL_PATH",
    str(Path(__file__).resolve().parent.parent / "datasets" / "raw" / "FinalFooDrugs_v4.sql"),
)

# Data structures to match our existing seeder interface
FOODS_DB = []
DRUGS_DB = []
INTERACTIONS = []

def append_fallback_data(seen_drugs, seen_foods):
    fallback_drugs = [
        {"id": "atorvastatin", "name": "Atorvastatin (Lipitor)", "description": "CYP3A4 metabolized.", "moleculeName": "Atorvastatin"},
        {"id": "warfarin", "name": "Warfarin (Coumadin)", "description": "Vitamin K antagonist.", "moleculeName": "Warfarin"},
        {"id": "ciprofloxacin", "name": "Ciprofloxacin (Cipro)", "description": "Chelates with calcium.", "moleculeName": "Ciprofloxacin"},
        {"id": "nardil", "name": "Phenelzine (Nardil)", "description": "MAO inhibitor.", "moleculeName": "Phenelzine"},
        {"id": "lisinopril", "name": "Lisinopril (Prinivil)", "description": "Retains potassium.", "moleculeName": "Lisinopril"},
        {"id": "acetaminophen", "name": "Acetaminophen (Tylenol)", "description": "Liver toxic with alcohol.", "moleculeName": "Acetaminophen"}
    ]
    fallback_foods = [
        {"id": "grapefruit", "name": "Grapefruit & Juice", "category": "Fruit", "description": "Bergamottin is a cyp3a4 inhibitor.", "moleculeName": "Bergamottin"},
        {"id": "leafy_greens", "name": "Leafy Greens (Spinach, Kale)", "category": "Vegetables", "description": "High in Vitamin K1.", "moleculeName": "Vitamin K1"},
        {"id": "dairy", "name": "Dairy Products (Milk, Yogurt)", "category": "Dairy", "description": "Rich in Calcium.", "moleculeName": "Calcium Carbonate"},
        {"id": "aged_cheese", "name": "Aged Cheese & Cured Meats", "category": "Fermented", "description": "High in Tyramine.", "moleculeName": "Tyramine"},
        {"id": "bananas", "name": "Bananas & Avocados", "category": "Fruit", "description": "High in potassium.", "moleculeName": "Potassium Ion"},
        {"id": "alcohol", "name": "Alcoholic Beverages", "category": "Beverage", "description": "Interacts with sedatives and acetaminophen.", "moleculeName": "Ethanol"}
    ]
    fallback_interactions = [
        ("grapefruit", "atorvastatin", "severe", "Severe Grapefruit-Statin Toxicity", "Furanocoumarins inhibit CYP3A4.", "Avoid grapefruit."),
        ("leafy_greens", "warfarin", "severe", "Vitamin K Counteraction", "Vitamin K counters warfarin.", "Keep greens intake stable."),
        ("dairy", "ciprofloxacin", "moderate", "Antibiotic Chelation", "Calcium chelates ciprofloxacin.", "Avoid dairy with drug."),
        ("aged_cheese", "nardil", "severe", "Hypertensive Crisis", "Tyramine triggers hypertensive crisis.", "Low tyramine diet."),
        ("bananas", "lisinopril", "moderate", "Hyperkalemia Warning", "ACE inhibitors retain potassium.", "Avoid high potassium foods."),
        ("alcohol", "acetaminophen", "severe", "Hepatotoxic Risk", "Alcohol shifts acetaminophen metabolism to NAPQI.", "Avoid alcohol.")
    ]

    for d in fallback_drugs:
        if d["id"] not in seen_drugs:
            seen_drugs.add(d["id"])
            DRUGS_DB.append(d)
            
    for f in fallback_foods:
        if f["id"] not in seen_foods:
            seen_foods.add(f["id"])
            FOODS_DB.append(f)
            
    for item in fallback_interactions:
        INTERACTIONS.append(item)

def parse_zenodo_dataset(limit=50000):
    """
    Parses the massive 415MB Zenodo MySQL dump line-by-line using high-performance string splitting.
    Extracts up to `limit` unique drug-food interactions.
    """
    if not os.path.exists(ZENODO_SQL_PATH):
        print(f"⚠ Zenodo SQL file not found at {ZENODO_SQL_PATH}")
        return

    print(f"📖 Parsing Zenodo database (Limit: {limit} unique interactions)...")
    
    unique_pairs = set()
    seen_drugs = set()
    seen_foods = set()
    
    with open(ZENODO_SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("INSERT INTO `TM_interactions`"):
                # Fast split by '), ('
                parts = line.split("),(")
                for part in parts:
                    sub_parts = part.split("'")
                    if len(sub_parts) >= 5:
                        food_raw = sub_parts[-4]
                        drug_raw = sub_parts[-2]
                        
                        # Clean strings (escape sequences from SQL)
                        food_clean = food_raw.replace("\\'", "'").strip()
                        drug_clean = drug_raw.replace("\\'", "'").strip()
                        
                        if not food_clean or not drug_clean:
                            continue
                            
                        # Create IDs
                        food_id = re.sub(r'[^a-z0-9]', '_', food_clean.lower())
                        drug_id = re.sub(r'[^a-z0-9]', '_', drug_clean.lower())
                        
                        pair = (drug_id, food_id)
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            
                            # Populate DRUGS_DB
                            if drug_id not in seen_drugs:
                                seen_drugs.add(drug_id)
                                DRUGS_DB.append({
                                    "id": drug_id,
                                    "name": drug_clean.capitalize(),
                                    "description": "Extracted from Zenodo FooDrugs text-mining dataset.",
                                    "moleculeName": drug_clean.capitalize()
                                })
                                
                            # Populate FOODS_DB
                            if food_id not in seen_foods:
                                seen_foods.add(food_id)
                                FOODS_DB.append({
                                    "id": food_id,
                                    "name": food_clean.capitalize(),
                                    "moleculeName": food_clean.capitalize(),
                                    "description": "Extracted from Zenodo FooDrugs dataset."
                                })
                                
                            # Add interaction (Tuple format for seed.py)
                            INTERACTIONS.append(
                                (food_id, drug_id, "moderate", f"Interaction: {drug_clean} + {food_clean}", "Text-mined co-occurrence", "Monitor usage")
                            )
                            
                            if len(unique_pairs) >= limit:
                                print(f"✅ Reached extraction limit of {limit} unique pairs.")
                                append_fallback_data(seen_drugs, seen_foods)
                                return

    print(f"✅ Finished parsing. Extracted {len(unique_pairs)} pairs.")
    append_fallback_data(seen_drugs, seen_foods)

# Automatically parse upon import to provide the DB dictionaries
parse_zenodo_dataset(limit=50000)
