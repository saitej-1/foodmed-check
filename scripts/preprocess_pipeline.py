import os
import sys
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from sqlalchemy import text

# Add path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_config import SessionLocal
from pipeline.feature_extractor import validate_smiles, get_smiles_from_pubchem

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"

def parse_drugbank_if_exists(db_session):
    """
    If the admin manually uploaded the DrugBank XML zip, we unzip and parse it
    to load official DrugBank IDs, chemical attributes, and warnings.
    """
    zip_path = RAW_DIR / "drugbank_all_full_database.xml.zip"
    if not zip_path.exists():
        print("ℹ No manually uploaded DrugBank zip file found. Skipping DrugBank parser.")
        return
    
    print("🔓 Extracting DrugBank raw database XML...")
    try:
        # Extract XML
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            xml_name = [name for name in zip_ref.namelist() if name.endswith('.xml')][0]
            xml_path = zip_ref.extract(xml_name, path=str(RAW_DIR))
            
        print(f"🎬 Parsing DrugBank XML: {xml_name} (this may take a minute)...")
        # Parse tree (streaming parser for large files)
        context = ET.iterparse(xml_path, events=('end',))
        
        # NS namespace
        ns = '{http://www.drugbank.ca}'
        
        drug_count = 0
        for event, elem in context:
            if elem.tag == f'{ns}drug':
                db_id = elem.findtext(f'{ns}drugbank-id[@primary="true"]')
                name = elem.findtext(f'{ns}name')
                description = elem.findtext(f'{ns}description')
                
                # Check for SMILES in properties
                smiles = None
                properties = elem.find(f'{ns}calculated-properties')
                if properties is not None:
                    for prop in properties.findall(f'{ns}property'):
                        kind = prop.findtext(f'{ns}kind')
                        if kind == 'SMILES':
                            smiles = prop.findtext(f'{ns}value')
                            break
                            
                # Check for food warnings
                food_interactions = []
                interactions = elem.find(f'{ns}food-interactions')
                if interactions is not None:
                    for interact in interactions.findall(f'{ns}food-interaction'):
                        food_interactions.append(interact.text)
                
                if name and smiles and validate_smiles(smiles):
                    # Ingest or update drug in DB
                    db_session.execute(
                        text("INSERT INTO drugs (drugbank_id, name, smiles, description) "
                             "VALUES (:dbid, :name, :s, :desc) "
                             "ON CONFLICT (name) DO UPDATE SET drugbank_id = EXCLUDED.drugbank_id, smiles = EXCLUDED.smiles"),
                        {"dbid": db_id, "name": name, "s": smiles, "desc": description}
                    )
                    
                    # Store warnings if present
                    if food_interactions:
                        print(f"   Fetched {len(food_interactions)} food warnings for {name}")
                        
                    drug_count += 1
                
                elem.clear() # Free memory
                
        db_session.commit()
        print(f"✅ Successfully parsed and synced {drug_count} drugs from manual DrugBank database!")
        
        # Clean up extracted XML file to save disk space
        os.remove(xml_path)
    except Exception as e:
        print(f"⚠ Error parsing DrugBank XML: {e}")
        db_session.rollback()

def normalize_names_and_smiles():
    """Validates, removes duplicates, and fills in missing chemical properties from PubChem."""
    print("🧹 Normalizing and validating database names & SMILES...")
    db = SessionLocal()
    try:
        # Load drugs with missing smiles
        drugs = db.execute(text("SELECT id, name FROM drugs WHERE smiles IS NULL OR smiles = ''")).fetchall()
        for d in drugs:
            did, name = d
            print(f"  Attempting to resolve missing structure for: '{name}'")
            smiles = get_smiles_from_pubchem(name)
            if smiles and validate_smiles(smiles):
                db.execute(text("UPDATE drugs SET smiles = :s WHERE id = :id"), {"s": smiles, "id": did})
                print(f"    Saved SMILES: {smiles[:30]}...")
        
        # Load food compounds with missing smiles
        compounds = db.execute(text("SELECT id, name FROM food_compounds WHERE smiles IS NULL OR smiles = ''")).fetchall()
        for c in compounds:
            cid, name = c
            print(f"  Attempting to resolve missing structure for: '{name}'")
            smiles = get_smiles_from_pubchem(name)
            if smiles and validate_smiles(smiles):
                db.execute(text("UPDATE food_compounds SET smiles = :s WHERE id = :id"), {"s": smiles, "id": cid})
                print(f"    Saved SMILES: {smiles[:30]}...")

        # Remove duplicate records
        print("  Removing duplicate drug records...")
        db.execute(
            text("DELETE FROM drugs a USING drugs b WHERE a.id < b.id AND a.name = b.name")
        )
        print("  Removing duplicate food records...")
        db.execute(
            text("DELETE FROM foods a USING foods b WHERE a.id < b.id AND a.name = b.name")
        )
        
        db.commit()
        print("✅ Normalization and deduplication complete!")
    except Exception as e:
        print(f"❌ Error during preprocessing normalization: {e}")
        db.rollback()
    finally:
        db.close()

def main():
    print("🎬 Starting DFI Platform Ingestion & Preprocessing Pipeline...")
    db = SessionLocal()
    parse_drugbank_if_exists(db)
    db.close()
    normalize_names_and_smiles()
    print("🎉 Ingestion & preprocessing pipeline finished!")

if __name__ == "__main__":
    main()
