import os
import sys
import json
from pathlib import Path
from sqlalchemy import text

# Add path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_config import SessionLocal

# Import RDKit modules safely
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("❌ Error: RDKit is required to generate 3D models.")
    sys.exit(1)

def generate_3d_coordinates(smiles: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Add hydrogens
        mol3d = Chem.AddHs(mol)
        # Embed in 3D space
        if AllChem.EmbedMolecule(mol3d, randomSeed=42) < 0:
            return None
        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol3d)
        
        # Get coordinates
        conf = mol3d.GetConformer()
        atoms = []
        for i, atom in enumerate(mol3d.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "id": i,
                "element": atom.GetSymbol(),
                "pos": [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)]
            })
        
        bonds = []
        for bond in mol3d.GetBonds():
            val = bond.GetBondTypeAsDouble()
            bonds.append({
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx(),
                "value": int(val) if val.is_integer() else val
            })
            
        return {"atoms": atoms, "bonds": bonds}
    except Exception as e:
        print(f"    ⚠ Conformer generation failed for SMILES: {smiles}. Error: {e}")
        return None

def main():
    print("🔬 Generating 3D Cartesian coordinates for database compounds...")
    
    db = SessionLocal()
    compounds = []
    
    try:
        # Fetch drug compounds
        drug_rows = db.execute(text("SELECT name, smiles FROM drug_compounds WHERE smiles IS NOT NULL")).fetchall()
        for name, smiles in drug_rows:
            compounds.append((name, smiles))
            
        # Fetch food compounds
        food_rows = db.execute(text("SELECT name, smiles FROM food_compounds WHERE smiles IS NOT NULL")).fetchall()
        for name, smiles in food_rows:
            compounds.append((name, smiles))
            
        print(f"  Loaded {len(compounds)} unique active compounds from PostgreSQL.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        db.close()
        sys.exit(1)
    finally:
        db.close()
        
    molecules_3d = {}
    for name, smiles in compounds:
        print(f"  🧬 Computing 3D structure for: '{name}'...")
        mol_data = generate_3d_coordinates(smiles)
        if mol_data:
            molecules_3d[name] = mol_data
        else:
            print(f"    ❌ Failed to compute conformers for '{name}'")
            
    print(f"✅ Generated {len(molecules_3d)} 3D models.")
    
    # 2. Update src/data/interactions.js
    frontend_path = Path(os.getenv(
        "INTERACTIONS_JS_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "food-drug-interact-app" / "src" / "data" / "interactions.js"),
    ))
    if not frontend_path.exists():
        print(f"❌ Frontend interactions.js not found at: {frontend_path}")
        sys.exit(1)
        
    # Read current interactions.js content
    with open(frontend_path, "r") as f:
        js_content = f.read()
        
    # Find the position of 'export const MOLECULES ='
    idx = js_content.find("export const MOLECULES =")
    if idx == -1:
        print("❌ Could not find MOLECULES export in interactions.js")
        sys.exit(1)
        
    # Generate JS output for MOLECULES
    molecules_js = "export const MOLECULES = " + json.dumps(molecules_3d, indent=2) + ";\n"
    
    # Concatenate the file content up to export const MOLECULES, and append the new molecules
    updated_content = js_content[:idx] + molecules_js
    
    with open(frontend_path, "w") as f:
        f.write(updated_content)
        
    print(f"🎉 Updated {frontend_path} with 3D coordinates for all active compounds!")

if __name__ == "__main__":
    main()
