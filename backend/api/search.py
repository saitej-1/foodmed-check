from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from database.db_config import get_db

router = APIRouter(prefix="/search", tags=["search"])

@router.get("/drugs")
def search_drugs(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Search drugs by name or category with prefix matching."""
    query = f"%{q}%"
    results = db.execute(
        text("SELECT id, name, smiles, formula, molecular_weight, description, class_name, drugbank_id "
             "FROM drugs "
             "WHERE name ILIKE :q OR class_name ILIKE :q "
             "ORDER BY name ASC LIMIT 10"),
        {"q": query}
    ).fetchall()
    
    return [
        {
            "id": r[0],
            "name": r[1],
            "smiles": r[2],
            "formula": r[3],
            "molecular_weight": r[4],
            "description": r[5],
            "class_name": r[6],
            "drugbank_id": r[7]
        }
        for r in results
    ]

@router.get("/foods")
def search_foods(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Search foods by name or category with prefix matching."""
    query = f"%{q}%"
    # Search foods
    results = db.execute(
        text("SELECT id, name, category, description "
             "FROM foods "
             "WHERE name ILIKE :q OR category ILIKE :q "
             "ORDER BY name ASC LIMIT 10"),
        {"q": query}
    ).fetchall()
    
    # For each food, join its bioactive compounds
    foods_list = []
    for r in results:
        fid = r[0]
        compounds = db.execute(
            text("SELECT name, smiles, pubchem_cid, description FROM food_compounds WHERE food_id = :fid"),
            {"fid": fid}
        ).fetchall()
        
        foods_list.append({
            "id": fid,
            "name": r[1],
            "category": r[2],
            "description": r[3],
            "compounds": [
                {
                    "name": c[0],
                    "smiles": c[1],
                    "pubchem_cid": c[2],
                    "description": c[3]
                }
                for c in compounds
            ]
        })
        
    return foods_list
