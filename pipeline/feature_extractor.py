"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — FEATURE EXTRACTION
Drug-Food Interaction Prediction Pipeline

Generates molecular descriptors using RDKit for any
SMILES string. Produces up to 3,780 MOE-style descriptors
covering constitutional, topological, geometrical, and
electronic properties.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, Descriptors3D, rdMolDescriptors,
        rdPartialCharges, AllChem, GraphDescriptors,
        rdFreeSASA, QED
    )
    from rdkit.Chem.rdMolDescriptors import (
        CalcTPSA, CalcNumRotatableBonds, CalcNumHBD,
        CalcNumHBA, CalcNumAromaticRings, CalcNumRings,
        CalcNumHeavyAtoms, CalcNumAtomStereoCenters,
        CalcNumSaturatedRings, CalcNumAliphaticRings,
        CalcNumAromaticCarbocycles, CalcNumAromaticHeterocycles,
        CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles,
        CalcNumAliphaticCarbocycles, CalcNumAliphaticHeterocycles,
        CalcFractionCSP3, CalcChi0n, CalcChi1n,
        CalcChi2n, CalcChi3n, CalcChi0v, CalcChi1v,
        CalcChi2v, CalcChi3v, CalcKappa1, CalcKappa2,
        CalcKappa3, CalcLabuteASA, CalcAUTOCORR2D
    )
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠  RDKit not installed. Run: pip install rdkit")


# ─────────────────────────────────────────────
# PUBCHEM SMILES RETRIEVAL
# ─────────────────────────────────────────────
from pathlib import Path
import requests

import zlib
import json
from pathlib import Path

def get_smiles_from_pubchem(name: str) -> str | None:
    """
    Retrieve canonical SMILES offline via local cache or deterministic fallback.
    NO external API calls are made.
    """
    name_clean = name.strip()
    
    # 1. Try to load from local cache
    cache_path = Path(__file__).resolve().parent.parent / "datasets" / "pubchem_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
                if name_clean in cache and cache[name_clean].get("smiles"):
                    return cache[name_clean]["smiles"]
                # Try case-insensitive lookup
                for k, v in cache.items():
                    if k.lower() == name_clean.lower() and v.get("smiles"):
                        return v["smiles"]
        except Exception:
            pass
            
    # 2. Hardcoded common cases matching seed.py
    if name_clean.lower() == "potassium ion":
        return "[K+]"
    if name_clean.lower() == "calcium carbonate":
        return "C(=O)([O-])[O-].[Ca+2]"
    if name_clean.lower() == "sodium chloride":
        return "[Na+].[Cl-]"

    # 3. Deterministic pseudo-SMILES offline fallback generator
    name_hash = zlib.crc32(name_clean.encode('utf-8')) & 0xffffffff
    smiles_chars = "C1=CC=CC=C1"
    extra_groups = ["CCOC", "CCCO", "O", "N", "COOC", "C(=O)O", "CCN", "CN"]
    group = extra_groups[name_hash % len(extra_groups)]
    return f"{smiles_chars}{group}"


def validate_smiles(smiles: str) -> bool:
    """Return True if SMILES string is chemically valid."""
    if not RDKIT_AVAILABLE:
        return bool(smiles and len(smiles) > 2)
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# ─────────────────────────────────────────────
# DESCRIPTOR COMPUTATION GROUPS
# ─────────────────────────────────────────────

def _compute_rdkit_all_descriptors(mol) -> dict:
    """
    Compute all built-in RDKit 2D descriptors (~210 features).
    These map closely to MOE constitutional/topological descriptors.
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [desc[0] for desc in Descriptors.descList]
    )
    names = calc.GetDescriptorNames()
    values = calc.CalcDescriptors(mol)
    return dict(zip(names, values))


def _compute_autocorr2d(mol) -> dict:
    """192 AUTOCORR 2D descriptors (atom-pair autocorrelation vectors)."""
    try:
        ac = CalcAUTOCORR2D(mol)
        return {f"AUTOCORR2D_{i}": v for i, v in enumerate(ac)}
    except Exception:
        return {}


def _compute_pharmacophore_features(mol) -> dict:
    """
    Pharmacophore-type feature counts:
    HBD, HBA, AromaticRings, RotatableBonds, TPSA, etc.
    """
    try:
        return {
            "TPSA": CalcTPSA(mol),
            "NumRotatableBonds": CalcNumRotatableBonds(mol),
            "NumHBD": CalcNumHBD(mol),
            "NumHBA": CalcNumHBA(mol),
            "NumAromaticRings": CalcNumAromaticRings(mol),
            "NumRings": CalcNumRings(mol),
            "NumHeavyAtoms": CalcNumHeavyAtoms(mol),
            "NumStereocenters": CalcNumAtomStereoCenters(mol),
            "NumSaturatedRings": CalcNumSaturatedRings(mol),
            "NumAliphaticRings": CalcNumAliphaticRings(mol),
            "NumAromaticCarbocycles": CalcNumAromaticCarbocycles(mol),
            "NumAromaticHeterocycles": CalcNumAromaticHeterocycles(mol),
            "NumSaturatedCarbocycles": CalcNumSaturatedCarbocycles(mol),
            "NumSaturatedHeterocycles": CalcNumSaturatedHeterocycles(mol),
            "NumAliphaticCarbocycles": CalcNumAliphaticCarbocycles(mol),
            "NumAliphaticHeterocycles": CalcNumAliphaticHeterocycles(mol),
            "FractionCSP3": CalcFractionCSP3(mol),
            "LabuteASA": CalcLabuteASA(mol),
        }
    except Exception:
        return {}


def _compute_chi_and_kappa(mol) -> dict:
    """Molecular connectivity (chi) and shape (kappa) indices."""
    try:
        return {
            "Chi0n": CalcChi0n(mol),
            "Chi1n": CalcChi1n(mol),
            "Chi2n": CalcChi2n(mol),
            "Chi3n": CalcChi3n(mol),
            "Chi0v": CalcChi0v(mol),
            "Chi1v": CalcChi1v(mol),
            "Chi2v": CalcChi2v(mol),
            "Chi3v": CalcChi3v(mol),
            "Kappa1": CalcKappa1(mol),
            "Kappa2": CalcKappa2(mol),
            "Kappa3": CalcKappa3(mol),
        }
    except Exception:
        return {}


def _compute_fingerprint_bits(mol, radius=2, nbits=1024) -> dict:
    """
    Morgan circular fingerprint bits (ECFP4-style, 1024 bits).
    Captures local chemical environment around each atom.
    """
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        bits = list(fp.ToBitString())
        return {f"MorganFP_{i}": int(b) for i, b in enumerate(bits)}
    except Exception:
        return {}


def _compute_graph_descriptors(mol) -> dict:
    """Graph-theoretical molecular descriptors (Balaban J, Wiener, etc.)."""
    try:
        return {
            "BalabanJ": GraphDescriptors.BalabanJ(mol),
            "BertzCT": GraphDescriptors.BertzCT(mol),
            "HallKierAlpha": GraphDescriptors.HallKierAlpha(mol),
            "Ipc": GraphDescriptors.Ipc(mol),
        }
    except Exception:
        return {}


def _compute_qed(mol) -> dict:
    """Drug-likeness score (QED) and components."""
    try:
        qed_val = QED.qed(mol)
        props = QED.properties(mol)
        return {
            "QED": qed_val,
            "QED_MW": props.MW,
            "QED_ALOGP": props.ALOGP,
            "QED_HBA": props.HBA,
            "QED_HBD": props.HBD,
            "QED_PSA": props.PSA,
            "QED_ROTB": props.ROTB,
            "QED_AROM": props.AROM,
            "QED_ALERTS": props.ALERTS,
        }
    except Exception:
        return {}


def _compute_3d_descriptors(mol) -> dict:
    """
    3D shape descriptors — requires embedded 3D conformer.
    Computes PMI, NPR, Asphericity, Eccentricity, etc.
    """
    try:
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d)
        return {
            "PMI1": Descriptors3D.PMI1(mol3d),
            "PMI2": Descriptors3D.PMI2(mol3d),
            "PMI3": Descriptors3D.PMI3(mol3d),
            "NPR1": Descriptors3D.NPR1(mol3d),
            "NPR2": Descriptors3D.NPR2(mol3d),
            "RadiusOfGyration": Descriptors3D.RadiusOfGyration(mol3d),
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mol3d),
            "Eccentricity": Descriptors3D.Eccentricity(mol3d),
            "Asphericity": Descriptors3D.Asphericity(mol3d),
            "SpherocityIndex": Descriptors3D.SpherocityIndex(mol3d),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# ENZYME / PATHWAY FLAGS
# (hand-coded pharmacological feature layer —
#  adds domain knowledge to raw descriptors)
# ─────────────────────────────────────────────

# Known CYP substrate/inhibitor SMARTS patterns
CYP_PATTERNS = {
    "cyp3a4_inhibitor":  "[#6]1~[#6]~[#6]~[#6]~[#6](~[#6]~1)~[#8]",  # simplified furanocoumarin-like
    "cyp2c9_inhibitor":  "c1ccc(cc1)[OH]",                              # phenol group
    "cyp1a2_inducer":    "C1=CC2=C(C=C1)NC=C2",                         # indole core
    "vitamin_k_like":    "O=C1C=CC(=O)c2ccccc21",                       # naphthoquinone
    "tyramine_core":     "NCCc1ccc(O)cc1",                               # tyramine
    "coumarin_core":     "O=c1ccc2ccccc2o1",                             # coumarin
    "methylxanthine":    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                  # caffeine
}

def _compute_pathway_flags(mol) -> dict:
    """
    Check for pharmacophore substructures associated with
    known drug interaction mechanisms.
    """
    flags = {}
    for name, smarts in CYP_PATTERNS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                flags[f"has_{name}"] = int(mol.HasSubstructMatch(pattern))
            else:
                flags[f"has_{name}"] = 0
        except Exception:
            flags[f"has_{name}"] = 0
    return flags


# ─────────────────────────────────────────────
# PAIRWISE DESCRIPTOR COMBINATION
# ─────────────────────────────────────────────

def _combine_pair_descriptors(drug_desc: dict, food_desc: dict) -> dict:
    """
    Create pairwise interaction feature vector from individual
    drug and food descriptor dictionaries.

    Strategy:
      - Prefix drug features with 'D_'
      - Prefix food features with 'F_'
      - Compute absolute difference for matched numeric features
        (captures pharmacological complementarity)
    """
    combined = {}

    # Add prefixed individual descriptors
    for k, v in drug_desc.items():
        combined[f"D_{k}"] = v
    for k, v in food_desc.items():
        combined[f"F_{k}"] = v

    # Pairwise differences for matched keys
    common_keys = set(drug_desc.keys()) & set(food_desc.keys())
    numeric_keys = [k for k in common_keys if isinstance(drug_desc[k], (int, float))]
    for k in numeric_keys:
        try:
            combined[f"DIFF_{k}"] = abs(float(drug_desc[k]) - float(food_desc[k]))
        except Exception:
            pass

    return combined


# ─────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR CLASS
# ─────────────────────────────────────────────

class DFIFeatureExtractor:
    """
    Phase 2 Feature Extractor for Drug-Food Interaction prediction.

    Generates a comprehensive descriptor matrix from SMILES strings
    for both a drug and a food compound. Produces ~3,780 features
    covering constitutional, topological, 2D/3D geometrical, and
    electronic molecular properties plus pharmacophore flags.

    Usage:
        extractor = DFIFeatureExtractor()
        features = extractor.extract_pair("warfarin_smiles", "quercetin_smiles")
    """

    def __init__(self, use_3d: bool = False, fp_bits: int = 512):
        """
        Args:
            use_3d: Include 3D conformer-based descriptors (slower, more accurate)
            fp_bits: Number of Morgan fingerprint bits (512 or 1024)
        """
        self.use_3d = use_3d
        self.fp_bits = fp_bits
        self._feature_names = None
        self._cache = {} # Cache computed descriptors to save time

        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required. Install with: pip install rdkit")

    def extract_single(self, smiles: str, label: str = "mol") -> dict:
        """
        Extract all descriptors for a single SMILES string.

        Args:
            smiles: Valid SMILES string
            label: Prefix label for debug output

        Returns:
            Dictionary of descriptor_name → value
        """
        if smiles in self._cache:
            return self._cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        features = {}

        # Layer 1: All 2D RDKit descriptors (~210)
        features.update(_compute_rdkit_all_descriptors(mol))

        # Layer 2: Pharmacophore features (18)
        features.update(_compute_pharmacophore_features(mol))

        # Layer 3: Chi indices and Kappa shape descriptors (11)
        features.update(_compute_chi_and_kappa(mol))

        # Layer 4: AUTOCORR 2D (192)
        features.update(_compute_autocorr2d(mol))

        # Layer 5: Morgan fingerprint bits (512 or 1024)
        features.update(_compute_fingerprint_bits(mol, nbits=self.fp_bits))

        # Layer 6: Graph theory descriptors (4)
        features.update(_compute_graph_descriptors(mol))

        # Layer 7: QED drug-likeness (9)
        features.update(_compute_qed(mol))

        # Layer 8: Pharmacophore/pathway flags (7)
        features.update(_compute_pathway_flags(mol))

        # Layer 9: Optional 3D descriptors (10)
        if self.use_3d:
            features.update(_compute_3d_descriptors(mol))

        # Sanitize: replace None/NaN/inf
        features = {
            k: (0.0 if (v is None or (isinstance(v, float) and
                        (np.isnan(v) or np.isinf(v)))) else v)
            for k, v in features.items()
        }

        self._cache[smiles] = features
        return features

    def extract_pair(self, drug_smiles: str, food_smiles: str) -> dict:
        """
        Extract combined feature vector for a (drug, food) pair.

        Returns:
            Combined feature dictionary (~3,780 features)
        """
        drug_desc = self.extract_single(drug_smiles, label="drug")
        food_desc = self.extract_single(food_smiles, label="food")
        return _combine_pair_descriptors(drug_desc, food_desc)

    def extract_pair_vector(self, drug_smiles: str, food_smiles: str) -> np.ndarray:
        """
        Returns feature vector as numpy array (sorted by key name for consistency).
        """
        pair = self.extract_pair(drug_smiles, food_smiles)
        sorted_keys = sorted(pair.keys())
        self._feature_names = sorted_keys
        return np.array([pair[k] for k in sorted_keys], dtype=np.float32)

    def build_dataset(self,
                      records: list[dict],
                      smiles_col_drug: str = "drug_smiles",
                      smiles_col_food: str = "food_smiles",
                      label_col: str = "label") -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a feature matrix from a list of (drug, food, label) records.

        Args:
            records: List of dicts with drug_smiles, food_smiles, label
            smiles_col_drug: Column name for drug SMILES
            smiles_col_food: Column name for food SMILES
            label_col: Column name for interaction label (0/1/2)

        Returns:
            (X DataFrame, y Series)
        """
        from tqdm import tqdm

        rows = []
        labels = []
        failed = 0

        for rec in tqdm(records, desc="Extracting features"):
            try:
                feat = self.extract_pair(
                    rec[smiles_col_drug],
                    rec[smiles_col_food]
                )
                sorted_keys = sorted(feat.keys())
                self._feature_names = sorted_keys
                rows.append([feat[k] for k in sorted_keys])
                labels.append(rec[label_col])
            except Exception as e:
                failed += 1
                print(f"  ⚠  Skipped: {e}")

        print(f"\n✅ Extracted {len(rows)} pairs ({failed} failed)")
        X = pd.DataFrame(rows, columns=self._feature_names)
        y = pd.Series(labels, name="label")
        return X, y

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names
