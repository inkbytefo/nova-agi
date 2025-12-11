## Developer: inkbytefo
## Modified: 2025-12-11

import numpy as np
import pandas as pd
import requests
import io
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    Chem = None
    Descriptors = None

logger = logging.getLogger(__name__)

# Expanded fallback SMILES list (>100 entries) for robust offline testing
DEFAULT_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccccc1",
    "C1CCCCC1",
    "CCO",
    "CC(=O)O",
    "C1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)N",
    "CC1=CC=CC=C1",
    "ClC1=CC=CC=C1",
    "OC1=CC=CC=C1O",
    "COC1=CC=CC=C1",
    "CCN(CC)CC",
    "C(C(C(C(C(C(=O)O)O)O)O)O)O",
    "C1=CN=CN1",
    "C1=CC=C2C(=C1)C=CC3=CC=CC=C32",
    "C1=CC=C2C(=C1)C=CC=C2",
    "C1=CC=C(C=C1)C(=O)O",
    "C1=CC=C(C=C1)N(=O)=O",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCCCCCC",
    "CCCCCCCC",
    "CC(C)C",
    "CC(C)CC",
    "CC(C)CCC",
    "CC(C)CCCC",
    "CC(C)CCCCC",
    "CCC(CC)CC",
    "CCCO",
    "CCCOC",
    "CCOCC",
    "CCOC(C)=O",
    "CC(C)O",
    "CC(C)CO",
    "CC(C)C(=O)O",
    "CC(C)C(=O)OC",
    "CC(C)C(=O)N",
    "CC(C)C(N)O",
    "CC(C)C(O)O",
    "CC(C)C(O)CO",
    "CC(O)C(O)C",
    "CNC",
    "CN(C)C",
    "CNC(=O)C",
    "CCNCC",
    "CCN(CC)CC",
    "C1CCOCC1",
    "C1COCCO1",
    "C1CCCCC1O",
    "C1CCCCC1N",
    "C1=CC=CC=C1O",
    "C1=CC=CC=C1N",
    "C1=CC=CC=C1Cl",
    "C1=CC=CC=C1Br",
    "C1=CC=CC=C1F",
    "FC1=CC=CC=C1",
    "CC(F)C(F)F",
    "CC(Cl)Cl",
    "CCBr",
    "CCI",
    "CCF",
    "CCCl",
    "CC(C)F",
    "CC(C)Cl",
    "CC(C)Br",
    "CC(C)I",
    "CC(C)N",
    "CC(C)CN",
    "CC(C)NC",
    "CC(C)N(C)C",
    "CC(C)C#N",
    "CC#N",
    "CC(=O)N",
    "CC(=O)NC",
    "CC(=O)NCC",
    "NC(=O)C",
    "NC(=O)CC",
    "NC(=O)CCC",
    "O=C(O)C",
    "O=C(O)CC",
    "O=C(O)CCC",
    "O=C(O)C(C)C",
    "O=C(O)C(C)CC",
    "CCOC(=O)C",
    "CCOC(=O)CC",
    "CCOC(=O)CCC",
    "COC(=O)C",
    "COC(=O)CC",
    "COC(=O)CCC",
    "C1=CC=C(C=C1)C",
    "C1=CC=C(C=C1)CC",
    "C1=CC=C(C=C1)CCC",
    "C1=CC=C(C=C1)CO",
    "C1=CC=C(C=C1)CN",
    "CC(C)C1=CC=CC=C1",
    "CC(C)C1=CC=CC=C1O",
    "CC(C)C1=CC=CC=C1N",
    "c1ccncc1",
    "c1ccncc1O",
    "c1ccncc1N",
    "c1ccoc1",
    "c1ccoc1O",
    "c1ccoc1N",
    "c1ncccc1",
    "c1ncccc1O",
    "c1ncccc1N",
    "c1cccc2ccccc12",
    "c1ccc2c(c1)cccc2",
    "c1ccc2c(c1)ccn2",
    "c1ccc2c(c1)cco2",
    "c1ccc2c(c1)ccoc2",
    "c1ccc2c(c1)ccnc2",
    "c1ccc(cc1)C(=O)O",
    "c1ccc(cc1)O",
    "c1ccc(cc1)N",
    "c1ccc(cc1)F",
    "c1ccc(cc1)Cl",
    "c1ccc(cc1)Br",
    "COC(=O)OC",
    "COC(=O)OCC",
    "COC(=O)OCCC",
    "CCOC(=O)OC",
    "CCOC(=O)OCC",
    "CCOC(=O)OCCC",
]

def smiles_to_hypergraph(smiles_string: str, target_name: str = "logP"):
    """
    Converts a SMILES string to a hypergraph representation suitable for NovaNet.
    
    Args:
        smiles_string: The SMILES string of the molecule.
        target_name: 'logP' or 'QED'.
        
    Returns:
        Tuple (x, H, y) or None if parsing fails.
        x: Node features (n, feature_dim)
        H: Incidence matrix (n, m)
        y: Target value (n, 1) - broadcasted graph property
    """
    if Chem is None:
        raise ImportError("rdkit is not installed. Please install it via 'pip install rdkit'.")

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # 1. Node Features (Atoms)
    # Feature dim = 2 (Atomic Number, Is Aromatic) - simple for now
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    
    x = []
    for atom in atoms:
        # Features: [AtomicNum, IsAromatic]
        # Normalize atomic number roughly (e.g., / 100)
        z = atom.GetAtomicNum() / 100.0
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        x.append([z, is_aromatic])
    
    x = np.array(x, dtype=np.float32) # (n, 2)
    
    # 2. Incidence Matrix H (Edges = Bonds + Rings)
    # We need to collect all hyperedges
    
    # Type 1: Bonds (connect 2 atoms)
    bonds = mol.GetBonds()
    bond_edges = []
    for bond in bonds:
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bond_edges.append([idx1, idx2])
        
    # Type 2: Rings (connect N atoms)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    ring_edges = []
    for ring in atom_rings:
        ring_edges.append(list(ring))
        
    num_edges = len(bond_edges) + len(ring_edges)
    
    # If no edges (single atom), create a self-loop or empty H?
    # NovaNet expects (n, m). If m=0, it might crash.
    # Let's add a self-loop for every node if no edges exist? 
    # Or just return None? Single atom molecules are rare in ZINC (e.g. methane).
    # Let's handle m=0 by adding a dummy hyperedge containing all nodes?
    if num_edges == 0:
        # Create one hyperedge containing all nodes (or just the single node)
        H = np.ones((num_atoms, 1), dtype=np.float32)
    else:
        H = np.zeros((num_atoms, num_edges), dtype=np.float32)
        
        current_edge_idx = 0
        
        # Add Bonds
        for edge_nodes in bond_edges:
            for node_idx in edge_nodes:
                H[node_idx, current_edge_idx] = 1.0
            current_edge_idx += 1
            
        # Add Rings
        for edge_nodes in ring_edges:
            for node_idx in edge_nodes:
                H[node_idx, current_edge_idx] = 1.0
            current_edge_idx += 1
            
    # 3. Target
    if target_name == "logP":
        val = Descriptors.MolLogP(mol)
    elif target_name == "QED":
        val = Descriptors.qed(mol)
    else:
        val = 0.0
        
    # Broadcast to (n, 1)
    y = np.full((num_atoms, 1), val, dtype=np.float32)
    
    return x, H, y

def load_zinc_subset(n_samples: int = 1000, target_name: str = "logP"):
    """
    Loads a subset of ZINC dataset.
    Downloads from a source if possible, otherwise uses a fallback list.
    
    Args:
        n_samples: Number of samples to load.
        target_name: Target property to predict.
        
    Returns:
        List of (x, H, y) tuples.
    """
    smiles_list = []
    
    # Try downloading a small CSV
    # Using a known URL for a small molecular dataset (e.g. from DeepChem or similar)
    # For stability, let's use the hardcoded list first if n_samples is small, 
    # or expand the hardcoded list.
    # But to be "Real-World", we should try to get real data.
    
    # Let's try to fetch a CSV from a public raw git or similar. 
    # Example: equivalent to zinc_test
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    
    try:
        logger.info(f"Attempting to download ZINC data from {url}...")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # The CSV structure is typically smiles,logP,qed,SAS
            # Let's check the first few lines
            content = response.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            
            # Adjust column names based on the specific CSV
            # This specific CSV has columns: "smiles", "logP", "qed", "SAS"
            if "smiles" in df.columns:
                smiles_list = df["smiles"].values.tolist()
            elif "SMILES" in df.columns:
                smiles_list = df["SMILES"].values.tolist()
            
            logger.info(f"Successfully downloaded {len(smiles_list)} molecules.")
        else:
            logger.warning("Failed to download ZINC data. Using fallback.")
            smiles_list = DEFAULT_SMILES
            
    except Exception as e:
        logger.warning(f"Error downloading ZINC data: {e}. Using fallback.")
        smiles_list = DEFAULT_SMILES

    # If too few, augment with randomized SMILES using RDKit when available
    if len(smiles_list) < n_samples:
        base_list = smiles_list if smiles_list else DEFAULT_SMILES
        augmented = []
        if Chem is not None:
            for s in base_list:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        continue
                    for _ in range(3):
                        rs = Chem.MolToSmiles(m, doRandom=True)
                        augmented.append(rs)
                except Exception:
                    continue
        pool = list(set(base_list + augmented))
        if not pool:
            pool = DEFAULT_SMILES
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(pool), size=n_samples, replace=True)
        smiles_list = [pool[i] for i in idxs]
    else:
        smiles_list = smiles_list[:n_samples]
        
    data = []
    valid_count = 0
    
    logger.info("Converting molecules to hypergraphs...")
    for smiles in smiles_list:
        try:
            res = smiles_to_hypergraph(smiles, target_name=target_name)
            if res is not None:
                data.append(res)
                valid_count += 1
        except Exception as e:
            # Skip invalid molecules
            continue
            
    logger.info(f"Loaded {valid_count} valid molecules.")
    return data
