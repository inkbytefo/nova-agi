## Developer: inkbytefo
## Modified: 2025-12-11

import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# Mock rdkit modules so we can test logic without installing rdkit
mock_rdkit = MagicMock()
mock_chem = MagicMock()
mock_descriptors = MagicMock()

sys.modules["rdkit"] = mock_rdkit
sys.modules["rdkit.Chem"] = mock_chem
sys.modules["rdkit.Chem.Descriptors"] = mock_descriptors

# Now import the module under test
from nova.data.zinc import smiles_to_hypergraph

def test_smiles_to_hypergraph_structure():
    # Setup mock molecule
    mol = MagicMock()
    mock_chem.MolFromSmiles.return_value = mol
    
    # Mock Atoms
    atom1 = MagicMock()
    atom1.GetAtomicNum.return_value = 6 # C
    atom1.GetIsAromatic.return_value = False
    
    atom2 = MagicMock()
    atom2.GetAtomicNum.return_value = 8 # O
    atom2.GetIsAromatic.return_value = False
    
    mol.GetAtoms.return_value = [atom1, atom2]
    
    # Mock Bonds
    bond = MagicMock()
    bond.GetBeginAtomIdx.return_value = 0
    bond.GetEndAtomIdx.return_value = 1
    mol.GetBonds.return_value = [bond]
    
    # Mock Rings
    ring_info = MagicMock()
    # A ring involving both atoms
    ring_info.AtomRings.return_value = [(0, 1)] 
    mol.GetRingInfo.return_value = ring_info
    
    # Mock Descriptors
    mock_descriptors.MolLogP.return_value = 2.5
    
    # Run the function
    # We pass any string, the mock MolFromSmiles returns our mock mol
    x, H, y = smiles_to_hypergraph("dummy_smiles", target_name="logP")
    
    # Verify Output Shapes
    # x: (num_atoms, 2)
    assert x.shape == (2, 2)
    # H: (num_atoms, num_edges) -> 1 bond + 1 ring = 2 edges
    assert H.shape == (2, 2)
    # y: (num_atoms, 1)
    assert y.shape == (2, 1)
    
    # Verify Content
    # Check atomic numbers (normalized / 100)
    assert np.isclose(x[0, 0], 0.06)
    assert np.isclose(x[1, 0], 0.08)
    
    # Check Incidence Matrix
    # Column 0: Bond (0-1)
    assert H[0, 0] == 1.0
    assert H[1, 0] == 1.0
    
    # Column 1: Ring (0-1)
    assert H[0, 1] == 1.0
    assert H[1, 1] == 1.0
    
    # Check Target
    assert np.allclose(y, 2.5)

def test_smiles_to_hypergraph_no_edges():
    # Test case for single atom (no bonds, no rings)
    mol = MagicMock()
    mock_chem.MolFromSmiles.return_value = mol
    
    atom1 = MagicMock()
    atom1.GetAtomicNum.return_value = 6
    atom1.GetIsAromatic.return_value = False
    
    mol.GetAtoms.return_value = [atom1]
    mol.GetBonds.return_value = []
    
    ring_info = MagicMock()
    ring_info.AtomRings.return_value = []
    mol.GetRingInfo.return_value = ring_info
    
    mock_descriptors.MolLogP.return_value = 0.5
    
    x, H, y = smiles_to_hypergraph("C")
    
    assert x.shape == (1, 2)
    # Should have 1 column (self-loop/dummy)
    assert H.shape == (1, 1)
    assert H[0, 0] == 1.0
    assert y.shape == (1, 1)

