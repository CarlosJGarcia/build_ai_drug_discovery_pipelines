from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, Draw, PandasTools)
 
# SMILES string: A molecule written as text
aspirine_smiles_string = "CC(=O)Oc1ccccc1C(=O)O"
caffeine_smiles_string = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


mol = Chem.MolFromSmiles(caffeine_smiles_string)
 
print("Number of atoms:", mol.GetNumAtoms())
# Output -> Number of atoms: 14

print("Number of bonds:", mol.GetNumBonds())
# Output -> Number of bonds: 15
 
# Print the SMILES string
print(Chem.MolToSmiles(mol)) 

# Print the molecular formula
#print(mol.GetFormula())

# Print the weight
print(f"Molecular weight: {Descriptors.MolWt(mol)}")
 
"""
img = Draw.MolsToGridImage(mols=[mol], molsPerRow=1, useSVG=True)
with open("figures/ch01/caffeine.svg", "w") as f:
  f.write(img.data)
"""