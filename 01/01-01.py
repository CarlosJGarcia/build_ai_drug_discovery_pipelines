from rdkit import Chem
from rich.console import Console
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdMolDescriptors
 
# SMILES string: A molecule written as text
aspirine_smiles_string = "CC(=O)Oc1ccccc1C(=O)O"
caffeine_smiles_string = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

console = Console()
console.print("\nCaffeine", style="gold1")
mol = Chem.MolFromSmiles(caffeine_smiles_string)

# Print the SMILES string
print("SMILES string:", Chem.MolToSmiles(mol)) 
print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}\n")

# Show properties
print("Number of atoms:", mol.GetNumAtoms())
print("Number of bonds:", mol.GetNumBonds())
print(f"Molecular weight: {Descriptors.MolWt(mol)}\n")
 


 
"""
img = Draw.MolsToGridImage(mols=[mol], molsPerRow=1, useSVG=True)
with open("figures/ch01/caffeine.svg", "w") as f:
  f.write(img.data)
"""