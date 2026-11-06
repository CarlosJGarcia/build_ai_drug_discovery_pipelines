import json
from pathlib import Path
# from rdkit import Chem
from rich.console import Console
# from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdMolDescriptors
 
# File name
MOLECULES_PATH = "data/molecules.json"

def print_molecule(molecule):
  # Print the SMILES string and molecular formula
  print("SMILES string:", Chem.MolToSmiles(molecule)) 
  print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(molecule)}\n")

  # Show properties
  print("Number of atoms:", molecule.GetNumAtoms())
  print("Number of bonds:", molecule.GetNumBonds())
  print(f"Molecular weight: {Descriptors.MolWt(molecule)}\n")



# *main*
console = Console()

# Load data from file
console.print("\nReading molecules dataset\n", style="gold1")
path = Path(MOLECULES_PATH)
molecules_load = path.read_text()
molecules_dict = json.loads(molecules_load)

# Loop through the Dictionary
for name, smiles_string in molecules_dict.items():
    print(f"{name.title()} - {smiles_string.title()}.")

print()

"""
console.print("\nCaffeine", style="gold1")
mol = Chem.MolFromSmiles(caffeine_smiles_string)

# Print the SMILES string
print("SMILES string:", Chem.MolToSmiles(mol)) 
print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}\n")

# Show properties
print("Number of atoms:", mol.GetNumAtoms())
print("Number of bonds:", mol.GetNumBonds())
print(f"Molecular weight: {Descriptors.MolWt(mol)}\n")
 
console.print("Oxigen", style="gold1")
mol = Chem.MolFromSmiles(oxygen_smiles_string)
print_molecule(mol)
"""



 
"""
img = Draw.MolsToGridImage(mols=[mol], molsPerRow=1, useSVG=True)
with open("figures/ch01/caffeine.svg", "w") as f:
  f.write(img.data)
"""