import json
import pandas as pd
from rdkit import Chem
from pathlib import Path
from rich.console import Console
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdMolDescriptors
 
# File name
MOLECULES_JSON_PATH = "data/molecules.json"
MOLECULES_CSV_PATH = "data/molecules.csv"

def print_molecule(molecule):
  # Print the SMILES string and molecular formula
  print("SMILES string (canonical):", Chem.MolToSmiles(molecule)) 
  print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(molecule)}")

  # Show properties
  print("Number of atoms:", molecule.GetNumAtoms())
  print("Number of bonds:", molecule.GetNumBonds())
  print(f"Molecular weight: {Descriptors.MolWt(molecule)}\n")


# *main*
console = Console()

# Load the file to a Dictonary
console.print("\nReading molecules dataset\n", style="gold1")
path = Path(MOLECULES_JSON_PATH)
molecules_load = path.read_text()
molecules_dict = json.loads(molecules_load)

"""
# Loop through the Dictionary, calculate and print values
for name, smiles in molecules_dict.items():
  console.print(name.title(), style="gold1")        # Print molecule name in plain text
  print("SMILES string:", smiles)                   # Print SMILEs string 
  mol = Chem.MolFromSmiles(smiles)    
  print_molecule(mol)                               # Print molecular formula, number of atoms, bonds and molecular weight               
"""

# Create a Pandas dataframe with the Dictionary
console.print("Creating Pandas Dataframe\n", style="gold1")
df = pd.DataFrame(list(molecules_dict.items()), columns=["Name", "SMILES"])
df.set_index("Name", inplace=True)

# 2. Create new columns using the RDKit functions
console.print("Calculating RDKit data and extending Dataframe\n", style="gold1")
df["Formula"] = df["SMILES"].apply(lambda x: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(x)))
df["Atoms"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms())
df["Bonds"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumBonds())
df["Weight"] = df["SMILES"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

console.print("Saving molecules dataset\n", style="gold1")
df.to_csv(MOLECULES_CSV_PATH)




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