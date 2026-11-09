# Loads the JSON file to a Dictionary, moves the information to a Pandas Dataframe and adds numerical values calculated with RDKit
# Saves the dataframe as a molecules.csv file

import json
import pandas as pd
from rdkit import Chem
from pathlib import Path
from rich.console import Console
from rdkit.Chem import Descriptors, rdMolDescriptors
 
# File names
MOLECULES_CSV_PATH = "data/molecules.csv"
MOLECULES_JSON_PATH = "data/molecules.json"

def print_molecule(molecule):
  # Print the SMILES string and molecular formula
  print("SMILES string (canonical):", Chem.MolToSmiles(molecule)) 
  print(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(molecule)}")

  # Show properties
  print("Number of atoms:", molecule.GetNumAtoms())
  print("Number of bonds:", molecule.GetNumBonds())
  print(f"Molecular weight: {Descriptors.MolWt(molecule)} g/mol\n")


# *main*
console = Console()

# Load the file to a Dictionary
console.print("\nReading molecules dataset\n", style="gold1")
path = Path(MOLECULES_JSON_PATH)
molecules_load = path.read_text()
molecules_dict = json.loads(molecules_load)

# Loop through the Dictionary, calculate and print values
for name, smiles in molecules_dict.items():
  console.print(name.title(), style="gold1")        # Print molecule name in plain text
  print("SMILES string:", smiles)                   # Print SMILEs string 
  mol = Chem.MolFromSmiles(smiles)   

  # If the molecule is Water add hydrogens
  if name.lower() == "water":
      mol = Chem.AddHs(mol)

  print_molecule(mol)                               # Print molecular formula, number of atoms, bonds and molecular weight               


# Create a Pandas dataframe with the Dictionary
console.print("Creating Pandas Dataframe\n", style="gold1")
df = pd.DataFrame(list(molecules_dict.items()), columns=["Name", "SMILES"])
df.set_index("Name", inplace=True)

# Add new columns with the RDKit values
console.print("Calculating RDKit data and extending Dataframe\n", style="gold1")

# Create empty lists to hold the calculated data
formulas = []
atoms_count = []
bonds_count = []
weights = []

# Loop through the SMILES column, calculate values and add them to the lists
# Use .items() to get both the Name (index) and the SMILES string, so I can add hydrogens to water

for name, smiles_string in df["SMILES"].items():
    # Create the molecule object (just once per row)
    mol = Chem.MolFromSmiles(smiles_string)

    # If the molecule is Water add hydrogens
    if name.lower() == "water":
        mol = Chem.AddHs(mol)
    
    # Calculate the properties and add them to the lists
    formulas.append(rdMolDescriptors.CalcMolFormula(mol))
    atoms_count.append(mol.GetNumAtoms())
    bonds_count.append(mol.GetNumBonds())
    weights.append(Descriptors.MolWt(mol))

# Assign the populated lists to new columns in the DataFrame
df["Formula"] = formulas
df["Atoms"] = atoms_count
df["Bonds"] = bonds_count
df["Weight"] = weights

# This would be a more concise approach, using lambda function, but it is difficult to understand
#df["Formula"] = df["SMILES"].apply(lambda x: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(x)))
#df["Atoms"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms())
#df["Bonds"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumBonds())
#df["Weight"] = df["SMILES"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

# Save Dataframe on file
console.print("Saving molecules dataset\n", style="gold1")
df.to_csv(MOLECULES_CSV_PATH)