# Loads the CSV file to a Dataframe and generates a JPG for each molecule 

import pandas as pd
from rdkit import Chem
#from pathlib import Path
from rich.console import Console
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdMolDescriptors
 
# File name
MOLECULES_CSV_PATH = "data/molecules.csv"


# *main*
console = Console()

# Load the file to a Dictonary
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


#df["Formula"] = df["SMILES"].apply(lambda x: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(x)))
#df["Atoms"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms())
#df["Bonds"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumBonds())
#df["Weight"] = df["SMILES"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

# Save Dataframe on file
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