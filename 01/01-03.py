# Loads the CSV file to a Dataframe and generates a JPG for each molecule 

import pandas as pd
from rdkit import Chem
from rich.console import Console
from rdkit.Chem import Draw
 
# File name
MOLECULES_CSV_PATH = "data/molecules.csv"

# *main*
console = Console()

# Load the CSV file to a Dataframe
console.print("\nReading molecules dataset from CSV\n", style="gold1")
df = pd.read_csv(MOLECULES_CSV_PATH)

console.print("Generating images for each molecule\n", style="gold1")

# Loop through the DataFrame rows
for index, row in df.iterrows():
    # Get the name from the first column and SMILES string
    name = str(row.iloc[0])
    smiles_string = row["SMILES"]
    
    console.print(name.title(), style="gold1")        
    #print("SMILES string:", smiles_string)                   
    
    # Create the molecule object
    mol = Chem.MolFromSmiles(smiles_string)   

    # If the molecule is Water add hydrogens
    if name.lower() == "water":
        mol = Chem.AddHs(mol)

    # Generate the image as JPG. useSVG=False returns a PIL (Python Imaging Library) image that can be saved directly
    img = Draw.MolsToGridImage(mols=[mol], molsPerRow=1, useSVG=False)
    
    # Save the JPG using the molecule name
    jpg_filename = f"{name}.jpg"
    img.save(jpg_filename)
    print(f"Saved image to: {jpg_filename}\n")
