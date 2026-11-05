from rdkit import Chem
from rich.console import Console
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdMolDescriptors
 
# SMILES string: A molecule written as text
oxygen_smiles_string = "O=O"                             # Oxigen molecule (02). Two O atoms with double covalent bond
hydrogen_smiles_string = "[H][H]"                        # Hidrogen molecule (H2). Two H atoms with single covalent bond
water_smiles_string = "O"                                # SMILES notation assumes the standard bonds, which for one atom of O is two atoms of H
salt_smiles_string = "[Na+].[Cl-]"                       # Ionic bond. Cl takes one electron from Na
aspirine_smiles_string = "CC(=O)Oc1ccccc1C(=O)O"
caffeine_smiles_string = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

# Dictionary
molecules_list = [
    "Caffeine", caffeine_smiles_string,
    "Oxygen", oxygen_smiles_string,
    "Hydrogen", hydrogen_smiles_string,
    "Water", water_smiles_string,
    "Salt", salt_smiles_string,
    "Aspirine", aspirine_smiles_string
]

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
img = Draw.MolsToGridImage(mols=[mol], molsPerRow=1, useSVG=True)
with open("figures/ch01/caffeine.svg", "w") as f:
  f.write(img.data)
"""