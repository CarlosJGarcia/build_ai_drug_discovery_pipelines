# Defines a dictionary and saves it as JSON

import json
from pathlib import Path
from rich.console import Console
 
# SMILES string: A molecule written as text
hydrogen_smiles_string = "[H][H]"                          # Hydrogen molecule (H2). Two H atoms with single covalent bond
oxygen_smiles_string = "O=O"                               # Oxygen molecule (O2). Two O atoms with double covalent bond
carbon_dioxide_smiles_string = "O=C=O"                     # CO2
nitrogen_smiles_string = "N#N"                             # Nitrogen (N2). Two N atoms with a triple bond, represented by the hash symbol.
water_smiles_string = "O"                                  # SMILES notation assumes the standard bonds, which for one atom of O is two atoms of H
salt_smiles_string = "[Na+].[Cl-]"                         # Cl and Na with an ionic bond. Cl takes one electron from Na
aspirin_smiles_string = "CC(=O)Oc1ccccc1C(=O)O"
caffeine_smiles_string = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
benzene_smiles_string = "c1ccccc1"                                        # C6H6 Each carbon atom has an H and is linked to two other carbons with one single and one double bond. The molecule is planar because the double bonds are rigid.
cyclohexane_smiles_string = "C1CCCCC1"                                    # C6H12 Each carbon atom has two Hydrogens and is linked to two other carbons with single bonds. The molecule folds over itself as all the bonds are single and they allow that.

# Samples of the three main molecule types in biochemistry
lactose_smiles_string = "C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O"     # Carbohydrate
cholesterol_smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"    # Lipid. Barely any oxygen, so it does not dissolve in water.
oxytocin_smiles = "CCC(C)C1C(=O)NC(C(=O)NC(C(=O)NC(CSSCC(C(=O)NC(C(=O)N2CCCC2C(=O)NC(C(=O)NCC(=O)N)CC(C)C)CC(=O)N)N)C(=O)NC(C(=O)N1)CCC(=O)N)CC(=O)N)CC3=CC=C(C=C3)O" # Small protein. Proteins are much, much bigger than carbohydrates or lipids

# Isomers: Molecules that have the same molecular formula but differ in their structural arrangement or spatial orientation. The @ denotes the spatial orientation
# In vivo chiral inversion. Even if a patient is given 100% pure R-thalidomide, the metabolic system 'automatically' converts R to the dangerous S and back in the bloodstream
r_thalidomide_smiles = "C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O"      # R-thalidomide (therapeutic, women morning sickness)
s_thalidomide_smiles = "C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O"       #S-thalidomide (birth defects)


# File name
MOLECULES_PATH = "data/molecules.json"

# *main*
console = Console()

# Dictionary
molecules_dict = {
    "Caffeine": caffeine_smiles_string,
    "Oxygen": oxygen_smiles_string,
    "Hydrogen": hydrogen_smiles_string,
    "Nitrogen": nitrogen_smiles_string,
    "Water": water_smiles_string,
    "Salt": salt_smiles_string,
    "Aspirin": aspirin_smiles_string,
    "Carbon dioxide": carbon_dioxide_smiles_string,
    "Benzene": benzene_smiles_string,
    "Cyclohexane": cyclohexane_smiles_string,
    "Lactose": lactose_smiles_string,
    "Cholesterol": cholesterol_smiles,
    "Oxytocin": oxytocin_smiles,
    "R-Thalidomide": r_thalidomide_smiles,
    "S-Thalidomide": s_thalidomide_smiles
}

# Save data on file
console.print("Saving molecules dataset\n", style="gold1")
path = Path(MOLECULES_PATH)
molecules_dump = json.dumps(molecules_dict)
path.write_text(molecules_dump)
