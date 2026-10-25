# RDKit: Rational Discovery Kit
# Open-source cheminformatics and machine learning toolkit for analyzing, manipulating, and calculating properties of chemical structures

from rich.console import Console
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools

# RO5_PROPS Python list containing the string names of the four RDKit descriptors used for Lipinski's Rule of Five
RO5_PROPS = ['MolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP']

# Molecular descriptors are useful for interpreting a molecule’s properties and for encoding a molecule’s chemical information
def list_available_descriptors(limit=3):
    descriptors = Descriptors._descList
    console.print(f"\nRDKit has {len(descriptors)} molecular descriptors available", style="gold1", highlight=False)
    
    if limit > 0:
        print(f"Showing first {min(limit, len(descriptors))} descriptors:")
        for name, func in descriptors[:limit]:
            print(f"- {name}")
    return {desc: func for desc, func in descriptors}



# Calculate 4 descriptors that describe each molecule’s molecular weight, number of hydrogen bond acceptors (HBAs), number of hydrogen bond donors (HBDs), and logP
def calculate_ro5_descriptors(df, mol_col='mol'):
    def compute_descriptor(mol, func_name, missing_val=None):
        if mol is None: return missing_val
        try:
            func = RDKIT_DESCRIPTORS[func_name]
            return func(mol)
        except Exception as e:    
            print(f"Descriptor {func_name} failed: {e}")
            return missing_val    
    
    for desc in RO5_PROPS:
        df[desc] = df[mol_col].apply(
            lambda x: compute_descriptor(x, desc)
        )
    
    row_count_before = len(df)
    df = df.dropna(subset=RO5_PROPS)    
    row_count_after = len(df)
    return df


# Load a SDF (Structure Data File) to a Pandas dataframe
console = Console()
console.print("\nLoading SDF", style="gold1", highlight=False)
specs = PandasTools.LoadSDF("data/Specs.sdf.gz", smilesName='smiles', molColName=None)[["PUBCHEM_SUBSTANCE_ID", "smiles"]]
print(f"Total entries loaded: {len(specs)}")

# Generate RDKit Mol objects from SMILES...
PandasTools.AddMoleculeColumnToFrame(specs, 'smiles', 'mol')

# List RDKit Molecular descriptors
RDKIT_DESCRIPTORS = list_available_descriptors(limit=10)

# Calculate RDKit Molecular descriptors
specs = calculate_ro5_descriptors(specs)
console.print("\nCalculated RO5 Descriptors (First 5 rows):", style="gold1", highlight=False)
print(specs[["PUBCHEM_SUBSTANCE_ID", "smiles"] + RO5_PROPS].head())

print()