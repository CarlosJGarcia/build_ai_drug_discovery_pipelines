from rich.console import Console
from rdkit.Chem import PandasTools


# Load a SDF (Structure Data File) to a Pandas dataframe
console = Console()
console.print("\nLoading SDF", style="gold1", highlight=False)
specs = PandasTools.LoadSDF("data/Specs.sdf.gz", smilesName='smiles', molColName=None)[["PUBCHEM_SUBSTANCE_ID", "smiles"]]
print(f"Total entries loaded: {len(specs)}")

# Generate RDKit Mol objects from SMILES...
PandasTools.AddMoleculeColumnToFrame(specs, 'smiles', 'mol')

# Check how many molecules successfully parsed into RDKit Mol objects
valid_mols = specs['mol'].notnull().sum()
invalid_mols = specs['mol'].isnull().sum()
print(f"Valid molecules: {valid_mols} | Failed parsing: {invalid_mols}")

console.print("\nFirst 2 rows excluding ROMol object representation:", style="gold1", highlight=False)
print(specs[["PUBCHEM_SUBSTANCE_ID", "smiles"]].head(2))