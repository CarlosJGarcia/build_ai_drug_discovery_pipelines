from rich.console import Console
from rdkit.Chem import PandasTools, Draw


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

# Use the mol object to generate an image
# Extract the first valid molecule from the dataframe
first_valid_mol = specs['mol'].dropna().iloc[0]

# Generate an SVG image and save it to a file
# Note: drawOptions=dopts was removed to prevent a NameError unless you explicitly define it
svg_data = Draw.MolsToGridImage(mols=[first_valid_mol], molsPerRow=1, useSVG=True)

# Write the SVG string to a file for viewing off-server
with open("molecule_output.svg", "w") as f:
    f.write(svg_data)

console.print("\nMolecule image saved to molecule_output.svg\n", style="gold1", highlight=False)