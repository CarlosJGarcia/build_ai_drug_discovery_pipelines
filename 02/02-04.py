# Molecular fingerprinting: Encode the strucure of a molecule to represent it as a UUID number

from rdkit import Chem
from rich.console import Console
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def genmol(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Failed to generate a molecule from SMILES: {smiles_string}")
    return mol

# Define your example molecule (e.g., Aspirin)
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

console = Console()
example_mol = genmol(CAFFEINE)

# Fingerprinting
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
binary_fp = morgan_gen.GetFingerprint(example_mol)
count_fp = morgan_gen.GetCountFingerprint(example_mol)

# Display
console.print("\nBinary fingerprint", style="gold1", highlight=False)
print(f"Total bits: {len(binary_fp)}")
print(f"Number of 'On' bits: {binary_fp.GetNumOnBits()}")

console.print("\nCount fingerprint", style="gold1", highlight=False)
nonzero_elements = count_fp.GetNonzeroElements()
print(f"Total unique substructures found: {len(nonzero_elements)}")
# Show a sample of the structural features found (Bit ID: Count)
print(f"First 5 features (Bit ID : Count): {list(nonzero_elements.items())[:5]}")

print()
