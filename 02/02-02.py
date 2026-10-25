from rich.console import Console
from rdkit.Chem import Descriptors

# Molecular descriptors are useful for interpreting a molecule’s properties and for encoding a molecule’s chemical information
def list_available_descriptors(limit=3):
    descriptors = Descriptors._descList
    console.print(f"\nRDKit has {len(descriptors)} molecular descriptors available", style="gold1", highlight=False)
    
    if limit > 0:
        print(f"Showing first {min(limit, len(descriptors))} descriptors:")
        for name, func in descriptors[:limit]:
            print(f"- {name}")
    return {desc: func for desc, func in descriptors}

console = Console()
RDKIT_DESCRIPTORS = list_available_descriptors(limit=10)
