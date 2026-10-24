from rdkit.Chem import Descriptors

def list_available_descriptors(limit=3):
    descriptors = Descriptors._descList
    print(f"RDKit has {len(descriptors)} molecular descriptors available")
    
    if limit > 0:
        print(f"Showing first {min(limit, len(descriptors))} descriptors:")
        for name, func in descriptors[:limit]:
            print(f"- {name}")
    return {desc: func for desc, func in descriptors}

RDKIT_DESCRIPTORS = list_available_descriptors(limit=10)
