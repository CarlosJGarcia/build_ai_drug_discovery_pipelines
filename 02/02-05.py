# RDKit: Rational Discovery Kit
# Open-source cheminformatics and ML library for analisys and calculation of the properties of chemical structures

# Future improvement: replace pandas with pandarallel for faster execution

import pandas as pd
from rich.console import Console
from rdkit import RDLogger, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import PandasTools, Descriptors, FilterCatalog, Draw, MolFromSmarts, AllChem


RO5_PROPS = ['MolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP']  # RO5_PROPS Python list containing the string names of the four RDKit descriptors used for Lipinski's Rule of Five
SPECS_SDF_PATH = "data/Specs.sdf.gz"                             # Specs (commercial company) dataset of chemical screening compounds for life science and pharma
GLAXO_ALERTS_PATH ="data/glaxo_structural_alerts.csv"

# Molecular descriptors are useful for interpreting a molecule’s properties and for encoding a molecule’s chemical information
def list_available_descriptors(limit=3):
    descriptors = Descriptors._descList
    console.print(f"\nRDKit has {len(descriptors)} molecular descriptors", style="gold1", highlight=False)
    
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


# Filter (remove) molecules that violate Lipinski’s Ro5 criteria
def apply_lipinski_filter(df):
    def count_ro5_violations(row):
        violations = 0
        if row['MolWt'] > 500: violations += 1
        if row['MolLogP'] > 5: violations += 1
        if row['NumHDonors'] > 5: violations += 1
        if row['NumHAcceptors'] > 10: violations += 1
        return violations
    
    df['ro5_violations'] = df.apply(count_ro5_violations, axis=1)
    df['ro5_compliant'] = (df['ro5_violations'] <= 1)
    compliant_df = df[df['ro5_compliant']]
    violated_df = df[~df['ro5_compliant']]
    return df, compliant_df, violated_df


# Filter molecules by structure, applying PAINS and BRENK filters
def apply_pains_brenk_filters(df, mol_col='mol'):
    df_copy = df.copy()
    filter_params = FilterCatalog.FilterCatalogParams()    
    filter_params.AddCatalog(filter_params.FilterCatalogs.PAINS)
    filter_params.AddCatalog(filter_params.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(filter_params)
    df_copy.loc[:, 'PAINS_BRENK_match'] = df_copy[mol_col].apply(catalog.HasMatch)    
    df_copy.loc[:, 'PAINS_BRENK_compliant'] = ~df_copy['PAINS_BRENK_match']
    filtered_df = df_copy[df_copy['PAINS_BRENK_compliant']]
    return df_copy, filtered_df


# Filter molecules by REOS (Rapid Elimination Of Swill)
def load_glaxo_alerts():
    # Remove sep='\t' so Pandas defaults to comma separation
    alerts_df = pd.read_csv(GLAXO_ALERTS_PATH)
    
    # Clean column names: strip hidden spaces and make everything lowercase
    alerts_df.columns = alerts_df.columns.str.strip().str.lower()
    
    # Safety check: If 'smarts' is still missing, fallback to the first column
    if 'smarts' not in alerts_df.columns:
        print(f"Column 'smarts' not found. Available columns: {list(alerts_df.columns)}")
        # Rename the first column to 'smarts' just in case there is no header
        alerts_df.rename(columns={alerts_df.columns[0]: 'smarts'}, inplace=True)

    # Use bracket notation instead of dot notation
    alerts_df["ROMol"] = alerts_df["smarts"].apply(MolFromSmarts)    
        
    invalid_count = alerts_df.ROMol.isnull().sum()
    if invalid_count > 0:    
        print(f"Warning: {invalid_count} invalid SMARTS patterns")
        alerts_df = alerts_df.dropna(subset=['ROMol'])
        
    print(f"Loaded {len(alerts_df)} Glaxo Wellcome structural alerts")
    return alerts_df

# Generate a Morgan fingerprint for a molecule
def generate_morgan_fingerprints(    
    df, mol_col='mol', radius=2, n_bits=2048    
):    
    df_copy = df.copy()
    fp_col_name = f"morgan_fp_r{radius}_b{n_bits}"
    morgan_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    def compute_fingerprint(mol):
        if mol is None: return None
        return morgan_generator.GetFingerprint(mol)
    
    df_copy.loc[:,fp_col_name] = \
        df_copy[mol_col].apply(compute_fingerprint)
    return df_copy, fp_col_name


# Load a SDF (Structure Data File) to a Pandas dataframe
console = Console()
console.print("\nLoading Specs SDF dataset", style="gold1", highlight=False)

# Create the Pandas dataframe and filter it so it has only two columns
RDLogger.DisableLog('rdApp.*')
specs = PandasTools.LoadSDF(SPECS_SDF_PATH, smilesName='smiles', molColName=None)[["PUBCHEM_SUBSTANCE_ID", "smiles"]]
print(f"Total entries loaded: {len(specs):,}")

# Generate RDKit Mol objects from SMILES...
PandasTools.AddMoleculeColumnToFrame(specs, 'smiles', 'mol')

# List RDKit Molecular descriptors
RDKIT_DESCRIPTORS = list_available_descriptors(limit=5)

# Calculate RDKit Molecular descriptors
specs = calculate_ro5_descriptors(specs)
console.print("\nCalculated RO5 Descriptors (First 5 rows):", style="gold1", highlight=False)
print(specs[["PUBCHEM_SUBSTANCE_ID", "smiles"] + RO5_PROPS].head())


# Filter (remove) molecules that violate Lipinski’s Ro5 criteria
console.print("\nFilter the molecules that violate Lipinski’s Ro5 criteria", style="gold1", highlight=False)
specs, specs_ro5_compliant, specs_ro5_violated = apply_lipinski_filter(specs)
print(f"Total entries remaining: {len(specs_ro5_compliant):,}")


# Filter molecules by structure
console.print("\nNow filter the molecules with PAINS & BRENK criteria", style="gold1", highlight=False)
specs_ro5_compliant, specs_ro5_pains_brenk_compliant = apply_pains_brenk_filters(specs_ro5_compliant)
print(f"Total entries remaining: {len(specs_ro5_pains_brenk_compliant):,}")

# Filter molecules by REOS (Rapid Elimination Of Swill)
console.print("\nLoad Glaxo Wellcome structural alerts", style="gold1", highlight=False)
glaxo_alerts = load_glaxo_alerts()

console.print("\nApply Glaxo structural alerts", style="gold1", highlight=False)
specs_glaxo_compliant = apply_glaxo_alerts(specs_ro5_pains_brenk_compliant, glaxo_alerts)
print(f"Total entries remaining: {len(specs_glaxo_compliant):,}")

"""
# Molecular molecules by fingerprinting
console.print("\nFilter by fingerprinting", style="gold1", highlight=False)
specs_filtered, fp_col_name = generate_morgan_fingerprints(specs_ro5_pains_brenk_compliant)
print(f"Total entries remaining: {len(specs_filtered):,}")
"""

print()