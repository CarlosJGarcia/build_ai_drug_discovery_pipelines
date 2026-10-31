# Pytorch workflow

# 1. Data processing: Convert SMILES strings into tensors - numerical representations (e.g., Morgan fingerprints) that can be ingested by a neural network 
# 2. Define the model: Structure of the neural network model, referred to as its architecture. Loss function, optimizer and learning rate scheduler. 
# 3. Training the model: With data and model, we fit the model to the data. The model to learn patterns in the training data to make predictions on new inputs
# 4. Evaluating the model: We compare the model’s predictions on test data against the actual values.
# 5. Improve the model: Experiment with changes such as modifications to the model architecture, hyperparameter sweeps, etc.
# 6. Saving the model: Once we’ve settled on a final model, so we can reuse it

# ML is 80% about clearing data and the other 20% is complaning about clearing data

import numpy as np
import pandas as pd
from rdkit import Chem
from pathlib import Path
from rdkit import DataStructs
from rich.console import Console
from collections import defaultdict
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # En lugar de definir la función, usamos esta que gestiona mejor los edge cases


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau


# Umplementation Bemis-Murcko scaffold split using RDKit
# Split 80/20 train/test
# Posteriormente del 80, reservamos 10% para validación
def scaffold_split(smiles_series, activities_series, train_fraction=0.8):
    scaffolds = defaultdict(list)
    smiles = smiles_series.reset_index(drop=True)
    activities = activities_series.reset_index(drop=True)
    
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            scaffolds[scaffold].append(i)
            
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_indices, test_indices = [], []
    train_cutoff = int(len(smiles) * train_fraction)
    
    for scaffold_group in scaffold_sets:
        if len(train_indices) + len(scaffold_group) <= train_cutoff:
            train_indices.extend(scaffold_group)
        else:
            test_indices.extend(scaffold_group)
            
    return (smiles.iloc[train_indices], activities.iloc[train_indices], 
            smiles.iloc[test_indices], activities.iloc[test_indices])


def smiles_to_fp(smiles, fp_size=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fp_size)
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros((fp_size,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
 

class MoleculeDataset(Dataset):
    def __init__(self, smiles, activities):
        self.smiles = smiles
        self.activities = activities
        self.fingerprints = [smiles_to_fp(s) for s in smiles]
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        fp = torch.tensor(self.fingerprints[idx], dtype=torch.float32)
        activity = torch.tensor([self.activities[idx]], dtype=torch.float32)
        return fp, activity

CHEMBL_EGFR_PATH ="data/raw/EGFR-activities-chembl37.csv.gz"

WRITE_PATH = "data/processed"
train_file_path = Path(WRITE_PATH) / "EGFR-train.csv"
validation_file_path = Path(WRITE_PATH) / "EGFR-validation.csv"
test_file_path = Path(WRITE_PATH) / "EGFR-test.csv"

# Import dataset
console = Console()
console.print(f"\nReading with Pandas ./{CHEMBL_EGFR_PATH}\n", style="gold1", highlight=False)
egfr_kinodata = pd.read_csv(CHEMBL_EGFR_PATH, sep=';', low_memory=False)  # Specify ; instead of , as field separator and load file in memory to prevent a warning when fields have None as value 

egfr_kinodata["Standard Value"] = pd.to_numeric(egfr_kinodata["Standard Value"], errors="coerce")
clean_data = egfr_kinodata.dropna(subset=["Smiles", "Standard Value"])

console.print("Dataframe:", style="gold1", highlight=False)
print(clean_data[["Smiles", "Standard Value"]])
print()

smiles = clean_data["Smiles"]
activities = clean_data["Standard Value"]
train_smiles, train_activities, test_smiles, test_activities = scaffold_split(smiles, activities)
train_smiles, val_smiles, train_activities, val_activities = train_test_split(train_smiles, train_activities, test_size=0.1, random_state=42)

# Reset indices for PyTorch compatibility
train_smiles = train_smiles.reset_index(drop=True)
train_activities = train_activities.reset_index(drop=True)
val_smiles = val_smiles.reset_index(drop=True)
val_activities = val_activities.reset_index(drop=True)
test_smiles = test_smiles.reset_index(drop=True)
test_activities = test_activities.reset_index(drop=True)

# Create MoleculeDataset objects for training, validation, and test data
# Each dataset has SMILES strings and their corresponding pIC50 activities
train_dataset = MoleculeDataset(train_smiles, train_activities)
test_dataset = MoleculeDataset(test_smiles, test_activities)
val_dataset = MoleculeDataset(val_smiles, val_activities)


# Combine Series into DataFrames and save without the index
pd.DataFrame({"Smiles": train_smiles, "Standard Value": train_activities}).to_csv(train_file_path, index=False)
pd.DataFrame({"Smiles": val_smiles, "Standard Value": val_activities}).to_csv(validation_file_path, index=False)
pd.DataFrame({"Smiles": test_smiles, "Standard Value": test_activities}).to_csv(test_file_path, index=False)

console.print(f"Files saved successfully to {WRITE_PATH}:", style="gold1", highlight=False)
print(f"- {train_file_path} ({train_smiles.shape[0]} rows)")
print(f"- {validation_file_path} ({val_smiles.shape[0]} rows)")
print(f"- {test_file_path} ({test_smiles.shape[0]} rows)")
print(f"Total: {train_smiles.shape[0] + val_smiles.shape[0] + test_smiles.shape[0]}")
print()

"""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KinaseBinderNN().to(device)
 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
 
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50)
evaluate_model(model, test_loader, criterion, device)

torch.save(model.state_dict(), 'artifacts/ch08/kinase_binder_model.pth')
print("Model saved successfully.")
"""