# Pytorch workflow
# EGFR (Epidermal Growth Factor Receptor) activity dataset - highly specialized molecules (kinase inhibitors targeted for cancer research)
# ML is 80% about clearing data and the other 20% is complaning about clearing data

# 1. Data processing: Convert SMILES strings into tensors - numerical representations (e.g., Morgan fingerprints) that can be ingested by a neural network 
# 2. Define the model: Structure of the neural network model, referred to as its architecture. Loss function, optimizer and learning rate scheduler. 
# 3. Training the model: With data and model, we fit the model to the data. The model to learn patterns in the training data to make predictions on new inputs
# 4. Evaluating the model: We compare the model’s predictions on test data against the actual values.
# 5. Improve the model: Experiment with changes such as modifications to the model architecture, hyperparameter sweeps, etc.
# 6. Saving the model: Once we’ve settled on a final model, so we can reuse it

import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator





"""
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split  # En lugar de definir la función, usamos esta que gestiona mejor los edge cases








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

"""

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


# Kinase Binder Neural Network Architecture
class KinaseBinderNN(nn.Module):
    def __init__(self, input_size=2048):
        super(KinaseBinderNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_model(
    model, train_loader, val_loader, criterion, optimizer, 
    scheduler, device, num_epochs=10
):
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}', 
              f'Train Loss: {train_loss:.4f}',
              f'Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        model.train()


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss() 

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate batch loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Keep as tensors, move to CPU to prevent GPU memory bloat, and append
            predictions.append(outputs.cpu())
            actuals.append(targets.cpu())
    
    # Combine lists of tensors into single continuous tensors and flatten
    predictions = torch.cat(predictions).view(-1)
    actuals = torch.cat(actuals).view(-1)
    
    # Calculate overall metrics
    mse = mse_criterion(predictions, actuals)
    mae = mae_criterion(predictions, actuals)
    
    # Use .item() to extract standard Python floats for printing
    print(f'Test Loss: {total_loss/len(test_loader):,.4f}')
    print(f'Mean Squared Error: {mse.item():,.4f}')
    print(f'Mean Absolute Error: {mae.item():,.4f}')
    

WRITE_PATH = "data/processed"
train_file_path = Path(WRITE_PATH) / "EGFR-train.csv"
validation_file_path = Path(WRITE_PATH) / "EGFR-validation.csv"
test_file_path = Path(WRITE_PATH) / "EGFR-test.csv"


# Initialize PyTorch Device 
console = Console()
console.print(f"\nInitialize Pytorch device", style="gold1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
        console.print(f"CUDA enabled, GPU {torch.cuda.get_device_name(0)}\n", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available, using CPU.\n", style="gold1") 

# Read the CSV files using Pandas
console.print(f"Reading datasets with Pandas", style="gold1")
train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(validation_file_path)
test_df = pd.read_csv(test_file_path)

# Instantiate the MoleculeDataset objects
# We pass the 'Smiles' and 'Standard Value' columns directly to the dataset class
train_dataset = MoleculeDataset(train_df["Smiles"], train_df["Standard Value"])
val_dataset = MoleculeDataset(val_df["Smiles"], val_df["Standard Value"])
test_dataset = MoleculeDataset(test_df["Smiles"], test_df["Standard Value"])

console.print(f"Loaded {len(train_dataset)} training samples.", style="green")
console.print(f"Loaded {len(val_dataset)} validation samples.", style="green")
console.print(f"Loaded {len(test_dataset)} test samples.", style="green")

# Create the DataLoaders for PyTorch training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instance the KinaseBinder Neural Network class
model = KinaseBinderNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training
console.print(f"\nTraining", style="gold1")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50)

# Evaluate with test dataset
console.print(f"\nEvaluation", style="gold1")
evaluate_model(model, test_loader, criterion, device)


print("\nTodo bien.")

"""

 

 

evaluate_model(model, test_loader, criterion, device)

torch.save(model.state_dict(), 'artifacts/ch08/kinase_binder_model.pth')
print("Model saved successfully.")
"""