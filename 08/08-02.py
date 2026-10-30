# Pytorch workflow

# 1. Data processing: Convert SMILES strings into tensors - numerical representations (e.g., Morgan fingerprints) that can be ingested by a neural network 
# 2. Define the model: Structure of the neural network model, referred to as its architecture. Loss function, optimizer and learning rate scheduler. 
# 3. Training the model: With data and model, we fit the model to the data. The model to learn patterns in the training data to make predictions on new inputs
# 4. Evaluating the model: We compare the model’s predictions on test data against the actual values.
# 5. Improve the model: Experiment with changes such as modifications to the model architecture, hyperparameter sweeps, etc.
# 6. Saving the model: Once we’ve settled on a final model, so we can reuse it

# ML is 80% about clearing data and the other 20% is complaning about clearing data

import pandas as pd
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold

# Umplementation Bemis-Murcko scaffold split using RDKit
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


CHEMBL_EGFR_PATH ="data/EGFR-activities-chembl37.csv.gz"

# Import dataset
egfr_kinodata = pd.read_csv(CHEMBL_EGFR_PATH, sep=';', low_memory=False)  # Specify ; instead of , as field separator and load file in memory to prevent a warning when fields have None as value 

egfr_kinodata["Standard Value"] = pd.to_numeric(egfr_kinodata["Standard Value"], errors="coerce")
clean_data = egfr_kinodata.dropna(subset=["Smiles", "Standard Value"])

smiles = clean_data["Smiles"]
activities = clean_data["Standard Value"]
train_smiles, train_activities, test_smiles, test_activities = scaffold_split(smiles, activities)



print("Todo bien")
"""

train_dataset = MoleculeDataset(train_smiles, train_activities)
val_dataset = MoleculeDataset(val_smiles, val_activities)
test_dataset = MoleculeDataset(test_smiles, test_activities)
 
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