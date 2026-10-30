# Pytorch workflow

# 1. Data processing: Convert SMILES strings into tensors - numerical representations (e.g., Morgan fingerprints) that can be ingested by a neural network 
# 2. Define the model: Structure of the neural network model, referred to as its architecture. Loss function, optimizer and learning rate scheduler. 
# 3. Training the model: With data and model, we fit the model to the data. The model to learn patterns in the training data to make predictions on new inputs
# 4. Evaluating the model: We compare the model’s predictions on test data against the actual values.
# 5. Improve the model: Experiment with changes such as modifications to the model architecture, hyperparameter sweeps, etc.
# 6. Saving the model: Once we’ve settled on a final model, so we can reuse it

import pandas as pd

CHEMBL_EGFR_PATH ="data/EGFR-activities-chembl33.csv"

# Dataset import
egfr_kinodata = pd.read_csv(CHEMBL_EGFR_PATH, sep=';', low_memory=False)  # Specify ; instead of , as field separator and load file in memory to prevent a warning when fields have None as value 
print("Todo bien")


"""
smiles = egfr_kinodata["compound_structures.canonical_smiles"]
activities = pd.to_numeric(egfr_kinodata["activities.standard_value"], errors="coerce")




train_smiles, train_activities, test_smiles, test_activities = scaffold_split(smiles, activities)
    
train_smiles, val_smiles, train_activities, val_activities = train_test_split(train_smiles, train_activities, test_size=0.1, random_state=42)
 
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