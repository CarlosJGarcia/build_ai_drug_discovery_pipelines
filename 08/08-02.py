# Pytorch workflow

# 1. Data processing: Convert SMILES strings into tensors - numerical representations (e.g., Morgan fingerprints) that can be ingested by a neural network 
# 2. Define the model: Structure of the neural network model, referred to as its architecture. Loss function, optimizer and learning rate scheduler. 
# 3. Training the model: With data and model, we fit the model to the data. The model to learn patterns in the training data to make predictions on new inputs
# 4. Evaluating the model: We compare the model’s predictions on test data against the actual values.
# 5. Improve the model: Experiment with changes such as modifications to the model architecture, hyperparameter sweeps, etc.
# 6. Saving the model: Once we’ve settled on a final model, so we can reuse it