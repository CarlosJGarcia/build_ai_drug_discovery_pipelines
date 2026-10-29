# PyTorch
# When the Python script finishes, the Linux and the CUDA driver automatically reclaim all allocated memory, including RAM and VRAM
# Deleting variables (del tensor_2d) or clearing the PyTorch cache (torch.cuda.empty_cache()) at the end is redundant and offers no benefit

import torch  
import torch.nn as nn                             # Neural network module
import torch.optim as optim                       # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Datasets and data loading
from rich.console import Console

console = Console()
console.print("\nVersions and device selection", style="gold1", highlight=False)
print(torch.__version__)
print(f"CUDA enabled: {torch.cuda.is_available()}")

# Configuración GPU RTX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

console.print("\nPytorch two-dimensional tensor (matrix):", style="gold1", highlight=False)
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(tensor_2d)
print(f"Shape: {tensor_2d.shape}")
print(f"Data type: {tensor_2d.dtype}")
print(f"Stored on: {tensor_2d.device}")

# Move the tensor to the GPU
console.print("\nMove the tensor to the GPU", style="gold1", highlight=False)
tensor_2d = tensor_2d.to(device)
print(f"Stored on: {tensor_2d.device}")

# Convert the tensor data type fron Integer to Float
console.print("\nConvert from Integer to Float", style="gold1", highlight=False)
tensor_2d = tensor_2d.to(torch.float32)
print(tensor_2d)
print(f"Data type: {tensor_2d.dtype}")
print(f"Stored on: {tensor_2d.device}")


print()