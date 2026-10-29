# PyTorch

import torch  
import torch.nn as nn                             # Neural network module
import torch.optim as optim                       # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Datasets and data loading
from rich.console import Console

console = Console()
console.print("\nVersions", style="gold1", highlight=False)
print(torch.__version__)
print(f"CUDA enabled: {torch.cuda.is_available()}")

console.print("\nPytorch two-dimensional tensor (matrix):", style="gold1", highlight=False)
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(tensor_2d)

print()