# GNN: Graph Neural Network
# DTA: Drug-Target Affinity

import os
import numpy as np
import networkx as nx
from rdkit import Chem
from typing import Tuple, List
from rich.console import Console

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool

# Convert SMILES String to Graph Representation
def smile_to_graph(smiles: str) -> Tuple[int, List, List]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    num_atoms = mol.GetNumAtoms()
 
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / (sum(feature) + 1e-10))
 
    # Extract bonds
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
 
    g = nx.Graph(edges).to_directed()
    edge_index = []
 
    mol_adj = np.zeros((num_atoms, num_atoms))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
 
    mol_adj += np.matrix(
        np.eye(mol_adj.shape[0])
    )
 
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
 
    return num_atoms, features, edge_index


# Generate 78-dimensional Feature Vector for a Single Atom
def atom_features(atom):
    features = []
    features += one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
    features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += [int(atom.GetIsAromatic())]
    return np.array(features, dtype=np.float32)


# Construct Protein Graph from Contact Map and Sequence
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    assert contact_map.shape[0] == target_size
    contact_map += np.eye(contact_map.shape[0])
 
    threshold = CONFIG['contact_map_threshold']
    index_row, index_col = np.where(contact_map >= threshold)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
 
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    target_feature = compute_target_feature(aln_file, target_sequence)
    target_edge_index = np.array(target_edge_index)
 
    return target_size, target_feature, target_edge_index


# Generating Protein Feature Matrices
def seq_feature(pro_seq: str) -> np.ndarray:
    pro_hot = np.zeros((len(pro_seq), len(AMINO_ACIDS)))
    pro_property = np.zeros((len(pro_seq), 12))
 
    for i in range(len(pro_seq)):
        pro_hot[i, :] = one_of_k_encoding_unk(pro_seq[i], AMINO_ACIDS)
        pro_property[i, :] = residue_features(pro_seq[i])
 
    return np.concatenate((pro_hot, pro_property), axis=1)


# Normalization
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
 
    out = {}
    for key in dic:    
        out[key] = (dic[key] - min_value) / interval
    out['X'] = (max_value + min_value) / 2.0
    return out


# Calculating a Position-Specific Scoring Matrix (PSSM)
def PSSM_calculation(aln_file: str, pro_seq: str) -> np.ndarray:
    pfm_mat = np.zeros((len(AMINO_ACIDS), len(pro_seq)))
    with open(aln_file, 'r') as f:
        lines = f.readlines()
        line_count = sum(1 for line in lines if len(line.strip()) == len(pro_seq))
 
        for line in lines:
            if len(line.strip()) != len(pro_seq):
                continue
 
            count = 0
            for res in line.strip():
                if res not in AMINO_ACIDS:
                    count += 1
                    continue
                pfm_mat[AMINO_ACIDS.index(res), count] += 1
                count += 1
 
    pseudocount = CONFIG['pssm_pseudocount']
    ppm_mat = ((pfm_mat + pseudocount / 20) / (float(line_count) + pseudocount))
    pssm_mat = ppm_mat
    return pssm_mat


# Generate 54-dimensional Feature Matrix for Protein
def compute_target_feature(aln_file: str, pro_seq: str) -> np.ndarray:
    pssm = PSSM_calculation(aln_file, pro_seq)
    pssm_transposed = np.transpose(pssm, (1, 0))
    other_feature = seq_feature(pro_seq)
    return np.concatenate


# Definition of the Graph Neural Network 
class DrugTargetGNN(nn.Module):
    def __init__(self, num_features_mol=78, num_features_pro=54, hidden_dim=128):
        super(DrugTargetGNN, self).__init__()
        
        # Molecular Graph Layers
        self.mol_conv1 = GINConv(nn.Sequential(nn.Linear(num_features_mol, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.mol_conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.mol_conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.mol_fc_g1 = nn.Linear(hidden_dim, hidden_dim)
        self.mol_fc_g2 = nn.Linear(hidden_dim, hidden_dim)

        # Protein Graph Layers (required for forward_pro)
        self.pro_fc1 = nn.Linear(num_features_pro, hidden_dim)
        self.pro_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Combined Fully Connected Layers (128 from mol + 128 from pro = 256)
        self.fc1 = nn.Linear(hidden_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    # Process Molecular Graph through GCN Layers
    def forward_mol(self, data_mol):
        mol_x, mol_edge_index, mol_batch = \
            data_mol.x, data_mol.edge_index, data_mol.batch
     
        x = self.relu(self.mol_conv1(mol_x, mol_edge_index))
        x = self.relu(self.mol_conv2(x, mol_edge_index))
        x = self.relu(self.mol_conv3(x, mol_edge_index))
     
        x = global_mean_pool(x, mol_batch)
     
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x

    # Process Protein Graph
    def forward_pro(self, data_pro):
        pro_x, pro_batch = data_pro.x, data_pro.batch
        
        x = self.relu(self.pro_fc1(pro_x))
        x = global_mean_pool(x, pro_batch)
        x = self.dropout(x)
        x = self.relu(self.pro_fc2(x))
        return x

    # Forward Pass through the Dual GNN
    def forward(self, data_mol, data_pro):
        mol_embedding = self.forward_mol(data_mol)
        protein_embedding = self.forward_pro(data_pro)
        
        combined = torch.cat((mol_embedding, protein_embedding), dim=1)
     
        combined = self.relu(self.fc1(combined)) # 256 -> 1024
        combined = self.dropout(combined)
        combined = self.relu(self.fc2(combined)) # 1024 -> 512
        combined = self.dropout(combined)
        out = self.out(combined) # 512 -> 1
        return out


# Main

# Initialize PyTorch Device 
console = Console()
console.print(f"\nInitialize Pytorch device", style="gold1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
        console.print(f"CUDA enabled, GPU {torch.cuda.get_device_name(0)}\n", style="bright_blue", highlight=False)
else:
        console.print(f"CUDA not available, using CPU.\n", style="gold1") 

# Initialize the model
num_features_mol = 78
num_features_pro = 54
model = DrugTargetGNN(num_features_mol=num_features_mol, num_features_pro=num_features_pro)
model = model.to(device)
    
# Mock sample generation to test forward pass on GPU
mock_mol = Data(x=torch.randn((10, num_features_mol)), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long))
mock_pro = Data(x=torch.randn((50, num_features_pro)))

# Wrap in batches to assign proper batch indices for pooling
batch_mol = Batch.from_data_list([mock_mol]).to(device)
batch_pro = Batch.from_data_list([mock_pro]).to(device)

# Forward pass
output = model(batch_mol, batch_pro)
console.print(f"Output shape: {output.shape} | Tensor: {output.item():.4f}", style="green")