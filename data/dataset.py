import json
import torch
from torch.utils.data import Dataset

class BindingAffinityDataset(Dataset):
    def __init__(self, json_file):
        self.data = []
        with open(json_file, 'r') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        protein_embedding = torch.tensor(item['prot_embedding'], dtype=torch.float32).squeeze(0)
        chemical_embedding = torch.tensor(item['mol_embedding'], dtype=torch.float32).squeeze(0)
        affinity = torch.tensor(item['affinity'], dtype=torch.float32)
        return protein_embedding, chemical_embedding, affinity
    
