import torch
from model.model import BAPULM

def load_model(checkpoint_path, device):
    model = BAPULM()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
