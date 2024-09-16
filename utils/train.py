import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.utils import set_seed

def train(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, num_epochs=60):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as train_progress:
            for sequences, smiles, affinities in train_progress:
                optimizer.zero_grad()
                sequences, smiles = sequences.to(device), smiles.to(device)
                predictions = model(sequences, smiles)
                loss = criterion(predictions.squeeze(), affinities.to(device).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_progress.set_postfix(loss=train_loss / len(train_loader))

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for sequences, smiles, affinities in validation_loader:
                sequences, smiles = sequences.to(device), smiles.to(device)
                predictions = model(sequences, smiles)
                loss = criterion(predictions.squeeze(), affinities.to(device).float())
                validation_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {validation_loss / len(validation_loader)}")
        scheduler.step(validation_loss)

    return model