import torch
from data.dataset import BindingAffinityDataset
from model.model import BAPULM
from utils.train import train
import yaml
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import set_seed

def main():
    # Load config
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = BindingAffinityDataset(config['dataset_path'])

    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(valid_data)}")

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], shuffle=True)
    validation_loader = DataLoader(valid_data, batch_size=config['train_batch_size'], shuffle=False)

    # Model setup
    model = BAPULM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler_factor'], patience=config['scheduler_patience'])

    # Training
    model = train(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, num_epochs=config['num_epochs'])

    # Save model
    torch.save(model.state_dict(), config['model_train_save_path'])
    print("Model training complete and saved.")

if __name__ == '__main__':
    main()