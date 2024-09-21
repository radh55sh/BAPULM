import torch
import torch.nn as nn
import numpy as np
from utils.utils import set_seed
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model.model import BAPULM
from utils.preprocessing import preprocess_function, EmbeddingExtractor
import yaml



class BindingAffinityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        protein_seq = item['seq']
        ligand_smiles = item['smiles_can']
        log_affinity = item['neg_log10_affinity_M']
        return protein_seq, ligand_smiles, log_affinity


def main():
    # Load the configuration file
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    # set seed
    set_seed(2102)

    # Initialize device and model
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = BAPULM().to(device)
    model.load_state_dict(torch.load(config['model_inference_path']))
    model.eval()

    # Initialize the embedding extractor
    embedding_extractor = EmbeddingExtractor(device)
    
    # Process each CSV file
    for csv_file in config['benchmark_files']:
        df = pd.read_csv(csv_file)
        df = preprocess_function(df)
        dataset = BindingAffinityDataset(df)
        validation_loader = DataLoader(dataset, batch_size=config['inference_batch_size'], shuffle=False)

        true_affinities = []
        predicted_affinities = []

        with torch.no_grad():
            for sequences, smiles, affinities in tqdm(validation_loader, desc=f"Processing {csv_file}"):
                prot_embeddings = []
                mol_embeddings = []
                for seq, smile in zip(sequences, smiles):
                    prot_embedding, mol_embedding = embedding_extractor.get_combined_embedding(seq, smile)
                    prot_embeddings.append(prot_embedding)
                    mol_embeddings.append(mol_embedding)

                prot_embeddings = torch.cat(prot_embeddings, dim=0).to(device)
                mol_embeddings = torch.cat(mol_embeddings, dim=0).to(device)
                affinities = affinities.to(device)

                predictions = model(prot_embeddings, mol_embeddings).squeeze().cpu().numpy()
                true_affinities.extend(affinities.cpu().numpy())
                predicted_affinities.extend(predictions)

        true_affinities = np.array(true_affinities)
        predicted_affinities = np.array(predicted_affinities)

        # Apply scaling and mean adjustments
        mean = 6.51286529169358
        scale = 1.5614094578916633
        predicted_affinities = predicted_affinities * scale + mean

        # Calculate metrics
        mse = mean_squared_error(true_affinities, predicted_affinities)
        mae = mean_absolute_error(true_affinities, predicted_affinities)
        pearson_corr, _ = pearsonr(true_affinities, predicted_affinities)
        rmse = np.sqrt(mse)

        print(f"Results for {csv_file}:")
        print(f"RMSE: {rmse}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"Pearson Correlation Coefficient: {pearson_corr}")

if __name__ == '__main__':
    main()
