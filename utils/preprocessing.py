import torch
import re 
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel


class EmbeddingExtractor:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.prot_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
        
        self.mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).to(self.device)

    def get_protein_embedding(self, sequence):
        tokens = self.prot_tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=3200).to(self.device)
        with torch.no_grad():
            embedding = self.prot_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding
    
    def get_molecule_embedding(self, smiles):
        tokens = self.mol_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=278).to(self.device)
        with torch.no_grad():
            embedding = self.mol_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding

    def get_combined_embedding(self, sequence, smiles):
        prot_embedding = self.get_protein_embedding(sequence)
        mol_embedding = self.get_molecule_embedding(smiles)
        return prot_embedding, mol_embedding

def preprocess_function(df):
    df['seq'] = df['seq'].apply(lambda x: " ".join(re.sub(r"[UZOB]", "X", x)))
    return df
