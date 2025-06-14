<div align="center">

# BAPULM: Binding Affinity Prediction Using Language Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bapulm-binding-affinity-prediction-using/protein-ligand-affinity-prediction-on-csar)](https://paperswithcode.com/sota/protein-ligand-affinity-prediction-on-csar?p=bapulm-binding-affinity-prediction-using)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bapulm-binding-affinity-prediction-using/protein-ligand-affinity-prediction-on-pdbbind)](https://paperswithcode.com/sota/protein-ligand-affinity-prediction-on-pdbbind?p=bapulm-binding-affinity-prediction-using)

<img src="https://drive.google.com/uc?export=view&id=1Q7TE4yWg-gGDO6mB6N64irMjlJti1nDd" alt="BAPULM Logo" width="320"/>
</div>


Welcome to the BAPULM repository! This repository corresponds to the prediction of protein-ligand complex binding affinity.

## Getting Started

1. **Clone the Repository:**
    Run the following command in your terminal:

   ```bash
   git clone https://github.com/radh55sh/BAPULM.git
   cd BAPULM
2. **Install the required packages:**
   Using conda:
   ```bash
     conda create --name bapulm-env python=3.10
     conda activate bapulm-env
     pip install -r requirements.txt
  
4. **Download the datset:**
    Download the prottrans_molformer dataset from the [Hugging Face Platform](https://huggingface.co/datasets/radh25sh/BAPULM/tree/main) and place it in the data/ directory.
   
5. **To train the model and inference:**
   First, train the model, and furthermore, to do inference on the model, download the model parameters from the [Hugging Face Platform](https://huggingface.co/datasets/radh25sh/BAPULM/tree/main) and place it in the data/ directory.
   
     ```bash
     python main.py # To train the model
     python inference.py # To perform inference 
   
