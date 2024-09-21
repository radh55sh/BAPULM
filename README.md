## BAPULM: Binding Affinity Prediction Using Language Models

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
   First, train the model and furthermore to do inference on the model change the config file model_inference_path parameter with the save model weights
     ```bash
     python main.py # To train the model
     python inference.py # To perform inference 
   
