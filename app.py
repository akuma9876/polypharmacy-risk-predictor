from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import random
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# --- 1. Load Model and Data Structures ---
# Load drug name mapping
with open('cid_to_name.json', 'r') as f:
    cid_to_name = json.load(f)

# Load id_to_smiles
with open('id_to_smiles.json', 'r') as f:
    id_to_smiles = json.load(f)

# Load feature_dict
with open('feature_dict.json', 'r') as f:
    feature_dict = json.load(f)

class CocktailRiskNet(nn.Module):
    def __init__(self, input_dim):
        super(CocktailRiskNet, self).__init__()

        # Shared Encoder: Processes each drug in the cocktail independently
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Risk Head: Takes the aggregated cocktail info and predicts toxicity
        self.risk_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_list):
        """
        drug_list: List of Tensors, each (batch_size, input_dim)
        This handles permutation invariance (A+B+C == C+B+A)
        """
        # Encode each drug using the same shared weights
        encoded_drugs = [self.encoder(d) for d in drug_list]

        # AGGREGATION: We use 'Max' pooling because polypharmacy risk
        # is often driven by the "weakest link" (the most toxic interaction).
        # We stack and take max across the drug dimension
        stacked = torch.stack(encoded_drugs, dim=0) # [num_drugs, batch, features]
        cocktail_repr, _ = torch.max(stacked, dim=0)

        return self.risk_head(cocktail_repr)


# Try to load models
# Assuming your input dimension is Morgan (1024) + GNN (73) = 1097
INPUT_DIM = 1097 

try:
    # Load the dictionary of state_dicts
    checkpoint = torch.load('polypharmacy_top10_models.pt', map_location='cpu')
    
    side_effect_models = {}
    for effect_name, state_dict in checkpoint.items():
        # 1. Create the model instance
        model = CocktailRiskNet(INPUT_DIM)
        # 2. Load the weights (the 'dict' that was causing the error)
        model.load_state_dict(state_dict)
        # 3. Set to evaluation mode immediately
        model.eval()
        side_effect_models[effect_name] = model
        
    MODELS_LOADED = True
    print("Models initialized and weights loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    MODELS_LOADED = False


class DrugFeaturizer:
    def __init__(self, gnn_map, fp_size=1024):
        self.gnn_map = gnn_map
        self.fp_size = fp_size
        
    def get_features(self, smiles, cid):
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_size))
        gnn_feat = self.gnn_map.get(cid)
        if gnn_feat is None:
            gnn_feat = np.zeros(73)
        else:
            gnn_feat = np.array(gnn_feat)
        return np.concatenate([fp, gnn_feat])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_drugs', methods=['GET'])
def get_drugs():
    """Return list of available drugs with CID and name"""
    drugs = [{'cid': cid, 'name': cid_to_name.get(cid, cid)} for cid in id_to_smiles.keys()]
    return jsonify(drugs)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        drug_ids = data.get('drug_ids', []) # e.g., ['CID000005467', 'CID000002244']
        
        print(f"Received drug_ids: {drug_ids}")
        
        results = {}
        
        # If models aren't loaded, return demo results
        if not MODELS_LOADED:
            # Generate mock predictions for demonstration
            demo_effects = [
                'Hepatotoxicity', 'Nephrotoxicity', 'Cardiotoxicity', 
                'Neurotoxicity', 'Hematotoxicity', 'Gastrointestinal',
                'Dermatological', 'Respiratory', 'Metabolic', 'Immunological'
            ]
            # Use absolute value of hash for consistent positive seed
            random.seed(abs(hash(str(sorted(drug_ids)))))
            for effect in demo_effects:
                results[effect] = round(random.uniform(0.1, 0.9), 4)
            print(f"Returning demo results: {results}")
            return jsonify(results)
        
        # 1. Featurize the cocktail
        featurizer = DrugFeaturizer(feature_dict)
        drug_tensors = []
        for cid in drug_ids:
            smiles = id_to_smiles.get(cid)
            if smiles:
                # Get the combined Morgan Fingerprint + GNN feature
                feat = torch.tensor(featurizer.get_features(smiles, cid)).float()
                drug_tensors.append(feat)

        if not drug_tensors:
            return jsonify({'error': 'No valid drugs found'}), 400

        # Aggregate the drugs into a single representation (e.g., Mean Pooling)
        # Resulting shape: [1, feature_dim]
        cocktail_tensor = torch.stack(drug_tensors).mean(dim=0).unsqueeze(0)

        # 2. Run through all 10 models
        for effect_name, model in side_effect_models.items():
            with torch.no_grad():
                logits = model(cocktail_tensor)
                score = torch.sigmoid(logits).item() 
                results[effect_name] = round(score, 4)

        return jsonify(results)
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)