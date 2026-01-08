# Polypharmacy Risk Predictor

A web application that uses Graph Neural Networks (GNN) to predict potential side effects from drug cocktails (polypharmacy).

![Demo Screenshot](screenshot.png)

## Features

- 🔍 **Searchable Drug Selection** - Type to search from 150+ drugs with autocomplete
- 🧬 **GNN-Powered Predictions** - Uses molecular fingerprints and graph neural network features
- 📊 **Risk Assessment** - Predicts 10 different types of toxicity/side effects
- 🎨 **Modern Dark UI** - Clean, responsive interface built with Tailwind CSS

## Tech Stack

- **Backend**: Flask, PyTorch, RDKit
- **Frontend**: HTML, Tailwind CSS, Select2
- **ML**: Graph Neural Networks, Morgan Fingerprints

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/polypharmacy-risk-predictor.git
cd polypharmacy-risk-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser to `http://127.0.0.1:5000`

## Usage

1. Click on the drug selection box
2. Type to search for drugs by name
3. Select multiple drugs to analyze
4. Click "Run Simulation" to get risk predictions
5. View the predicted risk percentages for each side effect type

## Side Effects Predicted

- Hepatotoxicity (liver damage)
- Nephrotoxicity (kidney damage)
- Cardiotoxicity (heart damage)
- Neurotoxicity (nervous system damage)
- Hematotoxicity (blood disorders)
- Gastrointestinal effects
- Dermatological effects
- Respiratory effects
- Metabolic effects
- Immunological effects

## License

MIT License

## Acknowledgments

- Drug data from PubChem
- RDKit for molecular fingerprinting
