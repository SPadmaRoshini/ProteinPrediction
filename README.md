# Protein Structure Prediction and Stability Analysis

This repository contains a Streamlit web application for protein structure prediction and stability analysis using a Transformer-GNN model. The app integrates ESM-2 embeddings and various molecular visualization tools to provide insights into protein structures, residue interactions, and stability factors.


# Features

Protein Structure Prediction: Uses a Transformer-GNN model to predict protein structures based on sequence data.

Molecular Visualization: Interactive 3D visualization of protein structures.

Residue-Level Contact Maps: Displays interactions between residues.

Secondary Structure Interactions: Identifies and visualizes secondary structural elements.

Ramachandran Plots: Analyzes backbone dihedral angles for structure validation.

Stability Estimation: Evaluates environmental factors affecting protein stability.

API Integration: Fetches protein structure and sequence data dynamically.

# Installation

Prerequisites

Ensure you have Python installed (>=3.8). This project is hosted and developed in GitHub Codespaces.

Clone the Repository : 

git clone https://github.com/SPadmaRoshini/ProteinPrediction.git

cd ProteinPrediction

Install Dependencies : 

pip install -r requirements.txt

# Usage

Running the Streamlit App : 

streamlit run app.py

API Integration

This app fetches protein structure and sequence data via APIs instead of manual uploads. Ensure you have network access to fetch data dynamically.

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

For issues and contributions, please open an issue in the repository or contact padma.roshini04@gmail.com.

