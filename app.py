# app.py
import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Melting Point Predictor 🔥")
st.write("Enter a SMILES string to predict the melting point.")

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(424)  # fallback for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=424)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

smiles_input = st.text_input("SMILES", value="CCO")

# Prediction
if st.button("Predict Melting Point"):
    features = smiles_to_features(smiles_input).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    st.success(f"Predicted Melting Point: {prediction:.2f} K")
