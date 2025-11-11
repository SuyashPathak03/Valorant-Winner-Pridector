import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Valorant Winner Predictor", page_icon="", layout="centered")
st.title(" Valorant Match Winner Predictor")
st.markdown("Predict which team will win based on performance stats and map selection!")

# ---- Load model safely ----
try:
    if os.path.exists("valorant_winner_final_xgb.pkl"):
        model = joblib.load("valorant_winner_final_xgb.pkl")
        st.success(" Model loaded successfully!")
    else:
        st.error(" Model file not found.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ---- Inputs ----
st.sidebar.header("Input Match Details")
team1 = st.sidebar.text_input("Team 1 Name", "sentinels")
team2 = st.sidebar.text_input("Team 2 Name", "drx")

acs_team1 = st.number_input("Team 1 ACS", min_value=100, max_value=300, value=210)
acs_team2 = st.number_input("Team 2 ACS", min_value=100, max_value=300, value=190)

if st.button("Predict Winner"):
    try:
        input_data = np.array([[acs_team1, acs_team2]])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success(f" {team1.upper()} is likely to WIN!")
        else:
            st.error(f" {team2.upper()} might take the match!")
    except Exception as e:
        st.error(f" Prediction error: {e}")
