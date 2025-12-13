import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Valorant Winner Predictor", layout="centered")
st.title("🎮 Valorant Match Winner Predictor")
st.markdown("Select both teams and a map to predict who’s likely to win the match!")

# ---------------- Load Model ----------------
if os.path.exists("valorant_winner_final_xgb.pkl"):
    model = joblib.load("valorant_winner_final_xgb.pkl")
    st.success("Model loaded successfully!")
else:
    st.error("Model file not found.")
    st.stop()

# ---------------- Load Data ----------------
try:
    team_stats = pd.read_csv("dataset/team_aggregated_stats.csv")
    maps = pd.read_csv("dataset/maps_stats.csv")
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Select Match Details")

team_list = sorted(team_stats["team"].dropna().unique())
map_list = sorted(maps["map_name"].dropna().unique())

team1 = st.sidebar.selectbox("Select Team 1", team_list)
team2 = st.sidebar.selectbox("Select Team 2", [t for t in team_list if t != team1])
selected_map = st.sidebar.selectbox("Select Map", map_list)

# ---------------- Prepare Input ----------------
if st.button("Predict Winner"):
    t1 = team_stats[team_stats["team"] == team1].iloc[0]
    t2 = team_stats[team_stats["team"] == team2].iloc[0]

    input_data = np.array([[  
        t1["rating"], t1["acs"], t1["adr"], t1["kast"], t1["hs_percent"],
        t1["fk"], t1["fd"], t1["fk_fd_diff"],
        t2["rating"], t2["acs"], t2["adr"], t2["kast"], t2["hs_percent"],
        t2["fk"], t2["fd"], t2["fk_fd_diff"],
        0,  # picked_by
        0,  # picked_by_team1
        0   # picked_by_team2
    ]])

    try:
        pred = model.predict(input_data)[0]
        if pred == 1:
            st.success(f"🏆 {team1.upper()} is likely to WIN on {selected_map.upper()}!")
        else:
            st.error(f"🏆 {team2.upper()} is likely to WIN on {selected_map.upper()}!")
    except Exception as e:
        st.error(f"Prediction error: {e}")
