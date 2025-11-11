import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Valorant Winner Predictor", layout="centered")
st.title(" Valorant Match Winner Predictor")
st.markdown("Select both teams and a map to predict who’s likely to win the match!")

# ---- Load model ----
try:
    if os.path.exists("valorant_winner_final_xgb.pkl"):
        model = joblib.load("valorant_winner_final_xgb.pkl")
        st.success(" Model loaded successfully!")
    else:
        st.error(" Model file not found.")
except Exception as e:
    st.error(f" Error loading model: {e}")

# ---- Load dataset ----
try:
    path = "./dataset"  # Adjust if your path differs
    players = pd.read_csv(path + "/player_stats.csv")
    maps = pd.read_csv(path + "/maps_stats.csv")
    st.success(" Dataset loaded successfully!")
except Exception as e:
    st.error(f" Error loading dataset: {e}")


# ---- Compute average team stats ----
if 'players' in locals() and not players.empty:
    # Auto-detect correct team column
    team_col = None
    for c in players.columns:
        if c.lower() in ["player_team", "team", "team_name", "playerteam"]:
            team_col = c
            break

    if team_col:
        st.info(f" Using team column: `{team_col}`")

        # Check for required numeric columns
        required_stats = [col for col in ["rating", "acs", "adr"] if col in players.columns]
        if len(required_stats) < 3:
            st.warning(" Some expected stat columns (rating, acs, adr) are missing in player_stats.csv.")
        else:
            # Calculate team averages
            team_stats = players.groupby(team_col)[required_stats].mean().reset_index().rename(columns={team_col: "team"})

            # Sidebar selections
            st.sidebar.header("Select Match Details")
            team_list = sorted(team_stats["team"].dropna().unique())

            if len(team_list) == 0:
                st.error(" No team names found in the dataset.")
            else:
                maps_col = None
                for c in maps.columns:
                    if c.lower() in ["map_name", "map", "mapname"]:
                        maps_col = c
                        break

                if maps_col:
                    map_list = sorted(maps[maps_col].dropna().unique())
                else:
                    st.warning(" Could not detect map column; using default maps.")
                    map_list = ["ascent", "haven", "lotus", "split", "bind", "sunset", "breeze", "abyss"]

                team1 = st.sidebar.selectbox("Select Team 1", team_list)
                team2 = st.sidebar.selectbox("Select Team 2", [t for t in team_list if t != team1])
                selected_map = st.sidebar.selectbox("Select Map", map_list)

                # Prediction button
                if st.button("Predict Winner"):
                    t1 = team_stats.loc[team_stats["team"] == team1].mean()
                    t2 = team_stats.loc[team_stats["team"] == team2].mean()

                    input_data = np.array([[
                        t1.get("rating", 1.0), t2.get("rating", 1.0),
                        t1.get("acs", 200), t2.get("acs", 200),
                        t1.get("adr", 120), t2.get("adr", 120),
                        0
                    ]])

                    try:
                        pred = model.predict(input_data)[0]
                        st.subheader(f"🗺️ Map: {selected_map.upper()}")
                        if pred == 1:
                            st.success(f" {team1.upper()} is likely to WIN!")
                        else:
                            st.error(f" {team2.upper()} might WIN!")
                    except Exception as e:
                        st.error(f" Prediction error: {e}")
    else:
        st.error(" Could not find a valid team column in player_stats.csv")
else:
    st.error(" Player dataset not loaded or empty.")
