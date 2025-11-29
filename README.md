# 🎮 Valorant Match Winner Predictor  

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?logo=streamlit)](https://valorant-winner-predictor-nddkfnr6tlxfpfzb6xynna.streamlit.app/)  

## 🧠 Overview  
This project predicts the **winner of a Valorant match** based on team performance statistics and map selection.  
The goal is to combine real competitive data with machine learning to forecast match outcomes — all through a simple, interactive web app.  

The project uses data from the **Valorant Champions 2024** dataset on Kaggle and builds a predictive model using **XGBoost**.

---

## 🚀 Live Demo  
👉 **Try it here:**  
🔗 [https://valorant-winner-predictor-nddkfnr6tlxfpfzb6xynna.streamlit.app/](https://valorant-winner-predictor-nddkfnr6tlxfpfzb6xynna.streamlit.app/)

---

## 🧩 Features  
- 🏆 Predicts match winner between two selected teams  
- 📊 Uses team-level performance features such as:
  - Rating  
  - ACS (Average Combat Score)  
  - ADR (Average Damage per Round)  
  - KAST% (Kill, Assist, Survive, Trade %)  
  - HS% (Headshot %)  
  - FK / FD (First Kill / First Death)  
  - FK-FD Difference  
- 🗺️ Incorporates map information and pick indicators  
- 💻 Built with **Streamlit** for an easy web interface  

---

   Model  
Model: **XGBoost Classifier**  
- Trained on 19 engineered features (team & map-based)
- Achieved ~86% cross-validation accuracy  
- Saved as: `valorant_winner_final_xgb.pkl`  

Feature list used in training:  
