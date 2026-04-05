import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

model = pickle.load(open("ipl_model.pkl", "rb"))

st.title("🏏 IPL Powerplay Win Predictor")

st.info("Prediction based on Powerplay performance (first 6 overs)")

st.write("Enter match details at 6 overs:")

score = st.number_input("Current Score (runs)", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)

run_rate = score / 6 if score > 0 else 0
pressure = score / (wickets + 1)

teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Daredevils', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Kings XI Punjab', 'Deccan Chargers'
]

team = st.selectbox("Select Batting Team", teams)

if wickets > 6:
    st.warning("⚠️ Too many wickets lost in Powerplay! Winning chances are low.")

if st.button("Predict Win Probability"):

    input_data = {
        'current_score': score,
        'wickets_lost': wickets,
        'run_rate': run_rate,
        'pressure': pressure
    }

    for t in teams:
        col_name = f"batting_team_{t}"
        input_data[col_name] = 1 if t == team else 0

    input_df = pd.DataFrame([input_data])

    model_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=model_cols, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]

    if prob > 0.5:
        st.success(f"🔥 High Chance to Win: {prob*100:.2f}%")
    else:
        st.error(f"❌ Low Chance to Win: {prob*100:.2f}%")

    fig = px.bar(
        x=["Win", "Lose"],
        y=[prob, 1 - prob],
        title="Win Probability Distribution",
        labels={"x": "Outcome", "y": "Probability"}
    )

    st.plotly_chart(fig)