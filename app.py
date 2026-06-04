import streamlit as st
import pickle
import pandas as pd

with open('f1_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encodings.pkl', 'rb') as f:
    encodings = pickle.load(f)

drivers = list(encodings['drivers'].keys())
teams = list(encodings['teams'].keys())

st.title("🏎️ F1 Podium Predictor")
st.write("Predict whether a driver will finish on the podium!")

st.subheader("Enter Race Details")

driver = st.selectbox("Select Driver", sorted(drivers))
team = st.selectbox("Select Team", sorted(teams))
grid_position = st.slider("Grid Position", 1, 20, 1)
recent_form = st.slider("Recent Form (0=no podiums, 1=all podiums)", 0.0, 1.0, 0.0)
constructor_strength = st.slider("Constructor Strength (avg points per race)", 0.0, 25.0, 5.0)
circuit_history = st.slider("Circuit History (0=never podiumed here, 1=always)", 0.0, 1.0, 0.0)

if st.button("Predict Podium 🏆"):
    driver_encoded = encodings['drivers'].get(driver, 0)
    team_encoded = encodings['teams'].get(team, 0)

    input_data = pd.DataFrame(
        [[grid_position, team_encoded, driver_encoded,
          recent_form, constructor_strength, circuit_history]],
        columns=['GridPosition', 'TeamEncoded', 'DriverEncoded',
                 'RecentForm', 'ConstructorStrength', 'CircuitHistory']
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🏆 {driver} is predicted to PODIUM! ({probability*100:.1f}% confidence)")
    else:
        st.error(f"❌ {driver} is NOT predicted to podium. ({probability*100:.1f}% confidence)")