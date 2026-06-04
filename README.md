# F1 Podium Predictor

A Machine Learning app that predicts whether an F1 driver will finish on the podium.

## Features
- Trained on 500+ race results across 5 seasons (2019-2023)
- 92% prediction accuracy
- Uses Random Forest classifier
- Built with Python, scikit-learn, and Streamlit

## Tech Stack
- Python
- FastF1 (data collection)
- Pandas (data processing)
- Scikit-learn (machine learning)
- Streamlit (web app)

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Model Features
- Grid position
- Driver encoded
- Team encoded
- Recent form (last 3 races)
- Constructor strength
- Circuit history
