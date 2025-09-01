import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- MODERN CUSTOM CSS -------------------
st.markdown("""
<style>
    /* Global Smooth Transition */
    * {
        transition: all 0.3s ease-in-out;
    }

    /* Modern Header */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        text-shadow: 0px 0px 15px rgba(79,172,254,0.7);
    }

    /* Prediction Box */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        color: #fff;
        font-weight: bold;
        box-shadow: 0px 5px 25px rgba(0,0,0,0.25);
        transform: scale(1);
    }
    .prediction-box:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 25px rgba(255,255,255,0.6), 0px 0px 50px rgba(0,255,255,0.4);
    }

    /* Fraud Styling */
    .fraud {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        box-shadow: 0 0 15px rgba(255, 65, 108, 0.7), 0 0 30px rgba(255, 75, 43, 0.7);
    }
    .fraud:hover {
        box-shadow: 0 0 25px #ff416c, 0 0 50px #ff4b2b;
    }

    /* Legitimate Styling */
    .legitimate {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        box-shadow: 0 0 15px rgba(0,176,155,0.7), 0 0 30px rgba(150,201,61,0.7);
    }
    .legitimate:hover {
        box-shadow: 0 0 25px #00b09b, 0 0 50px #96c93d;
    }

    /* Buttons Neon Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0px 0px 15px rgba(106,17,203,0.7);
    }
    div.stButton > button:hover {
        box-shadow: 0px 0px 25px rgba(37,117,252,0.9), 0px 0px 50px rgba(106,17,203,0.7);
        transform: translateY(-3px);
    }

    /* Sidebar Title Glow */
    section[data-testid="stSidebar"] .css-1d391kg {
        font-size: 1.2rem;
        font-weight: bold;
        color: #fff;
        text-shadow: 0px 0px 10px rgba(0,255,255,0.7);
    }

</style>
""", unsafe_allow_html=True)


# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    try:
        try:
            model = joblib.load('model.pkl')
        except:
            model = pickle.load(open('model.pkl', 'rb'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except:
        return None

# ------------------- MAIN APP -------------------
def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)

    model = load_model()
    scaler = load_scaler()

    if model is None:
        st.error("⚠️ Model could not be loaded. Please ensure 'model.pkl' exists.")
        return

    st.sidebar.title("📌 Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Single Transaction", "Batch Upload", "About"])

    if app_mode == "Single Transaction":
        single_transaction_prediction(model, scaler)
    elif app_mode == "Batch Upload":
        batch_prediction(model, scaler)
    else:
        about_page()

# ------------------- SINGLE TRANSACTION -------------------
def single_transaction_prediction(model, scaler):
    st.subheader("🔎 Single Transaction Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input("Time (seconds)", value=0.0, format="%.2f")
        v1 = st.number_input("V1", 0.0, format="%.6f")
        v2 = st.number_input("V2", 0.0, format="%.6f")
        v3 = st.number_input("V3", 0.0, format="%.6f")
        v4 = st.number_input("V4", 0.0, format="%.6f")
        v5 = st.number_input("V5", 0.0, format="%.6f")
        v6 = st.number_input("V6", 0.0, format="%.6f")
        v7 = st.number_input("V7", 0.0, format="%.6f")
        v8 = st.number_input("V8", 0.0, format="%.6f")
        v9 = st.number_input("V9", 0.0, format="%.6f")

    with col2:
        v10 = st.number_input("V10", 0.0, format="%.6f")
        v11 = st.number_input("V11", 0.0, format="%.6f")
        v12 = st.number_input("V12", 0.0, format="%.6f")
        v13 = st.number_input("V13", 0.0, format="%.6f")
        v14 = st.number_input("V14", 0.0, format="%.6f")
        v15 = st.number_input("V15", 0.0, format="%.6f")
        v16 = st.number_input("V16", 0.0, format="%.6f")
        v17 = st.number_input("V17", 0.0, format="%.6f")
        v18 = st.number_input("V18", 0.0, format="%.6f")

    with col3:
        v19 = st.number_input("V19", 0.0, format="%.6f")
        v20 = st.number_input("V20", 0.0, format="%.6f")
        v21 = st.number_input("V21", 0.0, format="%.6f")
        v22 = st.number_input("V22", 0.0, format="%.6f")
        v23 = st.number_input("V23", 0.0, format="%.6f")
        v24 = st.number_input("V24", 0.0, format="%.6f")
        v25 = st.number_input("V25", 0.0, format="%.6f")
        v26 = st.number_input("V26", 0.0, format="%.6f")
        v27 = st.number_input("V27", 0.0, format="%.6f")
        v28 = st.number_input("V28", 0.0, format="%.6f")
        amount = st.number_input("Amount", 0.0, format="%.2f")

    if st.button("🚀 Predict Fraud"):
        features = np.array([[time,
            v1, v2, v3, v4, v5, v6, v7, v8, v9,
            v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
            amount
        ]])

        if scaler is not None:
            features = scaler.transform(features)

        try:
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)[0]

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba[1]*100,
                title={'text': "Fraud Probability (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "crimson"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            if prediction[0] == 1:
                st.markdown(f"""
                <div class="prediction-box fraud">
                    🚨 FRAUD DETECTED <br>
                    Probability: {prediction_proba[1]:.4f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box legitimate">
                    ✅ LEGITIMATE TRANSACTION <br>
                    Probability: {prediction_proba[0]:.4f}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ------------------- BATCH PREDICTION -------------------
def batch_prediction(model, scaler):
    st.subheader("📊 Batch Transaction Analysis")

    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            missing_columns = [c for c in required_columns if c not in df.columns]

            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                return

            features = df[required_columns].values
            if scaler is not None:
                features = scaler.transform(features)

            predictions = model.predict(features)
            predictions_proba = model.predict_proba(features)

            df['Prediction'] = predictions
            df['Fraud_Probability'] = predictions_proba[:, 1]
            df['Status'] = df['Prediction'].map({1: "Fraud", 0: "Legitimate"})

            st.metric("Total Transactions", len(df))
            st.metric("Fraudulent Transactions", df['Prediction'].sum())

            # ---- Charts ----
            col1, col2 = st.columns(2)

            with col1:
                pie = px.pie(df, names="Status", title="Fraud vs Legitimate Distribution",
                             color="Status", color_discrete_map={"Fraud": "red", "Legitimate": "green"})
                st.plotly_chart(pie, use_container_width=True)

            with col2:
                hist = px.histogram(df, x="Fraud_Probability", nbins=50,
                                    color="Status", title="Fraud Probability Distribution")
                st.plotly_chart(hist, use_container_width=True)

            trend = px.line(df, x="Time", y="Fraud_Probability", color="Status",
                            title="Fraud Probability Over Time")
            st.plotly_chart(trend, use_container_width=True)

            st.subheader("🔍 Prediction Table (first 20 rows)")
            st.dataframe(df.head(20).style.apply(
                lambda row: ['background-color: #ffcccc' if row.Status == "Fraud" else '' for _ in row], axis=1))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ------------------- ABOUT PAGE -------------------
def about_page():
    st.subheader("ℹ️ About This App")
    st.markdown("""
    This **Credit Card Fraud Detection Dashboard** helps analyze transactions for potential fraud.

    **Features**
    - Real-time single transaction fraud prediction
    - Bulk transaction analysis with CSV uploads
    - Advanced visualizations: Fraud probability gauges, pie charts, histograms, trends
    - Downloadable fraud prediction reports

    **Dataset**
    - Based on the popular Kaggle dataset
    - Features: `Time`, `V1–V28` (PCA features), `Amount`
    - Target: Fraud (1) or Legitimate (0)

    ⚠️ *Disclaimer: For demonstration purposes only. Not for production banking use.*
    """)

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    main()
