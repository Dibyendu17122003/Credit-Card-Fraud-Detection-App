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
    st.set_page_config(page_title="About - Credit Card Fraud Detection", layout="wide")
    
    st.title("ℹ️ About Credit Card Fraud Detection Dashboard")
    
    # App Overview
    st.markdown("""
    ---
    ## 🚀 Dashboard Overview
    Welcome to the **Credit Card Fraud Detection Dashboard**, a modern, interactive tool designed to help detect suspicious activities in credit card transactions.  
    Whether you want to analyze **a single transaction** or **bulk data**, this dashboard provides detailed insights through advanced visualizations.  

    **Key Features**
    - 🔹 Real-time single transaction fraud prediction  
    - 🔹 Bulk transaction analysis with CSV uploads  
    - 🔹 Interactive visualizations: Fraud probability gauges, pie charts, histograms, trend lines  
    - 🔹 Downloadable fraud prediction reports for easy sharing and compliance  
    - 🔹 Insights on model confidence and feature importance
    """)
    
    # Theoretical Background
    st.markdown("""
    ---
    ## 📚 Credit Card & Fraud Theoretical Background
    ### What is a Credit Card?
    A **credit card** is a payment card issued by financial institutions allowing users to borrow funds to pay for goods and services. It provides convenience, rewards, and financial flexibility.  

    ### Types of Credit Card Fraud
    Credit card fraud occurs when **unauthorized users gain access to a cardholder's information** and perform illegal transactions. Common types include:  
    - **Stolen card data**: Using lost or stolen physical cards  
    - **Counterfeit cards**: Cloning or creating fake cards  
    - **Card-not-present (CNP) fraud**: Online or phone transactions  
    - **Identity theft fraud**: Using someone’s personal info to obtain a credit card  

    ### Impact of Credit Card Fraud
    - 💳 Financial losses for consumers and banks  
    - 🏦 Damage to banking credibility and trust  
    - 💰 Increased operational costs for fraud prevention  
    - ⚖️ Regulatory penalties for non-compliance in financial systems  

    ### Why Fraud Detection Matters
    Detecting fraud is **critical** to protect users and institutions. Advanced detection methods help:  
    - Identify suspicious behavior patterns in real-time  
    - Minimize financial losses  
    - Ensure regulatory compliance  
    - Improve customer trust and satisfaction  
    """)
    
    # Importance of Machine Learning
    st.markdown("""
    ---
    ## 🤖 Role of Machine Learning in Fraud Detection
    Traditional rule-based systems are **limited** because fraud patterns evolve rapidly. Machine learning offers:  
    - **Adaptive detection:** Learns from historical transactions  
    - **Predictive power:** Detects anomalies before financial damage occurs  
    - **Probabilistic scoring:** Assigns a likelihood of fraud for better decision-making  
    - **Scalability:** Handles millions of transactions efficiently  

    **Popular ML Techniques in Fraud Detection:**  
    - Logistic Regression & Decision Trees  
    - Random Forest & Gradient Boosting  
    - Support Vector Machines (SVM)  
    - Neural Networks & Deep Learning  
    - Anomaly detection algorithms (Isolation Forest, Autoencoders)
    """)
    
    # Dataset Info
    st.markdown("""
    ---
    ## 🗄️ Dataset Overview
    Our models are trained on the **Kaggle Credit Card Fraud dataset**:  
    - Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
    - Features: `Time`, `V1–V28` (PCA-transformed features), `Amount`  
    - Target: `Class` — Fraud (1) or Legitimate (0)  

    **Dataset Highlights**  
    - Highly imbalanced: Fraud transactions are <0.2% of total data  
    - Data anonymized for privacy  
    - Suitable for demonstrating ML-based fraud detection
    """)
    
    # Model Info
    st.markdown("""
    ---
    ## 🧠 Model Information
    Our dashboard uses **state-of-the-art machine learning models** trained on historical transaction data.  

    **Model Highlights**  
    - Handles imbalanced datasets with specialized techniques  
    - Provides fraud probability scores for each transaction  
    - Enables detailed insights through **interactive charts**  
    - Continuous learning potential for evolving fraud patterns
    """)
    
    # Developed By
    st.markdown("""
    ---
    ## 👨‍💻 Developed By
    **Dibyendu Karmahapatra**  
    Passionate about **Data Science**, **Machine Learning**, and creating intuitive dashboards for **real-world applications**.  
    This project reflects expertise in **Python, Streamlit, ML modeling, and data visualization**.
    """)
    
    # Why Choose This Dashboard
    st.markdown("""
    ---
    ## 🌟 Why Choose Our Dashboard?
    - **Interactive & Modern UI:** Clear, responsive, and easy-to-navigate interface  
    - **Advanced Visualizations:** Fraud probability gauges, histograms, pie charts, and trend lines  
    - **Accurate Predictions:** ML models trained on real-world transaction data  
    - **Educational Value:** Learn how fraud detection works and understand patterns in data  
    - **Downloadable Reports:** Export predictions for documentation or analysis  
    - **Secure & Privacy-Focused:** No personal card information stored
    """)
    
    # Modern Extra Section (Optional Infographics / Tips)
    st.markdown("""
    ---
    ## 📈 Quick Tips for Fraud Prevention
    - Never share your card PIN or CVV  
    - Enable two-factor authentication for online transactions  
    - Monitor bank statements regularly  
    - Use trusted and secure payment gateways  
    - Report suspicious activity immediately to your bank
    """)
    
    # Disclaimer
    st.markdown("""
    ---
    ## ⚠️ Disclaimer
    This application is for **educational and demonstration purposes only**.  
    **Do NOT use it for real banking decisions or financial advice.**
    """)
# ------------------- RUN APP -------------------
if __name__ == "__main__":
    main()
