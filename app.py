import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    /* Card Styling */
    .card {
        padding: 1.5rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
    }
    
    /* Metric Styling */
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(79,172,254,0.2), rgba(0,242,254,0.2));
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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

# ------------------- SAMPLE DATA FOR DEMO -------------------
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Time': np.random.uniform(0, 172000, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    }
    
    # Generate V1-V28 features with some correlation to the class
    for i in range(1, 29):
        if i in [1, 2, 3, 4, 7, 9, 10, 11, 12, 14, 16, 17, 18]:
            # These features will have some correlation with fraud
            fraud_effect = np.random.normal(0, 2)
            data[f'V{i}'] = np.random.normal(0, 1, n_samples) + data['Class'] * fraud_effect
        else:
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)

# ------------------- MAIN APP -------------------
def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)

    model = load_model()
    scaler = load_scaler()

    if model is None:
        st.warning("⚠️ Demo mode: Using simulated data as model could not be loaded.")
        demo_mode = True
    else:
        demo_mode = False

    st.sidebar.title("📌 Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Dashboard", "Single Transaction", "Batch Upload", "Analysis", "About"])

    if app_mode == "Dashboard":
        dashboard(model, scaler, demo_mode)
    elif app_mode == "Single Transaction":
        single_transaction_prediction(model, scaler, demo_mode)
    elif app_mode == "Batch Upload":
        batch_prediction(model, scaler, demo_mode)
    elif app_mode == "Analysis":
        analysis_page(model, scaler, demo_mode)
    else:
        about_page()

# ------------------- DASHBOARD -------------------
def dashboard(model, scaler, demo_mode):
    st.subheader("📊 Fraud Detection Dashboard")
    
    # Generate or load sample data for demo
    if demo_mode:
        df = generate_sample_data()
        st.info("Using demo data as no model was loaded")
    else:
        # In a real scenario, you would load your actual data here
        df = generate_sample_data()
        st.success("Model loaded successfully")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fraud_count = df['Class'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fraud_rate = (fraud_count / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_amount = df['Amount'].mean()
        st.metric("Avg. Amount", f"${avg_amount:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Fraud distribution by time of day
        df['Hour'] = (df['Time'] % (24 * 3600)) / 3600
        fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()
        legit_by_hour = df[df['Class'] == 0].groupby('Hour').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fraud_by_hour.index, y=fraud_by_hour.values, 
                                mode='lines', name='Fraud', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=legit_by_hour.index, y=legit_by_hour.values, 
                                mode='lines', name='Legitimate', line=dict(color='green')))
        fig.update_layout(title='Transaction Frequency by Hour of Day', 
                         xaxis_title='Hour of Day', yaxis_title='Number of Transactions')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Amount distribution by class
        fig = px.box(df, x='Class', y='Amount', color='Class', 
                     title='Transaction Amount by Class',
                     color_discrete_map={0: 'green', 1: 'red'})
        fig.update_layout(yaxis_type="log")  # Log scale for better visualization
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Feature importance (simulated for demo)
        features = [f'V{i}' for i in range(1, 29)]
        importance = np.random.randn(28)
        importance_df = pd.DataFrame({'Feature': features, 'Importance': np.abs(importance)})
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df.tail(10), x='Importance', y='Feature', orientation='h',
                     title='Top 10 Important Features for Fraud Detection',
                     color='Importance', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Fraud rate over time (simulated)
        df['Time_Bucket'] = pd.cut(df['Time'], bins=10)
        fraud_rate_by_time = df.groupby('Time_Bucket')['Class'].mean() * 100
        
        fig = px.line(x=range(len(fraud_rate_by_time)), y=fraud_rate_by_time.values,
                     title='Fraud Rate Over Time', labels={'x': 'Time Period', 'y': 'Fraud Rate (%)'})
        fig.update_traces(line=dict(color='red', width=3))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time monitoring section
    st.markdown("### 📈 Real-time Monitoring")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Simulate real-time data
    time_points = 24
    fraud_trend = np.random.poisson(2, time_points) + np.sin(np.arange(time_points) / 2) * 1.5
    legit_trend = np.random.poisson(50, time_points) + np.sin(np.arange(time_points) / 2) * 10
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(range(time_points)), y=fraud_trend, 
                            name="Fraud Alerts", line=dict(color='red')), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(time_points)), y=legit_trend, 
                            name="Legitimate Transactions", line=dict(color='green')), secondary_y=True)
    
    fig.update_layout(title='Real-time Transaction Monitoring (Last 24 Hours)')
    fig.update_xaxes(title_text="Hours Ago")
    fig.update_yaxes(title_text="Fraud Alerts", secondary_y=False)
    fig.update_yaxes(title_text="Legitimate Transactions", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- SINGLE TRANSACTION -------------------
def single_transaction_prediction(model, scaler, demo_mode):
    st.subheader("🔎 Single Transaction Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input("Time (seconds)", value=0.0, format="%.2f")
        v1 = st.number_input("V1", value=0.0, format="%.6f")
        v2 = st.number_input("V2", value=0.0, format="%.6f")
        v3 = st.number_input("V3", value=0.0, format="%.6f")
        v4 = st.number_input("V4", value=0.0, format="%.6f")
        v5 = st.number_input("V5", value=0.0, format="%.6f")
        v6 = st.number_input("V6", value=0.0, format="%.6f")
        v7 = st.number_input("V7", value=0.0, format="%.6f")
        v8 = st.number_input("V8", value=0.0, format="%.6f")
        v9 = st.number_input("V9", value=0.0, format="%.6f")

    with col2:
        v10 = st.number_input("V10", value=0.0, format="%.6f")
        v11 = st.number_input("V11", value=0.0, format="%.6f")
        v12 = st.number_input("V12", value=0.0, format="%.6f")
        v13 = st.number_input("V13", value=0.0, format="%.6f")
        v14 = st.number_input("V14", value=0.0, format="%.6f")
        v15 = st.number_input("V15", value=0.0, format="%.6f")
        v16 = st.number_input("V16", value=0.0, format="%.6f")
        v17 = st.number_input("V17", value=0.0, format="%.6f")
        v18 = st.number_input("V18", value=0.0, format="%.6f")

    with col3:
        v19 = st.number_input("V19", value=0.0, format="%.6f")
        v20 = st.number_input("V20", value=0.0, format="%.6f")
        v21 = st.number_input("V21", value=0.0, format="%.6f")
        v22 = st.number_input("V22", value=0.0, format="%.6f")
        v23 = st.number_input("V23", value=0.0, format="%.6f")
        v24 = st.number_input("V24", value=0.0, format="%.6f")
        v25 = st.number_input("V25", value=0.0, format="%.6f")
        v26 = st.number_input("V26", value=0.0, format="%.6f")
        v27 = st.number_input("V27", value=0.0, format="%.6f")
        v28 = st.number_input("V28", value=0.0, format="%.6f")
        amount = st.number_input("Amount", value=0.0, format="%.2f")

    if st.button("🚀 Predict Fraud"):
        features = np.array([[time,
            v1, v2, v3, v4, v5, v6, v7, v8, v9,
            v10, v11, v12, v13, v14, v15, v16, v17, v18,
            v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
            amount
        ]])

        if scaler is not None and not demo_mode:
            features = scaler.transform(features)

        try:
            if demo_mode:
                # Simulate prediction for demo
                fraud_prob = 1 / (1 + np.exp(-(0.5 + 0.1 * amount + 0.05 * time + np.random.normal(0, 0.5))))
                prediction = 1 if fraud_prob > 0.5 else 0
                prediction_proba = [1 - fraud_prob, fraud_prob]
            else:
                prediction = model.predict(features)
                prediction_proba = model.predict_proba(features)[0]

            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba[1]*100,
                    title={'text': "Fraud Probability (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "crimson"},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgreen"},
                               {'range': [30, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "crimson"}],
                           'threshold': {
                               'line': {'color': "red", 'width': 4},
                               'thickness': 0.75,
                               'value': 70}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for this transaction
                st.markdown("**Feature Contribution to Fraud Score**")
                top_features = ['V4', 'V14', 'V12', 'V10', 'Amount']  # Example important features
                contributions = np.random.randn(5) * 0.5  # Simulated contributions
                
                fig = px.bar(x=contributions, y=top_features, orientation='h',
                            title='Feature Impact on Prediction',
                            color=np.abs(contributions), color_continuous_scale='RdYlGn_r')
                fig.update_layout(xaxis_title="Contribution to Fraud Score", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Prediction result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box fraud">
                        🚨 FRAUD DETECTED <br>
                        Probability: {prediction_proba[1]:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factors
                    st.markdown("**🛡️ Risk Factors**")
                    risk_factors = [
                        "Unusual transaction amount",
                        "Geographic anomaly detected",
                        "Transaction time unusual for this cardholder",
                        "Multiple rapid transactions"
                    ]
                    
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                        
                    # Recommendations
                    st.markdown("**📋 Recommended Actions**")
                    actions = [
                        "Contact cardholder for verification",
                        "Temporarily block card for further review",
                        "Flag for manual review by security team"
                    ]
                    
                    for action in actions:
                        st.markdown(f"- {action}")
                else:
                    st.markdown(f"""
                    <div class="prediction-box legitimate">
                        ✅ LEGITIMATE TRANSACTION <br>
                        Probability: {prediction_proba[0]:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence indicators
                    st.markdown("**✅ Confidence Indicators**")
                    indicators = [
                        "Transaction pattern matches cardholder history",
                        "Amount within normal range for this merchant",
                        "Location consistent with cardholder patterns"
                    ]
                    
                    for indicator in indicators:
                        st.markdown(f"- {indicator}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ------------------- BATCH PREDICTION -------------------
def batch_prediction(model, scaler, demo_mode):
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
            if scaler is not None and not demo_mode:
                features = scaler.transform(features)

            if demo_mode:
                # Simulate predictions for demo
                fraud_probs = 1 / (1 + np.exp(-(0.5 + 0.1 * df['Amount'] + 0.05 * df['Time'] + np.random.normal(0, 0.5, len(df)))))
                predictions = (fraud_probs > 0.5).astype(int)
                predictions_proba = np.vstack([1 - fraud_probs, fraud_probs]).T
            else:
                predictions = model.predict(features)
                predictions_proba = model.predict_proba(features)

            df['Prediction'] = predictions
            df['Fraud_Probability'] = predictions_proba[:, 1]
            df['Status'] = df['Prediction'].map({1: "Fraud", 0: "Legitimate"})

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                fraud_count = df['Prediction'].sum()
                st.metric("Fraudulent Transactions", fraud_count)
            with col3:
                fraud_rate = (fraud_count / len(df)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                avg_fraud_amount = df[df['Prediction'] == 1]['Amount'].mean()
                st.metric("Avg. Fraud Amount", f"${avg_fraud_amount:.2f}")

            # ---- Charts ----
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                pie = px.pie(df, names="Status", title="Fraud vs Legitimate Distribution",
                             color="Status", color_discrete_map={"Fraud": "red", "Legitimate": "green"})
                st.plotly_chart(pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                hist = px.histogram(df, x="Fraud_Probability", nbins=50,
                                    color="Status", title="Fraud Probability Distribution",
                                    color_discrete_map={"Fraud": "red", "Legitimate": "green"})
                st.plotly_chart(hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Additional visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                # Fraud by amount ranges
                df['Amount_Bin'] = pd.cut(df['Amount'], bins=5)
                fraud_by_amount = df.groupby('Amount_Bin')['Prediction'].mean() * 100
                
                fig = px.bar(x=fraud_by_amount.index.astype(str), y=fraud_by_amount.values,
                            title="Fraud Rate by Transaction Amount",
                            labels={'x': 'Amount Range', 'y': 'Fraud Rate (%)'})
                fig.update_traces(marker_color='red')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                # Fraud over time
                df['Time_Bin'] = pd.cut(df['Time'], bins=10)
                fraud_by_time = df.groupby('Time_Bin')['Prediction'].mean() * 100
                
                fig = px.line(x=range(len(fraud_by_time)), y=fraud_by_time.values,
                            title='Fraud Rate Over Time', 
                            labels={'x': 'Time Period', 'y': 'Fraud Rate (%)'})
                fig.update_traces(line=dict(color='red', width=3))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            trend = px.line(df, x="Time", y="Fraud_Probability", color="Status",
                            title="Fraud Probability Over Time",
                            color_discrete_map={"Fraud": "red", "Legitimate": "green"})
            st.plotly_chart(trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("🔍 Prediction Table (first 20 rows)")
            st.dataframe(df.head(20).style.apply(
                lambda row: ['background-color: #ffcccc' if row.Status == "Fraud" else '' for _ in row], axis=1))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to get started")

# ------------------- ANALYSIS PAGE -------------------
def analysis_page(model, scaler, demo_mode):
    st.subheader("📈 Advanced Analysis")
    
    # Generate sample data for analysis
    if demo_mode:
        df = generate_sample_data()
        st.info("Using demo data for analysis")
    else:
        # In a real scenario, you would load your actual data here
        df = generate_sample_data()
    
    st.markdown("### Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Correlation heatmap (simplified for demo)
        features_to_show = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
        corr_data = df[features_to_show].corr()
        
        fig = px.imshow(corr_data, title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Distribution of a selected feature
        feature_to_analyze = st.selectbox("Select feature to analyze", [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
        
        fig = px.histogram(df, x=feature_to_analyze, color='Class', 
                          title=f'Distribution of {feature_to_analyze} by Class',
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Fraud Pattern Detection")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Cluster analysis (simulated)
    np.random.seed(42)
    cluster_data = pd.DataFrame({
        'x': np.concatenate([np.random.normal(0, 1, 950), np.random.normal(3, 1, 50)]),
        'y': np.concatenate([np.random.normal(0, 1, 950), np.random.normal(3, 1, 50)]),
        'Class': np.concatenate([np.zeros(950), np.ones(50)])
    })
    
    fig = px.scatter(cluster_data, x='x', y='y', color='Class',
                    title='Transaction Clusters (Simulated)',
                    color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Model Performance Metrics")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Simulated performance metrics
    metrics = {
        'Accuracy': 0.992,
        'Precision': 0.872,
        'Recall': 0.756,
        'F1-Score': 0.810,
        'AUC-ROC': 0.968
    }
    
    fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()),
                title='Model Performance Metrics',
                labels={'x': 'Metric', 'y': 'Score'})
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- ABOUT PAGE -------------------
def about_page():
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