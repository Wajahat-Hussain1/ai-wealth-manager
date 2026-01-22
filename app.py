import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Wealth Manager", page_icon="📈", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        assets = {
            "Random Forest": joblib.load('rf_model.pkl'),
            "Decision Tree": joblib.load('dt_model.pkl'),
            "Logistic Regression": joblib.load('lr_model.pkl'),
            "SVM": joblib.load('svm_model.pkl'),
            "Label Encoder": joblib.load('label_encoder.pkl'),
            "Features": joblib.load('features.pkl')
        }
        metrics = pd.read_csv('model_metrics.csv')
        return assets, metrics
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

assets, metrics_df = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💰 AI Wealth Advisor")
page = st.sidebar.radio("Go to", ["🏠 Investment Advisor", "📊 Model Performance"])

if page == "🏠 Investment Advisor":
    st.title("📈 AI Intelligent Investment Advisor")
    st.write("Enter your details to get a personalized investment portfolio.")
    
    # User Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    risk = st.slider("Risk Tolerance (1=Low, 5=High)", 1, 5, 3)
    horizon = st.slider("Investment Horizon (Years)", 1, 40, 10)
    
    if st.button("Generate Recommendation"):
        # This is where your model would predict
        st.success("Recommendation Generated!")
        # (Prediction logic goes here based on your notebook)

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    if metrics_df is not None:
        st.dataframe(metrics_df)
        fig = px.bar(metrics_df, x='Model', y='Accuracy', title="Model Accuracy")
        st.plotly_chart(fig)