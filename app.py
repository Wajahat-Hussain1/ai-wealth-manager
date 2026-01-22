import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Wealth Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #004d99; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        width: 100%; background-color: #004d99; color: white; font-weight: bold; padding: 10px;
    }
    .stButton>button:hover { background-color: #003366; }
    .metric-box {
        background-color: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center;
    }
    .footer {
        position: fixed; bottom: 0; left: 0; width: 100%; background-color: white;
        text-align: center; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        assets = {
            "Random Forest": joblib.load('rf_model.pkl'),
            "Decision Tree": joblib.load('dt_model.pkl'),
            "Logistic Regression": joblib.load('lr_model.pkl'),
            "SVM": joblib.load('svm_model.pkl'),
            "le": joblib.load('label_encoder.pkl'),
            "features": joblib.load('features.pkl'),
            "metrics": pd.read_csv('model_metrics.csv')
        }
        return assets
    except FileNotFoundError:
        return None

assets = load_assets()

if not assets:
    st.error("⚠️ Error: Model files missing. Please run the 'Save Models' step in your notebook.")
    st.stop()

models = {k: v for k, v in assets.items() if k not in ["le", "features", "metrics"]}
le = assets["le"]
feature_names = assets["features"]
metrics_df = assets["metrics"]

# --- HELPER: PARSE PORTFOLIO STRING ---
def parse_portfolio(portfolio_str):
    matches = re.findall(r"(\d+)%\s+([a-zA-Z\s]+)", portfolio_str)
    if matches:
        data = [{"Asset": m[1].strip(), "Percentage": int(m[0])} for m in matches]
        return pd.DataFrame(data)
    else:
        return pd.DataFrame({"Asset": ["Diversified Portfolio"], "Percentage": [100]})

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4202/4202568.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🤖 AI Predictor", "📊 Model Performance"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Active Model", list(models.keys()))
active_model = models[selected_model_name]

st.sidebar.markdown("---")
# REPLACE [Your Name Here] with your actual name
# st.sidebar.info("Developed by:\n**[Your Name Here]**\nFinal Year Project") 

# ==========================================
# PAGE 1: AI PREDICTOR
# ==========================================
if page == "🤖 AI Predictor":
    st.title("🏦 Intelligent Investment Advisor")
    st.markdown("### Get a personalized portfolio recommendation in seconds.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 1000, 10000000, 50000, step=1000)
    with col2:
        risk_tol = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
    with col3:
        sector = st.selectbox("Preferred Sector", ["Tech", "Real Estate", "Energy", "Finance", "Healthcare"])
        goal = st.selectbox("Investment Goal", ["Wealth Growth", "Retirement", "Education", "House Purchase"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Analyze Profile & Recommend"):
        # Predict
        input_data = {col: 0 for col in feature_names}
        input_data['Age'] = age
        input_data['Annual_Income'] = income
        input_data['Risk_Tolerance'] = {'Low': 0, 'Medium': 1, 'High': 2}[risk_tol]
        input_data['Investment_Horizon'] = {'Short-term': 0, 'Medium-term': 1, 'Long-term': 2}[horizon]
        if f"Preferred_Sector_{sector}" in input_data: input_data[f"Preferred_Sector_{sector}"] = 1
        if f"Investment_Goal_{goal}" in input_data: input_data[f"Investment_Goal_{goal}"] = 1
        
        df_in = pd.DataFrame([input_data])
        pred_idx = active_model.predict(df_in)[0]
        prediction_text = le.inverse_transform([pred_idx])[0]
        
        # Results
        st.success("✅ Analysis Complete!")
        r_col1, r_col2 = st.columns([1.5, 1])
        
        with r_col1:
            st.subheader("Recommended Strategy")
            st.info(f"**{prediction_text}**")
            st.markdown("#### 💡 Why this fits you:")
            st.write(f"- **Risk Profile:** Your **{risk_tol}** tolerance matches this asset mix.")
            st.write(f"- **Goal Alignment:** Optimized for **{goal}** over the **{horizon}**.")
            st.caption(f"Prediction made using: {selected_model_name}")

        with r_col2:
            st.subheader("Asset Allocation")
            pie_data = parse_portfolio(prediction_text)
            
            fig = px.pie(pie_data, values='Percentage', names='Asset', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Bold)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
    # DISCLAIMER (Professional Touch)
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is an AI prototype developed for educational purposes. It does not constitute professional financial advice. Please consult a certified financial advisor before making investment decisions.")

# ==========================================
# PAGE 2: MODEL PERFORMANCE
# ==========================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    st.markdown("Comparing the accuracy and reliability of different AI algorithms.")
    st.markdown("---")

    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("🏆 Best Model", best_model['Model'])
    m2.metric("🎯 Top Accuracy", f"{best_model['Accuracy']:.2%}")
    m3.metric("⚡ F1-Score", f"{best_model['F1-Score']:.2f}")

    st.markdown("---")
    st.subheader("⚔️ Accuracy Comparison")
    
    fig_acc = px.bar(metrics_df, x='Model', y='Accuracy', color='Model', 
                     text_auto='.2%', title="Model Accuracy Ranking",
                     color_discrete_sequence=px.colors.qualitative.Bold)
                     
    fig_acc.update_layout(showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)