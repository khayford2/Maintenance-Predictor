import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="HPU Maintenance Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources with caching
@st.cache_resource
def load_ann_resources():
    try:
        model = load_model('ann_opt.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"ANN Load Error: {str(e)}")
        return None, None, False

@st.cache_resource
def load_rf_resources():
    try:
        model = joblib.load('RF_opt_model.pkl')
        scaler = joblib.load('scaler_X.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"RF Load Error: {str(e)}")
        return None, None, False

# Load both models
ann_model, ann_scaler, ann_loaded = load_ann_resources()
rf_model, rf_scaler, rf_loaded = load_rf_resources()

# Define features, ranges, and default values
features = ["Cumulative Runtime (days)", "LP Pump Pressure (PSI)", "HP Pump Pressure (PSI)", 
            "Vibration (mm/s)", "Fluid Temp (°C)", "Pressure Cycles"]
ranges = [(1, 2000), (1000, 5000), (5000, 15000), (1, 12), (25, 35), (0, 1500)]
defaults = [500, 4000, 8000, None, 30, 400]
icons = ["⏱️", "📊", "📈", "📳", "🌡️", "🔄"]
help_texts = [
    "Total operational time since installation or last major overhaul",
    "Low Pressure pump operating pressure",
    "High Pressure pump operating pressure",
    "Automatically calculated based on other parameters",
    "Operating temperature of hydraulic fluid",
    "Number of pressure cycling events"
]

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #1e6091;
        --primary-light: #2a7cb1;
        --secondary: #f0f5f9;
        --accent: #ffb703;
        --text: #2c3e50;
        --success: #2ecc71;
        --warning: #f39c12;
        --error: #e74c3c;
    }
    .main {background-color: #f5f7fa; color: var(--text); font-family: 'Segoe UI', sans-serif;}
    .card {background-color: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); padding: 24px; margin-bottom: 20px;}
    .header {color: var(--primary); font-size: 2.5rem; font-weight: 700; text-align: center;}
    .subheader {color: var(--text); font-size: 1.2rem; opacity: 0.8; text-align: center; margin-bottom: 1.5rem;}
    .section-header {color: var(--primary); font-size: 1.3rem; font-weight: 600; margin: 1rem 0; border-left: 4px solid var(--primary); padding-left: 10px;}
    .stNumberInput > div > div > input {border-radius: 8px; border: 1px solid #e0e0e0; padding: 10px;}
    .stButton>button {background-color: var(--primary); color: white; font-weight: 600; border-radius: 8px; padding: 10px 16px; border: none; transition: all 0.3s ease; width: 100%; margin-top: 10px;}
    .stButton>button:hover {background-color: var(--primary-light); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); transform: translateY(-2px);}
    .result-box {background-color: var(--secondary); border-left: 5px solid var(--primary); padding: 20px; border-radius: 8px; text-align: center; font-size: 1.5rem; color: var(--primary); margin-top: 20px;}
    .param-label {font-weight: 600; color: var(--primary); margin-bottom: 5px;}
    .param-icon {font-size: 1.3rem; margin-right: 8px; vertical-align: middle;}
    .footer {text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9rem; color: #888;}
    .info-box {background-color: #e6f3ff; border-radius: 8px; padding: 15px; margin: 10px 0;}
    .css-1aumxhk {background-color: #f0f5f9;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image("https://www.hydraulicspneumatics.com/sites/hydraulicspneumatics.com/files/uploads/2013/04/Hydraulic-Power-Unit.jpg", use_column_width=True)
    
    st.markdown('<div class="section-header">About This Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    This application predicts the remaining time until maintenance is required for Hydraulic Pressure Units using either an Optimized Artificial Neural Network (ANN) or Random Forest (RF) model.
    """)
    
    # Model selection
    model_options = []
    if ann_loaded:
        model_options.append("Optimized ANN")
    if rf_loaded:
        model_options.append("Optimized Random Forest")
    if not model_options:
        model_options = ["Demo Mode"]
    
    selected_model = st.selectbox("Select Prediction Model", model_options, index=0)
    
    if selected_model == "Demo Mode" and (ann_loaded or rf_loaded):
        st.warning("⚠️ Using Demo Mode despite available models.")
    elif not ann_loaded and not rf_loaded:
        st.error("⚠️ No models loaded. Running in Demo Mode.")
    else:
        st.success(f"✅ Using {selected_model}")
    
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Select a prediction model
    2. Enter HPU parameters
    3. Vibration is auto-calculated
    4. Click "Predict" for results
    """)
    
    st.markdown('<div class="section-header">Quick Presets</div>', unsafe_allow_html=True)
    preset = st.selectbox("Load parameter presets", ["Custom Settings", "New Unit", "Mid-Life Unit", "Aging Unit"])
    
    if preset == "New Unit":
        presets = [100, 3500, 7000, None, 28, 150]
    elif preset == "Mid-Life Unit":
        presets = [800, 4200, 9000, None, 31, 600]
    elif preset == "Aging Unit":
        presets = [1500, 4800, 12000, None, 33, 1200]
    else:
        presets = defaults.copy()
    
    if preset != "Custom Settings":
        st.success(f"✅ {preset} preset loaded")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="header">HPU Maintenance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Intelligent maintenance timeline estimation</div>', unsafe_allow_html=True)

# Session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'inputs' not in st.session_state:
    st.session_state.inputs = defaults.copy()
if 'vibration' not in st.session_state:
    st.session_state.vibration = 0

# Input area
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">HPU Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
inputs = presets.copy()

with col1:
    for i in [0, 3]:
        if i == 3:
            continue
        st.markdown(f'<div class="param-label"><span class="param-icon">{icons[i]}</span>{features[i]}</div>', unsafe_allow_html=True)
        inputs[i] = st.number_input(
            label=f"Enter {features[i]}",
            label_visibility="collapsed",
            min_value=float(ranges[i][0]),
            max_value=float(ranges[i][1]),
            value=float(presets[i]),
            step=1.0,
            help=help_texts[i]
        )

with col2:
    for i in [1, 4]:
        st.markdown(f'<div class="param-label"><span class="param-icon">{icons[i]}</span>{features[i]}</div>', unsafe_allow_html=True)
        inputs[i] = st.number_input(
            label=f"Enter {features[i]}",
            label_visibility="collapsed",
            min_value=float(ranges[i][0]),
            max_value=float(ranges[i][1]),
            value=float(presets[i]),
            step=1.0,
            help=help_texts[i]
        )

with col3:
    for i in [2, 5]:
        st.markdown(f'<div class="param-label"><span class="param-icon">{icons[i]}</span>{features[i]}</div>', unsafe_allow_html=True)
        inputs[i] = st.number_input(
            label=f"Enter {features[i]}",
            label_visibility="collapsed",
            min_value=float(ranges[i][0]),
            max_value=float(ranges[i][1]),
            value=float(presets[i]),
            step=1.0,
            help=help_texts[i]
        )

# Compute Vibration
lp_pressure = inputs[1]
hp_pressure = inputs[2]
runtime = inputs[0]
base_vibration = 1.5 + 0.005 * runtime + 0.002 * lp_pressure/1000 + 0.001 * hp_pressure/1000
vibration = min(max(base_vibration, 1), 12)
inputs[3] = vibration
st.session_state.vibration = vibration

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f'<div class="param-label"><span class="param-icon">{icons[3]}</span>{features[3]}</div>', unsafe_allow_html=True)
    st.info(f"**{vibration:.2f} mm/s**")

with col2:
    vibration_percentage = int((vibration - 1) / 11 * 100)
    color = "#2ecc71" if vibration < 5 else "#f39c12" if vibration < 8 else "#e74c3c"
    st.markdown(f"""
    <div style="margin-top: 10px; background-color: #e0e0e0; border-radius: 10px; height: 15px;">
        <div style="width: {vibration_percentage}%; background-color: {color}; height: 15px; border-radius: 10px;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 5px;">
        <span>Normal</span>
        <span>Warning</span>
        <span>Critical</span>
    </div>
    """, unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict Maintenance Timeline", use_container_width=True):
    try:
        st.session_state.inputs = inputs.copy()
        input_array = np.array(inputs).reshape(1, -1)
        
        if selected_model == "Optimized ANN" and ann_loaded:
            input_scaled = ann_scaler.transform(input_array)
            prediction = ann_model.predict(input_scaled, verbose=0)[0][0]  # ANN: 2D output
            prediction = round(max(0, min(300, prediction)))
        elif selected_model == "Optimized Random Forest" and rf_loaded:
            input_scaled = rf_scaler.transform(input_array)
            prediction = rf_model.predict(input_scaled)[0]  # RF: 1D output
            prediction = round(max(0, min(300, prediction)))
        else:
            # Demo mode
            age_factor = runtime / 2000
            pressure_factor = (hp_pressure - 5000) / 10000
            temp_factor = (inputs[4] - 25) / 10
            cycles_factor = inputs[5] / 1500
            wear = 0.4 * age_factor + 0.3 * pressure_factor + 0.15 * temp_factor + 0.15 * cycles_factor
            prediction = round(300 * (1 - wear))
        
        st.session_state.prediction = prediction
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.session_state.prediction = None

# Display result
if st.session_state.prediction is not None:
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
    
    prediction = st.session_state.prediction
    
    if prediction > 200:
        status = "Excellent"
        color = "#2ecc71"
        emoji = "🟢"
    elif prediction > 100:
        status = "Good"
        color = "#3498db"
        emoji = "🔵"
    elif prediction > 50:
        status = "Plan Maintenance"
        color = "#f39c12"
        emoji = "🟠"
    else:
        status = "Urgent Service Required"
        color = "#e74c3c"
        emoji = "🔴"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="result-box" style="border-left-color: {color};">
            <div style="font-size: 0.9rem; color: #666;">Predicted Time to Maintenance</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {color};">{prediction} days</div>
            <div style="font-size: 1.2rem; margin-top: 10px;">{emoji} {status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        
        if prediction > 200:
            st.success("✅ No immediate action required")
            st.markdown("• Continue regular monitoring")
            st.markdown("• Perform standard fluid checks")
        elif prediction > 100:
            st.info("ℹ️ Medium-term planning")
            st.markdown("• Schedule maintenance in coming months")
            st.markdown("• Monitor vibration levels")
        elif prediction > 50:
            st.warning("⚠️ Near-term action needed")
            st.markdown("• Plan maintenance within 7 weeks")
            st.markdown("• Inspect seals and connections")
        else:
            st.error("🚨 Immediate attention required")
            st.markdown("• Schedule maintenance ASAP")
            st.markdown("• Reduce operational load")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        percentage = min(100, max(0, prediction / 3))
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.bar(x=np.pi/2, height=100, width=2*np.pi, bottom=0, color='#e0e0e0', alpha=0.5)
        ax.bar(x=np.pi/2, height=100, width=2*np.pi * percentage/100, bottom=0, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.text(0, 0, f"{prediction}\ndays", ha='center', va='center', fontsize=20, fontweight='bold')
        ax.set_ylim(0, 120)
        st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(f"Developed by Kelvin Hayford | Powered by {selected_model} | v2.0", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)