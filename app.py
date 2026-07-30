import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from src.recommendations import get_recommendations
from src.train import ClassifierWrapper

# Page Configuration
st.set_page_config(
    page_title="Post-Harvest Food Loss Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .card-low {
        background: rgba(46, 204, 113, 0.15);
        border: 2px solid rgba(46, 204, 113, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #2ecc71;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(46, 204, 113, 0.05);
        text-align: center;
    }
    
    .card-med {
        background: rgba(241, 196, 15, 0.15);
        border: 2px solid rgba(241, 196, 15, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #f1c40f;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(241, 196, 15, 0.05);
        text-align: center;
    }
    
    .card-high {
        background: rgba(231, 76, 60, 0.15);
        border: 2px solid rgba(231, 76, 60, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #e74c3c;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(231, 76, 60, 0.05);
        text-align: center;
    }
    
    .section-title {
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 5px;
    }
    
    .rec-item {
        background-color: #f8f9fa;
        border-left: 4px solid #11998e;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Load Pipeline and Metadata
@st.cache_resource
def load_assets():
    pipeline_path = "models/food_loss_pipeline.joblib"
    metadata_path = "models/metadata.joblib"
    
    if not os.path.exists(pipeline_path) or not os.path.exists(metadata_path):
        return None, None
        
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(metadata_path)
    return pipeline, metadata

pipeline, metadata = load_assets()

if pipeline is None or metadata is None:
    st.error("Critical error: Model asset files are missing. Please ensure models/food_loss_pipeline.joblib and models/metadata.joblib exist.")
    st.stop()


# Header
st.markdown("<div class='main-title'>🌾 Post-Harvest Food Loss Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time agricultural supply chain risk prediction and mitigation system</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Risk Calculator", "🚀 Logistics Simulator", "📊 Insights Dashboard"])

# SLIDER LIMITS to ensure session state and slider boundaries are always synchronized and safe
SLIDER_LIMITS = {
    "Storage_Temperature": (-10.0, 45.0),
    "Storage_Humidity": (10.0, 100.0),
    "Warehouse_Storage_Time": (0.0, 45.0),
    "Route_Distance": (5.0, 2000.0),
    "Delivery_Time": (0.5, 120.0),
    "Queue_Time": (0.0, 24.0),
    "Vibration_Level": (0.0, 800.0),
    "Weather_Impact": (0.0, 10.0),
    "Traffic_Level": (0.0, 10.0),
    "Fuel_Consumption": (0.0, 500.0),
    "Fuel_Costs": (0.0, 3000.0),
    "Temperature": (-15.0, 50.0),
    "Humidity": (10.0, 100.0),
    "IoT_Sensor_Reading_Temperature": (-10.0, 45.0),
    "IoT_Sensor_Reading_Humidity": (10.0, 100.0),
    "IoT_Sensor_Reading_Light": (0.0, 100.0),
    "Quality_Maintenance_Ratio": (0.0, 100.0),
}

# Reusable helper function to safely initialize and clamp numeric inputs in session state
def safe_number_input(label, min_value, max_value, default_value, key, step=None):
    if key not in st.session_state:
        st.session_state[key] = default_value
    else:
        val = st.session_state[key]
        try:
            val_float = float(val)
            clamped_val = max(min_value, min(max_value, val_float))
            # Preserve type of default_value (e.g., int vs float)
            if isinstance(default_value, int):
                st.session_state[key] = int(round(clamped_val))
            else:
                st.session_state[key] = float(clamped_val)
        except (ValueError, TypeError):
            st.session_state[key] = default_value
            
    return st.number_input(label, min_value=min_value, max_value=max_value, step=step, key=key)

# Default values setup from metadata training medians (safely clamped)
defaults = {}
for col in metadata["numeric_cols"]:
    val = float(metadata["medians"][col])
    if col in SLIDER_LIMITS:
        min_v, max_v = SLIDER_LIMITS[col]
        defaults[col] = max(min_v, min(max_v, val))
    else:
        defaults[col] = val

# Sidebar / Scenario Selector
with st.sidebar:
    if os.path.exists("reports/feature_importance.png"):
        st.image("reports/feature_importance.png", caption="Top Predictor Features", width="stretch")
    else:
        st.warning("Chart file not found: reports/feature_importance.png")
    st.markdown("### Model Properties")
    st.info(f"**Model**: {metadata['best_model_name']}\n\n**Training 33rd Percentile**: {metadata['thresholds']['33rd_percentile']:.4f}\n\n**Training 66th Percentile**: {metadata['thresholds']['66th_percentile']:.4f}")

# Logistics Simulator (Demo Cases) Tab
with tab2:
    st.markdown("<div class='section-title'>Select a Pre-configured Shipment Scenario</div>", unsafe_allow_html=True)
    
    demo_cases = {
        "Optimal Cold Chain Logistics (Low Risk)": {
            "Crop_Type": "Wheat", "Vehicle_Type": "Truck", "Storage_Temperature": 12.0, "Storage_Humidity": 55.0,
            "Warehouse_Storage_Time": 2.0, "Delivery_Time": 6.0, "Route_Distance": 120.0, "Queue_Time": 1.0,
            "Weather_Impact": 1.5, "Vibration_Level": 35.0, "Fuel_Costs": 120.0, "Quality_Maintenance_Ratio": 95.0,
            "Fuel_Consumption": 15.0, "Traffic_Level": 1.5, "Temperature": 18.0, "Humidity": 50.0,
            "IoT_Sensor_Reading_Temperature": 13.0, "IoT_Sensor_Reading_Humidity": 56.0, "IoT_Sensor_Reading_Light": 5.0
        },
        "High-Temperature Delivery Delay (Medium Risk)": {
            "Crop_Type": "Corn", "Vehicle_Type": "Van", "Storage_Temperature": 22.0, "Storage_Humidity": 68.0,
            "Warehouse_Storage_Time": 8.0, "Delivery_Time": 28.0, "Route_Distance": 650.0, "Queue_Time": 5.5,
            "Weather_Impact": 4.8, "Vibration_Level": 180.0, "Fuel_Costs": 580.0, "Quality_Maintenance_Ratio": 72.0,
            "Fuel_Consumption": 65.0, "Traffic_Level": 3.8, "Temperature": 26.0, "Humidity": 65.0,
            "IoT_Sensor_Reading_Temperature": 23.0, "IoT_Sensor_Reading_Humidity": 69.0, "IoT_Sensor_Reading_Light": 12.0
        },
        "Failing Environment & Extended Storage (High Risk)": {
            "Crop_Type": "Rice", "Vehicle_Type": "Motorbike", "Storage_Temperature": 32.0, "Storage_Humidity": 85.0,
            "Warehouse_Storage_Time": 24.0, "Delivery_Time": 48.0, "Route_Distance": 950.0, "Queue_Time": 8.0,
            "Weather_Impact": 7.5, "Vibration_Level": 320.0, "Fuel_Costs": 980.0, "Quality_Maintenance_Ratio": 38.0,
            "Fuel_Consumption": 110.0, "Traffic_Level": 8.2, "Temperature": 34.0, "Humidity": 82.0,
            "IoT_Sensor_Reading_Temperature": 33.0, "IoT_Sensor_Reading_Humidity": 86.0, "IoT_Sensor_Reading_Light": 18.0
        }
    }
    
    selected_scenario = st.selectbox(
        "Choose a demo scenario to load into the Calculator:",
        list(demo_cases.keys()),
        key="scenario_select"
    )
    
    if st.button("Load Selected Scenario into Calculator", key="btn_load"):
        sanitized_scenario = {}
        for k, v in demo_cases[selected_scenario].items():
            if k in SLIDER_LIMITS:
                min_v, max_v = SLIDER_LIMITS[k]
                try:
                    v_float = float(v)
                    sanitized_scenario[k] = max(min_v, min(max_v, v_float))
                except (ValueError, TypeError):
                    sanitized_scenario[k] = defaults.get(k, min_v)
            else:
                sanitized_scenario[k] = v
        st.session_state.update(sanitized_scenario)
        st.success(f"Successfully loaded: '{selected_scenario}'. Open the 'Risk Calculator' tab to view and run prediction!")

# Risk Calculator Tab
with tab1:
    st.markdown("<div class='section-title'>Enter Logistics & Environmental Parameters</div>", unsafe_allow_html=True)
    
    # We initialize session state for all parameters if not already present
    # and ensure existing session state values are clamped within valid slider ranges.
    for key, val in demo_cases["Optimal Cold Chain Logistics (Low Risk)"].items():
        if key not in st.session_state:
            if key in ["Crop_Type", "Vehicle_Type"]:
                st.session_state[key] = val
            else:
                st.session_state[key] = defaults.get(key, val)
        else:
            # If already present, clamp to valid ranges
            if key in SLIDER_LIMITS:
                min_v, max_v = SLIDER_LIMITS[key]
                try:
                    curr_val = float(st.session_state[key])
                    st.session_state[key] = max(min_v, min(max_v, curr_val))
                except (ValueError, TypeError):
                    st.session_state[key] = defaults.get(key, min_v)
                
    # Input layouts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🌾 Crop & Vehicle Profile")
        crop_type = st.selectbox("Crop Type", sorted(metadata['one_hot_categories']['Crop_Type']), key="Crop_Type")
        vehicle_type = st.selectbox("Transportation Vehicle Type", sorted(metadata['one_hot_categories']['Vehicle_Type']), key="Vehicle_Type")
        
        st.markdown("##### 🌡️ Storage Chamber Conditions")
        storage_temp = safe_number_input("Storage Temperature (°C)", -10.0, 45.0, defaults.get("Storage_Temperature", 7.5), "Storage_Temperature", step=0.5)
        storage_humidity = safe_number_input("Storage Humidity (%)", 10.0, 100.0, defaults.get("Storage_Humidity", 60.0), "Storage_Humidity", step=1.0)
        warehouse_time = safe_number_input("Warehouse Storage Time (days)", 0.0, 45.0, defaults.get("Warehouse_Storage_Time", 3.0), "Warehouse_Storage_Time", step=0.5)
        
        st.markdown("##### 🚚 Journey Details")
        route_dist = safe_number_input("Route Distance (km)", 5.0, 2000.0, defaults.get("Route_Distance", 100.0), "Route_Distance", step=10.0)
        delivery_time = safe_number_input("Delivery Time (hours)", 0.5, 120.0, defaults.get("Delivery_Time", 6.0), "Delivery_Time", step=0.5)
        
    with col2:
        st.markdown("##### 🛣️ Route & Logistics Factors")
        queue_time = safe_number_input("Terminal Queue/Unloading Time (hours)", 0.0, 24.0, defaults.get("Queue_Time", 1.0), "Queue_Time", step=0.5)
        vibration_level = safe_number_input("Vibration Level (m/s²)", 0.0, 800.0, defaults.get("Vibration_Level", 40.0), "Vibration_Level", step=5.0)
        weather_impact = safe_number_input("Weather Impact Severity", 0.0, 10.0, defaults.get("Weather_Impact", 2.0), "Weather_Impact", step=0.1)
        traffic_level = safe_number_input("Traffic Congestion Level", 0.0, 10.0, defaults.get("Traffic_Level", 2.0), "Traffic_Level", step=0.1)
        
        st.markdown("##### 🔋 Energy & Fuel Profile")
        fuel_consumption = safe_number_input("Fuel Consumption (Liters)", 0.0, 500.0, defaults.get("Fuel_Consumption", 20.0), "Fuel_Consumption", step=1.0)
        fuel_costs = safe_number_input("Fuel Costs ($)", 0.0, 3000.0, defaults.get("Fuel_Costs", 150.0), "Fuel_Costs", step=10.0)
        
    with col3:
        st.markdown("##### 🌦️ Ambient Weather Conditions")
        ambient_temp = safe_number_input("Ambient Temperature (°C)", -15.0, 50.0, defaults.get("Temperature", 25.0), "Temperature", step=0.5)
        ambient_humidity = safe_number_input("Ambient Humidity (%)", 10.0, 100.0, defaults.get("Humidity", 55.0), "Humidity", step=1.0)
        
        st.markdown("##### 📡 IoT Cargo Sensor Readings")
        iot_temp = safe_number_input("IoT Cargo Temp (°C)", -10.0, 45.0, defaults.get("IoT_Sensor_Reading_Temperature", 8.0), "IoT_Sensor_Reading_Temperature", step=0.5)
        iot_humidity = safe_number_input("IoT Cargo Humidity (%)", 10.0, 100.0, defaults.get("IoT_Sensor_Reading_Humidity", 60.0), "IoT_Sensor_Reading_Humidity", step=1.0)
        iot_light = safe_number_input("IoT Cargo Light Level (Lux)", 0.0, 100.0, defaults.get("IoT_Sensor_Reading_Light", 5.0), "IoT_Sensor_Reading_Light", step=1.0)
        
        st.markdown("##### 📋 Quality Metrics")
        quality_ratio = safe_number_input("Quality Maintenance Ratio (%)", 0.0, 100.0, defaults.get("Quality_Maintenance_Ratio", 90.0), "Quality_Maintenance_Ratio", step=1.0)

    # Run Prediction
    user_inputs = {
        'Crop_Type': crop_type,
        'Vehicle_Type': vehicle_type,
        'Storage_Temperature': storage_temp,
        'Storage_Humidity': storage_humidity,
        'Fuel_Consumption': fuel_consumption,
        'Route_Distance': route_dist,
        'Delivery_Time': delivery_time,
        'Traffic_Level': traffic_level,
        'Temperature': ambient_temp,
        'Humidity': ambient_humidity,
        'Vibration_Level': vibration_level,
        'Queue_Time': queue_time,
        'Weather_Impact': weather_impact,
        'IoT_Sensor_Reading_Temperature': iot_temp,
        'IoT_Sensor_Reading_Humidity': iot_humidity,
        'IoT_Sensor_Reading_Light': iot_light,
        'Warehouse_Storage_Time': warehouse_time,
        'Fuel_Costs': fuel_costs,
        'Quality_Maintenance_Ratio': quality_ratio
    }
    
    input_df = pd.DataFrame([user_inputs])
    
    # Calculate predicted class
    pred_class = pipeline.predict(input_df)[0]
    
    # Compute pseudo-probabilities using decision_function since Ridge doesn't support predict_proba
    classifier = pipeline.named_steps['classifier']
    classes = classifier.classes_
    decision = pipeline.decision_function(input_df)[0]
    
    # Softmax conversion
    exp_dec = np.exp(decision - np.max(decision))
    probs = exp_dec / np.sum(exp_dec)
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    # Display Risk Results
    st.markdown("<div class='section-title'>Risk Diagnostics & Mitigation Center</div>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown("##### Predicted Spoilage Risk Category")
        if pred_class == "Low Risk":
            st.markdown(f"<div class='card-low'>🟢 LOW RISK ({prob_dict['Low Risk']*100:.1f}%)</div>", unsafe_allow_html=True)
        elif pred_class == "Medium Risk":
            st.markdown(f"<div class='card-med'>🟡 MEDIUM RISK ({prob_dict['Medium Risk']*100:.1f}%)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card-high'>🔴 HIGH RISK ({prob_dict['High Risk']*100:.1f}%)</div>", unsafe_allow_html=True)
            
        # Class probabilities bars
        st.markdown("##### Risk Class Confidence")
        for cls in ['Low Risk', 'Medium Risk', 'High Risk']:
            st.write(f"**{cls}**")
            st.progress(prob_dict[cls])
            
    with res_col2:
        st.markdown("##### Actionable Mitigation Recommendations")
        rec_data = get_recommendations(user_inputs, pred_class, prob_dict)
        
        # Display Summary
        if pred_class != "Low Risk":
            st.warning(rec_data['risk_summary'])
        else:
            st.success(rec_data['risk_summary'])
        
        # Display List
        for rec in rec_data['recommendations'][1:]: # skip summary
            st.markdown(f"<div class='rec-item'>👉 {rec}</div>", unsafe_allow_html=True)

# Insights & Performance Dashboard Tab
with tab3:
    st.markdown("<div class='section-title'>Scientific Research Findings & Model Performance</div>", unsafe_allow_html=True)
    
    dash_col1, dash_col2 = st.columns([1, 1])
    
    with dash_col1:
        st.markdown("### 🧬 Preprocessing & Reconstruction")
        st.markdown(r"""
        * **Exponential Scaling Recovery**: The raw Kaggle dataset features contained values up to $10^{308}$, causing numerical overflow during computation. Our preprocessor resolves this by applying a natural log transform ($x = \ln(y)$) to recover physical coordinates (such as Storage Temperature in the $5\text{--}18^\circ\text{C}$ range).
        * **Statistical Independence (Null Result)**: Correlation analysis shows that `Spoilage_Risk` in the source dataset is mathematically independent of all individual input features (Pearson and Spearman correlations are all $< 0.01$). 
        * **Prediction Performance**: As a result of this independence, the classifiers achieve test performance matching the dummy baseline (~33.3% accuracy). This transparent "null result" represents the true data characteristics and is documented for scientific integrity.
        """)
        
        # Load Metrics
        st.markdown("### 📊 Test Set Metrics")
        if os.path.exists("reports/metrics.json"):
            with open("reports/metrics.json") as f:
                metrics = json.load(f)
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Score"}), width="stretch")
        else:
            st.warning("Metrics file not found: reports/metrics.json")
            
        # Load Comparison
        st.markdown("### ⚖️ Classifier Model Comparison")
        if os.path.exists("reports/model_comparison.csv"):
            comp_df = pd.read_csv("reports/model_comparison.csv")
            st.dataframe(comp_df, width="stretch")
        else:
            st.warning("Model comparison file not found: reports/model_comparison.csv")
            
    with dash_col2:
        # Render Heatmap and Feature Importance
        if os.path.exists("reports/feature_importance.png"):
            st.image("reports/feature_importance.png", caption="Feature Importance (Ridge Coefficients)", width="stretch")
        else:
            st.warning("Chart file not found: reports/feature_importance.png")
            
        if os.path.exists("reports/correlation_heatmap.png"):
            st.image("reports/correlation_heatmap.png", caption="Feature Correlation Heatmap", width="stretch")
        else:
            st.warning("Chart file not found: reports/correlation_heatmap.png")
            
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", caption="Model Confusion Matrix", width="stretch")
        else:
            st.warning("Chart file not found: reports/confusion_matrix.png")
