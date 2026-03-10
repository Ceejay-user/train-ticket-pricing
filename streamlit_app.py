import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
import os

# --- CONFIGURATION ---
# Use an environment variable for the API URL, default to the container name in your docker-compose
# API_URL = os.getenv("API_URL", "http://api:8000/predict") 
API_URL = "http://localhost:8000/predict"




@st.cache_data(show_spinner="Retrieving SHAP insights from MLflow...")
def get_cached_artifact(run_id, artifact_path):
    """Downloads the artifact once and caches the local path."""
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

def display_mlflow_artifact(run_id, artifact_path):
    """Renders the cached image in the UI."""
    try:
        local_path = get_cached_artifact(run_id, artifact_path)
        img = Image.open(local_path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load {artifact_path}: {e}")



st.set_page_config(page_title="Revenue Management Demo", layout="wide")
# --- UI STYLING ---
st.markdown("""
    <style>
    .stMetric { border: 1px solid #e1e4e8; padding: 15px; border-radius: 8px; background: #ffffff; }
    h1 { color: #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Input Parameters")

def get_inputs():
    # Constructing the exact JSON structure provided
    data = {
        "Route_ID": st.sidebar.text_input("Route ID", "R0001"),
        "Travel_Date": "2026-03-10T11:32:55.101Z",
        "Booking_Date": "2026-03-10T11:32:55.101Z",
        "Booking_Timestamp": "2026-03-10T11:32:55.101Z",
        "Seat_Class": st.sidebar.selectbox("Seat Class", ["Standard", "Flex", "First"]),
        "Booking_Channel": st.sidebar.selectbox("Booking Channel", ["Web", "Mobile", "Phone", "Agent"]),
        "Origin": st.sidebar.text_input("Origin", "Aberdeen"),
        "Destination": st.sidebar.text_input("Destination", "Leeds"),
        "Distance_km": st.sidebar.slider("Distance (km)", 0, 1000, 220),
        "Route_Category": st.sidebar.selectbox("Route Category", ["Short", "Medium", "Long"]),
        "Customer_Segment": st.sidebar.radio("Segment", ["Leisure", "Business", "Commuter", "Group"], horizontal=True),
        "Loyalty_Status": st.sidebar.selectbox("Loyalty", ["Gold", "Silver", "Bronze", "Not loyal"]),
        "Booking_Frequency_Qtr": st.sidebar.number_input("Freq/Qtr", 0, 10, 3),
        "Average_Spend_GBP": st.sidebar.number_input("Avg Spend (£)", 0.0, 5000.0, 45.0),
        "Total_Seats": 200,
        "Seats_Sold_Realized": st.sidebar.number_input("Seats Sold", 0, 200, 120),
        "Remaining_Seats_Realized": 80,
        "Demand_Index": st.sidebar.slider("Demand Index", 0.0, 1.0, 0.65),
        "Base_Price_At_Booking": st.sidebar.number_input("Base Price", 0.0, 2000.0, 10.0),
        "Days_Before_Travel": st.sidebar.number_input("Days Before", 0, 365, 30),
        "Price_Premium": 0.1,
        "Load_Factor": 0.6
    }
    return data

payload = get_inputs()

# --- MAIN DASHBOARD ---
st.title("🚀 Price Optimization Engine")
st.markdown("Dynamic pricing predictions powered by your live MLflow model.")

if st.sidebar.button("Run Prediction", type="primary"):
    try:
        # POST to your API container
        response = requests.post(API_URL, json=payload)
                    
        response.raise_for_status()
        result = response.json()
        price = result.get('ticket_price_gbp', 0)
        current_run_id = result.get('run_id')
        
        # Display Prediction Results
        display_price = round(price, 2)

        # Render the Hero Card
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
                padding: 40px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
                margin: 20px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h3 style="
                    color: #94a3b8; 
                    text-transform: uppercase; 
                    letter-spacing: 2px; 
                    font-size: 14px; 
                    margin-bottom: 10px;
                    font-weight: 600;
                ">
                    Predicted Optimal Fare
                </h3>
                <h1 style="
                    color: #ffffff; 
                    font-size: 72px; 
                    font-weight: 800; 
                    margin: 0;
                    line-height: 1;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                ">
                    £{display_price:,.2f}
                </h1>
                <div style="
                    margin-top: 20px;
                    display: inline-block;
                    background: rgba(16, 185, 129, 0.2);
                    color: #34d399;
                    padding: 6px 16px;
                    border-radius: 50px;
                    font-size: 14px;
                    font-weight: 600;
                    border: 1px solid rgba(16, 185, 129, 0.3);
                ">
                    ● AI Model Inference Successful
                </div>
            </div>
            """, unsafe_allow_html=True)


        waterfall_path = "plots/shap_waterfall_row0.png"
        summary_path = "plots/shap_summary.png"

        st.divider()

        # SHAP VISUALIZATIONS
        st.header("🔍 Interpretability Insights")
        col_wf, col_sum = st.columns(2)
        with col_wf:
            st.subheader("Summary: Global Importance")
            display_mlflow_artifact(current_run_id, summary_path)
            
        with col_sum:
            st.subheader("Waterfall: Local Impact")
            display_mlflow_artifact(current_run_id, waterfall_path)

    except Exception as e:
        st.error(f"Error connecting to API at {API_URL}: {e}")

# Payload View for Technical Demo
with st.expander("View Request JSON"):
    st.json(payload)
