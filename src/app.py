import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from utils import (
    process_submission, predict_with_model, generate_chatbot_response
)

# Page Config
st.set_page_config(
    page_title="Insurance Fraud AI Analyst",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths
MODEL_DIR = "models/"
MODELS = {
    "XGBoost (Best Performance)": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl"
}

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'home' # home, input, result

# Respect URL query param for quick navigation (e.g. ?page=input)
try:
    params = st.experimental_get_query_params()
    if 'page' in params and params['page']:
        requested = params['page'][0]
        if requested in ('home', 'input', 'result'):
            st.session_state.page = requested
except Exception:
    pass

def set_page(page_name):
    st.session_state.page = page_name

def reset_app():
    st.session_state.page = 'input'
    st.session_state.analysis_done = False
    st.session_state.show_chat = False
    if "messages" in st.session_state:
        del st.session_state.messages

def go_to_input():
    set_page('input')
    st.session_state.analysis_done = False
    # update URL so back/forward and deep links work
    try:
        st.experimental_set_query_params(page='input')
    except Exception:
        pass
    st.rerun()

# Load Assets (Cached)
@st.cache_resource
def load_assets():
    assets = {}
    
    # Load Models
    for name, filename in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                assets[name] = joblib.load(path)
            except Exception as e:
                st.error(f"Error loading model {name}: {e}")
                return None
        else:
            st.error(f"Model file not found: {path}")
            return None
            
    # Load Encoders
    encoder_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    if os.path.exists(encoder_path):
        try:
            assets["encoders"] = joblib.load(encoder_path)
        except Exception as e:
             st.error(f"Error loading encoders: {e}")
             return None
    else:
        st.error(f"Encoders not found at {encoder_path}. Please re-run training.")
        return None
        
    return assets

assets = load_assets()

# Default model selection
selected_model_name = "XGBoost (Best Performance)"

if assets:
    model = assets[selected_model_name]
    encoders = assets.get("encoders")

# Chat dialog handled inline on result page (no floating dialog)

# --- Pages ---

def render_home_page():
    # Eye-catching professional home/landing page
    st.markdown("""
    <div style="background: linear-gradient(90deg,#0b3d91, #00aaff); padding:40px; border-radius:12px; color: white; box-shadow: 0 6px 30px rgba(0,0,0,0.15);">
        <h1 style="font-size:42px; margin:0; letter-spacing:1px;">Insurance fraud detector</h1>
        <p style="font-size:16px; opacity:0.95; margin-top:6px;">Advanced AI-driven claim assessment — fast, professional, and audit-ready.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="background: #ffffff; padding:24px; border-radius:10px; box-shadow: 0 6px 18px rgba(12,40,80,0.06);">
            <h2 style="color:#0b3d91; margin-top:0;">Get started — Assess a claim</h2>
            <p style="color:#333;">Upload claim details and receive a clear, categorical fraud risk assessment and a professionally formatted report for investigators.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    feat1, feat2, feat3 = st.columns(3)
    feat1.metric("Speed", "Analysis in seconds")
    feat2.metric("Professional", "Audit-friendly reports")
    feat3.metric("Insightful", "Model + heuristic drivers")

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA button styling (highlighted primary action, refined visual)
    st.markdown(
    """
    <style>
    /* Center wrapper */
    .home-cta-wrapper {
        display: flex;
        justify-content: center;
    }

    /* Target Streamlit button properly */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #0077DA 0%, #00AEEF 100%) !important;
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 16px 44px !important;
        border-radius: 14px !important;
        border: 0.5px solid rgba(255,255,255,0.12) !important;
        box-shadow: 0 18px 40px rgba(0,120,220,0.18) !important;
        transition: transform 0.16s cubic-bezier(.2,.9,.3,1), box-shadow 0.16s !important;
        display: inline-block !important;
        margin: 12px auto !important;
        min-width: 280px !important;
        max-width: 68% !important;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 28px 60px rgba(0,120,220,0.24) !important;
    }

    div[data-testid="stButton"] > button:focus {
        outline: 3px solid rgba(0,170,255,0.18) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

    # Centered Streamlit CTA button (calls go_to_input)
    st.markdown('<div class="home-cta-wrapper"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 0.6, 1])
    with c2:
        if st.button("Get Started", key="home_get_started"):
            # set session state and URL, then rerun (not using callback rerun)
            st.session_state.page = 'input'
            st.session_state.analysis_done = False
            try:
                st.experimental_set_query_params(page='input')
            except Exception:
                pass
            # experimental_rerun may not exist in some Streamlit versions; stop execution
            # The page state is set to 'input' so the next run will render the form.
            st.stop()


def render_input_page():
    st.title("New Claim Assessment")
    st.markdown("Enter claim details below to generate a fraud risk analysis.")
    
    with st.form("claim_form"):
        # Section 1: Policy Info
        st.subheader("1. Policy Information")
        c1, c2, c3 = st.columns(3)
        months_as_customer = c1.number_input("Months as Customer", min_value=0, value=12)
        age = c2.number_input("Insured Age", min_value=18, value=35)
        policy_state = c3.selectbox("Policy State", ["OH", "IL", "IN"])
        
        c4, c5 = st.columns(2)
        policy_bind_date = c4.date_input("Policy Bind Date", value=datetime(2020, 1, 1))
        policy_annual_premium = c5.number_input("Annual Premium ($)", value=1000.0)
        policy_deductable = c5.number_input("Deductible ($)", value=1000)

        st.markdown("---")

        # Section 2: Incident Details
        st.subheader("2. Incident Details")
        i1, i2 = st.columns(2)
        incident_date = i1.date_input("Incident Date", value=datetime(2021, 1, 1))
        incident_type = i2.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"])
        
        i3, i4, i5 = st.columns(3)
        collision_type = i3.selectbox("Collision Type", ["Side Collision", "Rear Collision", "Front Collision", "?"])
        incident_severity = i4.selectbox("Severity", ["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"])
        authorities_contacted = i5.selectbox("Authorities", ["Police", "Fire", "Ambulance", "Other", "None"])
        
        i6, i7 = st.columns(2)
        state = i6.selectbox("Incident State", ["NY", "SC", "WV", "VA", "NC", "PA", "OH"])
        city = i7.text_input("Incident City", value="Springfield")
        
        st.markdown("---")

        # Section 3: Financials & Asset
        st.subheader("3. Financials & Asset")
        f1, f2, f3 = st.columns(3)
        total_claim_amount = f1.number_input("Total Claim Amount ($)", value=50000)
        injury_claim = f2.number_input("Injury Claim ($)", value=5000)
        property_claim = f3.number_input("Property Claim ($)", value=5000)
        vehicle_claim = st.number_input("Vehicle Claim ($)", value=40000)
        
        a1, a2, a3 = st.columns(3)
        auto_make = a1.text_input("Auto Make", "Saab")
        auto_model = a2.text_input("Auto Model", "92x")
        auto_year = a3.number_input("Auto Year", 1990, 2024, 2010)
        
        st.markdown("---")
        
        # Section 4: Other Indicators
        st.subheader("4. Risk Indicators")
        o1, o2, o3 = st.columns(3)
        witnesses = o1.number_input("Witnesses", 0, 10, 0)
        police_report = o2.selectbox("Police Report Available?", ["YES", "NO", "?"])
        property_damage = o3.selectbox("Property Damage?", ["YES", "NO", "?"])

        # Advanced / Hidden Features (Expandable)
        with st.expander("Advanced Risk Factors (Optional)"):
            a1, a2 = st.columns(2)
            umbrella_limit = a1.number_input("Umbrella Limit", step=1000000, value=0)
            capital_gains = a2.number_input("Capital Gains", step=1000, value=0)
            capital_loss = a1.number_input("Capital Loss", step=1000, value=0)
            incident_hour = a2.slider("Incident Hour (24h)", 0, 23, 12)
            
            num_vehicles = 1
            if incident_type == "Multi-vehicle Collision":
                num_vehicles = st.slider("Vehicles Involved", 2, 4, 3)
            elif incident_type == "Single Vehicle Collision":
                 num_vehicles = 1
            else:
                 num_vehicles = 1
            
            bodily_injuries = st.number_input("Bodily Injuries", 0, 2, 1)

        submitted = st.form_submit_button("Analyze Claim", type="primary")
        
    if submitted and assets:
        # Use imported process_submission logic
        process_submission(
             months_as_customer, age, policy_bind_date, policy_state, policy_deductable, policy_annual_premium,
             incident_date, incident_type, collision_type, incident_severity, authorities_contacted, state, city,
             total_claim_amount, injury_claim, property_claim, vehicle_claim,
             auto_make, auto_model, auto_year,
             witnesses, police_report, property_damage,
             'MALE', 'MD', 'sales', # Default hidden fields
             umbrella_limit, capital_gains, capital_loss, incident_hour, num_vehicles, bodily_injuries,
             assets['encoders']
        )

def render_result_page():
    st.title("Risk Assessment Results")
    
    # New Assessment button
    if st.button("New Assessment", on_click=reset_app):
        pass
    
    st.markdown("---")
    
    # Retrieve preprocessed data from session state
    df_model_input = st.session_state.get('df_model_input')
    df_input = st.session_state.get('df_input')
    
    # Dynamically predict with currently selected model (using imported function)
    if df_model_input is not None and model is not None:
        probability = predict_with_model(df_model_input, model)
    else:
        probability = 0.0
        
    # Store probability for global chat context
    st.session_state['probability'] = probability
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Key Drivers (calculate first for risk assessment)
        drivers = []
        # Re-calc heuristics for display
        if df_input is not None:
            if df_input['days_since_policy_bind'].iloc[0] < 30:
                drivers.append(f"Recent Policy (Bind < 30 days)")
            if st.session_state.get('police_report') == "NO" and st.session_state.get('total_claim_amount') > 20000:
                drivers.append("High Value Claim without Police Report")
            if st.session_state.get('incident_severity') == "Major Damage" and st.session_state.get('witnesses') == 0:
                drivers.append("Major Incident with No Witnesses")
            if st.session_state.get('incident_type') == "Single Vehicle Collision":
                 drivers.append("Single Vehicle Incident Category")
        
        # Store drivers for global chat context
        st.session_state['drivers'] = drivers
        
        # Determine risk level based on both model probability and heuristic drivers
        num_drivers = len(drivers)
        if probability > 0.7 or num_drivers >= 3:
            risk_color = "red"
            risk_label = "HIGH RISK"
            risk_description = "High"
        elif probability > 0.3 or num_drivers >= 2:
            risk_color = "orange" 
            risk_label = "MODERATE RISK"
            risk_description = "Medium"
        elif probability > 0.1 or num_drivers >= 1:
            risk_color = "#FFD700"  # Gold/yellow
            risk_label = "LOW-MODERATE RISK"
            risk_description = "Low-Medium"
        else:
            risk_color = "green"
            risk_label = "LOW RISK"
            risk_description = "Low"
        
        # Score Card
        st.markdown(f"""
            <div style="text-align: center; padding: 30px; border-radius: 15px; background-color: #f0f2f6; border: 2px solid {risk_color}; margin-bottom: 20px;">
                <h4 style="color: #555; margin: 0;">FRAUD RISK ASSESSMENT</h4>
                <h1 style="color: {risk_color}; font-size: 60px; margin: 10px 0; font-weight: 800;">{risk_description}</h1>
                <h2 style="color: {risk_color}; margin: 0; font-size: 24px;">{risk_label}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Key Drivers
        st.subheader("Key Risk Drivers")
        
        if drivers:
            for d in drivers:
                st.warning(f"⚠️ {d}")
        else:
            st.success("No standard heuristic red flags detected.")

        # PDF Generation
        st.markdown("---")
        st.subheader("Official Report")
        if st.button("Generate & Download PDF"):
            from pdf_gen import generate_pdf_report
            with st.spinner("Generating Report..."):
                pdf_path = generate_pdf_report(df_input, probability, risk_label, drivers)
            
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path))

    with col2:
        # Chatbot Section
        st.subheader("🤖 AI Assistant")
        st.markdown("Ask me questions about this claim analysis.")
        
        # Initialize chat history if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I've analyzed the claim. What would you like to know?"}]

        # Display chat messages
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get context
            df_input = st.session_state.get('df_input')
            probability = st.session_state.get('probability', 0.0)
            drivers = st.session_state.get('drivers', [])
            
            # Generate response
            response = generate_chatbot_response(prompt, df_input, probability, drivers)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# --- Main App Logic ---

def main():
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'input':
        render_input_page()
    elif st.session_state.page == 'result':
        render_result_page()

if __name__ == "__main__":
    main()

