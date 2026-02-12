import streamlit as st
import pandas as pd
import numpy as np

# --- Floating Chatbot Button CSS ---
FLOATING_CSS = """
<style>
@keyframes bot-pulse {
    0% { 
        box-shadow: 0 15px 50px rgba(0, 153, 255, 0.8), 0 0 30px rgba(0, 217, 255, 0.6);
        transform: scale(1);
    }
    50% { 
        box-shadow: 0 20px 60px rgba(0, 217, 255, 1), 0 0 50px rgba(0, 217, 255, 0.9);
        transform: scale(1.05);
    }
    100% { 
        box-shadow: 0 15px 50px rgba(0, 153, 255, 0.8), 0 0 30px rgba(0, 217, 255, 0.6);
        transform: scale(1);
    }
}

/* Target the button container */
div[data-testid="column"]:has(button[kind="primary"][key="floating_chat_btn"]) {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    z-index: 999999 !important;
    width: auto !important;
}

/* Style the actual button */
button[key="floating_chat_btn"] {
    width: 85px !important;
    height: 85px !important;
    border-radius: 28px !important;
    background: linear-gradient(135deg, #00D9FF 0%, #0099FF 50%, #0066FF 100%) !important;
    color: white !important;
    font-size: 42px !important;
    border: 4px solid rgba(255, 255, 255, 1) !important;
    box-shadow: 0 15px 50px rgba(0, 153, 255, 0.8), 0 0 30px rgba(0, 217, 255, 0.6) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    padding: 0 !important;
    animation: bot-pulse 2.5s infinite ease-in-out !important;
}

button[key="floating_chat_btn"]:hover {
    transform: scale(1.15) rotate(8deg) !important;
    box-shadow: 0 20px 60px rgba(0, 153, 255, 1), 0 0 40px rgba(0, 217, 255, 0.9) !important;
    background: linear-gradient(135deg, #00EAFF 0%, #00AAFF 50%, #0077FF 100%) !important;
}
</style>
"""

FLOATING_HTML = ""

# Field Mappings for Chatbot Lookup
FIELD_MAPPINGS = {
    'premium': 'policy_annual_premium',
    'deductible': 'policy_deductable',
    'policy state': 'policy_state',
    'months as customer': 'months_as_customer',
    'age': 'age',
    'incident date': 'incident_date',
    'incident type': 'incident_type',
    'collision type': 'collision_type',
    'severity': 'incident_severity',
    'authorities': 'authorities_contacted',
    'incident state': 'incident_state',
    'incident city': 'incident_city',
    'total claim': 'total_claim_amount',
    'injury claim': 'injury_claim',
    'property claim': 'property_claim',
    'vehicle claim': 'vehicle_claim',
    'auto make': 'auto_make',
    'auto model': 'auto_model',
    'auto year': 'auto_year',
    'witnesses': 'witnesses',
    'police report': 'police_report_available',
    'property damage': 'property_damage'
}

def generate_chatbot_response(prompt, df, prob, drivers):
    """Generate precise, context-aware chatbot responses"""
    if df is None:
        return "Please analyze a claim first to get specific insights."
    
    p = prompt.lower()
    
    # 1. Check for specific field queries
    for keyword, col in FIELD_MAPPINGS.items():
        # Check if keyword (e.g., "incident date") is in prompt
        if keyword in p:
             try:
                 val = df[col].iloc[0]
                 # Smart Formatting
                 if isinstance(val, (int, float)) and ('claim' in keyword or 'premium' in keyword or 'deductible' in keyword or 'limit' in keyword):
                     formatted_val = f"${val:,.2f}"
                 elif 'date' in keyword and hasattr(val, 'strftime'):
                     formatted_val = val.strftime('%Y-%m-%d')
                 else:
                     formatted_val = str(val)
                 
                 return f"The **{keyword.title()}** for this claim is **{formatted_val}**."
             except Exception:
                 continue # Skip if column issue

    # 2. Risk/Fraud Explanation
    if any(x in p for x in ['why', 'reason', 'flag', 'risk', 'fraud', 'score']):
        return f"This claim is assessed at **{prob:.1%}** fraud probability. " + (f"Key risk drivers include: {', '.join(drivers)}." if drivers else "This is based on complex model patterns.")
        
    # 3. Document Recommendations
    if 'document' in p or 'proof' in p or 'evidence' in p:
        return "Recommended Documents for verification: 1. Police Report. 2. Cell Tower Data. 3. Vehicle Maintenance Records."
        
    # 4. Fallback
    return f"I can give you precise details. Try asking 'What is the premium?', 'How much is the claim?', 'Who was contacted?', or 'Why is this a risk?'."

def predict_with_model(df_model_input, selected_model):
    """Run prediction with the specified model"""
    try:
        probability = selected_model.predict_proba(df_model_input)[0][1]
        return probability
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return 0.0

def process_submission(
    months_as_customer, age, policy_bind_date, policy_state, policy_deductable, policy_annual_premium,
    incident_date, incident_type, collision_type, incident_severity, authorities_contacted, state, city,
    total_claim_amount, injury_claim, property_claim, vehicle_claim,
    auto_make, auto_model, auto_year,
    witnesses, police_report, property_damage,
    insured_sex, insured_education_level, insured_occupation,
    umbrella_limit, capital_gains, capital_loss, incident_hour, num_vehicles, bodily_injuries,
    encoders
):
    # Construct DataFrame
    input_data = {
        'months_as_customer': [months_as_customer],
        'age': [age],
        'policy_bind_date': [pd.to_datetime(policy_bind_date)],
        'policy_state': [policy_state],
        'policy_deductable': [policy_deductable],
        'policy_annual_premium': [policy_annual_premium],
        'umbrella_limit': [umbrella_limit],
        'insured_sex': [insured_sex],
        'insured_education_level': [insured_education_level],
        'insured_occupation': [insured_occupation],
        'insured_hobbies': ['sleeping'],
        'insured_relationship': ['husband'],
        'capital-gains': [capital_gains],
        'capital-loss': [capital_loss],
        'incident_date': [pd.to_datetime(incident_date)],
        'incident_type': [incident_type],
        'collision_type': [collision_type],
        'incident_severity': [incident_severity],
        'authorities_contacted': [authorities_contacted],
        'incident_state': [state],
        'incident_city': [city],
        'incident_hour_of_the_day': [incident_hour],
        'number_of_vehicles_involved': [num_vehicles],
        'property_damage': [property_damage],
        'bodily_injuries': [bodily_injuries],
        'witnesses': [witnesses],
        'police_report_available': [police_report],
        'total_claim_amount': [total_claim_amount],
        'injury_claim': [injury_claim],
        'property_claim': [property_claim],
        'vehicle_claim': [vehicle_claim],
        'auto_make': [auto_make],
        'auto_model': [auto_model],
        'auto_year': [auto_year]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Feature Engineering
    df_input['days_since_policy_bind'] = (df_input['incident_date'] - df_input['policy_bind_date']).dt.days
    df_input['incident_month'] = df_input['incident_date'].dt.month
    df_input['incident_day_of_week'] = df_input['incident_date'].dt.dayofweek
    
    if total_claim_amount > 0:
        df_input['injury_claim_ratio'] = injury_claim / total_claim_amount
        df_input['property_claim_ratio'] = property_claim / total_claim_amount
        df_input['vehicle_claim_ratio'] = vehicle_claim / total_claim_amount
    else:
        df_input['injury_claim_ratio'] = 0
        df_input['property_claim_ratio'] = 0
        df_input['vehicle_claim_ratio'] = 0

    drop_cols = ['policy_number', 'policy_csl', 'insured_zip', 'incident_location', '_c39', 'policy_bind_date', 'incident_date']
    df_model_input = df_input.drop([c for c in drop_cols if c in df_input.columns], axis=1, errors='ignore')
    
    # Encode Categoricals
    cat_cols = df_model_input.select_dtypes(include=['object']).columns
    
    if encoders:
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                try:
                    # Handle '?' inputs -> 'nan' -> text
                    df_model_input[col] = df_model_input[col].apply(lambda x: 'nan' if x == '?' else str(x))
                    
                    # Transform (handle unseen)
                    df_model_input[col] = df_model_input[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                except Exception:
                    df_model_input[col] = 0
    
    # Save preprocessed data and metadata to session state
    st.session_state['analysis_done'] = True
    st.session_state['df_input'] = df_input
    st.session_state['df_model_input'] = df_model_input  # Store preprocessed data
    st.session_state['police_report'] = police_report
    st.session_state['incident_severity'] = incident_severity
    st.session_state['witnesses'] = witnesses
    st.session_state['incident_type'] = incident_type
    st.session_state['total_claim_amount'] = total_claim_amount
    
    # Change Page
    st.session_state.page = 'result'
    st.rerun()
