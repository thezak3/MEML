import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import torch
import torch.nn as nn
import warnings

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# ============= ANN MODEL WRAPPER CLASS =============
# This class must be defined before loading the ANN model
class PavementANN(nn.Module):
    """ANN architecture for pavement cracking prediction"""
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.2):
        super(PavementANN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ANNModelWrapper:
    """Wrapper to make PyTorch model pickle-compatible"""
    def __init__(self, model, input_size, hidden_sizes, dropout_rate, device='cpu'):
        self.model = model
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, X):
        """Predict method compatible with sklearn-style interface"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

# Page configuration
st.set_page_config(
    page_title="Pavement Design Tool",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load models and scaler
@st.cache_resource
def load_models():
    with open('xgboost_model_row_level.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('lightgbm_model_row_level.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    with open('random_forest_model_row_level.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('ann_model_row_level.pkl', 'rb') as f:
        ann_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return xgb_model, lgb_model, rf_model, ann_model, scaler

# Binder grade and mix type data
binder_data = {
    ('64-22', 'Type B'): (6.4359E-06, 3.8374),
    ('70-22', 'Type B'): (6.8551E-06, 3.8201),
    ('76-22', 'Type B'): (7.317E-06, 3.8023),
    ('64-28', 'Type B'): (3.1557E-06, 4.0323),
    ('70-28', 'Type B'): (3.7800E-06, 3.9828),
    ('64-22', 'Type C'): (5.2041E-06, 3.8948),
    ('70-22', 'Type C'): (5.5095E-06, 3.8792),
    ('76-22', 'Type C'): (5.8430E-06, 3.8630),
    ('64-28', 'Type C'): (2.8039E-06, 4.0645),
    ('70-28', 'Type C'): (3.3231E-06, 4.0179),
    ('64-22', 'Type D'): (4.2081E-06, 3.9531),
    ('70-22', 'Type D'): (4.4280E-06, 3.9391),
    ('76-22', 'Type D'): (4.6659E-06, 3.9248),
    ('64-28', 'Type D'): (2.4914E-06, 4.0969),
    ('70-28', 'Type D'): (2.9215E-06, 4.0532),
}

# AC Modulus lookup table
ac_modulus_lookup = {
    ('64-22', 'Type B'): 1389.46,
    ('76-22', 'Type B'): 1652.78,
    ('64-28', 'Type B'): 1244.88,
    ('64-28', 'Type C'): 1152.44,
    ('70-22', 'Type D'): 1346.05,
    ('76-22', 'Type D'): 1449.94,
    ('64-28', 'Type D'): 1101.56,
}

DEFAULT_AC_MODULUS = 837

# Training data ranges
VALID_RANGES = {
    'AC_Thickness': (4.0, 7.0),
    'Base_Thickness': (8.0, 24.0),
    'Base_Modulus': (36.5, 250),
    'Subgrade_Modulus': (5.0, 20.0),
    'RAP_Percent': (0, 30)
}

# Prediction function with monotonicity enforcement
def predict_cracking_with_monotonicity(inputs_dict, scaler, xgb_model, lgb_model, rf_model, ann_model, 
                                       total_esals, design_life_years, return_progression=False):
    """
    Make cracking prediction with monotonicity enforcement.
    If return_progression=True, returns progression curves for all models.
    """
    # Generate predictions for each year
    years = list(range(design_life_years + 1))
    
    predictions = {
        'xgb': [],
        'lgb': [],
        'rf': [],
        'ann': [],
        'ensemble': []
    }
    
    for year in years:
        year_months = year * 12
        year_esals = (total_esals / 240) * year_months if year > 0 else 0
        
        year_inputs = inputs_dict.copy()
        year_inputs['Pavement_Age_Months'] = year_months
        year_inputs['Cumulative_Monthly_ESALs'] = year_esals
        
        # Create feature dataframe with exact column order from training
        df_input = pd.DataFrame([year_inputs])
        
        # Add polynomial and interaction features
        df_input['Age_Squared'] = df_input['Pavement_Age_Months'] ** 2
        df_input['ESALs_Squared'] = df_input['Cumulative_Monthly_ESALs'] ** 2
        df_input['Age_x_ESALs'] = df_input['Pavement_Age_Months'] * df_input['Cumulative_Monthly_ESALs']
        df_input['AC_Thickness_x_Modulus'] = df_input['AC_Thickness'] * df_input['AC_Modulus_ksi']
        df_input['Base_Thickness_x_Modulus'] = df_input['Base_Thickness'] * df_input['Base_Modulus']
        
        # Add Paris Law features
        df_input['Paris_ESALs'] = df_input['A'] * (df_input['Cumulative_Monthly_ESALs'] ** df_input['n'])
        df_input['Paris_Age'] = df_input['A'] * (df_input['Pavement_Age_Months'] ** df_input['n'])
        df_input['Paris_Combined'] = df_input['A'] * ((df_input['Cumulative_Monthly_ESALs'] + df_input['Pavement_Age_Months']) ** df_input['n'])
        df_input['Paris_Modulus_ESALs'] = (df_input['A'] / (df_input['AC_Modulus_ksi'] + 1)) * (df_input['Cumulative_Monthly_ESALs'] ** df_input['n'])
        
        # Ensure columns are in exact training order
        feature_order = [
            'Pavement_Age_Months', 'Cumulative_Monthly_ESALs', 'AC_Thickness',
            'RAP_Percent', 'A', 'n', 'AC_Modulus_ksi', 'Base_Thickness',
            'Base_Modulus', 'Subgrade_Modulus',
            'Age_Squared', 'ESALs_Squared', 'Age_x_ESALs',
            'AC_Thickness_x_Modulus', 'Base_Thickness_x_Modulus',
            'Paris_ESALs', 'Paris_Age', 'Paris_Combined', 'Paris_Modulus_ESALs'
        ]
        df_input = df_input[feature_order]
        
        # Scale features
        X_scaled = scaler.transform(df_input)
        
        # Get predictions from all models
        pred_xgb = np.clip(xgb_model.predict(X_scaled)[0] ** 2, 0, 100)
        pred_lgb = np.clip(lgb_model.predict(X_scaled)[0] ** 2, 0, 100)
        pred_rf = np.clip(rf_model.predict(X_scaled)[0] ** 2, 0, 100)
        pred_ann = np.clip(ann_model.predict(X_scaled)[0] ** 2, 0, 100)
        
        # Apply monotonicity enforcement
        if year > 0:
            pred_xgb = max(pred_xgb, predictions['xgb'][-1])
            pred_lgb = max(pred_lgb, predictions['lgb'][-1])
            pred_rf = max(pred_rf, predictions['rf'][-1])
            pred_ann = max(pred_ann, predictions['ann'][-1])
        
        # Store predictions
        predictions['xgb'].append(pred_xgb)
        predictions['lgb'].append(pred_lgb)
        predictions['rf'].append(pred_rf)
        predictions['ann'].append(pred_ann)
        
        # Calculate ensemble average
        ensemble_pred = (pred_xgb + pred_lgb + pred_rf + pred_ann) / 4
        predictions['ensemble'].append(ensemble_pred)
    
    if return_progression:
        return predictions, years
    else:
        # Return final year predictions
        final_idx = design_life_years
        return (predictions['xgb'][final_idx], 
                predictions['lgb'][final_idx], 
                predictions['rf'][final_idx],
                predictions['ann'][final_idx])

# Check if value is out of range
def check_out_of_range(param_name, value):
    if param_name in VALID_RANGES:
        min_val, max_val = VALID_RANGES[param_name]
        return value < min_val or value > max_val
    return False

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'instructions'
if 'show_info' not in st.session_state:
    st.session_state.show_info = False
if 'show_xgb' not in st.session_state:
    st.session_state.show_xgb = False
if 'show_lgb' not in st.session_state:
    st.session_state.show_lgb = False
if 'show_rf' not in st.session_state:
    st.session_state.show_rf = False
if 'show_ann' not in st.session_state:
    st.session_state.show_ann = False

# Instructions Page
if st.session_state.page == 'instructions':
    st.title("Pavement Fatigue Cracking Design Tool")
    st.markdown("## Instructions")
    
    st.markdown("""
    ### Purpose
    This tool predicts fatigue cracking in flexible pavements using an ensemble of four machine learning models trained on pavement performance data.
    
    ### How to Use
    
    **1. Design Input Tab**
    - Select your pavement design parameters:
        - Design life (5, 10, 15, or 20 years)
        - Mix design (Performance Grade and Mix Type)
        - Pavement structure (layer thicknesses and moduli)
        - Traffic loading
    - Click "Calculate Prediction" to see results
    
    **2. Design Comparison Tab**
    - Compare three design alternatives side-by-side:
        - Your base design
        - Two custom alternative designs
    - View cracking progression over time for all three designs
    
    **3. Understanding Results**
    - **Good** (Green): Cracking < 15% - Design meets performance criteria
    - **Acceptable** (Yellow): Cracking 15-30% - Design is acceptable but monitor closely
    - **Early Failure** (Red): Cracking > 30% - Design may require modification
    
    **4. Cracking Progression**
    - Graph shows predicted cracking from year 0 to your selected design life
    - Toggle individual models on/off to compare predictions
    - Shaded area shows the range between models
    
    **5. Sensitivity Analysis**
    - Shows how changing key parameters affects cracking
    - Helps optimize your design
    
    ### Models
    - **XGBoost** - Gradient boosting with excellent interaction capture
    - **LightGBM** - Fast, efficient gradient boosting
    - **Random Forest** - Ensemble of decision trees, robust to outliers
    - **ANN (PyTorch)** - Neural network for complex non-linear patterns
    - **Ensemble** - Average of all four models for robust predictions
    - Trained on 260 pavement cases with 62,400 observations using row-level split with 6-fold cross-validation
    
    ### Model Confidence
    - **High**: Model spread < 5% - All models agree closely
    - **Medium**: Model spread 5-10% - Reasonable agreement
    - **Low**: Model spread > 10% - Significant disagreement, use caution
    
    ### Limitations
    - Predictions are most reliable within the training data ranges
    - Warnings appear if inputs are outside typical values
    - This tool aids engineering judgment but does not replace it
    
    ### Data Ranges
    - AC Thickness: 4.0 - 7.0 inches
    - Base Thickness: 8.0 - 24.0 inches
    - Base Modulus: 36.5 - 250 ksi
    - Subgrade Modulus: 5.0 - 20.0 ksi
    - RAP Content: 0 - 30%
    """)
    
    if st.button("I Understand - Proceed to Tool", type="primary", key="proceed_button"):
        st.session_state.page = 'design_input'
        st.rerun()

# Main App Pages
else:
    # Load models
    try:
        xgb_model, lgb_model, rf_model, ann_model, scaler = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Design Input", "Design Comparison Tool", "Batch Processing"])
    
    # ============================================================================
    # TAB 1: DESIGN INPUT
    # ============================================================================
    with tab1:
        st.title("Pavement Fatigue Cracking Design Tool")
        
        # Input Section
        st.markdown("## Design Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Design Life & Mix")
            design_life = st.selectbox("Design Life (years)", [5, 10, 15, 20], index=3)
            
            pg_options = sorted(list(set([key[0] for key in binder_data.keys()])))
            selected_pg = st.selectbox("Performance Grade (PG)", pg_options, index=0)
            
            available_mix_types = sorted(list(set([key[1] for key in binder_data.keys() if key[0] == selected_pg])))
            selected_mix = st.selectbox("Mix Type", available_mix_types, index=0)
            
            A_raw, n = binder_data[(selected_pg, selected_mix)]
            A_scaled = A_raw * 1e6
            
            ac_modulus = ac_modulus_lookup.get((selected_pg, selected_mix), DEFAULT_AC_MODULUS)
            
            rap_percent = st.number_input("RAP Content (%)", min_value=0, max_value=30, value=0, step=5)
        
        with col2:
            st.markdown("### Pavement Structure")
            ac_thickness = st.selectbox(
                "AC Thickness (inches)", 
                [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 
                index=3
            )
            base_thickness = st.selectbox(
                "Base Thickness (inches)", 
                [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0], 
                index=4
            )
            base_modulus = st.number_input("Base Modulus (ksi)", min_value=30, max_value=300, value=37, step=30)
            subgrade_modulus = st.number_input("Subgrade Modulus (ksi)", min_value=5, max_value=20, value=10, step=5)
        
        with col3:
            st.markdown("### Traffic Loading")
            traffic_level = st.selectbox("Traffic Level", ["Light", "Medium", "Heavy", "Custom"], index=1)
            
            if traffic_level == "Light":
                total_esals = 500000
                st.info(f"Total ESALs: {total_esals:,}")
            elif traffic_level == "Medium":
                total_esals = 2000000
                st.info(f"Total ESALs: {total_esals:,}")
            elif traffic_level == "Heavy":
                total_esals = 5000000
                st.info(f"Total ESALs: {total_esals:,}")
            else:
                total_esals = st.number_input("Total Design ESALs", 100000, 10000000, 2000000, 100000)
            
            st.caption("📊 Models were trained using 6,749 ESALs/month average. Predictions with significantly different traffic levels may be less reliable.")
        
        # Check for out-of-range values
        warnings = []
        if check_out_of_range('AC_Thickness', ac_thickness):
            warnings.append(f"AC Thickness ({ac_thickness} in) is outside typical range {VALID_RANGES['AC_Thickness']}")
        if check_out_of_range('Base_Thickness', base_thickness):
            warnings.append(f"Base Thickness ({base_thickness} in) is outside typical range {VALID_RANGES['Base_Thickness']}")
        if check_out_of_range('Base_Modulus', base_modulus):
            warnings.append(f"Base Modulus ({base_modulus} ksi) is outside typical range {VALID_RANGES['Base_Modulus']}")
        if check_out_of_range('Subgrade_Modulus', subgrade_modulus):
            warnings.append(f"Subgrade Modulus ({subgrade_modulus} ksi) is outside typical range {VALID_RANGES['Subgrade_Modulus']}")
        if check_out_of_range('RAP_Percent', rap_percent):
            warnings.append(f"RAP Content ({rap_percent}%) is outside typical range {VALID_RANGES['RAP_Percent']}")
        
        if warnings:
            st.warning("⚠️ Out-of-Range Inputs Detected:\n" + "\n".join([f"- {w}" for w in warnings]) + 
                      "\n\nPredictions may be less reliable for values outside training data ranges.")
        
        # Calculate button
        if st.button("Calculate Prediction", type="primary", key="calc_button"):
            # Prepare inputs
            inputs_dict = {
                'AC_Thickness': ac_thickness,
                'RAP_Percent': rap_percent,
                'A': A_scaled,
                'n': n,
                'AC_Modulus_ksi': ac_modulus,
                'Base_Thickness': base_thickness,
                'Base_Modulus': base_modulus,
                'Subgrade_Modulus': subgrade_modulus
            }
            
            # Get progression with monotonicity enforcement
            predictions, years = predict_cracking_with_monotonicity(
                inputs_dict, scaler, xgb_model, lgb_model, rf_model, ann_model,
                total_esals, design_life, return_progression=True
            )
            
            # Get final predictions
            final_idx = design_life
            pred_xgb = predictions['xgb'][final_idx]
            pred_lgb = predictions['lgb'][final_idx]
            pred_rf = predictions['rf'][final_idx]
            pred_ann = predictions['ann'][final_idx]
            avg_pred = predictions['ensemble'][final_idx]
            
            # Calculate model spread and confidence
            final_preds = [pred_xgb, pred_lgb, pred_rf, pred_ann]
            model_spread = max(final_preds) - min(final_preds)
            
            if model_spread < 5:
                confidence = "High"
            elif model_spread <= 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Store in session
            st.session_state['predictions'] = {
                'xgb': pred_xgb,
                'lgb': pred_lgb,
                'rf': pred_rf,
                'ann': pred_ann,
                'avg': avg_pred,
                'design_life': design_life,
                'confidence': confidence,
                'spread': model_spread
            }
            
            st.session_state['progression'] = {
                'years': years,
                'predictions': predictions
            }
            st.session_state['base_inputs'] = inputs_dict
            st.session_state['total_esals'] = total_esals
            st.session_state['selected_pg'] = selected_pg
            st.session_state['selected_mix'] = selected_mix
        
        # Display results
        if 'predictions' in st.session_state:
            preds = st.session_state['predictions']
            avg_pred = preds['avg']
            
            # Determine status
            if avg_pred < 15:
                status = "Good"
                status_color = "🟢"
            elif avg_pred <= 30:
                status = "Acceptable"
                status_color = "🟡"
            else:
                status = "Early Failure"
                status_color = "🔴"
            
            st.markdown("---")
            
            # Header with info button
            col_header, col_info_button = st.columns([0.95, 0.05])
            with col_header:
                st.markdown("## Prediction Results")
            with col_info_button:
                if st.button("ℹ️", key="info_toggle"):
                    st.session_state.show_info = not st.session_state.show_info
            
            # Expandable info section
            if st.session_state.show_info:
                with st.expander("Model Confidence Information", expanded=True):
                    st.markdown("### Understanding Model Confidence")
                    st.markdown(f"**Current Confidence Level:** {preds['confidence']}")
                    st.markdown(f"**Model Spread:** {preds['spread']:.2f}%")
                    
                    st.markdown("""
                    ---
                    **Confidence Levels:**
                    
                    - **High Confidence** (Model Spread < 5%):
                      - All four models agree closely on the prediction
                      - The prediction is highly reliable
                      - You can proceed with confidence in the results
                    
                    - **Medium Confidence** (Model Spread 5-10%):
                      - Models show reasonable agreement
                      - The prediction is reasonably reliable
                      - Consider the results carefully and verify assumptions
                    
                    - **Low Confidence** (Model Spread > 10%):
                      - Models disagree significantly
                      - Use caution when interpreting results
                      - May indicate inputs outside typical training ranges
                      - Consider consulting with a pavement engineer
                    
                    **Model Spread** is calculated as the difference between the highest and lowest individual model predictions. A smaller spread indicates better agreement among the models.
                    
                    ---
                    **Individual Models in Ensemble:**
                    - **XGBoost**: Gradient boosting, excellent at capturing complex interactions
                    - **LightGBM**: Fast and efficient gradient boosting with good generalization
                    - **Random Forest**: Ensemble of trees, robust against outliers and overfitting
                    - **ANN (Neural Network)**: Deep learning model for complex non-linear patterns
                    
                    The final prediction is the average of all four models, which provides a more robust estimate than any single model.
                    """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label=f"Predicted Cracking at {preds['design_life']} years", value=f"{avg_pred:.2f}%")
            
            with col2:
                st.metric(label="Design Status", value=f"{status_color} {status}")
            
            with col3:
                st.metric(label="Model Confidence", value=preds['confidence'], 
                         delta=f"Spread: {preds['spread']:.2f}%")
            
            # Individual predictions in expandable section
            with st.expander("📊 Individual Model Predictions", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("XGBoost", f"{preds['xgb']:.2f}%")
                with col2:
                    st.metric("LightGBM", f"{preds['lgb']:.2f}%")
                with col3:
                    st.metric("Random Forest", f"{preds['rf']:.2f}%")
                with col4:
                    st.metric("ANN", f"{preds['ann']:.2f}%")
            
            # Cracking progression chart with toggles
            if 'progression' in st.session_state:
                prog = st.session_state['progression']
                years = prog['years']
                predictions = prog['predictions']
                
                st.markdown("### Cracking Progression Over Time")
                
                # Toggle controls
                st.markdown("**Show Individual Models:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    show_xgb = st.checkbox("XGBoost", value=st.session_state.show_xgb, key="toggle_xgb")
                    st.session_state.show_xgb = show_xgb
                with col2:
                    show_lgb = st.checkbox("LightGBM", value=st.session_state.show_lgb, key="toggle_lgb")
                    st.session_state.show_lgb = show_lgb
                with col3:
                    show_rf = st.checkbox("Random Forest", value=st.session_state.show_rf, key="toggle_rf")
                    st.session_state.show_rf = show_rf
                with col4:
                    show_ann = st.checkbox("ANN", value=st.session_state.show_ann, key="toggle_ann")
                    st.session_state.show_ann = show_ann
                
                fig = go.Figure()
                
                # Add shaded area for model range
                min_preds = [min(predictions['xgb'][i], predictions['lgb'][i], 
                                predictions['rf'][i], predictions['ann'][i]) for i in range(len(years))]
                max_preds = [max(predictions['xgb'][i], predictions['lgb'][i], 
                                predictions['rf'][i], predictions['ann'][i]) for i in range(len(years))]
                
                fig.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=max_preds + min_preds[::-1],
                    fill='toself',
                    fillcolor='rgba(173, 216, 230, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Model Range',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add individual model lines (if toggled)
                if show_xgb:
                    fig.add_trace(go.Scatter(
                        x=years, y=predictions['xgb'],
                        mode='lines',
                        name='XGBoost',
                        line=dict(color='lightblue', width=1.5, dash='dot'),
                        opacity=0.7
                    ))
                
                if show_lgb:
                    fig.add_trace(go.Scatter(
                        x=years, y=predictions['lgb'],
                        mode='lines',
                        name='LightGBM',
                        line=dict(color='lightgreen', width=1.5, dash='dot'),
                        opacity=0.7
                    ))
                
                if show_rf:
                    fig.add_trace(go.Scatter(
                        x=years, y=predictions['rf'],
                        mode='lines',
                        name='Random Forest',
                        line=dict(color='orange', width=1.5, dash='dot'),
                        opacity=0.7
                    ))
                
                if show_ann:
                    fig.add_trace(go.Scatter(
                        x=years, y=predictions['ann'],
                        mode='lines',
                        name='ANN',
                        line=dict(color='purple', width=1.5, dash='dot'),
                        opacity=0.7
                    ))
                
                # Add ensemble line (always shown, bold)
                fig.add_trace(go.Scatter(
                    x=years, y=predictions['ensemble'],
                    mode='lines+markers',
                    name='Ensemble Average',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8, color='blue')
                ))
                
                # Add threshold lines
                fig.add_hline(y=15, line_dash="dash", line_color="green", 
                             annotation_text="Good/Acceptable (15%)", annotation_position="right")
                fig.add_hline(y=30, line_dash="dash", line_color="red", 
                             annotation_text="Acceptable/Failure (30%)", annotation_position="right")
                
                fig.update_layout(
                    xaxis_title="Pavement Age (Years)",
                    yaxis_title="Fatigue Cracking (%)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # SENSITIVITY ANALYSIS
            st.markdown("---")
            st.markdown("## Sensitivity Analysis")
            st.caption("Shows how changing key parameters affects predicted cracking (ensemble average)")
            
            base_inputs = st.session_state.get('base_inputs', {})
            base_total_esals = st.session_state.get('total_esals', 2000000)
            
            # AC Thickness sensitivity
            ac_options = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
            ac_results = []
            for ac in ac_options:
                test_inputs = base_inputs.copy()
                test_inputs['AC_Thickness'] = ac
                
                preds_ac, _ = predict_cracking_with_monotonicity(
                    test_inputs, scaler, xgb_model, lgb_model, rf_model, ann_model,
                    base_total_esals, design_life, return_progression=True
                )
                ac_results.append((ac, preds_ac['ensemble'][design_life]))
            
            # Base Thickness sensitivity
            base_options = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
            base_results = []
            for base_thick in base_options:
                test_inputs = base_inputs.copy()
                test_inputs['Base_Thickness'] = base_thick
                
                preds_base, _ = predict_cracking_with_monotonicity(
                    test_inputs, scaler, xgb_model, lgb_model, rf_model, ann_model,
                    base_total_esals, design_life, return_progression=True
                )
                base_results.append((base_thick, preds_base['ensemble'][design_life]))
            
            # Mix Type sensitivity
            current_pg = st.session_state.get('selected_pg', selected_pg)
            mix_results = []
            for mix_type in ['Type B', 'Type C', 'Type D']:
                if (current_pg, mix_type) in binder_data:
                    test_A_raw, test_n = binder_data[(current_pg, mix_type)]
                    test_A_scaled = test_A_raw * 1e6
                    test_ac_mod = ac_modulus_lookup.get((current_pg, mix_type), DEFAULT_AC_MODULUS)
                    
                    test_inputs = base_inputs.copy()
                    test_inputs['A'] = test_A_scaled
                    test_inputs['n'] = test_n
                    test_inputs['AC_Modulus_ksi'] = test_ac_mod
                    
                    preds_mix, _ = predict_cracking_with_monotonicity(
                        test_inputs, scaler, xgb_model, lgb_model, rf_model, ann_model,
                        base_total_esals, design_life, return_progression=True
                    )
                    mix_results.append((mix_type, preds_mix['ensemble'][design_life]))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sensitivity Table")
                
                sens_data = []
                
                # AC Thickness
                for ac, crack in ac_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'AC Thickness',
                        'Value': f"{ac} in",
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                # Base Thickness
                for base_t, crack in base_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'Base Thickness',
                        'Value': f"{base_t} in",
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                # Mix Type
                for mix, crack in mix_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'Mix Type',
                        'Value': mix,
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### What-If Summary")
                
                # Find best/worst for each parameter
                best_ac = min(ac_results, key=lambda x: x[1])
                worst_ac = max(ac_results, key=lambda x: x[1])
                
                best_base = min(base_results, key=lambda x: x[1])
                worst_base = max(base_results, key=lambda x: x[1])
                
                st.markdown(f"**AC Thickness:**")
                st.markdown(f"- Increasing from {worst_ac[0]} to {best_ac[0]} inches reduces cracking by {worst_ac[1] - best_ac[1]:.2f}%")
                
                st.markdown(f"**Base Thickness:**")
                st.markdown(f"- Increasing from {worst_base[0]} to {best_base[0]} inches reduces cracking by {worst_base[1] - best_base[1]:.2f}%")
                
                if mix_results:
                    best_mix = min(mix_results, key=lambda x: x[1])
                    worst_mix = max(mix_results, key=lambda x: x[1])
                    st.markdown(f"**Mix Type:**")
                    st.markdown(f"- Changing from {worst_mix[0]} to {best_mix[0]} reduces cracking by {worst_mix[1] - best_mix[1]:.2f}%")
    
    # ============================================================================
    # TAB 2: DESIGN COMPARISON TOOL
    # ============================================================================
    with tab2:
        st.title("Design Comparison Tool")
        st.markdown("Compare your base design with two design alternatives")
        
        if 'base_inputs' not in st.session_state:
            st.info("Please run a prediction in the Design Input tab first to enable comparison.")
        else:
            base = st.session_state['base_inputs']
            base_total_esals = st.session_state.get('total_esals', 2000000)
            design_life = st.session_state['predictions']['design_life']
            
            st.markdown("### Design Alternatives Input")
            st.caption(f"All designs will use the same design life: {design_life} years")
            
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Base Design (read-only display)
            with col1:
                st.markdown("#### Base Design")
                st.markdown(f"**Design Life:** {design_life} years")
                st.markdown(f"**PG Grade:** {st.session_state.get('selected_pg', 'N/A')}")
                st.markdown(f"**Mix Type:** {st.session_state.get('selected_mix', 'N/A')}")
                st.markdown(f"**AC Thickness:** {base['AC_Thickness']} in")
                st.markdown(f"**Base Thickness:** {base['Base_Thickness']} in")
                st.markdown(f"**Base Modulus:** {base['Base_Modulus']} ksi")
                st.markdown(f"**Subgrade Modulus:** {base['Subgrade_Modulus']} ksi")
                st.markdown(f"**RAP Content:** {base['RAP_Percent']}%")
                st.markdown(f"**Total ESALs:** {base_total_esals:,}")
            
            # Column 2: Alternative 1
            with col2:
                st.markdown("#### Alternative 1")
                st.markdown(f"**Design Life:** {design_life} years")
                
                pg_options = sorted(list(set([key[0] for key in binder_data.keys()])))
                alt1_pg = st.selectbox("PG Grade", pg_options, index=0, key="alt1_pg")
                
                available_mix_types_alt1 = sorted(list(set([key[1] for key in binder_data.keys() if key[0] == alt1_pg])))
                alt1_mix = st.selectbox("Mix Type", available_mix_types_alt1, index=0, key="alt1_mix")
                
                alt1_ac_thickness = st.selectbox(
                    "AC Thickness (in)", 
                    [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 
                    index=3,
                    key="alt1_ac"
                )
                
                alt1_base_thickness = st.selectbox(
                    "Base Thickness (in)", 
                    [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0], 
                    index=4,
                    key="alt1_base"
                )
                
                alt1_base_modulus = st.number_input(
                    "Base Modulus (ksi)", 
                    min_value=30, max_value=300, value=37, step=30,
                    key="alt1_base_mod"
                )
                
                alt1_subgrade_modulus = st.number_input(
                    "Subgrade Modulus (ksi)", 
                    min_value=5, max_value=20, value=10, step=5,
                    key="alt1_sub_mod"
                )
                
                alt1_rap = st.number_input(
                    "RAP Content (%)", 
                    min_value=0, max_value=30, value=0, step=5,
                    key="alt1_rap"
                )
                
                alt1_traffic = st.selectbox(
                    "Traffic Level", 
                    ["Light", "Medium", "Heavy", "Custom"], 
                    index=1,
                    key="alt1_traffic"
                )
                
                if alt1_traffic == "Light":
                    alt1_total_esals = 500000
                    st.caption(f"Total ESALs: {alt1_total_esals:,}")
                elif alt1_traffic == "Medium":
                    alt1_total_esals = 2000000
                    st.caption(f"Total ESALs: {alt1_total_esals:,}")
                elif alt1_traffic == "Heavy":
                    alt1_total_esals = 5000000
                    st.caption(f"Total ESALs: {alt1_total_esals:,}")
                else:
                    alt1_total_esals = st.number_input(
                        "Total Design ESALs", 
                        100000, 10000000, 2000000, 100000,
                        key="alt1_esals"
                    )
            
            # Column 3: Alternative 2
            with col3:
                st.markdown("#### Alternative 2")
                st.markdown(f"**Design Life:** {design_life} years")
                
                alt2_pg = st.selectbox("PG Grade", pg_options, index=0, key="alt2_pg")
                
                available_mix_types_alt2 = sorted(list(set([key[1] for key in binder_data.keys() if key[0] == alt2_pg])))
                alt2_mix = st.selectbox("Mix Type", available_mix_types_alt2, index=0, key="alt2_mix")
                
                alt2_ac_thickness = st.selectbox(
                    "AC Thickness (in)", 
                    [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 
                    index=3,
                    key="alt2_ac"
                )
                
                alt2_base_thickness = st.selectbox(
                    "Base Thickness (in)", 
                    [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0], 
                    index=4,
                    key="alt2_base"
                )
                
                alt2_base_modulus = st.number_input(
                    "Base Modulus (ksi)", 
                    min_value=30, max_value=300, value=37, step=30,
                    key="alt2_base_mod"
                )
                
                alt2_subgrade_modulus = st.number_input(
                    "Subgrade Modulus (ksi)", 
                    min_value=5, max_value=20, value=10, step=5,
                    key="alt2_sub_mod"
                )
                
                alt2_rap = st.number_input(
                    "RAP Content (%)", 
                    min_value=0, max_value=30, value=0, step=5,
                    key="alt2_rap"
                )
                
                alt2_traffic = st.selectbox(
                    "Traffic Level", 
                    ["Light", "Medium", "Heavy", "Custom"], 
                    index=1,
                    key="alt2_traffic"
                )
                
                if alt2_traffic == "Light":
                    alt2_total_esals = 500000
                    st.caption(f"Total ESALs: {alt2_total_esals:,}")
                elif alt2_traffic == "Medium":
                    alt2_total_esals = 2000000
                    st.caption(f"Total ESALs: {alt2_total_esals:,}")
                elif alt2_traffic == "Heavy":
                    alt2_total_esals = 5000000
                    st.caption(f"Total ESALs: {alt2_total_esals:,}")
                else:
                    alt2_total_esals = st.number_input(
                        "Total Design ESALs", 
                        100000, 10000000, 2000000, 100000,
                        key="alt2_esals"
                    )
            
            # Compare All Designs button
            st.markdown("---")
            if st.button("Compare All Designs", type="primary", key="compare_button"):
                # Get binder parameters for alternatives
                alt1_A_raw, alt1_n = binder_data[(alt1_pg, alt1_mix)]
                alt1_A_scaled = alt1_A_raw * 1e6
                alt1_ac_modulus = ac_modulus_lookup.get((alt1_pg, alt1_mix), DEFAULT_AC_MODULUS)
                
                alt2_A_raw, alt2_n = binder_data[(alt2_pg, alt2_mix)]
                alt2_A_scaled = alt2_A_raw * 1e6
                alt2_ac_modulus = ac_modulus_lookup.get((alt2_pg, alt2_mix), DEFAULT_AC_MODULUS)
                
                # Prepare inputs for each design
                alt1_inputs = {
                    'AC_Thickness': alt1_ac_thickness,
                    'RAP_Percent': alt1_rap,
                    'A': alt1_A_scaled,
                    'n': alt1_n,
                    'AC_Modulus_ksi': alt1_ac_modulus,
                    'Base_Thickness': alt1_base_thickness,
                    'Base_Modulus': alt1_base_modulus,
                    'Subgrade_Modulus': alt1_subgrade_modulus
                }
                
                alt2_inputs = {
                    'AC_Thickness': alt2_ac_thickness,
                    'RAP_Percent': alt2_rap,
                    'A': alt2_A_scaled,
                    'n': alt2_n,
                    'AC_Modulus_ksi': alt2_ac_modulus,
                    'Base_Thickness': alt2_base_thickness,
                    'Base_Modulus': alt2_base_modulus,
                    'Subgrade_Modulus': alt2_subgrade_modulus
                }
                
                # Generate progressions
                base_preds, years = predict_cracking_with_monotonicity(
                    base, scaler, xgb_model, lgb_model, rf_model, ann_model,
                    base_total_esals, design_life, return_progression=True
                )
                
                alt1_preds, _ = predict_cracking_with_monotonicity(
                    alt1_inputs, scaler, xgb_model, lgb_model, rf_model, ann_model,
                    alt1_total_esals, design_life, return_progression=True
                )
                
                alt2_preds, _ = predict_cracking_with_monotonicity(
                    alt2_inputs, scaler, xgb_model, lgb_model, rf_model, ann_model,
                    alt2_total_esals, design_life, return_progression=True
                )
                
                # Plot comparison
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years, y=base_preds['ensemble'],
                    mode='lines+markers',
                    name='Base Design',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=alt1_preds['ensemble'],
                    mode='lines+markers',
                    name='Alternative 1',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=alt2_preds['ensemble'],
                    mode='lines+markers',
                    name='Alternative 2',
                    line=dict(color='orange', width=3),
                    marker=dict(size=8)
                ))
                
                # Add threshold lines
                fig.add_hline(y=15, line_dash="dash", line_color="green", 
                             annotation_text="Good/Acceptable (15%)")
                fig.add_hline(y=30, line_dash="dash", line_color="red", 
                             annotation_text="Acceptable/Failure (30%)")
                
                fig.update_layout(
                    title="Design Comparison: Cracking Progression (Ensemble Predictions)",
                    xaxis_title="Pavement Age (Years)",
                    yaxis_title="Fatigue Cracking (%)",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                st.markdown("### Design Performance Summary")
                
                final_year_idx = design_life
                summary_data = {
                    'Design': ['Base', 'Alternative 1', 'Alternative 2'],
                    'PG Grade': [
                        st.session_state.get('selected_pg', 'N/A'),
                        alt1_pg,
                        alt2_pg
                    ],
                    'Mix Type': [
                        st.session_state.get('selected_mix', 'N/A'),
                        alt1_mix,
                        alt2_mix
                    ],
                    'AC Thickness (in)': [
                        base['AC_Thickness'],
                        alt1_ac_thickness,
                        alt2_ac_thickness
                    ],
                    'Base Thickness (in)': [
                        base['Base_Thickness'],
                        alt1_base_thickness,
                        alt2_base_thickness
                    ],
                    'RAP (%)': [
                        base['RAP_Percent'],
                        alt1_rap,
                        alt2_rap
                    ],
                    'Total ESALs': [
                        f"{base_total_esals:,}",
                        f"{alt1_total_esals:,}",
                        f"{alt2_total_esals:,}"
                    ],
                    f'Cracking at {design_life} Years (%)': [
                        f"{base_preds['ensemble'][final_year_idx]:.2f}",
                        f"{alt1_preds['ensemble'][final_year_idx]:.2f}",
                        f"{alt2_preds['ensemble'][final_year_idx]:.2f}"
                    ]
                }
                
                # Add status
                statuses = []
                for preds in [base_preds, alt1_preds, alt2_preds]:
                    crack = preds['ensemble'][final_year_idx]
                    if crack < 15:
                        statuses.append('🟢 Good')
                    elif crack <= 30:
                        statuses.append('🟡 Acceptable')
                    else:
                        statuses.append('🔴 Early Failure')
                
                summary_data['Design Status'] = statuses
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("### Key Insights")
                
                base_final = base_preds['ensemble'][final_year_idx]
                alt1_final = alt1_preds['ensemble'][final_year_idx]
                alt2_final = alt2_preds['ensemble'][final_year_idx]
                
                best_design = min([('Base', base_final), ('Alternative 1', alt1_final), 
                                  ('Alternative 2', alt2_final)], key=lambda x: x[1])
                worst_design = max([('Base', base_final), ('Alternative 1', alt1_final), 
                                   ('Alternative 2', alt2_final)], key=lambda x: x[1])
                
                st.markdown(f"- **Best performing design:** {best_design[0]} with {best_design[1]:.2f}% cracking")
                st.markdown(f"- **Worst performing design:** {worst_design[0]} with {worst_design[1]:.2f}% cracking")
                st.markdown(f"- **Range of predicted cracking:** {best_design[1]:.2f}% to {worst_design[1]:.2f}% (difference of {worst_design[1] - best_design[1]:.2f}%)")
    
    # ============================================================================
    # TAB 3: BATCH PROCESSING
    # ============================================================================
    with tab3:
        st.title("Batch Processing Tool")
        st.markdown("Upload a CSV file with multiple design cases to get predictions for all at once")
        
        # Show template
        with st.expander("📋 CSV Template and Instructions"):
            st.markdown("""
            ### Required Columns:
            Your CSV must have these columns (exact names):
            
            - `Design_Life_Years` - 5, 10, 15, or 20
            - `AC_Thickness` - 4.0 to 7.0 inches
            - `Base_Thickness` - 8.0 to 24.0 inches
            - `Base_Modulus` - 36.5 to 250 ksi
            - `Subgrade_Modulus` - 5.0 to 20.0 ksi
            - `RAP_Percent` - 0 to 30
            - `Total_ESALs` - Total design ESALs
            - `PG_Grade` - e.g., "64-22", "70-22", "76-22", "64-28", "70-28"
            - `Mix_Type` - e.g., "Type B", "Type C", "Type D"
            
            ### Optional Column:
            - `Case_Name` - Give each case a descriptive name
            """)
            
            # Create template
            template_data = {
                'Case_Name': ['Example 1', 'Example 2', 'Example 3'],
                'Design_Life_Years': [20, 15, 10],
                'AC_Thickness': [5.5, 6.0, 4.5],
                'Base_Thickness': [16.0, 18.0, 12.0],
                'Base_Modulus': [37, 100, 150],
                'Subgrade_Modulus': [10, 8, 12],
                'RAP_Percent': [15, 20, 0],
                'Total_ESALs': [2000000, 5000000, 1000000],
                'PG_Grade': ['64-22', '70-22', '76-22'],
                'Mix_Type': ['Type B', 'Type C', 'Type D']
            }
            template_df = pd.DataFrame(template_data)
            
            st.dataframe(template_df, use_container_width=True)
            
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name="batch_input_template.csv",
                mime="text/csv"
            )
        
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} cases from file")
                
                with st.expander("Preview uploaded data"):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                
                # Validate columns
                required_cols = ['Design_Life_Years', 'AC_Thickness', 'Base_Thickness', 
                               'Base_Modulus', 'Subgrade_Modulus', 'RAP_Percent', 
                               'Total_ESALs', 'PG_Grade', 'Mix_Type']
                
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                if 'Case_Name' not in batch_df.columns:
                    batch_df['Case_Name'] = [f"Case {i+1}" for i in range(len(batch_df))]
                
                # Process button
                if st.button("🚀 Run Batch Predictions", type="primary"):
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in batch_df.iterrows():
                        status_text.text(f"Processing {row['Case_Name']} ({idx+1}/{len(batch_df)})...")
                        
                        try:
                            pg_grade = row['PG_Grade']
                            mix_type = row['Mix_Type']
                            
                            if (pg_grade, mix_type) not in binder_data:
                                results_list.append({
                                    'Case_Name': row['Case_Name'],
                                    'Status': 'ERROR',
                                    'Error': f"Invalid PG/Mix combination"
                                })
                                continue
                            
                            A_raw, n = binder_data[(pg_grade, mix_type)]
                            A_scaled = A_raw * 1e6
                            ac_modulus = ac_modulus_lookup.get((pg_grade, mix_type), DEFAULT_AC_MODULUS)
                            
                            inputs_dict = {
                                'AC_Thickness': row['AC_Thickness'],
                                'RAP_Percent': row['RAP_Percent'],
                                'A': A_scaled,
                                'n': n,
                                'AC_Modulus_ksi': ac_modulus,
                                'Base_Thickness': row['Base_Thickness'],
                                'Base_Modulus': row['Base_Modulus'],
                                'Subgrade_Modulus': row['Subgrade_Modulus']
                            }
                            
                            design_life_years = int(row['Design_Life_Years'])
                            total_esals = row['Total_ESALs']
                            
                            # Get predictions
                            pred_xgb, pred_lgb, pred_rf, pred_ann = predict_cracking_with_monotonicity(
                                inputs_dict, scaler, xgb_model, lgb_model, rf_model, ann_model,
                                total_esals, design_life_years, return_progression=False
                            )
                            
                            avg_pred = (pred_xgb + pred_lgb + pred_rf + pred_ann) / 4
                            model_spread = max(pred_xgb, pred_lgb, pred_rf, pred_ann) - min(pred_xgb, pred_lgb, pred_rf, pred_ann)
                            
                            if model_spread < 5:
                                confidence = "High"
                            elif model_spread <= 10:
                                confidence = "Medium"
                            else:
                                confidence = "Low"
                            
                            if avg_pred < 15:
                                status = "Good"
                            elif avg_pred <= 30:
                                status = "Acceptable"
                            else:
                                status = "Early Failure"
                            
                            results_list.append({
                                'Case_Name': row['Case_Name'],
                                'Design_Life_Years': design_life_years,
                                'AC_Thickness': row['AC_Thickness'],
                                'Base_Thickness': row['Base_Thickness'],
                                'RAP_Percent': row['RAP_Percent'],
                                'Total_ESALs': row['Total_ESALs'],
                                'PG_Grade': pg_grade,
                                'Mix_Type': mix_type,
                                'Ensemble_Prediction_%': round(avg_pred, 2),
                                'XGBoost_%': round(pred_xgb, 2),
                                'LightGBM_%': round(pred_lgb, 2),
                                'RandomForest_%': round(pred_rf, 2),
                                'ANN_%': round(pred_ann, 2),
                                'Design_Status': status,
                                'Model_Confidence': confidence,
                                'Model_Spread_%': round(model_spread, 2)
                            })
                            
                        except Exception as e:
                            results_list.append({
                                'Case_Name': row['Case_Name'],
                                'Status': 'ERROR',
                                'Error': str(e)
                            })
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    status_text.text("✅ Batch processing complete!")
                    
                    results_df = pd.DataFrame(results_list)
                    
                    st.markdown("---")
                    st.markdown("## Batch Results")
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    successful = len(results_df[results_df.get('Design_Status', '') != ''])
                    good = len(results_df[results_df.get('Design_Status', '') == 'Good'])
                    acceptable = len(results_df[results_df.get('Design_Status', '') == 'Acceptable'])
                    failure = len(results_df[results_df.get('Design_Status', '') == 'Early Failure'])
                    
                    with col1:
                        st.metric("Total Cases", len(results_df))
                    with col2:
                        st.metric("Good Designs", good)
                    with col3:
                        st.metric("Acceptable", acceptable)
                    with col4:
                        st.metric("Early Failure", failure)
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_results,
                        file_name="batch_predictions_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    if successful > 0:
                        st.markdown("---")
                        st.markdown("## Results Visualization")
                        
                        plot_df = results_df[results_df.get('Ensemble_Prediction_%', 0).notna()].copy()
                        plot_df = plot_df.sort_values('Ensemble_Prediction_%')
                        
                        fig = go.Figure()
                        
                        colors = {'Good': 'green', 'Acceptable': 'orange', 'Early Failure': 'red'}
                        
                        for status in ['Good', 'Acceptable', 'Early Failure']:
                            mask = plot_df['Design_Status'] == status
                            if mask.any():
                                fig.add_trace(go.Bar(
                                    x=plot_df[mask]['Case_Name'],
                                    y=plot_df[mask]['Ensemble_Prediction_%'],
                                    name=status,
                                    marker_color=colors[status]
                                ))
                        
                        fig.add_hline(y=15, line_dash="dash", line_color="green", 
                                     annotation_text="Good/Acceptable (15%)")
                        fig.add_hline(y=30, line_dash="dash", line_color="red",
                                     annotation_text="Acceptable/Failure (30%)")
                        
                        fig.update_layout(
                            title="Predicted Cracking by Design Case (Ensemble)",
                            xaxis_title="Case Name",
                            yaxis_title="Predicted Cracking (%)",
                            barmode='group',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Pavement Fatigue Cracking Design Tool | 4-Model Ensemble (XGBoost, LightGBM, Random Forest, ANN)*")
