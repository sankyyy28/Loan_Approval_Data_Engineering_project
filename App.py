import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import sklearn
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #c3e6cb;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .prediction-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #f5c6cb;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .sidebar-profile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profile-name {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .profile-title {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    .contact-info {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .demo-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
    .success-banner {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with profile information
with st.sidebar:
    st.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
    
    # Profile Header
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    with col2:
        st.markdown('<div class="profile-name">Sanket Sanjay Sonparate</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-title">Data Scientist | ML Engineer</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact Information
    st.markdown('<div class="contact-info">', unsafe_allow_html=True)
    st.markdown("**📧 Email**")
    st.write("sonparatesanket@gmail.com")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Social Links
    st.markdown("**🔗 Social Links**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://github.com/sankyyy28" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Profile-black?logo=github&style=for-the-badge" width="140">
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://www.linkedin.com/in/sanket-sonparate-018350260" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&style=for-the-badge" width="140">
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Version info
    st.markdown("**🔧 Environment Info**")
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"numpy: {np.__version__}")

# Main content
st.markdown('<h1 class="main-header">🏦 Smart Loan Approval Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Loan Decision System • Fast & Accurate Predictions</p>', unsafe_allow_html=True)

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True  # Start in demo mode by default

# Enhanced model loading with multiple fallback options
@st.cache_resource
def load_model():
    model_files = ['loan_approval_model.pkl', 'loan_approval_model_new.pkl']
    
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as file:
                    model_dict = pickle.load(file)
                st.success(f"✅ Model loaded successfully from {model_file}!")
                return model_dict, False  # model_dict, is_demo_mode
        except Exception as e:
            st.warning(f"Could not load {model_file}: {str(e)}")
            continue
    
    # If no model files work, use demo mode
    st.markdown('<div class="demo-warning">', unsafe_allow_html=True)
    st.warning("🔧 Demo Mode: Using simulated predictions for demonstration purposes.")
    st.info("To use the real model, ensure a compatible model file is available.")
    st.markdown('</div>', unsafe_allow_html=True)
    return create_demo_model(), True

def create_demo_model():
    """Create a fully functional demo model"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create demo label encoders
    label_encoders = {
        'Gender': LabelEncoder().fit(['Male', 'Female']),
        'Married': LabelEncoder().fit(['Yes', 'No']),
        'Dependents': LabelEncoder().fit(['0', '1', '2', '3+']),
        'Education': LabelEncoder().fit(['Graduate', 'Not Graduate']),
        'Self_Employed': LabelEncoder().fit(['Yes', 'No']),
        'Property_Area': LabelEncoder().fit(['Urban', 'Rural', 'Semiurban'])
    }
    
    # Create demo scaler with proper initialization
    scaler = StandardScaler()
    # Mock the scaler parameters to avoid fitting
    scaler.mean_ = np.array([5000.0, 1000.0, 150.0, 360.0])
    scaler.scale_ = np.array([2000.0, 500.0, 50.0, 120.0])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 4
    
    # Create and "train" a simple demo model with dummy data
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    # Create dummy training data to avoid warnings
    X_dummy = np.random.randn(10, 11)  # 11 features
    y_dummy = np.random.choice(['Y', 'N'], 10)
    model.fit(X_dummy, y_dummy)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }

# Load model
with st.spinner("Loading AI model..."):
    model_dict, is_demo = load_model()

st.session_state.demo_mode = is_demo

# Extract components
scaler = model_dict.get('scaler')
label_encoders = model_dict.get('label_encoders')
model = model_dict.get('model')

if not st.session_state.demo_mode:
    st.markdown('<div class="success-banner">', unsafe_allow_html=True)
    st.success("🎯 Real AI Model Active - Making accurate predictions!")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("🔄 Demo Mode Active - Using intelligent simulation")

# Create input form
with st.form("loan_application_form"):
    st.markdown('<div class="feature-input">', unsafe_allow_html=True)
    st.subheader("📋 Applicant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Details**")
        gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
        married = st.selectbox("Married", options=label_encoders['Married'].classes_)
        education = st.selectbox("Education", options=label_encoders['Education'].classes_)
        self_employed = st.selectbox("Self Employed", options=label_encoders['Self_Employed'].classes_)
        
    with col2:
        st.markdown("**Financial Information**")
        dependents = st.selectbox("Dependents", options=label_encoders['Dependents'].classes_)
        credit_history = st.selectbox("Credit History", options=[1, 0], 
                                     format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)",
                                     help="1 = Good credit history, 0 = Bad credit history")
        property_area = st.selectbox("Property Area", options=label_encoders['Property_Area'].classes_)
        
    with col3:
        st.markdown("**Income & Loan Details**")
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=100000, value=5000, step=100,
                                          help="Monthly income of the applicant")
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=50000, value=0, step=100,
                                            help="Monthly income of co-applicant")
        loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0, max_value=1000, value=100, step=10,
                                     help="Loan amount in thousands")
        loan_term = st.slider("Loan Term (months)", min_value=12, max_value=480, value=360,
                             help="Duration of the loan")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

if submitted:
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    try:
        # Label encode categorical variables
        for column in label_encoders:
            if column in input_df.columns:
                input_df[column] = label_encoders[column].transform(input_df[column])
        
        # Scale numerical features
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        if st.session_state.demo_mode:
            # Enhanced demo mode with realistic rules
            base_score = 50
            
            # Credit history has highest impact
            if credit_history == 1:
                base_score += 30
            else:
                base_score -= 20
            
            # Income factors
            if applicant_income > 8000:
                base_score += 15
            elif applicant_income > 5000:
                base_score += 10
            elif applicant_income < 2000:
                base_score -= 10
                
            if coapplicant_income > 2000:
                base_score += 5
                
            # Loan amount factors
            if loan_amount > 400:
                base_score -= 15
            elif loan_amount > 200:
                base_score -= 5
            elif loan_amount < 100:
                base_score += 5
                
            # Other factors
            if education == 'Graduate':
                base_score += 5
            if married == 'Yes':
                base_score += 3
            if property_area == 'Urban':
                base_score += 5
            elif property_area == 'Semiurban':
                base_score += 3
                
            # Convert to probability (0-100 scale to 0-1 scale)
            prob_approved = max(0.05, min(0.95, base_score / 100))
            prediction = ['Y'] if prob_approved > 0.5 else ['N']
            prediction_proba = [[1-prob_approved, prob_approved]] if prediction[0] == 'Y' else [[prob_approved, 1-prob_approved]]
            
        else:
            # Real model prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Prediction Result")
            
            if prediction[0] == 'Y':
                st.markdown('<div class="prediction-approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
                st.balloons()
                st.success("Congratulations! Your loan application has been approved.")
            else:
                st.markdown('<div class="prediction-rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
                st.warning("We recommend reviewing your application details or contacting our support.")
            
            # Show probability
            prob_approved = prediction_proba[0][1] if prediction[0] == 'Y' else prediction_proba[0][0]
            st.metric("Confidence Score", f"{prob_approved:.2%}")
            
            # Progress bar for confidence
            st.progress(float(prob_approved))
            
            # Show input summary
            with st.expander("📋 Application Summary"):
                st.write("**Your Application Details:**")
                for key, value in input_data.items():
                    st.write(f"- **{key}**: {value}")

        with col2:
            st.subheader("🔍 Key Decision Factors")
            important_factors = {
                "Credit History": "★★★★★",
                "Applicant Income": "★★★★☆", 
                "Loan Amount": "★★★★☆",
                "Co-applicant Income": "★★★☆☆",
                "Property Area": "★★★☆☆",
                "Education Level": "★★☆☆☆",
                "Loan Term": "★★☆☆☆"
            }
            
            for factor, stars in important_factors.items():
                st.write(f"**{factor}**: {stars}")
                
            if st.session_state.demo_mode:
                st.info("💡 **Demo Insights**: Based on realistic banking rules")
            else:
                st.success("🎯 **AI Insights**: Based on trained machine learning model")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please try different input values or refresh the page.")

# Enhanced troubleshooting section
with st.expander("🔧 Troubleshooting & Model Information"):
    st.markdown(f"""
    **Current Status:** {'**Demo Mode** 🎭' if st.session_state.demo_mode else '**Real Model** 🎯'}
    
    **If you want to use the real model:**
    
    1. **Install compatible versions**:
    ```bash
    pip install scikit-learn==1.2.2 pandas==1.5.3 numpy==1.24.3
    ```
    
    2. **Ensure model file** `loan_approval_model.pkl` is in the correct directory
    
    3. **Current versions detected:**
       - scikit-learn: {sklearn.__version__}
       - pandas: {pd.__version__}
       - numpy: {np.__version__}
    
    **About this app:**
    - Built with Streamlit
    - Uses Decision Tree Classifier
    - Real-time predictions
    - Professional interface
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p> 
        <strong>Developed by Sanket Sonparate</strong> | 
        Built with ❤️ using Streamlit | 
        Machine Learning Model: Decision Tree Classifier
    </p>
    <div style="margin-top: 10px;">
        <a href="https://github.com/sankyyy28" style="margin: 0 10px;">GitHub</a> | 
        <a href="https://www.linkedin.com/in/sanket-sonparate-018350260" style="margin: 0 10px;">LinkedIn</a> | 
        <a href="mailto:sonparatesanket@gmail.com" style="margin: 0 10px;">Email</a>
    </div>
</div>
""", unsafe_allow_html=True)
