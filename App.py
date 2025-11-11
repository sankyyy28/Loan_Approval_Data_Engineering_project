import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    with open('loan_approval_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

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
    .social-badges {
        margin-top: 15px;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e9ecef;
    }
    .contact-info {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
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
    
    # Social Links with badges
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
    
    # About section
    st.markdown("### 🚀 About This App")
    st.write("""
    This intelligent loan approval predictor uses machine learning to analyze applicant information 
    and provide instant loan approval decisions using a trained Decision Tree Classifier.
    """)
    
    st.markdown("### 💡 Features")
    st.write("""
    • Real-time prediction
    • Confidence scoring
    • Key factor analysis
    • Professional interface
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h1 class="main-header">🏦 Smart Loan Approval Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Loan Decision System • Fast & Accurate Predictions</p>', unsafe_allow_html=True)

# Load model
try:
    model_dict = load_model()
    scaler = model_dict['scaler']
    label_encoders = model_dict['label_encoders']
    model = model_dict['model']
    
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
                                         format_func=lambda x: "Good" if x == 1 else "Bad")
            property_area = st.selectbox("Property Area", options=label_encoders['Property_Area'].classes_)
            
        with col3:
            st.markdown("**Income & Loan Details**")
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=100000, value=5000, step=100)
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=50000, value=0, step=100)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000, value=100, step=10)
            loan_term = st.slider("Loan Term (months)", min_value=12, max_value=480, value=360)
        
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
        
        # Label encode categorical variables
        for column in label_encoders:
            if column in input_df.columns:
                input_df[column] = label_encoders[column].transform(input_df[column])
        
        # Scale numerical features
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        with col2:
            st.markdown("### 📊 Prediction Result")
            
            if prediction[0] == 'Y':
                st.markdown('<div class="prediction-approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
                st.balloons()
                st.success("Congratulations! Your loan application has been approved.")
            else:
                st.markdown('<div class="prediction-rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
                st.warning("We recommend reviewing your application details.")
            
            # Show probability
            prob_approved = prediction_proba[0][1] if prediction[0] == 'Y' else prediction_proba[0][0]
            st.metric("Confidence Score", f"{prob_approved:.2%}")
            
            # Progress bar for confidence
            st.progress(int(prob_approved * 100))
            
            # Feature importance visualization
            st.markdown("### 🔍 Key Decision Factors")
            important_features = {
                "Credit History": "Very High Impact",
                "Applicant Income": "High Impact", 
                "Loan Amount": "High Impact",
                "Co-applicant Income": "Medium Impact",
                "Property Area": "Medium Impact",
                "Loan Term": "Medium Impact"
            }
            
            for feature, impact in important_features.items():
                st.write(f"• **{feature}**: {impact}")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Please ensure the model file 'loan_approval_model.pkl' is in the correct directory.")

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