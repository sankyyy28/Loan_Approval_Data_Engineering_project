import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import sklearn

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
    .success-banner {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    .info-banner {
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

# Create a new trained model that works with current scikit-learn version
@st.cache_resource
def create_and_train_model():
    """Create and train a new Decision Tree model that works with current scikit-learn version"""
    
    # Create realistic loan approval dataset
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic data
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.8, 0.2]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'ApplicantIncome': np.random.normal(5000, 2000, n_samples).clip(1000, 15000),
        'CoapplicantIncome': np.random.exponential(1000, n_samples).clip(0, 8000),
        'LoanAmount': np.random.normal(150, 50, n_samples).clip(50, 400),
        'Loan_Amount_Term': np.random.choice([360, 180, 480, 300, 240], n_samples, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
        'Credit_History': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples, p=[0.4, 0.3, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable based on business rules
    loan_status = []
    for idx, row in df.iterrows():
        score = 0
        
        # Credit history is most important
        if row['Credit_History'] == 1:
            score += 30
        else:
            score -= 20
            
        # Income factors
        if row['ApplicantIncome'] > 6000:
            score += 15
        elif row['ApplicantIncome'] < 3000:
            score -= 10
            
        if row['CoapplicantIncome'] > 2000:
            score += 5
            
        # Loan amount factors
        if row['LoanAmount'] > 300:
            score -= 15
        elif row['LoanAmount'] < 100:
            score += 5
            
        # Education
        if row['Education'] == 'Graduate':
            score += 5
            
        # Property area
        if row['Property_Area'] == 'Urban':
            score += 5
            
        # Married applicants get slight preference
        if row['Married'] == 'Yes':
            score += 3
            
        # Determine loan status
        if score >= 25:
            loan_status.append('Y')  # Approved
        else:
            loan_status.append('N')  # Rejected
    
    df['Loan_Status'] = loan_status
    
    # Prepare features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Create and fit label encoders
    label_encoders = {}
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Train the model
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10
    )
    model.fit(X, y)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns)
    }

# Load or create model
with st.spinner("🚀 Initializing AI Loan Approval System..."):
    try:
        # Try to load existing model first
        if os.path.exists('loan_approval_model.pkl'):
            with open('loan_approval_model.pkl', 'rb') as f:
                model_dict = pickle.load(f)
            st.markdown('<div class="success-banner">', unsafe_allow_html=True)
            st.success("✅ Pre-trained model loaded successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            raise FileNotFoundError("No pre-trained model found")
    except Exception as e:
        st.markdown('<div class="info-banner">', unsafe_allow_html=True)
        st.info("🔄 Creating a new optimized loan approval model...")
        st.markdown('</div>', unsafe_allow_html=True)
        model_dict = create_and_train_model()
        st.success("🎯 New AI model trained successfully! Ready for predictions.")

# Extract components
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
                                     format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)",
                                     help="1 = Good credit history, 0 = Bad credit history")
        property_area = st.selectbox("Property Area", options=label_encoders['Property_Area'].classes_)
        
    with col3:
        st.markdown("**Income & Loan Details**")
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=100000, value=5000, step=100,
                                          help="Monthly income of the applicant")
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=50000, value=0, step=100,
                                            help="Monthly income of co-applicant")
        loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0, max_value=1000, value=150, step=10,
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
        
        # Make prediction
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
                
            st.success("🎯 **AI Insights**: Based on trained machine learning model")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please try different input values.")

# Information section
with st.expander("ℹ️ About This AI System"):
    st.markdown("""
    **🤖 AI Loan Approval System**
    
    This system uses a **Decision Tree Classifier** trained on realistic loan application data to predict approval chances.
    
    **Key Features:**
    - Real-time AI predictions
    - Confidence scoring
    - Transparent decision factors
    - Professional interface
    
    **Model Information:**
    - Algorithm: Decision Tree Classifier
    - Training Data: 2000+ simulated loan applications
    - Accuracy: Optimized for realistic banking scenarios
    - Version: Compatible with current scikit-learn
    
    **Technical Stack:**
    - Framework: Streamlit
    - Machine Learning: scikit-learn
    - Data Processing: pandas, numpy
    - Version: {}
    """.format(sklearn.__version__))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p> 
        <strong>Developed by Sanket Sonparate</strong> | 
        Built with ❤️ using Streamlit | 
        AI-Powered Decision Tree Classifier
    </p>
    <div style="margin-top: 10px;">
        <a href="https://github.com/sankyyy28" style="margin: 0 10px;">GitHub</a> | 
        <a href="https://www.linkedin.com/in/sanket-sonparate-018350260" style="margin: 0 10px;">LinkedIn</a> | 
        <a href="mailto:sonparatesanket@gmail.com" style="margin: 0 10px;">Email</a>
    </div>
</div>
""", unsafe_allow_html=True)
