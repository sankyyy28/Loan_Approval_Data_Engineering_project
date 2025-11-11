import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Create sample data structure matching your original model
def recreate_model():
    # Create dummy data to retrain a similar model
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1500, 81000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 40000, n_samples),
        'LoanAmount': np.random.randint(9, 700, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 480, 300, 240], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Prepare features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Create label encoders
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
    
    # Train model
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Save the model
    model_dict = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }
    
    with open('loan_approval_model_new.pkl', 'wb') as file:
        pickle.dump(model_dict, file)
    
    print("New model created successfully!")
    return model_dict

if __name__ == "__main__":
    recreate_model()
