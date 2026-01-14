from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


@tool("Engineer features from customer data")
def feature_engineer_tool(file_path: str) -> str:
    """Engineers features from raw data for machine learning.
    
    Args:
        file_path: Path to the cleaned CSV file
    """
    try:
        df = pd.read_csv(file_path)
        initial_columns = df.columns.tolist()
        
        # Create new features if standard churn dataset
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', df['tenure'] * df['MonthlyCharges']), errors='coerce')
            df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
            df['ChargePerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # Encode categorical variables
        label_encoders = {}
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['customerID', 'Churn']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Encode target variable if exists
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Save engineered data
        engineered_path = file_path.replace('_cleaned.csv', '_engineered.csv')
        df.to_csv(engineered_path, index=False)
        
        # Save encoders
        os.makedirs('models', exist_ok=True)
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        
        new_features = [col for col in df.columns if col not in initial_columns]
        
        return f"""
Feature engineering completed!

Initial columns: {len(initial_columns)}
Final columns: {len(df.columns)}
New features created: {new_features if new_features else 'None (encoding applied)'}

Categorical columns encoded: {len(label_encoders)}

Data shape: {df.shape}
Engineered data saved to: {engineered_path}
Label encoders saved to: models/label_encoders.pkl
"""
    except Exception as e:
        return f"Error in feature engineering: {str(e)}"


@tool("Scale numerical features")
def feature_scale_tool(file_path: str) -> str:
    """Scales numerical features for machine learning models.
    
    Args:
        file_path: Path to the engineered CSV file
    """
    try:
        df = pd.read_csv(file_path)
        
        # Identify columns to exclude from scaling
        exclude_cols = ['customerID', 'Churn']
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_columns if col not in exclude_cols]
        
        # Scale features
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        # Save scaled data
        scaled_path = file_path.replace('_engineered.csv', '_scaled.csv')
        df.to_csv(scaled_path, index=False)
        
        # Save scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        
        return f"""
Feature scaling completed!

Columns scaled: {len(cols_to_scale)}
Scaled columns: {', '.join(cols_to_scale)}

Scaling method: StandardScaler (mean=0, std=1)

Data shape: {df.shape}
Scaled data saved to: {scaled_path}
Scaler saved to: models/scaler.pkl
"""
    except Exception as e:
        return f"Error in feature scaling: {str(e)}"
