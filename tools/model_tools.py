from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


@tool("Train machine learning models for churn prediction")
def model_train_tool(file_path: str) -> str:
    """Trains multiple machine learning models and selects the best one.
    
    Args:
        file_path: Path to the scaled CSV file
    """
    try:
        df = pd.read_csv(file_path)
        
        # Prepare features and target
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        if 'Churn' not in df.columns:
            return "Error: 'Churn' column not found in dataset"
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_model = None
        best_score = 0
        best_model_name = ""
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/churn_model.pkl')
        
        # Save feature names
        feature_names = X.columns.tolist()
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        # Create results summary
        summary = f"""
Model Training Completed!

Dataset split:
- Training samples: {len(X_train)}
- Testing samples: {len(X_test)}
- Features: {X.shape[1]}

Model Performance Comparison:
"""
        for name, metrics in results.items():
            summary += f"\n{name}:"
            summary += f"\n  Accuracy:  {metrics['accuracy']:.4f}"
            summary += f"\n  Precision: {metrics['precision']:.4f}"
            summary += f"\n  Recall:    {metrics['recall']:.4f}"
            summary += f"\n  F1-Score:  {metrics['f1']:.4f}"
            summary += f"\n  ROC-AUC:   {metrics['roc_auc']:.4f}\n"
        
        summary += f"\nBest Model: {best_model_name} (F1-Score: {best_score:.4f})"
        summary += f"\nModel saved to: models/churn_model.pkl"
        
        return summary
        
    except Exception as e:
        return f"Error training models: {str(e)}"


@tool("Make churn predictions using trained model")
def model_predict_tool(file_path: str) -> str:
    """Makes churn predictions using the trained model.
    
    Args:
        file_path: Path to the data file for prediction
    """
    try:
        df = pd.read_csv(file_path)
        
        # Load model
        model = joblib.load('models/churn_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Prepare features
        customer_ids = df['customerID'] if 'customerID' in df.columns else df.index
        
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Make predictions
        predictions = model.predict(df[feature_names])
        probabilities = model.predict_proba(df[feature_names])[:, 1]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'CustomerID': customer_ids,
            'ChurnProbability': probabilities,
            'ChurnPrediction': predictions,
            'RiskLevel': pd.cut(probabilities, 
                               bins=[0, 0.4, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High'])
        })
        
        # Save predictions
        os.makedirs('outputs', exist_ok=True)
        results_df.to_csv('outputs/predictions.csv', index=False)
        
        # Summary statistics
        high_risk = (results_df['RiskLevel'] == 'High').sum()
        medium_risk = (results_df['RiskLevel'] == 'Medium').sum()
        low_risk = (results_df['RiskLevel'] == 'Low').sum()
        
        summary = f"""
Predictions Completed!

Total customers analyzed: {len(results_df)}

Risk Distribution:
- High Risk (>70%):    {high_risk} customers ({high_risk/len(results_df)*100:.1f}%)
- Medium Risk (40-70%): {medium_risk} customers ({medium_risk/len(results_df)*100:.1f}%)
- Low Risk (<40%):     {low_risk} customers ({low_risk/len(results_df)*100:.1f}%)

Average churn probability: {probabilities.mean():.2%}

Top 5 highest risk customers:
{results_df.nlargest(5, 'ChurnProbability').to_string()}

Predictions saved to: outputs/predictions.csv
"""
        return summary
        
    except Exception as e:
        return f"Error making predictions: {str(e)}"
