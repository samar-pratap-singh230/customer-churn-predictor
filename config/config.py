import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")

# Data paths
DATA_PATH = "data/customer_data.csv"
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_PATH = "outputs/"

# Model Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Churn thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Business metrics
CUSTOMER_LIFETIME_VALUE = 1000  # Average CLV in dollars

INTERVENTION_COST = 100  # Cost per retention intervention
