# Customer Churn Prediction System

An AI-powered customer churn prediction system using **CrewAI**, **GROQ API**, and machine learning to predict customer churn and recommend retention strategies.

## 🎯 Features

- **Multi-Agent System**: 5 specialized AI agents working together
- **Automated ML Pipeline**: Data cleaning, feature engineering, model training
- **Risk Scoring**: Segments customers into High, Medium, and Low risk
- **Retention Strategies**: Personalized recommendations for each customer
- **ROI Analysis**: Calculates expected return on retention investments
- **Interactive Dashboard**: Streamlit-based visualization

## 🏗️ Project Structure

```
customer-churn-predictor/
├── agents/              # (Future: Custom agent definitions)
├── data/                # Customer data files
│   └── customer_data.csv
├── models/              # Trained models and encoders
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── outputs/             # Predictions and reports
│   ├── predictions.csv
│   ├── retention_strategies.csv
│   └── roi_analysis.csv
├── tools/               # Custom CrewAI tools
│   ├── data_tools.py
│   ├── feature_tools.py
│   ├── model_tools.py
│   └── strategy_tools.py
├── config/              # Configuration files
│   └── config.py
├── notebooks/           # Jupyter notebooks for EDA
├── .env                 # Environment variables (not in git)
├── .env.template        # Template for environment variables
├── main.py              # Main execution script
├── dashboard.py         # Streamlit dashboard
├── requirements.txt     # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- GROQ API key ([Get one here](https://console.groq.com))
- Customer churn dataset (CSV format)

### Installation

1. **Clone or create the project directory**:
```bash
mkdir customer-churn-predictor
cd customer-churn-predictor
```

2. **Create a virtual environment**:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# Copy the template
cp .env.template .env

# Edit .env and add your GROQ API key
GROQ_API_KEY=your_actual_groq_api_key_here
MODEL_NAME=mixtral-8x7b-32768
```

5. **Add your dataset**:
- Download a customer churn dataset (recommended: [Telco Customer Churn from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- Place the CSV file in the `data/` folder
- Rename it to `customer_data.csv`

### Running the System

1. **Execute the churn prediction workflow**:
```bash
python main.py
```

This will:
- Load and clean the data
- Engineer features
- Train multiple ML models
- Generate predictions
- Create retention strategies
- Calculate ROI

2. **Launch the dashboard**:
```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🤖 The Five Agents

### 1. Data Analyst Agent
- **Role**: Data exploration and cleaning
- **Tools**: DataLoadTool, DataCleanTool
- **Output**: Clean, validated dataset

### 2. Feature Engineer Agent
- **Role**: Feature creation and transformation
- **Tools**: FeatureEngineerTool, FeatureScaleTool
- **Output**: Engineered features ready for ML

### 3. Model Trainer Agent
- **Role**: ML model training and selection
- **Tools**: ModelTrainTool
- **Output**: Best performing trained model

### 4. Risk Scorer Agent
- **Role**: Customer risk assessment
- **Tools**: ModelPredictTool
- **Output**: Risk scores and segmentation

### 5. Strategy Advisor Agent
- **Role**: Retention strategy development
- **Tools**: RetentionStrategyTool, ROICalculatorTool
- **Output**: Personalized retention plans with ROI

## 📊 Expected Outputs

After running `main.py`, you'll get:

1. **predictions.csv**: Customer IDs with churn probabilities and risk levels
2. **retention_strategies.csv**: Recommended actions for each customer
3. **roi_analysis.csv**: Financial analysis of retention program

## 🎨 Dashboard Features

The Streamlit dashboard provides:
- **Overview Metrics**: Total customers, high-risk count, average churn probability
- **Risk Distribution**: Visual breakdown of customer risk levels
- **Churn Probability Distribution**: Histogram of predicted probabilities
- **ROI Analysis**: Cost vs. revenue by risk level
- **High-Risk Customers Table**: Immediate action list
- **Retention Strategies**: Recommended actions by risk level

## 🔧 Customization

### Adjust Business Metrics

Edit `config/config.py`:
```python
CUSTOMER_LIFETIME_VALUE = 1000  # Your average CLV
INTERVENTION_COST = 100         # Your intervention cost
HIGH_RISK_THRESHOLD = 0.7       # Adjust risk thresholds
```

### Use Different Models

Modify `tools/model_tools.py` to add more algorithms:
```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),  # Add this
    # Add more models...
}
```

### Add More Features

Extend `tools/feature_tools.py` to create custom features based on your domain knowledge.

## 📈 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to churners
- **F1-Score**: Balanced metric
- **ROC-AUC**: Discrimination ability

## 🐛 Troubleshooting

### "API key not found"
- Make sure `.env` file exists in project root
- Check that `GROQ_API_KEY` is set correctly

### "Dataset not found"
- Ensure `customer_data.csv` is in the `data/` folder
- Check file name is exactly `customer_data.csv`

### "Module not found"
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Poor model performance
- Check data quality (missing values, outliers)
- Add more relevant features
- Try different algorithms
- Tune hyperparameters

## 📚 Dataset Requirements

Your dataset should include:
- **Customer ID**: Unique identifier
- **Demographics**: Age, gender, location, etc.
- **Service Usage**: Features about product/service usage
- **Account Info**: Tenure, contract type, payment method
- **Charges**: Monthly charges, total charges
- **Target**: Churn status (Yes/No or 1/0)

## 🤝 Contributing

Feel free to:
- Add new agents for specialized tasks
- Implement additional ML algorithms
- Enhance the dashboard with more visualizations
- Add new retention strategy types

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- **CrewAI**: For the multi-agent framework
- **GROQ**: For fast LLM inference
- **Scikit-learn**: For ML algorithms
- **Streamlit**: For the interactive dashboard

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments
3. Consult CrewAI documentation: https://docs.crewai.com
4. Check GROQ documentation: https://console.groq.com/docs

---

**Built with ❤️ using CrewAI and GROQ**