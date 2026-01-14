import os
import sys
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# Import tools as functions (new format)
from tools.data_tools import data_load_tool, data_clean_tool
from tools.feature_tools import feature_engineer_tool, feature_scale_tool
from tools.model_tools import model_train_tool, model_predict_tool
from tools.strategy_tools import retention_strategy_tool, roi_calculator_tool

# Load environment variables
load_dotenv()

print("Loading environment variables...")
groq_api_key = os.getenv('GROQ_API_KEY')
if groq_api_key:
    print(f"API Key loaded: {groq_api_key[:15]}...")
else:
    print("ERROR: GROQ API Key not found!")
    sys.exit(1)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("Directories created/verified")

# Initialize LLM
print("Initializing GROQ LLM...")
try:
    # Use llama-3.3-70b-versatile (current supported model)
    llm = LLM(model="groq/llama-3.3-70b-versatile")
    print("✓ LLM initialized successfully with llama-3.3-70b-versatile")
except Exception as e:
    print(f"✗ Error initializing LLM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Define Agents
print("\nCreating AI agents...")

data_analyst = Agent(
    role='Data Analyst',
    goal='Load, explore, and clean customer data to prepare it for analysis',
    backstory="""You are an expert data analyst with 10 years of experience in 
    customer analytics. You have a keen eye for data quality issues and know how 
    to handle missing values, outliers, and data inconsistencies.""",
    verbose=True,
    allow_delegation=False,
    tools=[data_load_tool, data_clean_tool],
    llm=llm
)

feature_engineer = Agent(
    role='Feature Engineering Specialist',
    goal='Create and transform features that maximize predictive power',
    backstory="""You are a machine learning feature engineering expert who 
    specializes in extracting meaningful patterns from raw data. You understand 
    customer behavior deeply and know which features drive churn.""",
    verbose=True,
    allow_delegation=False,
    tools=[feature_engineer_tool, feature_scale_tool],
    llm=llm
)

model_trainer = Agent(
    role='Machine Learning Engineer',
    goal='Train and optimize predictive models for maximum accuracy',
    backstory="""You are a senior ML engineer with expertise in classification 
    problems. You've built churn prediction models for Fortune 500 companies 
    and know how to select the best algorithms.""",
    verbose=True,
    allow_delegation=False,
    tools=[model_train_tool, model_predict_tool],
    llm=llm
)

risk_scorer = Agent(
    role='Risk Assessment Specialist',
    goal='Score customers by churn risk and segment them appropriately',
    backstory="""You are a risk assessment expert who excels at translating 
    model predictions into actionable risk scores.""",
    verbose=True,
    allow_delegation=False,
    tools=[model_predict_tool],
    llm=llm
)

strategy_advisor = Agent(
    role='Customer Retention Strategist',
    goal='Design personalized retention strategies that maximize ROI',
    backstory="""You are a customer success strategist with 15 years of 
    experience in retention and loyalty programs.""",
    verbose=True,
    allow_delegation=False,
    tools=[retention_strategy_tool, roi_calculator_tool],
    llm=llm
)

print("✓ 5 agents created successfully")

# Define Tasks
print("\nDefining tasks...")

task1 = Task(
    description="""Load and analyze the customer data from 'data/customer_data.csv'.
    Perform initial exploration to understand the dataset structure, identify 
    data quality issues, and provide a summary of the dataset characteristics.
    Then clean the data by handling missing values and removing duplicates.""",
    agent=data_analyst,
    expected_output="A detailed report on data quality and a cleaned dataset"
)

task2 = Task(
    description="""Engineer features from the cleaned customer data at 'data/customer_data_cleaned.csv'. 
    Create new features that capture customer behavior patterns, encode categorical variables, 
    and scale numerical features for optimal model performance.""",
    agent=feature_engineer,
    expected_output="Engineered and scaled features ready for modeling"
)

task3 = Task(
    description="""Train multiple machine learning models (Logistic Regression, 
    Random Forest, Gradient Boosting) on the prepared data at 'data/customer_data_scaled.csv'. 
    Evaluate each model's performance and select the best performing model.""",
    agent=model_trainer,
    expected_output="A trained model with performance metrics and comparison report"
)

task4 = Task(
    description="""Use the trained model to score all customers in the dataset at 
    'data/customer_data_scaled.csv'. Calculate churn probabilities and segment customers 
    into risk levels (High, Medium, Low). Identify the highest risk customers who need 
    immediate attention.""",
    agent=risk_scorer,
    expected_output="Customer risk scores and segmentation with detailed analysis"
)

task5 = Task(
    description="""Based on the risk scores in 'outputs/predictions.csv', create personalized 
    retention strategies for each customer segment. Calculate the expected ROI of the 
    retention program including costs, expected saves, and net benefit.""",
    agent=strategy_advisor,
    expected_output="Comprehensive retention strategy with ROI analysis and action plan"
)

print("✓ 5 tasks defined")

# Create Crew
print("\nAssembling crew...")
crew = Crew(
    agents=[data_analyst, feature_engineer, model_trainer, risk_scorer, strategy_advisor],
    tasks=[task1, task2, task3, task4, task5],
    process=Process.sequential,
    verbose=True
)
print("✓ Crew assembled")


def main():
    """Main function to run the churn prediction system."""
    
    print("\n" + "=" * 80)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("Powered by CrewAI and GROQ")
    print("=" * 80)
    print()
    
    # Check if data file exists
    if not os.path.exists('data/customer_data.csv'):
        print("ERROR: data/customer_data.csv not found!")
        print("Please download a customer churn dataset and place it in the data/ folder.")
        return
    
    print("✓ Dataset found: data/customer_data.csv")
    print("\nStarting churn prediction workflow...")
    print("This may take 5-10 minutes depending on your internet speed...")
    print()
    
    try:
        # Execute the crew
        print("=" * 80)
        print("EXECUTING CREW WORKFLOW")
        print("=" * 80)
        result = crew.kickoff()
        
        print()
        print("=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Final Result:")
        print(result)
        print()
        print("Generated Files:")
        print("- data/customer_data_cleaned.csv")
        print("- data/customer_data_engineered.csv")
        print("- data/customer_data_scaled.csv")
        print("- models/churn_model.pkl")
        print("- models/scaler.pkl")
        print("- models/label_encoders.pkl")
        print("- outputs/predictions.csv")
        print("- outputs/retention_strategies.csv")
        print("- outputs/roi_analysis.csv")
        print()
        print("Next Steps:")
        print("1. Review the predictions in outputs/predictions.csv")
        print("2. Implement retention strategies from outputs/retention_strategies.csv")
        print("3. Monitor ROI using the analysis in outputs/roi_analysis.csv")
        print("4. Run the Streamlit dashboard: streamlit run dashboard.py")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")


if __name__ == "__main__":
    main()