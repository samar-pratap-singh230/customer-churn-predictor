from flask import Flask, render_template, request, jsonify, send_file, session
import os
import pandas as pd
import threading
import uuid
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Import tools
from tools.data_tools import data_load_tool, data_clean_tool
from tools.feature_tools import feature_engineer_tool, feature_scale_tool
from tools.model_tools import model_train_tool, model_predict_tool
from tools.strategy_tools import retention_strategy_tool, roi_calculator_tool

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Store for tracking job status
job_status = {}

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)


def initialize_crew():
    """Initialize the CrewAI agents and tasks"""
    llm = LLM(model="groq/llama-3.3-70b-versatile")
    
    data_analyst = Agent(
        role='Data Analyst',
        goal='Load, explore, and clean customer data to prepare it for analysis',
        backstory="""You are an expert data analyst with 10 years of experience in 
        customer analytics. You have a keen eye for data quality issues.""",
        verbose=True,
        allow_delegation=False,
        tools=[data_load_tool, data_clean_tool],
        llm=llm
    )
    
    feature_engineer = Agent(
        role='Feature Engineering Specialist',
        goal='Create and transform features that maximize predictive power',
        backstory="""You are a machine learning feature engineering expert who 
        specializes in extracting meaningful patterns from raw data.""",
        verbose=True,
        allow_delegation=False,
        tools=[feature_engineer_tool, feature_scale_tool],
        llm=llm
    )
    
    model_trainer = Agent(
        role='Machine Learning Engineer',
        goal='Train and optimize predictive models for maximum accuracy',
        backstory="""You are a senior ML engineer with expertise in classification 
        problems. You've built churn prediction models for Fortune 500 companies.""",
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
    
    task1 = Task(
        description="""Load and analyze the customer data from 'data/customer_data.csv'.
        Perform initial exploration and clean the data by handling missing values.""",
        agent=data_analyst,
        expected_output="A detailed report on data quality and a cleaned dataset"
    )
    
    task2 = Task(
        description="""Engineer features from the cleaned customer data at 'data/customer_data_cleaned.csv'. 
        Create new features and scale them for optimal model performance.""",
        agent=feature_engineer,
        expected_output="Engineered and scaled features ready for modeling"
    )
    
    task3 = Task(
        description="""Train multiple machine learning models on the prepared data at 
        'data/customer_data_scaled.csv'. Evaluate and select the best performing model.""",
        agent=model_trainer,
        expected_output="A trained model with performance metrics"
    )
    
    task4 = Task(
        description="""Use the trained model to score all customers. Calculate churn 
        probabilities and segment customers into risk levels.""",
        agent=risk_scorer,
        expected_output="Customer risk scores and segmentation"
    )
    
    task5 = Task(
        description="""Based on the risk scores in 'outputs/predictions.csv', create 
        personalized retention strategies. Calculate the expected ROI.""",
        agent=strategy_advisor,
        expected_output="Comprehensive retention strategy with ROI analysis"
    )
    
    crew = Crew(
        agents=[data_analyst, feature_engineer, model_trainer, risk_scorer, strategy_advisor],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=False
    )
    
    return crew


def run_prediction_job(job_id):
    """Run the prediction job in background"""
    try:
        job_status[job_id] = {'status': 'running', 'progress': 0, 'message': 'Initializing...'}
        
        job_status[job_id] = {'status': 'running', 'progress': 20, 'message': 'Loading data...'}
        
        crew = initialize_crew()
        
        job_status[job_id] = {'status': 'running', 'progress': 40, 'message': 'Running AI agents...'}
        
        result = crew.kickoff()
        
        job_status[job_id] = {'status': 'running', 'progress': 90, 'message': 'Finalizing results...'}
        
        # Load results
        predictions_df = pd.read_csv('outputs/predictions.csv')
        strategies_df = pd.read_csv('outputs/retention_strategies.csv')
        roi_df = pd.read_csv('outputs/roi_analysis.csv')
        
        summary = {
            'total_customers': len(predictions_df),
            'high_risk': int((predictions_df['RiskLevel'] == 'High').sum()),
            'medium_risk': int((predictions_df['RiskLevel'] == 'Medium').sum()),
            'low_risk': int((predictions_df['RiskLevel'] == 'Low').sum()),
            'avg_churn_prob': float(predictions_df['ChurnProbability'].mean()),
            'total_cost': float(roi_df['TotalCost'].sum()),
            'expected_revenue': float(roi_df['ExpectedRevenue'].sum()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        job_status[job_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Analysis completed successfully!',
            'summary': summary
        }
        
    except Exception as e:
        job_status[job_id] = {
            'status': 'failed',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save the uploaded file
        file.save('data/customer_data.csv')
        
        # Quick validation
        df = pd.read_csv('data/customer_data.csv')
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully! {len(df)} rows, {len(df.columns)} columns',
            'rows': len(df),
            'columns': len(df.columns)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    """Start analysis job"""
    try:
        if not os.path.exists('data/customer_data.csv'):
            return jsonify({'error': 'Please upload a dataset first'}), 400
        
        job_id = str(uuid.uuid4())
        
        # Start background job
        thread = threading.Thread(target=run_prediction_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Analysis started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_status[job_id])


@app.route('/results')
def results():
    """Results page"""
    try:
        if not os.path.exists('outputs/predictions.csv'):
            return render_template('results.html', error='No results available yet')
        
        predictions_df = pd.read_csv('outputs/predictions.csv')
        strategies_df = pd.read_csv('outputs/retention_strategies.csv')
        roi_df = pd.read_csv('outputs/roi_analysis.csv')
        
        # Calculate summary statistics
        summary = {
            'total_customers': len(predictions_df),
            'high_risk': int((predictions_df['RiskLevel'] == 'High').sum()),
            'medium_risk': int((predictions_df['RiskLevel'] == 'Medium').sum()),
            'low_risk': int((predictions_df['RiskLevel'] == 'Low').sum()),
            'avg_churn_prob': float(predictions_df['ChurnProbability'].mean()),
            'total_cost': float(roi_df['TotalCost'].sum()),
            'expected_revenue': float(roi_df['ExpectedRevenue'].sum())
        }
        
        # Top 10 high-risk customers
        high_risk_customers = predictions_df[predictions_df['RiskLevel'] == 'High'].nlargest(10, 'ChurnProbability')
        
        return render_template('results.html',
                             summary=summary,
                             high_risk_customers=high_risk_customers.to_dict('records'),
                             roi_data=roi_df.to_dict('records'))
        
    except Exception as e:
        return render_template('results.html', error=str(e))


@app.route('/download/<file_type>')
def download(file_type):
    """Download result files"""
    try:
        file_map = {
            'predictions': 'outputs/predictions.csv',
            'strategies': 'outputs/retention_strategies.csv',
            'roi': 'outputs/roi_analysis.csv'
        }
        
        if file_type not in file_map:
            return jsonify({'error': 'Invalid file type'}), 400
        
        file_path = file_map[file_type]
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)