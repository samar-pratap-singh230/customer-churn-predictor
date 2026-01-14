from crewai.tools import tool
import pandas as pd
import os


@tool("Generate retention strategies for at-risk customers")
def retention_strategy_tool(predictions_file: str) -> str:
    """Generates personalized retention strategies for at-risk customers.
    
    Args:
        predictions_file: Path to predictions CSV file
    """
    try:
        df = pd.read_csv(predictions_file)
        
        strategies = []
        
        for _, row in df.iterrows():
            customer_id = row['CustomerID']
            probability = row['ChurnProbability']
            risk_level = row['RiskLevel']
            
            if risk_level == 'High':
                strategy = {
                    'CustomerID': customer_id,
                    'RiskLevel': risk_level,
                    'ChurnProbability': f"{probability:.2%}",
                    'Priority': 'URGENT',
                    'RecommendedActions': 'Personal call, 30% discount, premium features, executive meeting',
                    'EstimatedCost': '$300',
                    'ExpectedRetentionRate': '60%',
                    'ROI': 'High (CLV: $1000 vs Cost: $300)'
                }
            elif risk_level == 'Medium':
                strategy = {
                    'CustomerID': customer_id,
                    'RiskLevel': risk_level,
                    'ChurnProbability': f"{probability:.2%}",
                    'Priority': 'High',
                    'RecommendedActions': 'Email campaign, 15% discount, survey, training',
                    'EstimatedCost': '$100',
                    'ExpectedRetentionRate': '45%',
                    'ROI': 'Positive (CLV: $1000 vs Cost: $100)'
                }
            else:
                strategy = {
                    'CustomerID': customer_id,
                    'RiskLevel': risk_level,
                    'ChurnProbability': f"{probability:.2%}",
                    'Priority': 'Monitor',
                    'RecommendedActions': 'Newsletter, satisfaction survey, standard engagement',
                    'EstimatedCost': '$20',
                    'ExpectedRetentionRate': '90%',
                    'ROI': 'Very High'
                }
            
            strategies.append(strategy)
        
        # Create strategies dataframe
        strategies_df = pd.DataFrame(strategies)
        strategies_df.to_csv('outputs/retention_strategies.csv', index=False)
        
        # Calculate overall metrics
        high_risk_count = (df['RiskLevel'] == 'High').sum()
        medium_risk_count = (df['RiskLevel'] == 'Medium').sum()
        
        total_intervention_cost = (high_risk_count * 300) + (medium_risk_count * 100)
        expected_saves = (high_risk_count * 0.6) + (medium_risk_count * 0.45)
        expected_revenue = expected_saves * 1000
        expected_roi = expected_revenue - total_intervention_cost
        
        summary = f"""
Retention Strategy Report Generated!

Total customers analyzed: {len(df)}

Action Summary:
- URGENT interventions (High Risk): {high_risk_count} customers
- High priority interventions (Medium Risk): {medium_risk_count} customers
- Monitor only (Low Risk): {(df['RiskLevel'] == 'Low').sum()} customers

Financial Analysis:
- Total intervention cost: ${total_intervention_cost:,.0f}
- Expected customers saved: {expected_saves:.0f}
- Expected revenue retained: ${expected_revenue:,.0f}
- Expected ROI: ${expected_roi:,.0f}
- ROI Percentage: {(expected_roi/total_intervention_cost)*100:.1f}%

Strategies saved to: outputs/retention_strategies.csv
"""
        return summary
        
    except Exception as e:
        return f"Error generating strategies: {str(e)}"


@tool("Calculate ROI for retention interventions")
def roi_calculator_tool(strategies_file: str) -> str:
    """Calculates detailed ROI for retention interventions.
    
    Args:
        strategies_file: Path to strategies CSV file
    """
    try:
        df = pd.read_csv(strategies_file)
        
        # Define costs and retention rates per risk level
        cost_mapping = {'High': 300, 'Medium': 100, 'Low': 20}
        retention_mapping = {'High': 0.60, 'Medium': 0.45, 'Low': 0.90}
        
        # Calculate for each risk level
        results = []
        for risk_level in ['High', 'Medium', 'Low']:
            customers = df[df['RiskLevel'] == risk_level]
            count = len(customers)
            
            if count > 0:
                cost_per_customer = cost_mapping[risk_level]
                total_cost = count * cost_per_customer
                retention_rate = retention_mapping[risk_level]
                expected_saves = count * retention_rate
                clv = 1000
                expected_revenue = expected_saves * clv
                net_benefit = expected_revenue - total_cost
                roi_percent = (net_benefit / total_cost) * 100
                
                results.append({
                    'RiskLevel': risk_level,
                    'Customers': count,
                    'CostPerCustomer': cost_per_customer,
                    'TotalCost': total_cost,
                    'RetentionRate': f"{retention_rate:.0%}",
                    'ExpectedSaves': f"{expected_saves:.1f}",
                    'ExpectedRevenue': expected_revenue,
                    'NetBenefit': net_benefit,
                    'ROI': f"{roi_percent:.1f}%"
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('outputs/roi_analysis.csv', index=False)
        
        # Overall calculations
        total_cost = results_df['TotalCost'].sum()
        total_revenue = results_df['ExpectedRevenue'].sum()
        total_benefit = total_revenue - total_cost
        overall_roi = (total_benefit / total_cost) * 100
        
        summary = f"""
ROI Analysis Completed!

Detailed Breakdown by Risk Level:
{results_df.to_string(index=False)}

Overall Business Impact:
- Total customers: {len(df)}
- Total intervention cost: ${total_cost:,.0f}
- Expected revenue retained: ${total_revenue:,.0f}
- Net benefit: ${total_benefit:,.0f}
- Overall ROI: {overall_roi:.1f}%

ROI analysis saved to: outputs/roi_analysis.csv
"""
        return summary
        
    except Exception as e:
        return f"Error calculating ROI: {str(e)}"
