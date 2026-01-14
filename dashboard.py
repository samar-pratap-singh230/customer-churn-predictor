import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**AI-Powered Customer Retention Analytics**")
st.markdown("---")

# Check if files exist
predictions_path = "outputs/predictions.csv"
strategies_path = "outputs/retention_strategies.csv"
roi_path = "outputs/roi_analysis.csv"

if not all([os.path.exists(p) for p in [predictions_path, strategies_path, roi_path]]):
    st.error("⚠️ Required data files not found. Please run main.py first!")
    st.info("Run: `python main.py` to generate predictions and strategies")
    st.stop()

# Load data
try:
    predictions_df = pd.read_csv(predictions_path)
    strategies_df = pd.read_csv(strategies_path)
    roi_df = pd.read_csv(roi_path)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=predictions_df['RiskLevel'].unique(),
    default=predictions_df['RiskLevel'].unique()
)

# Filter data
filtered_predictions = predictions_df[predictions_df['RiskLevel'].isin(risk_filter)]

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value=len(predictions_df)
    )

with col2:
    high_risk = (predictions_df['RiskLevel'] == 'High').sum()
    st.metric(
        label="High Risk Customers",
        value=high_risk,
        delta=f"{high_risk/len(predictions_df)*100:.1f}%"
    )

with col3:
    avg_churn_prob = predictions_df['ChurnProbability'].mean()
    st.metric(
        label="Avg Churn Probability",
        value=f"{avg_churn_prob:.1%}"
    )

with col4:
    if len(roi_df) > 0:
        total_cost = roi_df['TotalCost'].sum()
        st.metric(
            label="Total Intervention Cost",
            value=f"${total_cost:,.0f}"
        )

st.markdown("---")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution")
    
    # Risk distribution pie chart
    risk_counts = predictions_df['RiskLevel'].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Customers by Risk Level",
        color=risk_counts.index,
        color_discrete_map={'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc00'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Churn Probability Distribution")
    
    # Histogram of churn probabilities
    fig_hist = px.histogram(
        predictions_df,
        x='ChurnProbability',
        nbins=30,
        title="Distribution of Churn Probabilities",
        color='RiskLevel',
        color_discrete_map={'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc00'}
    )
    fig_hist.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ROI Analysis
st.subheader("💰 ROI Analysis")

col1, col2 = st.columns(2)

with col1:
    # ROI by risk level
    if len(roi_df) > 0:
        fig_roi = go.Figure()
        
        fig_roi.add_trace(go.Bar(
            name='Total Cost',
            x=roi_df['RiskLevel'],
            y=roi_df['TotalCost'],
            marker_color='#ff4b4b'
        ))
        
        fig_roi.add_trace(go.Bar(
            name='Expected Revenue',
            x=roi_df['RiskLevel'],
            y=roi_df['ExpectedRevenue'],
            marker_color='#00cc00'
        ))
        
        fig_roi.update_layout(
            title="Cost vs Expected Revenue by Risk Level",
            barmode='group',
            xaxis_title="Risk Level",
            yaxis_title="Amount ($)"
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)

with col2:
    # ROI metrics table
    if len(roi_df) > 0:
        st.dataframe(
            roi_df[['RiskLevel', 'Customers', 'TotalCost', 'ExpectedRevenue', 'ROI']],
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")

# High-risk customers table
st.subheader("🚨 High-Risk Customers (Immediate Action Required)")

high_risk_customers = predictions_df[predictions_df['RiskLevel'] == 'High'].sort_values(
    'ChurnProbability', ascending=False
)

if len(high_risk_customers) > 0:
    # Display top 10 high-risk customers
    display_df = high_risk_customers.head(10)[['CustomerID', 'ChurnProbability', 'RiskLevel']]
    display_df['ChurnProbability'] = display_df['ChurnProbability'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = high_risk_customers.to_csv(index=False)
    st.download_button(
        label="📥 Download All High-Risk Customers",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )
else:
    st.success("✅ No high-risk customers identified!")

st.markdown("---")

# Retention strategies
st.subheader("💡 Recommended Retention Strategies")

# Strategy summary by priority
if 'Priority' in strategies_df.columns:
    priority_counts = strategies_df['Priority'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        urgent = priority_counts.get('URGENT', 0)
        st.metric("URGENT Priority", urgent)
    
    with col2:
        high = priority_counts.get('High', 0)
        st.metric("High Priority", high)
    
    with col3:
        monitor = priority_counts.get('Monitor', 0)
        st.metric("Monitor Only", monitor)

# Sample strategies
st.write("**Sample Retention Actions:**")

# Display strategies for different risk levels
for risk_level in ['High', 'Medium', 'Low']:
    with st.expander(f"{risk_level} Risk Strategy"):
        risk_strategies = strategies_df[strategies_df['RiskLevel'] == risk_level].head(1)
        
        if len(risk_strategies) > 0:
            strategy = risk_strategies.iloc[0]
            st.write(f"**Priority:** {strategy.get('Priority', 'N/A')}")
            st.write(f"**Estimated Cost:** {strategy.get('EstimatedCost', 'N/A')}")
            st.write(f"**Expected Retention Rate:** {strategy.get('ExpectedRetentionRate', 'N/A')}")
            st.write(f"**ROI:** {strategy.get('ROI', 'N/A')}")
            
            if 'RecommendedActions' in strategy:
                st.write("**Recommended Actions:**")
                # Try to parse actions if they're in string format
                actions = str(strategy['RecommendedActions'])
                for action in actions.split(','):
                    st.write(f"- {action.strip()}")

st.markdown("---")

# Footer
st.markdown("### 📈 Next Steps")
st.info("""
1. **Immediate Actions**: Contact all URGENT priority customers within 24 hours
2. **Campaign Launch**: Set up automated retention campaigns for High priority customers
3. **Monitoring**: Set up alerts for customers moving into higher risk categories
4. **Review**: Analyze results weekly and adjust strategies based on outcomes
""")

# Refresh data button
if st.button("🔄 Refresh Data"):
    st.rerun()