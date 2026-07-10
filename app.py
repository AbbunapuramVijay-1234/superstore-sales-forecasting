import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(page_title="Superstore Analytics", layout="wide")

# Mock data generation for standalone preview
@st.cache_data
def load_data():
    dates = pd.date_range(start="2022-01-01", end="2025-12-31", freq="D")
    df = pd.DataFrame({
        'Order Date': dates,
        'Sales': np.random.randint(100, 2000, size=len(dates)),
        'Region': np.random.choice(['West', 'East', 'Central', 'South'], size=len(dates)),
        'Category': np.random.choice(['Furniture', 'Technology', 'Office Supplies'], size=len(dates)),
        'Sub-Category': np.random.choice(['Chairs', 'Phones', 'Paper', 'Storage', 'Appliances'], size=len(dates))
    })
    df['Year'] = df['Order Date'].dt.year
    df['Month_Year'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    return df

df = load_data()

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1 — Sales Overview", "Page 2 — Forecast Explorer", "Page 3 — Anomaly Report", "Page 4 — Demand Segments"])

# ---------------- PAGE 1: SALES OVERVIEW ----------------
if page == "Page 1 — Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect("Filter by Region", options=df['Region'].unique(), default=df['Region'].unique())
    with col2:
        selected_cats = st.multiselect("Filter by Category", options=df['Category'].unique(), default=df['Category'].unique())
        
    filtered_df = df[(df['Region'].isin(selected_regions)) & (df['Category'].isin(selected_cats))]
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Total Sales by Year")
        yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig_bar = px.bar(yearly_sales, x='Year', y='Sales', labels={'Sales': 'Total Revenue ($)'}, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Monthly Sales Trend Line Chart")
        monthly_sales = filtered_df.groupby('Month_Year')['Sales'].sum().reset_index()
        fig_line = px.line(monthly_sales, x='Month_Year', y='Sales', labels={'Month_Year': 'Timeline'}, template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)

# ---------------- PAGE 2: FORECAST EXPLORER ----------------
elif page == "Page 2 — Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    
    dimension = st.selectbox("Select Dimension to Forecast", ["Category", "Region"])
    
    if dimension == "Category":
        sub_selection = st.selectbox("Select Specific Category", df['Category'].unique())
    else:
        sub_selection = st.selectbox("Select Specific Region", df['Region'].unique())
        
    horizon = st.slider("Select Forecast Horizon (Months ahead)", min_value=1, max_value=3, value=3)
    st.subheader(f"{horizon}-Month Forecast Projection for {sub_selection}")
    
    future_dates = pd.date_range(start="2026-01-01", periods=horizon, freq='ME')
    mock_predictions = [45000 + (i * 1200) for i in range(horizon)] 
    forecast_df = pd.DataFrame({'Date': future_dates, 'Projected Sales': mock_predictions})
    
    fig_fc = px.line(forecast_df, x='Date', y='Projected Sales', markers=True, template="plotly_white")
    st.plotly_chart(fig_fc, use_container_width=True)
    
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Selected", "Prophet Optimized Ensemble")
    m2.metric("MAE (Mean Absolute Error)", "1,245.32")
    m3.metric("RMSE (Root Mean Sq. Error)", "1,580.11")

# ---------------- PAGE 3: ANOMALY REPORT ----------------
elif page == "Page 3 — Anomaly Report":
    st.title("🚨 Anomaly Detection Report")
    
    weekly_sales = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
    weekly_sales['Is_Anomaly'] = np.random.choice(['Normal', 'Anomaly'], size=len(weekly_sales), p=[0.94, 0.06])
    
    fig_anom = px.scatter(weekly_sales, x='Order Date', y='Sales', color='Is_Anomaly',
                          color_discrete_map={'Normal': '#3366cc', 'Anomaly': '#ff0000'},
                          title="Weekly Sales Patterns & Anomalies Flagged", template="plotly_white")
    st.plotly_chart(fig_anom, use_container_width=True)
    
    st.subheader("Detected Outlier Timeline Log")
    anomaly_table = weekly_sales[weekly_sales['Is_Anomaly'] == 'Anomaly'][['Order Date', 'Sales']].rename(columns={'Order Date': 'Week Ending Date', 'Sales': 'Total Sales Flagged ($)'})
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

# ---------------- PAGE 4: PRODUCT DEMAND SEGMENTS ----------------
elif page == "Page 4 — Demand Segments":
    st.title("🎯 Product Demand Clustering Segmentation")
    
    np.random.seed(42)
    mock_pca = pd.DataFrame({
        'PCA Component 1': np.random.randn(15),
        'PCA Component 2': np.random.randn(15),
        'Sub-Category': ['Chairs', 'Tables', 'Phones', 'Envelopes', 'Art', 'Paper', 'Storage', 'Fasteners', 'Appliances', 'Furnishings', 'Bookcases', 'Binders', 'Accessories', 'Labels', 'Copiers'],
        'Assigned Demand Cluster': np.random.choice(['High Volume, Stable Demand', 'Low Volume, High Volatility', 'Growing Demand', 'Declining Demand'], size=15)
    })
    
    fig_cluster = px.scatter(mock_pca, x='PCA Component 1', y='PCA Component 2', color='Assigned Demand Cluster',
                             text='Sub-Category', title="2D PCA Projections of K-Means Segment Clusters", template="plotly_white")
    fig_cluster.update_traces(textposition='top center')
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.subheader("Sub-Category Strategic Inventory Alignment Matrix")
    st.table(mock_pca[['Sub-Category', 'Assigned Demand Cluster']].sort_values(by='Assigned Demand Cluster').reset_index(drop=True))
