import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
# Try to import matplotlib and seaborn with error handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
import os
import re

# Set page config
st.set_page_config(
    page_title="PacWest Speeding Violations Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look like shadcn/ui
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #ffffff;
        color: #020817;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #020817;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        font-weight: 600;
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #020817;
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1a1a1a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #020817;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
    }
    
    /* Alert for high violations */
    .high-violations {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.375rem;
    }
    
    /* Card container */
    .card-container {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("PacWest Speeding Violations Dashboard")
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="font-size: 1.1rem; color: #64748b;">
        Comprehensive analysis of fleet vehicle speeding data updated through March 6, 2025.
        Vehicles with 40+ violations are above allowed threshold.
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load the specific file
        df = pd.read_excel("Excessive Speeding 3.6.25.xlsx")
        
        # Basic data cleaning and preparation
        # Convert duration to minutes for analysis if it exists
        if 'Total Duration hh:mm:ss' in df.columns:
            # Convert duration strings to minutes
            def duration_to_minutes(duration_str):
                try:
                    if pd.isna(duration_str):
                        return 0
                    # Check if it's already a timedelta
                    if isinstance(duration_str, timedelta):
                        return duration_str.total_seconds() / 60
                    
                    # Handle different formats
                    if ':' in str(duration_str):
                        parts = str(duration_str).split(':')
                        if len(parts) == 3:  # hh:mm:ss
                            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                        elif len(parts) == 2:  # mm:ss
                            return int(parts[0]) + int(parts[1]) / 60
                    return 0
                except:
                    return 0
            
            df['Duration_Minutes'] = df['Total Duration hh:mm:ss'].apply(duration_to_minutes)
        
        # Create a vehicle identifier column
        if 'Unit Number' in df.columns:
            df['Vehicle'] = df['Unit Number']
        elif 'CVN' in df.columns:
            df['Vehicle'] = df['CVN']
        
        # Create a driver column if it exists
        if 'Driver/Pool' in df.columns:
            df['Driver'] = df['Driver/Pool']
            # Identify pool and crew vehicles
            df['isPool'] = df['Driver/Pool'].str.contains('POOL', case=False, na=False)
            df['isCrew'] = df['Driver/Pool'].str.contains('Crew', case=False, na=False)
            df['Type'] = 'Individual'
            df.loc[df['isPool'], 'Type'] = 'Pool'
            df.loc[df['isCrew'], 'Type'] = 'Crew'
            
            # Create a combined identifier for display
            if 'Vehicle' in df.columns:
                df['Display_ID'] = df.apply(lambda row: f"{row['Vehicle']} - {row['Driver']}", axis=1)
            else:
                df['Display_ID'] = df['Driver']
        else:
            df['Display_ID'] = df['Vehicle']
        
        # Create a violation count column by vehicle
        if 'Vehicle' in df.columns:
            # Instead of using value_counts which just counts rows, we need to sum the actual events
            # First check if we have a 'Number Of Events' column
            if 'Number Of Events' in df.columns:
                # Group by Vehicle and sum the Number Of Events
                violation_counts = df.groupby('Vehicle')['Number Of Events'].sum().reset_index()
                violation_counts.columns = ['Vehicle', 'violation_count']
            else:
                # If no Number Of Events column, fall back to counting rows
                violation_counts = df['Vehicle'].value_counts().reset_index()
                violation_counts.columns = ['Vehicle', 'violation_count']
            
            df = df.merge(violation_counts, on='Vehicle', how='left')
        
        # Calculate average speed if possible
        if 'Total Distance' in df.columns and 'Duration_Minutes' in df.columns:
            df['Avg_Speed'] = df.apply(
                lambda row: (row['Total Distance'] / (row['Duration_Minutes'] / 60)) 
                if row['Duration_Minutes'] > 0 else 0, 
                axis=1
            )
        
        # Calculate events per mile
        if 'Total Distance' in df.columns and 'Number Of Events' in df.columns:
            df['Events_Per_Mile'] = df.apply(
                lambda row: (row['Number Of Events'] / row['Total Distance']) 
                if row['Total Distance'] > 0 else 0, 
                axis=1
            )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Please ensure the Excel file is accessible and in the expected format.")
        return None

# Load the data
df = load_data()

if df is not None:
    # Create metrics row
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_violations = len(df)
        st.metric("Total Violations", f"{total_violations:,}")
    
    with col2:
        if 'Vehicle' in df.columns:
            unique_vehicles = df['Vehicle'].nunique()
            st.metric("Unique Vehicles", f"{unique_vehicles:,}")
    
    with col3:
        if 'Number Of Events' in df.columns:
            total_events = df['Number Of Events'].sum()
            st.metric("Total Events", f"{total_events:,}")
    
    with col4:
        if 'Total Distance' in df.columns:
            total_distance = df['Total Distance'].sum()
            distance_unit = df['Total Distance Unit'].iloc[0] if 'Total Distance Unit' in df.columns else 'units'
            st.metric("Total Distance", f"{total_distance:,.1f} {distance_unit}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tabs = st.tabs(["Top Offenders", "High Risk Drivers", "By Driver Type", "Highest Speeds", "Correlation Analysis"])
    
    with tabs[0]:  # Top Offenders tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Top 40 Offenders by Number of Speeding Events")
        
        # Vehicles requiring corrective action (40+ violations)
        if 'Vehicle' in df.columns and 'violation_count' in df.columns:
            high_violation_vehicles = df.drop_duplicates(subset=['Vehicle'])[
                df['violation_count'] >= 40
            ].sort_values('violation_count', ascending=False)
            
            if not high_violation_vehicles.empty:
                st.markdown(f"""
                <div class="high-violations">
                    <h4>⚠️ {len(high_violation_vehicles)} vehicles require corrective action (40+ violations)</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Top 40 vehicles with most violations (changed from 20)
            top_vehicles = df.drop_duplicates(subset=['Vehicle']).nlargest(40, 'violation_count')
            
            # Create a bar chart with data labels using plotly graph objects
            fig = go.Figure()
            
            # Add the bar trace
            fig.add_trace(go.Bar(
                x=top_vehicles['Display_ID'],
                y=top_vehicles['violation_count'],
                marker=dict(
                    color=top_vehicles['violation_count'],
                    colorscale='Blues_r',
                ),
                text=top_vehicles['violation_count'],
                textposition='outside',
                textfont=dict(size=12),
            ))
            
            # Update layout
            fig.update_layout(
                title="Top 40 Vehicles by Number of Speeding Events",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': '#020817'},
                margin=dict(t=40, b=40, l=40, r=40),
                xaxis_tickangle=-45,
                xaxis_title="",
                yaxis_title="Number of Speeding Events",
                height=600  # Increased height to accommodate more bars
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about top offenders
            if not top_vehicles.empty:
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This chart shows the drivers and pools with the highest number of speeding events. 
                        {top_vehicles.iloc[0]['Display_ID']} leads with {int(top_vehicles.iloc[0]['violation_count'])} events, followed by 
                        {top_vehicles.iloc[1]['Display_ID']} with {int(top_vehicles.iloc[1]['violation_count'])} events.
                        The threshold for corrective action is 40 violations.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:  # High Risk Drivers tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Highest Risk Drivers (Events per Mile)")
        
        if 'Events_Per_Mile' in df.columns and 'Total Distance' in df.columns:
            # Filter for vehicles with at least 10 miles driven
            high_risk_drivers = df[df['Total Distance'] >= 10].drop_duplicates(subset=['Vehicle'])
            high_risk_drivers = high_risk_drivers.nlargest(15, 'Events_Per_Mile')
            
            if not high_risk_drivers.empty:
                # Create a bar chart with data labels using plotly graph objects
                fig = go.Figure()
                
                # Add the bar trace
                fig.add_trace(go.Bar(
                    x=high_risk_drivers['Display_ID'],
                    y=high_risk_drivers['Events_Per_Mile'],
                    marker=dict(
                        color=high_risk_drivers['Events_Per_Mile'],
                        colorscale='Oranges_r',
                    ),
                    text=[f"{x:.2f}" for x in high_risk_drivers['Events_Per_Mile']],
                    textposition='outside',
                    textfont=dict(size=12),
                ))
                
                # Update layout
                fig.update_layout(
                    title="Highest Risk Drivers (Events per Mile, min 10 miles driven)",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': '#020817'},
                    margin=dict(t=40, b=40, l=40, r=40),
                    xaxis_tickangle=-45,
                    xaxis_title="",
                    yaxis_title="Events per Mile",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insights about high risk drivers
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This chart shows drivers with the highest frequency of speeding events per mile driven.
                        These drivers represent the highest risk based on violation density rather than absolute counts.
                        {high_risk_drivers.iloc[0]['Display_ID']} has {high_risk_drivers.iloc[0]['Events_Per_Mile']:.2f} events per mile.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No vehicles with sufficient mileage (10+ miles) to calculate reliable risk metrics.")
        else:
            st.info("Required data for risk analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:  # By Driver Type tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Speeding Events by Driver Type")
        
        if 'Type' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Count events by type
                type_counts = df.groupby('Type')['Number Of Events'].sum().reset_index()
                type_counts.columns = ['Type', 'Event Count']
                
                fig = px.pie(
                    type_counts,
                    values='Event Count',
                    names='Type',
                    title="Distribution of Speeding Events by Driver Type",
                    color='Type',
                    color_discrete_map={
                        'Individual': '#0088FE',
                        'Pool': '#00C49F',
                        'Crew': '#FFBB28'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': '#020817'},
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                total_events = type_counts['Event Count'].sum()
                
                st.markdown("""
                <h3 class="text-lg font-medium mb-2">Key Insights:</h3>
                <ul class="list-disc pl-5 space-y-2">
                """, unsafe_allow_html=True)
                
                for idx, row in type_counts.iterrows():
                    percentage = (row['Event Count'] / total_events) * 100
                    st.markdown(f"""
                    <li>
                        <strong>{row['Type']} Drivers:</strong> {int(row['Event Count'])} events ({percentage:.1f}% of all violations)
                    </li>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <li>
                        <strong>Total Events:</strong> {int(total_events)}
                    </li>
                </ul>
                <p class="mt-4 text-sm text-gray-600">
                    This breakdown helps identify whether violations are primarily occurring with 
                    individual drivers or shared vehicles, which guides different intervention strategies.
                </p>
                """, unsafe_allow_html=True)
        else:
            st.info("Driver type information is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:  # Highest Speeds tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Drivers with Highest Average Speeds")
        
        if 'Avg_Speed' in df.columns and 'Total Distance' in df.columns:
            # Filter for vehicles with at least 10 miles driven
            top_speeders = df[df['Total Distance'] >= 10].drop_duplicates(subset=['Vehicle'])
            top_speeders = top_speeders.nlargest(15, 'Avg_Speed')
            
            if not top_speeders.empty:
                # Create a bar chart with data labels using plotly graph objects
                fig = go.Figure()
                
                # Add the bar trace
                fig.add_trace(go.Bar(
                    x=top_speeders['Display_ID'],
                    y=top_speeders['Avg_Speed'],
                    marker=dict(
                        color=top_speeders['Avg_Speed'],
                        colorscale='Greens_r',
                    ),
                    text=[f"{x:.1f}" for x in top_speeders['Avg_Speed']],
                    textposition='outside',
                    textfont=dict(size=12),
                ))
                
                # Update layout
                fig.update_layout(
                    title="Drivers with Highest Average Speeds (min 10 miles driven)",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': '#020817'},
                    margin=dict(t=40, b=40, l=40, r=40),
                    xaxis_tickangle=-45,
                    xaxis_title="",
                    yaxis_title=f"Average Speed ({distance_unit}/hour)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insights about top speeders
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This chart displays drivers with the highest average speeds. Average speed is calculated 
                        from total distance divided by total drive time. The highest average speed is 
                        {top_speeders.iloc[0]['Avg_Speed']:.1f} {distance_unit}/hour by {top_speeders.iloc[0]['Display_ID']}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No vehicles with sufficient mileage (10+ miles) to calculate reliable speed metrics.")
        else:
            st.info("Required data for speed analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[4]:  # Correlation Analysis tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Correlation: Distance Driven vs. Speeding Events")
        
        if 'Total Distance' in df.columns and 'Number Of Events' in df.columns:
            # Create scatter plot data
            scatter_data = df.drop_duplicates(subset=['Vehicle'])
            
            fig = px.scatter(
                scatter_data,
                x='Total Distance',
                y='Number Of Events',
                size='Number Of Events',
                color='Type',
                hover_name='Display_ID',
                size_max=50,
                title="Relationship Between Distance Driven and Number of Speeding Events",
                labels={
                    'Total Distance': f'Total Distance ({distance_unit})',
                    'Number Of Events': 'Number of Speeding Events'
                }
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': '#020817'},
                margin=dict(t=40, b=40, l=40, r=40),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about correlation
            st.markdown("""
            <div style="margin-top: 1rem;">
                <p class="text-sm text-gray-600">
                    This scatter plot shows the relationship between distance driven and number of speeding events.
                    Each point represents a driver or vehicle, and the size of the point corresponds to the number of events.
                    A clear positive correlation shows that more driving generally results in more violations, 
                    but outliers may indicate drivers who have disproportionately high violation rates.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Required data for correlation analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary and Recommendations
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("Summary and Recommendations")
    
    total_events = df['Number Of Events'].sum() if 'Number Of Events' in df.columns else len(df)
    unique_vehicles = df['Vehicle'].nunique() if 'Vehicle' in df.columns else 'N/A'
    
    top_offender = None
    top_offender_count = 0
    if 'Vehicle' in df.columns and 'violation_count' in df.columns:
        top_vehicle_data = df.drop_duplicates(subset=['Vehicle']).nlargest(1, 'violation_count')
        if not top_vehicle_data.empty:
            top_offender = top_vehicle_data.iloc[0]['Display_ID']
            top_offender_count = int(top_vehicle_data.iloc[0]['violation_count'])
    
    highest_risk = None
    highest_risk_rate = 0
    if 'Events_Per_Mile' in df.columns and 'Total Distance' in df.columns:
        high_risk_data = df[df['Total Distance'] >= 10].drop_duplicates(subset=['Vehicle']).nlargest(1, 'Events_Per_Mile')
        if not high_risk_data.empty:
            highest_risk = high_risk_data.iloc[0]['Display_ID']
            highest_risk_rate = high_risk_data.iloc[0]['Events_Per_Mile']
    
    st.markdown(f"""
    <ul class="list-disc pl-5 space-y-1">
        <li>Total of <strong>{total_events}</strong> speeding events recorded across <strong>{unique_vehicles}</strong> vehicles</li>
        {'<li>Top individual contributor: <strong>' + str(top_offender) + '</strong> with <strong>' + str(top_offender_count) + '</strong> events</li>' if top_offender else ''}
        {'<li>Highest risk driver (per mile): <strong>' + str(highest_risk) + '</strong> with <strong>' + f"{highest_risk_rate:.2f}" + '</strong> events per mile</li>' if highest_risk else ''}
        <li>Focus driver training on both high event count drivers and those with high events-per-mile ratios</li>
        <li>Consider specific intervention for pool vehicles if they show higher violation rates</li>
        <li><strong>Corrective action recommended</strong> for all vehicles with 40+ violations</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("Could not load the data. Please check if the Excel file is in the correct format and location.")
    
    # Placeholder for demo
    st.info("If you're having trouble loading your data, you can try running the app with the following command:")
    st.code("streamlit run app.py")
    
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
    <p style="color: #64748b; font-size: 0.875rem;">
        Speeding Violations Dashboard • Created by Mike Drake
    </p>
</div>
""", unsafe_allow_html=True)
