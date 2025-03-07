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

# Set page config with dark mode support
st.set_page_config(
    page_title="Speeding Violations Dashboard (through March 6, 2025)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look like shadcn/ui with dark mode support
st.markdown("""
    <style>
    /* Dark mode compatibility */
    [data-testid="stAppViewContainer"] {
        color: var(--text-color);
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    
    [data-testid="stToolbar"] {
        right: 2rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        font-weight: 600;
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #4e8df5;
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
    }
    
    /* Alert for high violations */
    .high-violations {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.375rem;
    }
    
    /* Card container */
    .card-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Safe driver highlight */
    .safe-driver-highlight {
        background-color: rgba(56, 161, 105, 0.1);
        border-left: 4px solid #38a169;
        border-radius: 0.375rem;
        padding: 1rem;
    }
    
    /* Explanation text */
    .explanation-text {
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    /* Fix for plotly charts in dark mode */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Speeding Violations Dashboard")
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p class="explanation-text">
        Comprehensive analysis of fleet vehicle speeding data updated through March 6, 2025.
        Vehicles with 40+ violations require corrective action.
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
    tabs = st.tabs(["Top Offenders", "High Risk Drivers", "By Driver Type", "Highest Speeds", "Correlation Analysis", "Safe Drivers", "Full Data"])
    
    with tabs[0]:  # Top Offenders tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Top 40 Offenders by Number of Speeding Events")
        
        # Simple explanation for presentation
        st.markdown("""
        <p class="explanation-text">
            <strong>What this shows:</strong> The vehicles with the most speeding events, ranked from highest to lowest. 
            Vehicles with 40+ violations (shown in red alert) require corrective action.
        </p>
        """, unsafe_allow_html=True)
        
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
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#333333'},
                margin=dict(t=40, b=40, l=40, r=40),
                xaxis_tickangle=-45,
                xaxis_title="",
                yaxis_title="Number of Speeding Events",
                height=600  # Increased height to accommodate more bars
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about top offenders
            if not top_vehicles.empty:
                # Get the count of vehicles above threshold
                vehicles_above_threshold = len(high_violation_vehicles)
                threshold_percentage = (vehicles_above_threshold / len(top_vehicles) * 100) if len(top_vehicles) > 0 else 0
                
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="explanation-text">
                        This chart shows the drivers and pools with the highest number of speeding events. 
                        {top_vehicles.iloc[0]['Display_ID']} leads with {int(top_vehicles.iloc[0]['violation_count'])} events, followed by 
                        {top_vehicles.iloc[1]['Display_ID']} with {int(top_vehicles.iloc[1]['violation_count'])} events.
                        Of the top 40 vehicles shown, {vehicles_above_threshold} ({threshold_percentage:.1f}%) have 40+ violations and require corrective action.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:  # High Risk Drivers tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Highest Risk Drivers (Events per Mile)")
        
        # Simple explanation for presentation
        st.markdown("""
        <p class="explanation-text">
            <strong>What this shows:</strong> Drivers with 40+ violations who have the highest frequency of speeding events per mile driven.
            These drivers may not have the highest total violations, but they speed more frequently when they drive.
        </p>
        """, unsafe_allow_html=True)
        
        if 'Events_Per_Mile' in df.columns and 'Total Distance' in df.columns and 'violation_count' in df.columns:
            # Filter for vehicles with at least 10 miles driven AND 40+ violations
            high_risk_drivers = df[(df['Total Distance'] >= 10) & (df['violation_count'] >= 40)].drop_duplicates(subset=['Vehicle'])
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
                    title="Highest Risk Drivers with 40+ Violations (Events per Mile, min 10 miles driven)",
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
                
                # Add insights about high risk drivers - Fixed string formatting
                top_driver = high_risk_drivers.iloc[0]['Display_ID'] if not high_risk_drivers.empty else 'No driver'
                top_events_per_mile = high_risk_drivers.iloc[0]['Events_Per_Mile'] if not high_risk_drivers.empty else 0
                
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This chart shows drivers with 40+ violations and the highest frequency of speeding events per mile driven.
                        These drivers represent the highest risk based on violation density rather than absolute counts.
                        {top_driver} has {top_events_per_mile:.2f} events per mile.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No vehicles with 40+ violations and sufficient mileage (10+ miles) to calculate reliable risk metrics.")
        else:
            st.info("Required data for risk analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:  # By Driver Type tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Speeding Events by Driver Type")
        
        # Simple explanation for presentation
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">
            <strong>What this shows:</strong> How speeding violations are distributed across different types of drivers.
            This helps identify whether problems are with individual drivers, pool vehicles, or crew vehicles.
        </p>
        """, unsafe_allow_html=True)
        
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
        
        # Simple explanation for presentation
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">
            <strong>What this shows:</strong> Drivers with 40+ violations who have the highest average speeds.
            These drivers may be consistently driving at dangerous speeds rather than occasional speeding.
        </p>
        """, unsafe_allow_html=True)
        
        if 'Avg_Speed' in df.columns and 'Total Distance' in df.columns and 'violation_count' in df.columns:
            # Filter for vehicles with at least 10 miles driven AND 40+ violations
            top_speeders = df[(df['Total Distance'] >= 10) & (df['violation_count'] >= 40)].drop_duplicates(subset=['Vehicle'])
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
                    title="Drivers with 40+ Violations and Highest Average Speeds (min 10 miles driven)",
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
                
                # Add insights about top speeders - Fixed string formatting
                top_speeder = top_speeders.iloc[0]['Display_ID'] if not top_speeders.empty else 'No driver'
                top_speed = top_speeders.iloc[0]['Avg_Speed'] if not top_speeders.empty else 0
                
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This chart displays drivers with 40+ violations and the highest average speeds. Average speed is calculated 
                        from total distance divided by total drive time. The highest average speed is 
                        {top_speed:.1f} {distance_unit}/hour by {top_speeder}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No vehicles with 40+ violations and sufficient mileage (10+ miles) to calculate reliable speed metrics.")
        else:
            st.info("Required data for speed analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[4]:  # Correlation Analysis tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Correlation: Distance Driven vs. Speeding Events")
        
        # Simple explanation for presentation
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">
            <strong>What this shows:</strong> The relationship between how much a vehicle is driven and how many speeding events it has.
            Points above the trend line speed more than expected for their mileage.
        </p>
        """, unsafe_allow_html=True)
        
        if 'Total Distance' in df.columns and 'Number Of Events' in df.columns:
            # Create scatter plot data
            scatter_data = df.drop_duplicates(subset=['Vehicle'])
            
            # Check if statsmodels is available for trendline
            try:
                import statsmodels.api as sm
                has_statsmodels = True
            except ImportError:
                has_statsmodels = False
            
            # Create the scatter plot with or without trendline
            if has_statsmodels:
                fig = px.scatter(
                    scatter_data,
                    x='Total Distance',
                    y='Number Of Events',
                    size='Number Of Events',
                    color='Type',
                    hover_name='Display_ID',
                    trendline='ols',
                    labels={
                        'Total Distance': f'Total Distance ({distance_unit})',
                        'Number Of Events': 'Number of Speeding Events',
                        'Type': 'Driver Type'
                    },
                    title=f"Correlation between Distance Driven and Speeding Events"
                )
                
                # Calculate the expected number of events based on the trend line
                X = scatter_data['Total Distance'].values.reshape(-1, 1)
                y = scatter_data['Number Of Events'].values
                
                # Add a constant to the X array for the intercept term
                X_with_const = sm.add_constant(X)
                
                # Fit the OLS model
                model = sm.OLS(y, X_with_const).fit()
                
                # Calculate expected values and deviations
                scatter_data['Expected_Events'] = model.predict(X_with_const)
                scatter_data['Deviation'] = scatter_data['Number Of Events'] - scatter_data['Expected_Events']
                scatter_data['Deviation_Percent'] = (scatter_data['Deviation'] / scatter_data['Expected_Events']) * 100
                
                # Create a second visualization to show deviation from expected
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create a bar chart showing deviation from expected
                    # Sort by deviation and get top 15 (positive deviation)
                    top_deviations = scatter_data.sort_values('Deviation', ascending=False).head(15)
                    
                    deviation_fig = go.Figure()
                    
                    # Add the bar trace
                    deviation_fig.add_trace(go.Bar(
                        x=top_deviations['Display_ID'],
                        y=top_deviations['Deviation'],
                        marker=dict(
                            color=top_deviations['Deviation'],
                            colorscale='Reds',
                        ),
                        text=[f"+{x:.0f}" for x in top_deviations['Deviation']],
                        textposition='outside',
                        textfont=dict(size=12),
                    ))
                    
                    # Update layout
                    deviation_fig.update_layout(
                        title="Top 15 Drivers Above Expected Speeding Events",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font={'color': '#020817'},
                        margin=dict(t=40, b=40, l=40, r=40),
                        xaxis_tickangle=-45,
                        xaxis_title="",
                        yaxis_title="Events Above Expected",
                        height=500
                    )
                    
                    st.plotly_chart(deviation_fig, use_container_width=True)
                
                # Add a table showing the top 10 drivers with highest deviation
                st.subheader("Drivers with Most Speeding Events Above Expected")
                
                # Prepare the data for the table
                deviation_table = top_deviations[['Display_ID', 'Total Distance', 'Number Of Events', 'Expected_Events', 'Deviation', 'Deviation_Percent']].head(10).copy()
                deviation_table.columns = ['Driver', 'Distance (miles)', 'Actual Events', 'Expected Events', 'Events Above Expected', '% Above Expected']
                
                # Format the numeric columns
                deviation_table['Expected Events'] = deviation_table['Expected Events'].map(lambda x: f"{x:.1f}")
                deviation_table['Events Above Expected'] = deviation_table['Events Above Expected'].map(lambda x: f"+{x:.1f}")
                deviation_table['% Above Expected'] = deviation_table['% Above Expected'].map(lambda x: f"+{x:.1f}%")
                
                # Display the table
                st.dataframe(deviation_table, use_container_width=True)
                
                # Add insights about correlation
                st.markdown("""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        <strong>How to interpret:</strong> The scatter plot shows the relationship between distance driven and speeding events.
                        The bar chart highlights drivers who have <strong>more speeding events than expected</strong> based on their mileage.
                        These drivers may need additional attention as they speed more frequently than their peers who drive similar distances.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                fig = px.scatter(
                    scatter_data,
                    x='Total Distance',
                    y='Number Of Events',
                    size='Number Of Events',
                    color='Type',
                    hover_name='Display_ID',
                    labels={
                        'Total Distance': f'Total Distance ({distance_unit})',
                        'Number Of Events': 'Number of Speeding Events',
                        'Type': 'Driver Type'
                    },
                    title=f"Correlation between Distance Driven and Speeding Events"
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': '#020817'},
                    margin=dict(t=40, b=40, l=40, r=40),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div style="margin-top: 1rem;">
                    <p class="text-sm text-gray-600">
                        This scatter plot shows the relationship between distance driven and number of speeding events.
                        Each point represents a driver, with the size indicating the number of events.
                        Drivers in the upper right have both high mileage and high violation counts.
                    </p>
                    <p class="text-sm text-gray-600">
                        <strong>Note:</strong> Install the statsmodels package to see additional analysis of drivers who speed more than expected for their mileage.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Required data for correlation analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[5]:  # Safe Drivers tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Safe Drivers - Good Driving Habits")
        
        # Simple explanation for presentation
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">
            <strong>What this shows:</strong> Drivers with fewer than 40 violations who have driven significant mileage (100+ miles).
            These drivers demonstrate good driving habits and could be recognized for their safe driving.
        </p>
        """, unsafe_allow_html=True)
        
        if 'Events_Per_Mile' in df.columns and 'Total Distance' in df.columns and 'violation_count' in df.columns and 'Avg_Speed' in df.columns:
            # Filter for vehicles with SIGNIFICANT mileage (at least 100 miles) AND less than 40 violations
            safe_drivers = df[(df['Total Distance'] >= 100) & (df['violation_count'] < 40)].drop_duplicates(subset=['Vehicle'])
            
            if not safe_drivers.empty:
                # Calculate a safety score (lower is better)
                # Low events per mile and reasonable average speed contribute to a better score
                # We'll also factor in total distance to favor drivers who have driven more
                safe_drivers['Safety_Score'] = (safe_drivers['Events_Per_Mile'] * 10) - (safe_drivers['Total Distance'] / 1000) + (safe_drivers['Avg_Speed'] / 20)
                
                # Get the top 15 safest drivers (lowest safety score)
                safest_drivers = safe_drivers.nsmallest(15, 'Safety_Score')
                
                # Create a bar chart with data labels using plotly graph objects
                fig = go.Figure()
                
                # Add the bar trace for events per mile (primary metric for safe driving)
                fig.add_trace(go.Bar(
                    x=safest_drivers['Display_ID'],
                    y=safest_drivers['Events_Per_Mile'],
                    marker=dict(
                        color='rgba(58, 171, 115, 0.8)',
                    ),
                    text=[f"{x:.2f}" for x in safest_drivers['Events_Per_Mile']],
                    textposition='outside',
                    textfont=dict(size=12),
                    name='Events Per Mile'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Safest Drivers (Lowest Events per Mile, min 100 miles driven)",
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
                
                # Create a table of safe drivers with their metrics
                st.subheader("Safe Driver Details")
                
                # Prepare the data for the table
                safe_table = safest_drivers[['Display_ID', 'Total Distance', 'Number Of Events', 'Events_Per_Mile', 'Avg_Speed', 'violation_count']].copy()
                safe_table.columns = ['Driver', 'Total Distance (miles)', 'Number Of Events', 'Events Per Mile', 'Avg Speed (mph)', 'Total Violations']
                
                # Format the numeric columns
                safe_table['Events Per Mile'] = safe_table['Events Per Mile'].map('{:.2f}'.format)
                safe_table['Avg Speed (mph)'] = safe_table['Avg Speed (mph)'].map('{:.1f}'.format)
                
                # Display the table
                st.dataframe(safe_table, use_container_width=True)
                
                # Add insights about safe drivers
                top_safe_driver = safest_drivers.iloc[0]['Display_ID'] if not safest_drivers.empty else 'No driver'
                top_safe_distance = safest_drivers.iloc[0]['Total Distance'] if not safest_drivers.empty else 0
                top_safe_events_per_mile = safest_drivers.iloc[0]['Events_Per_Mile'] if not safest_drivers.empty else 0
                top_safe_speed = safest_drivers.iloc[0]['Avg_Speed'] if not safest_drivers.empty else 0
                
                st.markdown(f"""
                <div class="safe-driver-highlight">
                    <h4>🏆 Safe Driving Recognition</h4>
                    <p>
                        These drivers have demonstrated good driving habits with fewer than 40 violations and low events per mile,
                        while maintaining significant mileage (100+ miles).
                    </p>
                    <p>
                        <strong>{top_safe_driver}</strong> leads with only {top_safe_events_per_mile:.2f} events per mile
                        over {top_safe_distance:.1f} miles driven, while maintaining a reasonable average speed of {top_safe_speed:.1f} mph.
                    </p>
                    <p style="margin-bottom: 0;">
                        Consider recognizing these drivers for their safe driving practices and using them as examples for drivers requiring corrective action.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No vehicles with less than 40 violations and significant mileage (100+ miles) to evaluate safe driving habits.")
        else:
            st.info("Required data for safe driver analysis is not available in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[6]:  # Full Data tab
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Complete Dataset")
        
        # Simple explanation for presentation
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">
            <strong>What this shows:</strong> The complete dataset with all vehicles and their metrics.
            Use the filters and search to find specific drivers or analyze different groups.
        </p>
        """, unsafe_allow_html=True)
        
        # Add filters in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Filter by violation count
            min_violations = st.number_input("Min Violations", min_value=0, value=0, step=1)
            max_violations = st.number_input("Max Violations", min_value=0, value=1000, step=1)
        
        with filter_col2:
            # Filter by driver type if available
            if 'Type' in df.columns:
                driver_types = ['All'] + sorted(df['Type'].unique().tolist())
                selected_type = st.selectbox("Driver Type", driver_types)
            
            # Filter by distance
            min_distance = st.number_input("Min Distance (miles)", min_value=0.0, value=0.0, step=10.0)
        
        with filter_col3:
            # Search by driver/vehicle
            search_term = st.text_input("Search Driver/Vehicle", "")
            
            # Sort options
            sort_options = {
                "Violation Count (High to Low)": ("violation_count", False),
                "Violation Count (Low to High)": ("violation_count", True),
                "Events Per Mile (High to Low)": ("Events_Per_Mile", False),
                "Events Per Mile (Low to High)": ("Events_Per_Mile", True),
                "Distance (High to Low)": ("Total Distance", False),
                "Distance (Low to High)": ("Total Distance", True),
                "Average Speed (High to Low)": ("Avg_Speed", False),
                "Average Speed (Low to High)": ("Avg_Speed", True)
            }
            sort_by = st.selectbox("Sort By", list(sort_options.keys()))
        
        # Apply filters to create a filtered dataframe
        filtered_df = df.copy()
        
        # Apply violation count filter
        if 'violation_count' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['violation_count'] >= min_violations) & 
                                     (filtered_df['violation_count'] <= max_violations)]
        
        # Apply driver type filter
        if 'Type' in filtered_df.columns and selected_type != 'All':
            filtered_df = filtered_df[filtered_df['Type'] == selected_type]
        
        # Apply distance filter
        if 'Total Distance' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Total Distance'] >= min_distance]
        
        # Apply search filter
        if search_term:
            if 'Display_ID' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Display_ID'].str.contains(search_term, case=False, na=False)]
            elif 'Vehicle' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Vehicle'].str.contains(search_term, case=False, na=False)]
        
        # Apply sorting
        if sort_by in sort_options:
            sort_col, sort_asc = sort_options[sort_by]
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)
        
        # Remove duplicates to show one row per vehicle
        if 'Vehicle' in filtered_df.columns:
            filtered_df = filtered_df.drop_duplicates(subset=['Vehicle'])
        
        # Prepare the display dataframe with selected columns
        display_columns = []
        
        # Add key identifier columns
        if 'Display_ID' in filtered_df.columns:
            display_columns.append('Display_ID')
        elif 'Vehicle' in filtered_df.columns:
            display_columns.append('Vehicle')
        
        # Add type column if available
        if 'Type' in filtered_df.columns:
            display_columns.append('Type')
        
        # Add key metrics
        for col in ['violation_count', 'Total Distance', 'Number Of Events', 'Events_Per_Mile', 'Avg_Speed']:
            if col in filtered_df.columns:
                display_columns.append(col)
        
        # Create the display dataframe
        if display_columns:
            display_df = filtered_df[display_columns].copy()
            
            # Rename columns for better readability
            column_rename = {
                'Display_ID': 'Driver/Vehicle',
                'Vehicle': 'Vehicle ID',
                'violation_count': 'Total Violations',
                'Total Distance': 'Distance (miles)',
                'Number Of Events': 'Events',
                'Events_Per_Mile': 'Events Per Mile',
                'Avg_Speed': 'Avg Speed (mph)'
            }
            
            display_df = display_df.rename(columns={col: column_rename.get(col, col) for col in display_df.columns})
            
            # Format numeric columns
            for col in display_df.columns:
                if 'Events Per Mile' in col:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                elif 'Speed' in col:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
                elif 'Distance' in col:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
            
            # Display the dataframe with pagination
            st.dataframe(display_df, use_container_width=True, height=500)
            
            # Show record count
            st.markdown(f"<p style='color: #64748b;'>Showing {len(display_df)} records</p>", unsafe_allow_html=True)
            
            # Add download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="fleet_speeding_data.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available with the current filters.")
        
        # Add a section for viewing the raw data
        with st.expander("View Original Format Data"):
            # Get the original columns from the Excel file
            original_columns = df.columns.tolist()
            
            # Remove the derived columns we added
            derived_columns = ['Vehicle', 'Driver', 'isPool', 'isCrew', 'Type', 'Display_ID', 
                              'violation_count', 'Duration_Minutes', 'Avg_Speed', 'Events_Per_Mile', 'Safety_Score']
            
            original_display_columns = [col for col in original_columns if col not in derived_columns]
            
            if original_display_columns:
                # Apply the same filters to the original format
                original_filtered_df = filtered_df[original_display_columns].copy()
                st.dataframe(original_filtered_df, use_container_width=True, height=400)
                
                # Add download button for original format
                original_csv = original_filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Original Format as CSV",
                    data=original_csv,
                    file_name="fleet_speeding_data_original.csv",
                    mime="text/csv",
                )
            else:
                st.info("Original format data not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary and Recommendations
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("Summary")
    
    # Get key metrics for the summary
    if not df.empty:
        total_vehicles = df['Vehicle'].nunique() if 'Vehicle' in df.columns else 0
        total_violations = len(df)
        
        # Count vehicles with 40+ violations
        if 'violation_count' in df.columns:
            high_violation_count = df[df['violation_count'] >= 40]['Vehicle'].nunique()
            high_violation_percentage = (high_violation_count / total_vehicles * 100) if total_vehicles > 0 else 0
        else:
            high_violation_count = 0
            high_violation_percentage = 0
        
        # Get driver types breakdown for vehicles with 40+ violations
        driver_type_breakdown = ""
        if 'Type' in df.columns and 'violation_count' in df.columns:
            high_violation_df = df[df['violation_count'] >= 40].drop_duplicates(subset=['Vehicle'])
            type_counts = high_violation_df['Type'].value_counts()
            
            type_percentages = []
            for driver_type, count in type_counts.items():
                percentage = (count / high_violation_count * 100) if high_violation_count > 0 else 0
                type_percentages.append(f"{driver_type}: {count} ({percentage:.1f}%)")
            
            driver_type_breakdown = ", ".join(type_percentages)
        
        # Get top individual contributor with 40+ violations
        top_individual = ""
        if 'Type' in df.columns and 'violation_count' in df.columns and 'Display_ID' in df.columns:
            individuals = df[(df['Type'] == 'Individual') & (df['violation_count'] >= 40)].drop_duplicates(subset=['Vehicle'])
            if not individuals.empty:
                top_individual_row = individuals.nlargest(1, 'violation_count').iloc[0]
                top_individual = f"{top_individual_row['Display_ID']} with {int(top_individual_row['violation_count'])} violations"
        
        # Get highest risk driver with 40+ violations
        highest_risk = ""
        if 'Events_Per_Mile' in df.columns and 'Total Distance' in df.columns and 'violation_count' in df.columns and 'Display_ID' in df.columns:
            risk_drivers = df[(df['Total Distance'] >= 10) & (df['violation_count'] >= 40)].drop_duplicates(subset=['Vehicle'])
            if not risk_drivers.empty:
                highest_risk_row = risk_drivers.nlargest(1, 'Events_Per_Mile').iloc[0]
                highest_risk = f"{highest_risk_row['Display_ID']} with {highest_risk_row['Events_Per_Mile']:.2f} events per mile"
        
        # Create the summary with bullet points
        st.markdown(f"""
        <ul class="list-disc pl-5 space-y-1">
            <li>Total of <strong>{total_violations}</strong> speeding events recorded across <strong>{total_vehicles}</strong> vehicles</li>
            <li><strong>{high_violation_count} vehicles ({high_violation_percentage:.1f}%)</strong> have 40+ violations and require corrective action</li>
            {f"<li>Breakdown of vehicles requiring corrective action by driver type: {driver_type_breakdown}</li>" if driver_type_breakdown else ""}
            {f"<li>Top individual contributor: <strong>{top_individual}</strong></li>" if top_individual else ""}
            {f"<li>Highest risk driver (per mile): <strong>{highest_risk}</strong></li>" if highest_risk else ""}
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.info("No data available to generate summary.")
    
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
