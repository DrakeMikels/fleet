# Fleet Vehicle Speeding Analytics Dashboard

A comprehensive dashboard for analyzing fleet vehicle speeding data updated through March 6, 2025, with a focus on identifying vehicles requiring corrective action (40+ violations).

## Features

- **Top Offenders**: Identify the top 40 vehicles with the most speeding violations
- **High Risk Drivers**: Analyze drivers with the highest events per mile driven
- **Driver Type Analysis**: Compare violations across individual, pool, and crew vehicles
- **Highest Speeds**: Identify drivers with the highest average speeds
- **Correlation Analysis**: Visualize the relationship between distance driven and speeding events
- **Modern UI**: Clean, shadcn-inspired interface with responsive design

## Screenshots

Take screenshots of the dashboard for your PowerPoint presentation by:
1. Running the dashboard (instructions below)
2. Using your system's screenshot tool (⌘+Shift+4 on Mac)
3. Capturing the visualizations you need

## Installation

1. Make sure you have Python 3.7+ installed
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard Locally

1. Ensure the "Excessive Speeding 3.6.25.xlsx" file is in the same directory as the app.py file
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The dashboard will open in your default web browser

## Deploying to Streamlit Cloud (Free Hobby Tier)

1. Create a GitHub repository:
   - Go to [GitHub](https://github.com) and sign in
   - Create a new public repository
   - Push your code to the repository:
     ```bash
     git init
     git add app.py requirements.txt "Excessive Speeding 3.6.25.xlsx" README.md
     git commit -m "Initial commit"
     git remote add origin https://github.com/yourusername/fleet-speeding-dashboard.git
     git push -u origin main
     ```

2. Deploy on Streamlit Cloud:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign up or sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and app file (app.py)
   - Click "Deploy"

3. Access your deployed app:
   - Once deployment is complete, you'll get a URL like:
     `https://yourusername-fleet-speeding-dashboard-app-xyz123.streamlit.app`
   - This URL is publicly accessible and can be shared with anyone

## Dashboard Sections

1. **Top Offenders**: Shows the top 40 vehicles with the most speeding violations, highlighting those requiring corrective action (40+ violations)
2. **High Risk Drivers**: Identifies drivers with the highest events per mile (minimum 10 miles driven)
3. **By Driver Type**: Breaks down violations by driver type (Individual, Pool, Crew)
4. **Highest Speeds**: Shows drivers with the highest average speeds
5. **Correlation Analysis**: Visualizes the relationship between distance driven and number of speeding events

## Customization

The dashboard automatically detects column names in your Excel file. If you need to customize the visualizations:

1. Open app.py in a text editor
2. Modify the column detection logic if needed
3. Restart the dashboard

## Troubleshooting

If you encounter issues:
1. Ensure the Excel file is not open in another program
2. Check that the file is in the same directory as app.py
3. Verify that all dependencies are installed correctly 