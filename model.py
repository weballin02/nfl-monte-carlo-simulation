# Import Libraries
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import streamlit as st
import matplotlib.dates as mdates

# Load and Explore Data
file_path = '/Users/matthewfox/ARIMA/traditional.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Step 2: Aggregate Points by Team and Date
team_data = data.groupby(['date', 'team'])['PTS'].sum().reset_index()
team_data.set_index('date', inplace=True)

# Step 3: Initialize, Train, Save, and Load ARIMA Models for Each Team
model_dir = '/Users/matthewfox/ARIMA/models/'
os.makedirs(model_dir, exist_ok=True)  # Ensure the model directory exists

team_models = {}
teams = team_data['team'].unique()
evaluation_metrics = {}

for team in teams:
    model_path = os.path.join(model_dir, f'{team}_arima_model.pkl')
    
    # Prepare team_points for both cases
    team_points = team_data[team_data['team'] == team]['PTS']
    team_points.reset_index(drop=True, inplace=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        # Load existing model
        model = joblib.load(model_path)
    else:
        # Train a new ARIMA model
        model = auto_arima(
            team_points,
            seasonal=False,
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
        model.fit(team_points)
        
        # Save the trained model
        joblib.dump(model, model_path)
    
    # Store the model in the dictionary
    team_models[team] = model

    # Evaluate Model Performance
    forecast_periods = 5
    if len(team_points) >= forecast_periods:
        last_n_periods = team_points.tail(forecast_periods)
        forecast = model.predict(n_periods=forecast_periods)
        if len(last_n_periods) == forecast_periods:
            evaluation_metrics[team] = {
                'MSE': mean_squared_error(last_n_periods, forecast),
                'MAE': mean_absolute_error(last_n_periods, forecast),
                'RMSE': np.sqrt(mean_squared_error(last_n_periods, forecast)),
                'MAPE': np.mean(
                    np.abs((last_n_periods - forecast) / last_n_periods.replace(0, np.nan))
                ) * 100
            }
        else:
            evaluation_metrics[team] = {
                'MSE': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan,
                'MAPE': np.nan
            }


# Step 4: Forecast the Next 5 Games for Each Team
team_forecasts = {}
forecast_periods = 5

for team, model in team_models.items():
    # Get the last date from the original data (team_data) for the specific team
    last_date = team_data[team_data['team'] == team].index.max()
    
    # Generate future dates for the forecast
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    
    # Forecast the next 5 games
    forecast, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)
    
    # Store the forecast in a DataFrame
    predictions = pd.DataFrame({'Date': future_dates, 'Predicted_PTS': forecast, 'Team': team})
    team_forecasts[team] = predictions

# Combine all forecasts
all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)

# Step 5: Create a User-Friendly Web App with Streamlit
st.title('NBA Team Points Prediction')

# Dropdown menu for selecting a team
team_name = st.selectbox('Select a team for prediction:', teams)

# Display forecast for the selected team
if team_name:
    team_points = team_data[team_data['team'] == team_name]['PTS']
    team_points.index = pd.to_datetime(team_points.index)

    st.write(f'### Historical Points for {team_name}')
    st.line_chart(team_points)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team_name]
    st.write(f'### Predicted Points for Next 5 Games ({team_name})')
    st.write(team_forecast[['Date', 'Predicted_PTS']])

    # Convert dates to Matplotlib's numeric format
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_points.index)

    # Ensure Predicted_PTS is numeric
    team_forecast['Predicted_PTS'] = pd.to_numeric(team_forecast['Predicted_PTS'], errors='coerce')

    # Handle missing values
    team_forecast = team_forecast.dropna(subset=['Predicted_PTS'])
    forecast_dates = mdates.date2num(team_forecast['Date'])  # Update forecast_dates after dropping NaNs

    # Calculate confidence interval bounds
    lower_bound = team_forecast['Predicted_PTS'] - 5
    upper_bound = team_forecast['Predicted_PTS'] + 5

    # Ensure no non-finite values
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)

    # Plot the historical and predicted points
    st.write(f'### Points Prediction for {team_name}')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        historical_dates,
        team_points.values,
        label=f'Historical Points for {team_name}',
        color='blue'
    )
    ax.plot(
        forecast_dates[finite_indices],
        team_forecast['Predicted_PTS'].values[finite_indices],
        label='Predicted Points',
        color='red'
    )
    ax.fill_between(
        forecast_dates[finite_indices],
        lower_bound.values[finite_indices],
        upper_bound.values[finite_indices],
        color='gray',
        alpha=0.2,
        label='Confidence Interval'
    )

    ax.xaxis_date()
    fig.autofmt_xdate()

    ax.set_title(f'Points Prediction for {team_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Points')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
