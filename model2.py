import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load and Explore Data
file_path = '/Users/matthewfox/ARIMA/traditional.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
print(data.head())  # Check data structure

# Step 2: Aggregate Points by Team and Date
team_data = data.groupby(['date', 'team'])['PTS'].sum().reset_index()
team_data.set_index('date', inplace=True)
print(team_data.head())  # Verify aggregation

# Step 3: Initialize and Train ARIMA Models for Each Team
team_models = {}
team_forecasts = {}
teams = team_data['team'].unique()
evaluation_metrics = {}

for team in teams:
    print(f"Training model for team: {team}")
    team_points = team_data[team_data['team'] == team]['PTS']
    
    # Reset index for ARIMA compatibility
    team_points.reset_index(drop=True, inplace=True)
    
    # Train ARIMA model
    model = auto_arima(team_points, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(team_points)
    team_models[team] = model
    
    # Forecast the next 5 games
    forecast_periods = 5
    forecast, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)

    # Get the last valid date for the future forecast
    last_date = team_data[team_data['team'] == team].index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    
    predictions = pd.DataFrame({'Date': future_dates, 'Predicted_PTS': forecast, 'Team': team})
    team_forecasts[team] = predictions
    
    # Evaluate Model Performance
    if len(team_points) >= forecast_periods:
        last_n_periods = team_points.tail(forecast_periods)
        if len(last_n_periods) == forecast_periods:
            evaluation_metrics[team] = {
                'MSE': mean_squared_error(last_n_periods, forecast),
                'MAE': mean_absolute_error(last_n_periods, forecast),
                'RMSE': np.sqrt(mean_squared_error(last_n_periods, forecast)),
                'MAPE': np.mean(np.abs((last_n_periods - forecast) / last_n_periods.replace(0, np.nan))) * 100
            }
        else:
            evaluation_metrics[team] = {'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    print(f"Evaluation Metrics for {team}: {evaluation_metrics[team]}")

# Combine all forecasts
all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
print(all_forecasts.head())

# Step 4: Visualize Forecasts for a Selected Team
team_name = 'BOS'  # Change to any team for visualization
team_points = team_data[team_data['team'] == team_name]['PTS']

plt.figure(figsize=(10, 6))
plt.plot(team_points.index, team_points.values, label=f'Historical Points for {team_name}', color='blue')
plt.plot(all_forecasts[all_forecasts['Team'] == team_name]['Date'], 
         all_forecasts[all_forecasts['Team'] == team_name]['Predicted_PTS'], 
         label='Predicted Points', color='red')

# Optional: Visualize Confidence Interval for Predicted Points
conf_int = all_forecasts[all_forecasts['Team'] == team_name]

# Convert Date to datetime if not already
conf_int['Date'] = pd.to_datetime(conf_int['Date'], errors='coerce')

# Ensure Predicted_PTS is numeric
conf_int['Predicted_PTS'] = pd.to_numeric(conf_int['Predicted_PTS'], errors='coerce')

# Drop rows with NaN values in important columns
conf_int = conf_int.dropna(subset=['Date', 'Predicted_PTS'])

# Convert to numpy arrays to ensure compatibility
dates = np.array(conf_int['Date'])
predicted_pts = np.array(conf_int['Predicted_PTS'])

plt.fill_between(dates, predicted_pts - 5, predicted_pts + 5, color='gray', alpha=0.2, label='Confidence Interval')

# Adding labels and legend for better visualization
plt.title(f'Points Prediction for {team_name}')
plt.xlabel('Date')
plt.ylabel('Points')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
