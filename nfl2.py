import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import streamlit as st
from pmdarima import auto_arima
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

# Enhanced data loading with duplicate handling
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        
        data = pd.read_csv(file_path)
        
        data['schedule_date'] = pd.to_datetime(data['schedule_date'], errors='coerce')
        
        # Drop rows with missing dates
        data.dropna(subset=['schedule_date'], inplace=True)
        
        # Prepare home and away data
        home_df = data[['schedule_date', 'team_home', 'score_home']].copy()
        home_df['is_home'] = 1
        home_df.rename(columns={'team_home': 'team', 'score_home': 'score'}, inplace=True)
        
        away_df = data[['schedule_date', 'team_away', 'score_away']].copy()
        away_df['is_home'] = 0
        away_df.rename(columns={'team_away': 'team', 'score_away': 'score'}, inplace=True)
        
        # Combine and sort data
        team_data = pd.concat([home_df, away_df], ignore_index=True)
        team_data.dropna(subset=['score'], inplace=True)
        team_data['score'] = pd.to_numeric(team_data['score'], errors='coerce')
        
        # Remove duplicates by averaging scores for duplicates on the same date
        team_data = team_data.groupby(['team', 'schedule_date'], as_index=False).agg({
            'score': 'mean', 
            'is_home': 'first'
        })
        
        # Sort data
        team_data.sort_values(by=['team', 'schedule_date'], inplace=True)
        
        # Add rolling averages
        team_data['rolling_avg_3'] = team_data.groupby('team')['score'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        team_data['rolling_avg_5'] = team_data.groupby('team')['score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        
        return team_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

# Enhanced model training with caching and ensemble approach
@st.cache_resource
def get_team_models(team_data):
    model_dir = '/Users/matthewfox/ARIMA/models/nfl/'
    os.makedirs(model_dir, exist_ok=True)
    
    team_models = {}
    teams_list = team_data['team'].unique()
    
    for team in teams_list:
        try:
            team_subset = team_data[team_data['team'] == team].copy()
            
            if len(team_subset) < 5:  # Reduced threshold to 5
                st.warning(f"Insufficient data for {team}. Skipping model training.")
                continue
            
            arima_path = os.path.join(model_dir, f'{team}_arima_model.pkl')
            xgb_path = os.path.join(model_dir, f'{team}_xgb_model.pkl')
            
            # ARIMA Model
            if os.path.exists(arima_path):
                arima_model = joblib.load(arima_path)
            else:
                arima_model = train_arima_model(team_subset.set_index('schedule_date')['score'])
                joblib.dump(arima_model, arima_path)
            
            # XGBoost Model
            if os.path.exists(xgb_path):
                xgb_model = joblib.load(xgb_path)
            else:
                xgb_model = train_xgb_model(team_subset)
                joblib.dump(xgb_model, xgb_path)
            
            team_models[team] = {
                'arima': arima_model,
                'xgb': xgb_model
            }
            
        except Exception as e:
            st.error(f"Error training models for {team}: {str(e)}")
            continue
    
    return team_models

def train_arima_model(series):
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_score = float('inf')
    
    for train_idx, val_idx in tscv.split(series):
        train = series.iloc[train_idx]
        val = series.iloc[val_idx]
        
        try:
            model = auto_arima(
                train,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            
            predictions = model.predict(n_periods=len(val))
            score = mean_squared_error(val, predictions)
            
            if score < best_score:
                best_score = score
                best_model = model
        
        except Exception:
            continue
    
    if best_model is None:
        raise ModelError("Failed to train ARIMA model")
    
    return best_model

def train_xgb_model(data):
    X = data[['is_home', 'rolling_avg_3', 'rolling_avg_5']].values
    y = data['score'].values
    
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_score = float('inf')
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        try:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = mean_squared_error(y_val, predictions)
            
            if score < best_score:
                best_score = score
                best_model = model
        
        except Exception:
            continue
    
    if best_model is None:
        raise ModelError("Failed to train XGBoost model")
    
    return best_model

# Prediction function using ensemble
def predict_team_score(team, team_data, team_models, periods=1):
    models = team_models.get(team)
    if not models:
        return None
    
    try:
        team_subset = team_data[team_data['team'] == team].copy()
        team_subset.sort_values(by='schedule_date', inplace=True)
        
        latest_data = team_subset.iloc[-1:]
        X_xgb = latest_data[['is_home', 'rolling_avg_3', 'rolling_avg_5']].values
        
        arima_pred = models['arima'].predict(n_periods=periods)[-1] if models['arima'] else None
        xgb_pred = models['xgb'].predict(X_xgb)[-1] if models['xgb'] else None
        
        if arima_pred is None and xgb_pred is None:
            return None
        elif arima_pred is None:
            return xgb_pred
        elif xgb_pred is None:
            return arima_pred
        else:
            return (arima_pred + xgb_pred) / 2
    
    except Exception:
        return None

# Generate forecasts for all teams
def generate_all_forecasts(team_data, team_models, periods=5):
    forecast_list = []

    for team in team_data['team'].unique():
        for period in range(1, periods + 1):
            forecast_date = team_data['schedule_date'].max() + timedelta(days=7 * period)  # assuming weekly games
            predicted_score = predict_team_score(team, team_data, team_models, periods=period)
            if predicted_score is not None:
                forecast_list.append({
                    'Team': team,
                    'Date': forecast_date,
                    'Predicted_Score': predicted_score
                })

    all_forecasts = pd.DataFrame(forecast_list)
    if not all_forecasts.empty:
        all_forecasts['Date'] = pd.to_datetime(all_forecasts['Date'])
        # Do not set index here to keep 'Team' as a column
    
    return all_forecasts

# Streamlit App Initialization
file_path = '/Users/matthewfox/ARIMA/nfl_data.csv'  # Update this path to your data file
team_data = load_and_preprocess_data(file_path)
team_models = get_team_models(team_data)
all_forecasts = generate_all_forecasts(team_data, team_models)

st.title('NFL Team Points Prediction')

teams_list = sorted(team_data['team'].unique())
team = st.selectbox('Select a team for prediction:', teams_list)

if team:
    team_scores = team_data[team_data['team'] == team].copy()
    team_scores.set_index('schedule_date', inplace=True)
    team_scores.index = pd.to_datetime(team_scores.index)
    
    st.write(f'### Historical Scores for {team}')
    
    plot_data = pd.DataFrame({
        'Score': team_scores['score'],
        'Rolling Avg (3 games)': team_scores['rolling_avg_3'],
        'Rolling Avg (5 games)': team_scores['rolling_avg_5']
    })
    
    st.line_chart(plot_data)
    
    if not all_forecasts.empty:
        team_forecast = all_forecasts[all_forecasts['Team'] == team].copy()
        team_forecast.set_index('Date', inplace=True)
        st.write(f'### Predicted Scores for Next 5 Games ({team})')
        
        if not team_forecast.empty:
            display_forecast = team_forecast[['Predicted_Score']].copy()
            display_forecast.index = display_forecast.index.strftime('%Y-%m-%d')
            display_forecast['Predicted_Score'] = display_forecast['Predicted_Score'].round(2)
            display_forecast.columns = ['Predicted Score']
            st.dataframe(display_forecast, use_container_width=True)
            
            st.write(f'### Score Prediction Visualization for {team}')
            fig, ax = plt.subplots(figsize=(12, 6))
            forecast_dates = mdates.date2num(pd.to_datetime(team_forecast.index))
            historical_dates = mdates.date2num(team_scores.index)
        
            ax.plot(
                historical_dates,
                team_scores['score'].values,
                label='Historical Scores',
                color='blue',
                alpha=0.7
            )
            
            ax.plot(
                historical_dates,
                team_scores['rolling_avg_3'].values,
                label='3-Game Average',
                color='green',
                alpha=0.5,
                linestyle='--'
            )
            
            ax.plot(
                historical_dates,
                team_scores['rolling_avg_5'].values,
                label='5-Game Average',
                color='purple',
                alpha=0.5,
                linestyle=':'
            )
        
            ax.plot(
                forecast_dates,
                team_forecast['Predicted_Score'].values,
                label='Ensemble Prediction',
                color='red',
                linewidth=2
            )
        
            std_dev = team_scores['score'].std()
            lower_bound = team_forecast['Predicted_Score'].values - 1.96 * std_dev
            upper_bound = team_forecast['Predicted_Score'].values + 1.96 * std_dev
        
            ax.fill_between(
                forecast_dates,
                lower_bound,
                upper_bound,
                color='red',
                alpha=0.1,
                label='95% Confidence Interval'
            )
        
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            ax.set_title(f'Historical Performance and Predictions for {team}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Score')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"No forecast available for {team}.")
    else:
        st.warning("No forecasts available.")
    
st.write('---')
st.header('NFL Game Predictions for Upcoming Games')

if not all_forecasts.empty:
    display_forecasts = all_forecasts[['Date', 'Team', 'Predicted_Score']].copy()
    display_forecasts['Date'] = display_forecasts['Date'].dt.strftime('%Y-%m-%d')
    display_forecasts['Predicted_Score'] = display_forecasts['Predicted_Score'].round(2)
    st.dataframe(display_forecasts, use_container_width=True)
else:
    st.warning("No forecasts available to display.")
