# modules/test.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from pmdarima import auto_arima
import joblib
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data(file_path):
    usecols = ['date', 'team', 'PTS']
    data = pd.read_csv(file_path, usecols=usecols)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data

def apply_team_name_mapping(data):
    team_name_mapping = {
        'ATL': 'Atlanta Hawks',
        'BOS': 'Boston Celtics',
        'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets',
        'CHH': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls',
        'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks',
        'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors',
        'HOU': 'Houston Rockets',
        'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers',
        'LAL': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks',
        'MIN': 'Minnesota Timberwolves',
        'NOH': 'New Orleans Pelicans',
        'NOK': 'New Orleans Pelicans',
        'NOP': 'New Orleans Pelicans',
        'NJN': 'Brooklyn Nets',
        'NYK': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers',
        'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings',
        'SAS': 'San Antonio Spurs',
        'SEA': 'Oklahoma City Thunder',
        'TOR': 'Toronto Raptors',
        'UTA': 'Utah Jazz',
        'VAN': 'Memphis Grizzlies',
        'WAS': 'Washington Wizards',
    }

    inverse_team_name_mapping = {v: k for k, v in team_name_mapping.items()}
    data['team_abbrev'] = data['team']
    data['team'] = data['team'].map(team_name_mapping)
    data.dropna(subset=['team'], inplace=True)

    return data, team_name_mapping, inverse_team_name_mapping

@st.cache_resource
def get_team_models(team_data, model_dir='models/nba/'):
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team_abbrev'].unique()

    for team_abbrev in teams_list:
        model_filename = f'{team_abbrev}_arima_model.pkl'
        model_path = os.path.join(model_dir, model_filename)

        team_points = team_data[team_data['team_abbrev'] == team_abbrev]['PTS']
        team_points.reset_index(drop=True, inplace=True)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            if len(team_points) < 10:
                continue
            model = auto_arima(
                team_points,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model.fit(team_points)
            joblib.dump(model, model_path)

        team_models[team_abbrev] = model

    return team_models

def predict_team_score(team_abbrev, team_models, periods=1):
    model = team_models.get(team_abbrev)
    if model:
        forecast = model.predict(n_periods=periods)
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        return forecast[0]
    else:
        return None

@st.cache_data
def compute_team_forecasts(team_models, team_data, forecast_periods=5, model_dir='models/nba/'):
    team_forecasts = {}
    for team_abbrev, model in team_models.items():
        team_points = team_data[team_data['team_abbrev'] == team_abbrev]['PTS']
        if team_points.empty:
            continue
        last_date = team_points.index.max()

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')

        forecast = model.predict(n_periods=forecast_periods)

        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_PTS': forecast,
            'Team': team_abbrev
        })
        team_forecasts[team_abbrev] = predictions

    if team_forecasts:
        all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
    else:
        all_forecasts = pd.DataFrame(columns=['Date', 'Predicted_PTS', 'Team'])
    return all_forecasts

@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().strftime('%m/%d/%Y')
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    return games

def main():
    st.title("Sports Monte Carlo Simulation App")

    # Sidebar Navigation
    menu = ["Login", "Register", "Monte Carlo Simulation", "Correlation Analysis", "NFL Predictions", "NBA Predictions"]
    choice = st.sidebar.selectbox("Menu", menu)

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if choice == "Register":
        registration_page()
    elif choice == "Login":
        login_page()
    elif choice == "Monte Carlo Simulation":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            monte_carlo_simulation_page()
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access the Monte Carlo Simulation.")
    elif choice == "Correlation Analysis":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            correlation_analysis_page()
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access the Correlation Analysis.")
    elif choice == "NFL Predictions":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            run_nfl_predictions()
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access NFL Predictions.")
    elif choice == "NBA Predictions":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            run_nba_predictions()
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access NBA Predictions.")

if __name__ == "__main__":
    main()
