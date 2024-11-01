# modules/nfl.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from pmdarima import auto_arima
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['schedule_date'] = pd.to_datetime(data['schedule_date'], errors='coerce')

    # Prepare home and away data
    home_df = data[['schedule_date', 'team_home', 'score_home']].copy()
    home_df.rename(columns={'team_home': 'team', 'score_home': 'score'}, inplace=True)

    away_df = data[['schedule_date', 'team_away', 'score_away']].copy()
    away_df.rename(columns={'team_away': 'team', 'score_away': 'score'}, inplace=True)

    # Combine both DataFrames
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data['score'] = pd.to_numeric(team_data['score'], errors='coerce')
    team_data.set_index('schedule_date', inplace=True)
    team_data.sort_index(inplace=True)

    return team_data

@st.cache_resource
def get_team_models(team_data, model_dir='models/nfl/'):
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team'].unique()

    for team in teams_list:
        model_path = os.path.join(model_dir, f'{team}_arima_model.pkl')

        team_scores = team_data[team_data['team'] == team]['score']
        team_scores.reset_index(drop=True, inplace=True)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            if len(team_scores) < 10:
                continue
            model = auto_arima(
                team_scores,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model.fit(team_scores)
            joblib.dump(model, model_path)

        team_models[team] = model

    return team_models

def predict_team_score(team, team_models, periods=1):
    model = team_models.get(team)
    if model:
        forecast = model.predict(n_periods=periods)
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        return forecast[0]
    else:
        return None

@st.cache_data
def compute_team_forecasts(team_models, team_data, forecast_periods=5, model_dir='models/nfl/'):
    team_forecasts = {}
    for team, model in team_models.items():
        team_scores = team_data[team_data['team'] == team]['score']
        if team_scores.empty:
            continue
        last_date = team_scores.index.max()

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_periods, freq='7D')

        forecast = model.predict(n_periods=forecast_periods)

        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Score': forecast,
            'Team': team
        })
        team_forecasts[team] = predictions

    if team_forecasts:
        all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
    else:
        all_forecasts = pd.DataFrame(columns=['Date', 'Predicted_Score', 'Team'])
    return all_forecasts

@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    # Assuming 'nfl_data_py' is used to fetch schedules
    import nfl_data_py as nfl  # Imported here to avoid top-level import issues
    now = datetime.now(pytz.UTC)
    schedule = nfl.import_schedules([now.year])

    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )

    schedule.dropna(subset=['game_datetime'], inplace=True)

    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'] >= now)
    ]

    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    team_abbrev_mapping = {
        'ARI': 'Arizona Cardinals',
        'ATL': 'Atlanta Falcons',
        'BAL': 'Baltimore Ravens',
        'BUF': 'Buffalo Bills',
        'CAR': 'Carolina Panthers',
        'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys',
        'DEN': 'Denver Broncos',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans',
        'IND': 'Indianapolis Colts',
        'JAX': 'Jacksonville Jaguars',
        'KC': 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers',
        'LAR': 'Los Angeles Rams',
        'LV': 'Las Vegas Raiders',
        'MIA': 'Miami Dolphins',
        'MIN': 'Minnesota Vikings',
        'NE': 'New England Patriots',
        'NO': 'New Orleans Saints',
        'NYG': 'New York Giants',
        'NYJ': 'New York Jets',
        'PHI': 'Philadelphia Eagles',
        'PIT': 'Pittsburgh Steelers',
        'SEA': 'Seattle Seahawks',
        'SF': 'San Francisco 49ers',
        'TB': 'Tampa Bay Buccaneers',
        'TEN': 'Tennessee Titans',
        'WAS': 'Washington Commanders',
    }

    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games
