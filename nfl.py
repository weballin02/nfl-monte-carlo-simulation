# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime
import streamlit as st
from pmdarima import auto_arima
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Load and Preprocess Data
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

# Load Data
file_path = '/Users/matthewfox/ARIMA/nfl_data.csv'  # Update this path
team_data = load_and_preprocess_data(file_path)

# Get list of teams
teams_list = team_data['team'].unique()

# Train or Load Models
@st.cache_resource
def get_team_models(team_data):
    model_dir = '/Users/matthewfox/ARIMA/models/nfl/'  # Update this path
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
            # Check if there are enough data points to train the model
            if len(team_scores) < 10:
                # Skip teams with insufficient data
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

# Get Team Models
team_models = get_team_models(team_data)

# Function to Predict Team Score
def predict_team_score(team, periods=1):
    model = team_models.get(team)
    if model:
        forecast = model.predict(n_periods=periods)
        # Ensure forecast is a numpy array
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        return forecast[0]
    else:
        return None

# Forecast the Next 5 Games for Each Team
@st.cache_data
def compute_team_forecasts(_team_models, team_data):
    team_forecasts = {}
    forecast_periods = 5

    for team, model in _team_models.items():
        # Get last date
        team_scores = team_data[team_data['team'] == team]['score']
        if team_scores.empty:
            continue
        last_date = team_scores.index.max()

        # Generate future dates (assuming games are played weekly)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_periods, freq='7D')

        # Forecast
        forecast = model.predict(n_periods=forecast_periods)

        # Store forecast
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Score': forecast,
            'Team': team
        })
        team_forecasts[team] = predictions

    # Combine all forecasts
    if team_forecasts:
        all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
    else:
        all_forecasts = pd.DataFrame(columns=['Date', 'Predicted_Score', 'Team'])
    return all_forecasts

# Compute Team Forecasts
all_forecasts = compute_team_forecasts(team_models, team_data)

# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    # Combine 'gameday' and 'gametime' to create 'game_datetime'
    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )

    # Drop rows where 'game_datetime' could not be parsed
    schedule.dropna(subset=['game_datetime'], inplace=True)

    # Get current time in UTC
    now = datetime.now(pytz.UTC)

    # Filter for upcoming regular-season games
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'] >= now)
    ]

    # Select necessary columns
    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    # Mapping from team abbreviations to full names
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

    # Apply mapping
    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    # Remove games where team names couldn't be mapped
    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)

    # Reset index
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games

# Fetch upcoming games
upcoming_games = fetch_upcoming_games()

# Streamlit App
st.title('NFL Team Points Prediction')

# Dropdown menu for selecting a team
teams_list = sorted(team_data['team'].unique())
team = st.selectbox('Select a team for prediction:', teams_list)

if team:
    team_scores = team_data[team_data['team'] == team]['score']
    team_scores.index = pd.to_datetime(team_scores.index)

    st.write(f'### Historical Scores for {team}')
    st.line_chart(team_scores)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team]
    st.write(f'### Predicted Scores for Next 5 Games ({team})')
    st.write(team_forecast[['Date', 'Predicted_Score']])

    # Plot the historical and predicted scores
    st.write(f'### Score Prediction for {team}')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert dates to Matplotlib's numeric format
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_scores.index)

    # Plot historical scores
    ax.plot(
        historical_dates,
        team_scores.values,
        label=f'Historical Scores for {team}',
        color='blue'
    )
    # Plot predicted scores
    ax.plot(
        forecast_dates,
        team_forecast['Predicted_Score'],
        label='Predicted Scores',
        color='red'
    )
    # Plot confidence interval (using +/- 5 as placeholder)
    lower_bound = team_forecast['Predicted_Score'] - 5
    upper_bound = team_forecast['Predicted_Score'] + 5

    # Ensure no non-finite values
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)

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

    ax.set_title(f'Score Prediction for {team}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# New functionality: NFL Game Predictions
st.write('---')
st.header('NFL Game Predictions for Upcoming Games')

# Create game labels
upcoming_games['game_label'] = [
    f"{row['away_team_full']} at {row['home_team_full']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]

# Let the user select a game
if not upcoming_games.empty:
    game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
    selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

    home_team = selected_game['home_team_full']
    away_team = selected_game['away_team_full']

    # Predict scores
home_team_score = predict_team_score(home_team)
away_team_score = predict_team_score(away_team)

if home_team_score is not None and away_team_score is not None:
    st.write(f"### Predicted Scores")
    st.write(f"**{home_team}:** {home_team_score:.2f}")
    st.write(f"**{away_team}:** {away_team_score:.2f}")

    if home_team_score > away_team_score:
        st.success(f"**Predicted Winner:** {home_team}")
    elif away_team_score > home_team_score:
        st.success(f"**Predicted Winner:** {away_team}")
    else:
        st.info("**Predicted Outcome:** Tie")
else:
    st.error("Prediction models for one or both teams are not available.")