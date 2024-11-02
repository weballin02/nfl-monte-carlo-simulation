# Import Libraries
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
import streamlit as st
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(file_path):
    usecols = ['date', 'team', 'PTS']
    data = pd.read_csv(file_path, usecols=usecols)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data

# Apply Team Name Mapping
def apply_team_name_mapping(data):
    # Mapping from team abbreviation to full team name
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

    # Inverse mapping from full team name to abbreviation
    inverse_team_name_mapping = {v: k for k, v in team_name_mapping.items()}

    # Apply mapping to data
    data['team_abbrev'] = data['team']
    data['team'] = data['team'].map(team_name_mapping)
    data.dropna(subset=['team'], inplace=True)

    return data, team_name_mapping, inverse_team_name_mapping

# Load Data
file_path = '/Users/matthewfox/ARIMA/traditional.csv'
data = load_and_preprocess_data(file_path)

# Apply Team Name Mapping
data, team_name_mapping, inverse_team_name_mapping = apply_team_name_mapping(data)

# Aggregate Points by Team and Date
team_data = data.groupby(['date', 'team_abbrev', 'team'])['PTS'].sum().reset_index()
team_data.set_index('date', inplace=True)

# Initialize, Train, Save, and Load ARIMA Models for Each Team
@st.cache_resource
def get_team_models(team_data):
    model_dir = '/Users/matthewfox/ARIMA/models/'
    os.makedirs(model_dir, exist_ok=True)  # Ensure the model directory exists

    team_models = {}
    teams_list = team_data['team_abbrev'].unique()

    for team_abbrev in teams_list:
        model_filename = f'{team_abbrev}_arima_model.pkl'
        model_path = os.path.join(model_dir, model_filename)

        # Prepare team_points
        team_points = team_data[team_data['team_abbrev'] == team_abbrev]['PTS']
        team_points.reset_index(drop=True, inplace=True)

        # Load or train model
        if os.path.exists(model_path):
            # Load existing model
            model = joblib.load(model_path)
        else:
            # Train a new ARIMA model
            model = auto_arima(
                team_points,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model.fit(team_points)

            # Save the trained model
            joblib.dump(model, model_path)

        # Store the model in the dictionary
        team_models[team_abbrev] = model

    return team_models

# Get Team Models
team_models = get_team_models(team_data)

# Forecast the Next 5 Games for Each Team
@st.cache_data
def compute_team_forecasts(_team_models, team_data):
    team_forecasts = {}
    forecast_periods = 5

    for team_abbrev, model in _team_models.items():
        # Get last date
        team_points = team_data[team_data['team_abbrev'] == team_abbrev]['PTS']
        if team_points.empty:
            continue
        last_date = team_points.index.max()

        # Generate future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')

        # Forecast
        forecast = model.predict(n_periods=forecast_periods)

        # Ensure forecast is an array
        if isinstance(forecast, pd.Series):
            forecast = forecast.values

        # Store forecast
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_PTS': forecast,
            'Team': team_abbrev
        })
        team_forecasts[team_abbrev] = predictions

    # Combine all forecasts
    if team_forecasts:
        all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
    else:
        all_forecasts = pd.DataFrame(columns=['Date', 'Predicted_PTS', 'Team'])
    return all_forecasts

# Compute Team Forecasts
all_forecasts = compute_team_forecasts(team_models, team_data)

# Streamlit App
st.title('NBA Team Points Prediction')

# Dropdown menu for selecting a team
teams_list = sorted(team_data['team_abbrev'].unique())
team_abbrev = st.selectbox('Select a team for prediction:', teams_list)

if team_abbrev:
    team_full_name = team_name_mapping[team_abbrev]
    team_points = team_data[team_data['team_abbrev'] == team_abbrev]['PTS']
    team_points.index = pd.to_datetime(team_points.index)

    st.write(f'### Historical Points for {team_full_name}')
    st.line_chart(team_points)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team_abbrev]
    st.write(f'### Predicted Points for Next 5 Games ({team_full_name})')
    st.write(team_forecast[['Date', 'Predicted_PTS']])

    # Plot the historical and predicted points
    st.write(f'### Points Prediction for {team_full_name}')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert dates to Matplotlib's numeric format
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_points.index)

    # Plot historical points
    ax.plot(
        historical_dates,
        team_points.values,
        label=f'Historical Points for {team_full_name}',
        color='blue'
    )
    # Plot predicted points
    ax.plot(
        forecast_dates,
        team_forecast['Predicted_PTS'],
        label='Predicted Points',
        color='red'
    )
    # Plot confidence interval (using +/- 5 as placeholder)
    lower_bound = team_forecast['Predicted_PTS'] - 5
    upper_bound = team_forecast['Predicted_PTS'] + 5

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

    ax.set_title(f'Points Prediction for {team_full_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Points')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# New functionality: Fetch upcoming games using nba_api and get predictions
st.write('---')
st.header('NBA Game Predictions for Today')

# Fetch upcoming games
@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().strftime('%m/%d/%Y')
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    return games

games = fetch_nba_games()

# Map team IDs to abbreviations
nba_team_list = nba_teams.get_teams()
nba_team_dict = {int(team['id']): team['abbreviation'] for team in nba_team_list}

# Map team IDs to your dataset's abbreviations
games['HOME_TEAM_ABBREV'] = games['HOME_TEAM_ID'].map(nba_team_dict)
games['VISITOR_TEAM_ABBREV'] = games['VISITOR_TEAM_ID'].map(nba_team_dict)

# Filter out any games where teams could not be mapped
games = games.dropna(subset=['HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV'])

# Process game data
game_list = []
for index, row in games.iterrows():
    game_id = row['GAME_ID']
    home_team_abbrev = row['HOME_TEAM_ABBREV']
    away_team_abbrev = row['VISITOR_TEAM_ABBREV']
    home_team_full = team_name_mapping.get(home_team_abbrev)
    away_team_full = team_name_mapping.get(away_team_abbrev)

    # Create game label
    game_label = f"{away_team_full} at {home_team_full}"
    game_list.append({
        'Game ID': game_id,
        'Game Label': game_label,
        'Home Team Abbrev': home_team_abbrev,
        'Away Team Abbrev': away_team_abbrev,
        'Home Team Full': home_team_full,
        'Away Team Full': away_team_full
    })

games_df = pd.DataFrame(game_list)

# Check if there are games today
if not games_df.empty:
    game_selection = st.selectbox('Select a game to get predictions:', games_df['Game Label'])

    # Get selected game details
    selected_game = games_df[games_df['Game Label'] == game_selection].iloc[0]
    home_team_abbrev = selected_game['Home Team Abbrev']
    away_team_abbrev = selected_game['Away Team Abbrev']
    home_team_full = selected_game['Home Team Full']
    away_team_full = selected_game['Away Team Full']

    # Get models
    home_team_model = team_models.get(home_team_abbrev)
    away_team_model = team_models.get(away_team_abbrev)

    if home_team_model and away_team_model:
        # Predict points
        home_team_forecast = home_team_model.predict(n_periods=1)
        away_team_forecast = away_team_model.predict(n_periods=1)

        # Access the prediction value
        if isinstance(home_team_forecast, pd.Series):
            home_team_forecast = home_team_forecast.iloc[0]
        else:
            home_team_forecast = home_team_forecast[0]

        if isinstance(away_team_forecast, pd.Series):
            away_team_forecast = away_team_forecast.iloc[0]
        else:
            away_team_forecast = away_team_forecast[0]

        st.write(f"### Predicted Points")
        st.write(f"**{home_team_full} ({home_team_abbrev}):** {home_team_forecast:.2f}")
        st.write(f"**{away_team_full} ({away_team_abbrev}):** {away_team_forecast:.2f}")

        if home_team_forecast > away_team_forecast:
            st.success(f"**Predicted Winner:** {home_team_full}")
        elif away_team_forecast > home_team_forecast:
            st.success(f"**Predicted Winner:** {away_team_full}")
        else:
            st.info("**Predicted Outcome:** Tie")
    else:
        st.error("Prediction models for one or both teams are not available.")
else:
    st.write("No games scheduled for today.")
