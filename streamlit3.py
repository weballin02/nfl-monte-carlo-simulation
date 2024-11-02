import os
import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pytz
from datetime import datetime, timedelta
from pmdarima import auto_arima
from nba_api.stats.endpoints import ScoreboardV2
from utils.auth import hash_password, check_password
from utils.user_database import initialize_database, add_user, verify_user
from utils.database import save_model, get_saved_models, load_model
from utils.correlation_analysis import correlation_analysis_page

# Initialize database
initialize_database()

@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['schedule_date'] = pd.to_datetime(data['schedule_date'], errors='coerce')
    home_df = data[['schedule_date', 'team_home', 'score_home']].copy()
    home_df.rename(columns={'team_home': 'team', 'score_home': 'score'}, inplace=True)
    away_df = data[['schedule_date', 'team_away', 'score_away']].copy()
    away_df.rename(columns={'team_away': 'team', 'score_away': 'score'}, inplace=True)
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
    now = datetime.now(pytz.UTC)
    schedule = nfl.import_schedules([now.year])
    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce', utc=True
    )
    schedule.dropna(subset=['game_datetime'], inplace=True)
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') & (schedule['game_datetime'] >= now)
    ]
    team_abbrev_mapping = {  # Sample mapping
        'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
        'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
        'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
        'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
        'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
    }
    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)
    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)
    upcoming_games.reset_index(drop=True, inplace=True)
    return upcoming_games[['game_id', 'game_datetime', 'home_team_full', 'away_team_full']]

@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().strftime('%m/%d/%Y')
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    return games

def login_page():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.sidebar.success(f"Welcome {username}!")
        else:
            st.sidebar.error("Login failed. Check username and password.")

def registration_page():
    st.sidebar.header("Register New Account")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Register"):
        hashed_password = hash_password(new_password)
        if add_user(new_username, hashed_password):
            st.sidebar.success("User registered successfully! Please log in.")
        else:
            st.sidebar.error("Username already exists.")

def monte_carlo_simulation_page():
    st.write("Select a matchup and run Monte Carlo simulations to predict game outcomes.")
    
    def calculate_team_stats():
        current_year = datetime.now().year
        games = nfl.import_schedules([current_year])
        past_games = games.dropna(subset=['home_score', 'away_score'])
        team_stats = {}
        for team in pd.concat([past_games['home_team'], past_games['away_team']]).unique():
            home_games = past_games[past_games['home_team'] == team]
            home_scores = home_games['home_score']
            away_games = past_games[past_games['away_team'] == team]
            away_scores = away_games['away_score']
            all_scores = pd.concat([home_scores, away_scores])
            team_stats[team] = {
                'avg_score': all_scores.mean(),
                'std_dev': all_scores.std(),
                'min_score': all_scores.min()
            }
        return team_stats
    
    def monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats):
        home_score_avg = team_stats[home_team]['avg_score']
        home_score_std = team_stats[home_team]['std_dev']
        home_min_score = team_stats[home_team]['min_score']
        away_score_avg = team_stats[away_team]['avg_score']
        away_score_std = team_stats[away_team]['std_dev']
        away_min_score = team_stats[away_team]['min_score']
        home_wins, away_wins = 0, 0
        total_home_scores, total_away_scores = [], []
        for _ in range(num_simulations):
            home_score = max(home_min_score, np.random.normal(home_score_avg + spread_adjustment, home_score_std))
            away_score = max(away_min_score, np.random.normal(away_score_avg, away_score_std))
            if home_score > away_score:
                home_wins += 1
            else:
                away_wins += 1
            total_home_scores.append(home_score)
            total_away_scores.append(away_score)
        avg_home_score = np.mean(total_home_scores)
        avg_away_score = np.mean(total_away_scores)
        avg_total_score = avg_home_score + avg_away_score
        return {
            "Home Win Percentage": round(home_wins / num_simulations * 100, 1),
            "Away Win Percentage": round(away_wins / num_simulations * 100, 1),
            "Average Home Score": round(avg_home_score, 1),
            "Average Away Score": round(avg_away_score, 1),
            "Average Total Score": round(avg_total_score, 1),
            "Score Differential (Home - Away)": round(np.mean(np.array(total_home_scores) - np.array(total_away_scores)), 1),
            "Home Max Score": round(np.max(total_home_scores), 1),
            "Home Min Score": round(np.min(total_home_scores), 1),
            "Away Max Score": round(np.max(total_away_scores), 1),
            "Away Min Score": round(np.min(total_away_scores), 1)
        }
    
    team_stats = calculate_team_stats()
    upcoming_games = fetch_upcoming_games()
    matchup = st.selectbox("Select Matchup", [(game['home_team_full'], game['away_team_full']) for idx, game in upcoming_games.iterrows()])
    spread_adjustment = -5.37  
    num_simulations = st.radio("Select Number of Simulations", ("1,000", "10,000", "100,000", "1,000,000", "Run All"))
    
    if st.button("Run Simulation"):
        home_team, away_team = matchup
        if num_simulations == "Run All":
            results = monte_carlo_simulation(home_team, away_team, spread_adjustment, 1000000, team_stats)
        else:
            results = monte_carlo_simulation(home_team, away_team, spread_adjustment, int(num_simulations.replace(",", "")), team_stats)
        st.write(results)

def run_nfl_predictions():
    nfl_file = st.file_uploader("Upload NFL CSV file", type="csv")
    if nfl_file is not None:
        nfl_data = load_and_preprocess_data(nfl_file)
        team_models = get_team_models(nfl_data)
        forecasts = compute_team_forecasts(team_models, nfl_data)
        st.write(forecasts)

def run_nba_predictions():
    nba_file = st.file_uploader("Upload NBA CSV file", type="csv")
    if nba_file is not None:
        nba_data = pd.read_csv(nba_file)
        nba_data, _, _ = apply_team_name_mapping(nba_data)
        team_models = get_team_models(nba_data, model_dir='models/nba/')
        forecasts = compute_team_forecasts(team_models, nba_data, forecast_periods=5)
        st.write(forecasts)

def main():
    st.title("Sports Monte Carlo Simulation App")
    menu = ["Login", "Register", "Monte Carlo Simulation", "Correlation Analysis", "NFL Predictions", "NBA Predictions"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Login":
        login_page()
    elif choice == "Register":
        registration_page()
    elif choice == "Monte Carlo Simulation":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            monte_carlo_simulation_page()
        else:
            st.warning("Please log in to access the Monte Carlo Simulation.")
    elif choice == "Correlation Analysis":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            correlation_analysis_page()
        else:
            st.warning("Please log in to access the Correlation Analysis.")
    elif choice == "NFL Predictions":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            run_nfl_predictions()
        else:
            st.warning("Please log in to access NFL Predictions.")
    elif choice == "NBA Predictions":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            run_nba_predictions()
        else:
            st.warning("Please log in to access NBA Predictions.")

if __name__ == "__main__":
    main()
