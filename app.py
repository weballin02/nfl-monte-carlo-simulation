from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st  # Import for correlation analysis
from utils.auth import hash_password, check_password  # User authentication functions
from utils.user_database import initialize_database
from utils.database import save_model, get_saved_models, load_model  # Model storage
from correlation_analysis import correlation_analysis_page  # Correlation analysis page

# Initialize the Flask app
app = Flask(__name__)

# Initialize the database
initialize_database()

# Fetch the current season's schedule for upcoming games
def get_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    today = datetime.now().date()
    day_of_week = today.weekday()  # 0 = Monday, 6 = Sunday

    if day_of_week < 3:  # Monday, Tuesday, or Wednesday
        days_until_thursday = (3 - day_of_week) % 7
        next_game_day = today + timedelta(days=days_until_thursday)
    else:  # Thursday to Saturday
        days_until_sunday = (6 - day_of_week) % 7
        next_game_day = today + timedelta(days=days_until_sunday)

    if 'gameday' in schedule.columns:
        schedule['gameday'] = pd.to_datetime(schedule['gameday']).dt.date
        upcoming_games = schedule[schedule['gameday'] == next_game_day]
        upcoming_games = upcoming_games[['gameday', 'home_team', 'away_team']]
    else:
        print("Warning: 'gameday' column not found. Available columns:", schedule.columns)
        upcoming_games = schedule[['home_team', 'away_team']]

    return upcoming_games.to_dict(orient='records')

# Calculate average scores, standard deviations, and minimum scores for the current season
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

# Monte Carlo simulation function including Average Total Score
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
        home_score = np.random.normal(home_score_avg + spread_adjustment, home_score_std)
        away_score = np.random.normal(away_score_avg, away_score_std)

        home_score = max(home_min_score, home_score)
        away_score = max(away_min_score, away_score)

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
        "Simulation Runs": num_simulations,
        "Home Win Percentage": home_wins / num_simulations * 100,
        "Away Win Percentage": away_wins / num_simulations * 100,
        "Average Home Score": avg_home_score,
        "Average Away Score": avg_away_score,
        "Average Total Score": avg_total_score,
        "Score Differential (Home - Away)": np.mean(np.array(total_home_scores) - np.array(total_away_scores)),
        "Home Max Score": np.max(total_home_scores),
        "Home Min Score": np.min(total_home_scores),
        "Away Max Score": np.max(total_away_scores),
        "Away Min Score": np.min(total_away_scores)
    }

# Run simulations for each preset when "run all" is selected
def run_all_simulations(home_team, away_team, spread_adjustment, team_stats):
    preset_iterations = [1000, 10000, 100000, 1000000]
    results_by_simulation = []

    for num_simulations in preset_iterations:
        result = monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats)
        results_by_simulation.append(result)

    # Calculate the aggregate average across all individual simulation runs
    aggregated_result = {
        "Simulation Runs": "Aggregate of All Runs",
        "Home Win Percentage": np.mean([res["Home Win Percentage"] for res in results_by_simulation]),
        "Away Win Percentage": np.mean([res["Away Win Percentage"] for res in results_by_simulation]),
        "Average Home Score": np.mean([res["Average Home Score"] for res in results_by_simulation]),
        "Average Away Score": np.mean([res["Average Away Score"] for res in results_by_simulation]),
        "Average Total Score": np.mean([res["Average Total Score"] for res in results_by_simulation]),
        "Score Differential (Home - Away)": np.mean([res["Score Differential (Home - Away)"] for res in results_by_simulation]),
        "Home Max Score": np.mean([res["Home Max Score"] for res in results_by_simulation]),
        "Home Min Score": np.mean([res["Home Min Score"] for res in results_by_simulation]),
        "Away Max Score": np.mean([res["Away Max Score"] for res in results_by_simulation]),
        "Away Min Score": np.mean([res["Away Min Score"] for res in results_by_simulation])
    }

    return {"Individual Results": results_by_simulation, "Aggregated Result": aggregated_result}

# Flask routes
@app.route('/upcoming_games', methods=['GET'])
def get_upcoming_games_endpoint():
    games = get_upcoming_games()
    return jsonify(games)

@app.route('/simulate', methods=['POST'])
def simulate_game():
    data = request.json
    num_simulations = data.get("num_simulations")
    spread_adjustment = -5.37  # Example value; adjust based on needs

    team_stats = calculate_team_stats()
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if num_simulations == "run_all":
        results = run_all_simulations(home_team, away_team, spread_adjustment, team_stats)
    else:
        results = monte_carlo_simulation(home_team, away_team, spread_adjustment, int(num_simulations), team_stats)

    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

# Streamlit Correlation Analysis page
def run_correlation_analysis():
    st.title("Correlation Analysis")
    correlation_analysis_page()

if __name__ == '__main__':
    # Initialize and run Streamlit for correlation analysis
    run_correlation_analysis()
    app.run(debug=True)
