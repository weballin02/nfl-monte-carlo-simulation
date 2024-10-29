import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title("NFL Monte Carlo Simulation")
st.write("Select a matchup and run Monte Carlo simulations to predict game outcomes.")

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

def get_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    today = datetime.now().date()
    schedule['gameday'] = pd.to_datetime(schedule['gameday']).dt.date
    upcoming_games = schedule[schedule['gameday'] >= today]
    return upcoming_games[['gameday', 'home_team', 'away_team']]

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

def run_all_simulations(home_team, away_team, spread_adjustment, team_stats):
    preset_iterations = [1000, 10000, 100000, 1000000]
    results_by_simulation = []

    for num_simulations in preset_iterations:
        result = monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats)
        result["Simulation Runs"] = num_simulations  
        results_by_simulation.append(result)

    aggregated_result = {
        "Simulation Runs": "Aggregate of All Runs",
        "Home Win Percentage": round(np.mean([res["Home Win Percentage"] for res in results_by_simulation]), 1),
        "Away Win Percentage": round(np.mean([res["Away Win Percentage"] for res in results_by_simulation]), 1),
        "Average Home Score": round(np.mean([res["Average Home Score"] for res in results_by_simulation]), 1),
        "Average Away Score": round(np.mean([res["Average Away Score"] for res in results_by_simulation]), 1),
        "Average Total Score": round(np.mean([res["Average Total Score"] for res in results_by_simulation]), 1),
        "Score Differential (Home - Away)": round(np.mean([res["Score Differential (Home - Away)"] for res in results_by_simulation]), 1),
        "Home Max Score": round(np.mean([res["Home Max Score"] for res in results_by_simulation]), 1),
        "Home Min Score": round(np.mean([res["Home Min Score"] for res in results_by_simulation]), 1),
        "Away Max Score": round(np.mean([res["Away Max Score"] for res in results_by_simulation]), 1),
        "Away Min Score": round(np.mean([res["Away Min Score"] for res in results_by_simulation]), 1)
    }

    return {"Individual Results": results_by_simulation, "Aggregated Result": aggregated_result}

team_stats = calculate_team_stats()
upcoming_games = get_upcoming_games()
matchup = st.selectbox("Select Matchup", [(game['home_team'], game['away_team']) for idx, game in upcoming_games.iterrows()])
spread_adjustment = -5.37  

num_simulations = st.radio("Select Number of Simulations", ("1,000", "10,000", "100,000", "1,000,000", "Run All"))

if st.button("Run Simulation"):
    home_team, away_team = matchup
    if num_simulations == "Run All":
        results = run_all_simulations(home_team, away_team, spread_adjustment, team_stats)
        
        st.subheader(f"Results for {results['Aggregated Result']['Simulation Runs']} Simulations")
        st.write(results['Aggregated Result'])
        
        for result in results["Individual Results"]:
            st.subheader(f"Results for {result['Simulation Runs']} Simulations")
            st.write(result)
    else:
        results = monte_carlo_simulation(home_team, away_team, spread_adjustment, int(num_simulations.replace(",", "")), team_stats)
        st.subheader(f"Results for {num_simulations} Simulations")
        st.write(results)
