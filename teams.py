# Install nba_api if you haven't already
# You can uncomment the following line to install
# !pip install nba_api

from nba_api.stats.static import teams

# Get all NBA teams
nba_teams = teams.get_teams()

# Print the total number of teams
print(f"Total NBA Teams: {len(nba_teams)}\n")

# Iterate over the teams and print their details
for team in nba_teams:
    team_id = team['id']
    full_name = team['full_name']
    abbreviation = team['abbreviation']
    city = team['city']
    state = team['state']
    year_founded = team['year_founded']
    print(f"Team ID: {team_id}")
    print(f"Full Name: {full_name}")
    print(f"Abbreviation: {abbreviation}")
    print(f"City: {city}")
    print(f"State: {state}")
    print(f"Year Founded: {year_founded}")
    print("-" * 30)

