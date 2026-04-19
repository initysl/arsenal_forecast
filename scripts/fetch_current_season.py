import pandas as pd
from src.data.api_client import APIFootballClient
import json

client = APIFootballClient()

print("Fetching 2025/26 Premier League data for Arsenal...")

# Get Arsenal fixtures
fixtures = client.get_current_season_fixtures(
    league_id=39,  # Premier League
    team_id=42,    # Arsenal
    season=2025
)

print(f"Found {len(fixtures)} fixtures")

# Parse fixtures into dataframe
matches = []
for fixture in fixtures:
    match = {
        'fixture_id': fixture['fixture']['id'],
        'date': fixture['fixture']['date'],
        'status': fixture['fixture']['status']['short'],  # FT, NS, LIVE
        'home_team': fixture['teams']['home']['name'],
        'away_team': fixture['teams']['away']['name'],
        'home_goals': fixture['goals']['home'],
        'away_goals': fixture['goals']['away'],
        'is_home': fixture['teams']['home']['id'] == 42
    }
    matches.append(match)

df = pd.DataFrame(matches)

# Save
df.to_csv('data/raw/arsenal_2025_26_live.csv', index=False)
print(f"Saved to data/raw/arsenal_2025_26_live.csv")

# Get current standings
print("\nFetching Premier League standings...")
standings = client.get_league_standings(league_id=39, season=2025)

standings_data = []
for team in standings:
    standings_data.append({
        'rank': team['rank'],
        'team': team['team']['name'],
        'points': team['points'],
        'played': team['all']['played'],
        'goal_diff': team['goalsDiff']
    })

standings_df = pd.DataFrame(standings_data)
standings_df.to_csv('data/raw/epl_standings_2025_26.csv', index=False)
print(f"\nCurrent Top 5:")
print(standings_df.head())