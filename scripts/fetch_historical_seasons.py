import pandas as pd
from src.data.api_client import APIFootballClient
import time

client = APIFootballClient()

print("Fetching EPL historical data (2022-2024)...")

def get_stat_value(statistics, stat_name):
    """Safely extract stat by name"""
    if not statistics:
        return None
    for stat in statistics:
        if stat['type'] == stat_name:
            return stat['value']
    return None

all_matches = []

for season in [2022, 2023, 2024]:
    print(f"\nFetching season {season}/{season+1}...")
    
    try:
        # Get all Premier League fixtures for the season
        params = {
            'league': 39,  # Premier League
            'season': season
        }
        
        response = client._get('fixtures', params)
        fixtures = response['response']
        
        print(f"Found {len(fixtures)} matches")
        
        for fixture in fixtures:
            # Only include finished matches
            if fixture['fixture']['status']['short'] not in ['FT', 'AET', 'PEN']:
                continue
            
            # Safely get statistics
            stats = fixture.get('statistics', [])
            home_stats = stats[0].get('statistics', []) if len(stats) > 0 else []
            away_stats = stats[1].get('statistics', []) if len(stats) > 1 else []
                
            match = {
                'season': f"{season}/{season+1}",
                'date': fixture['fixture']['date'],
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'home_shots': get_stat_value(home_stats, 'Total Shots'),
                'away_shots': get_stat_value(away_stats, 'Total Shots'),
            }
            all_matches.append(match)
        
        print(f"Processed {len([m for m in all_matches if m['season'] == f'{season}/{season+1}'])} completed matches")
        
        # Rate limiting - wait between seasons
        time.sleep(2)
        
    except Exception as e:
        print(f"Error fetching season {season}: {e}")
        continue

# Save to CSV
df = pd.DataFrame(all_matches)
print(f"\nTotal matches collected: {len(df)}")

df.to_csv('data/raw/epl_historical_2022_2024.csv', index=False)
print(f"Saved to data/raw/epl_historical_2022_2024.csv")

print("\nSample data:")
print(df.head())
if len(df) > 0:
    print(f"\nSeasons: {df['season'].unique()}")