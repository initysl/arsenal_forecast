import pandas as pd
from src.data.api_client import APIFootballClient
import time

client = APIFootballClient()

print("Fetching EPL historical data (2021-2024)...")

all_matches = []

for season in [2021, 2022, 2023, 2024]:
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
                
            match = {
                'season': f"{season}/{season+1}",
                'date': fixture['fixture']['date'],
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'home_shots': fixture['statistics'][0]['statistics'][2]['value'] if fixture.get('statistics') else None,
                'away_shots': fixture['statistics'][1]['statistics'][2]['value'] if fixture.get('statistics') else None,
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

df.to_csv('data/raw/epl_historical_2021_2024.csv', index=False)
print(f"Saved to data/raw/epl_historical_2021_2024.csv")

print("\nSample data:")
print(df.head())
print(f"\nSeasons: {df['season'].unique()}")