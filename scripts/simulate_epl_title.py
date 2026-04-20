import pandas as pd
import numpy as np
import joblib
from src.features.team_strength import calculate_team_strength

# Load model and team strength
model = joblib.load('outputs/models/match_predictor.pkl')
team_strength = pd.read_csv('outputs/models/team_strength.csv', index_col=0)

# Load Arsenal fixtures
arsenal_fixtures = pd.read_csv('data/raw/arsenal_epl_2025_26.csv')
arsenal_fixtures.columns = arsenal_fixtures.columns.str.strip()

# Load Man City fixtures
city_fixtures = pd.read_csv('data/raw/mancity_epl_2025_26.csv')
city_fixtures.columns = city_fixtures.columns.str.strip()

# Current points
ARSENAL_CURRENT = 70
CITY_CURRENT = 67

print("="*60)
print("PREMIER LEAGUE TITLE RACE SIMULATION")
print("="*60)
print(f"\nCurrent Standings:")
print(f"Arsenal:   {ARSENAL_CURRENT} points (5 games left)")
print(f"Man City:  {CITY_CURRENT} points (6 games left)")
print(f"\nMax possible: Arsenal 85 pts, Man City 85 pts")

# Get upcoming fixtures
arsenal_upcoming = arsenal_fixtures[arsenal_fixtures['Result'].isna()].copy()
city_upcoming = city_fixtures[city_fixtures['Result'].isna()].copy()

print(f"\nArsenal remaining fixtures:")
print(arsenal_upcoming[['Date', 'Home Team', 'Away Team']])

print(f"\nMan City remaining fixtures:")
print(city_upcoming[['Date', 'Home Team', 'Away Team']])

# Function to predict match probabilities
def predict_match_probs(home_team, away_team, model, team_strength):
    """Predict win/draw/loss probabilities"""
    
    # Get team strengths (default to 1.5 if missing)
    home_str = team_strength.loc[home_team, 'avg_points'] if home_team in team_strength.index else 1.5
    away_str = team_strength.loc[away_team, 'avg_points'] if away_team in team_strength.index else 1.5
    
    # Simple form estimate (use strength as proxy)
    home_form = home_str
    away_form = away_str
    
    features = pd.DataFrame([{
        'home_strength': home_str,
        'away_strength': away_str,
        'strength_diff': home_str - away_str,
        'home_form': home_form,
        'away_form': away_form,
        'form_diff': home_form - away_form,
        'home_advantage': 1.0
    }])
    
    probs = model.predict_proba(features)[0]
    
    # Return as dict {D, L, W}
    return dict(zip(model.classes_, probs))

# Monte Carlo simulation
N_SIMS = 10000
arsenal_wins = 0
city_wins = 0
draws = 0

for sim in range(N_SIMS):
    arsenal_points = ARSENAL_CURRENT
    city_points = CITY_CURRENT
    
    # Simulate Arsenal fixtures
    for _, match in arsenal_upcoming.iterrows():
        home = match['Home Team']
        away = match['Away Team']
        
        probs = predict_match_probs(home, away, model, team_strength)
        outcome = np.random.choice(['D', 'L', 'W'], p=[probs.get('D', 0), probs.get('L', 0), probs.get('W', 0)])
        
        # Points from home team perspective
        if outcome == 'W':
            pts = 3
        elif outcome == 'D':
            pts = 1
        else:
            pts = 0
        
        # Add points to Arsenal
        if home == 'Arsenal':
            arsenal_points += pts
        else:
            arsenal_points += (3 - pts if outcome == 'L' else (1 if outcome == 'D' else 0))
    
    # Simulate Man City fixtures
    for _, match in city_upcoming.iterrows():
        home = match['Home Team']
        away = match['Away Team']
        
        probs = predict_match_probs(home, away, model, team_strength)
        outcome = np.random.choice(['D', 'L', 'W'], p=[probs.get('D', 0), probs.get('L', 0), probs.get('W', 0)])
        
        if outcome == 'W':
            pts = 3
        elif outcome == 'D':
            pts = 1
        else:
            pts = 0
        
        # Add points to Man City
        if home == 'Man City':
            city_points += pts
        else:
            city_points += (3 - pts if outcome == 'L' else (1 if outcome == 'D' else 0))
    
    # Determine winner
    if arsenal_points > city_points:
        arsenal_wins += 1
    elif city_points > arsenal_points:
        city_wins += 1
    else:
        draws += 1

# Results
print("\n" + "="*60)
print("SIMULATION RESULTS (10,000 runs)")
print("="*60)
print(f"\nArsenal wins title: {arsenal_wins/N_SIMS*100:.1f}%")
print(f"Man City wins title: {city_wins/N_SIMS*100:.1f}%")
print(f"Finish level on points: {draws/N_SIMS*100:.1f}%")