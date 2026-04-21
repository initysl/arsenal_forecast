import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('outputs/models/match_predictor.pkl')

# Use CURRENT season strength instead of historical
team_strength = pd.read_csv('outputs/models/current_season_strength.csv', index_col=0)

# Load Arsenal fixtures
arsenal_fixtures = pd.read_csv('data/raw/arsenal_epl_2025_26.csv')
arsenal_fixtures.columns = arsenal_fixtures.columns.str.strip()

# Load Man City fixtures
city_fixtures = pd.read_csv('data/raw/mancity_epl_2025_26.csv')
city_fixtures.columns = city_fixtures.columns.str.strip()

# Current points
ARSENAL_CURRENT = 70
CITY_CURRENT = 67

print("-"*60)
print("PREMIER LEAGUE TITLE RACE SIMULATION")
print("-"*60)
print(f"\nCurrent Standings:")
print(f"Arsenal:   {ARSENAL_CURRENT} points (5 games left)")
print(f"Man City:  {CITY_CURRENT} points (6 games left)")
print(f"\nCurrent form (pts/game):")
print(f"Arsenal:   {team_strength.loc['Arsenal', 'avg_points']:.2f}")
print(f"Man City:  {team_strength.loc['Man City', 'avg_points']:.2f}")

# Get upcoming fixtures
arsenal_upcoming = arsenal_fixtures[arsenal_fixtures['Result'].isna()].copy()
city_upcoming = city_fixtures[city_fixtures['Result'].isna()].copy()

print(f"\nArsenal remaining fixtures:")
for _, match in arsenal_upcoming.iterrows():
    opp = match['Away Team'] if match['Home Team'] == 'Arsenal' else match['Home Team']
    loc = 'H' if match['Home Team'] == 'Arsenal' else 'A'
    opp_str = team_strength.loc[opp, 'avg_points'] if opp in team_strength.index else 1.0
    print(f"  {loc} vs {opp:15s} (strength: {opp_str:.2f})")

print(f"\nMan City remaining fixtures:")
for _, match in city_upcoming.iterrows():
    opp = match['Away Team'] if match['Home Team'] == 'Man City' else match['Home Team']
    loc = 'H' if match['Home Team'] == 'Man City' else 'A'
    opp_str = team_strength.loc[opp, 'avg_points'] if opp in team_strength.index else 1.0
    print(f"  {loc} vs {opp:15s} (strength: {opp_str:.2f})")

# Function to predict match probabilities
def predict_match_probs(home_team, away_team, model, team_strength):
    """Predict win/draw/loss probabilities using CURRENT season strength"""
    
    # Get current season strengths
    home_str = team_strength.loc[home_team, 'avg_points'] if home_team in team_strength.index else 1.5
    away_str = team_strength.loc[away_team, 'avg_points'] if away_team in team_strength.index else 1.5
    
    # Use current season form
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
    return dict(zip(model.classes_, probs))

print(f"\nArsenal match-by-match probabilities:")

for _, match in arsenal_upcoming.iterrows():
    home = match['Home Team']
    away = match['Away Team']
    probs = predict_match_probs(home, away, model, team_strength)
    
    loc = 'H' if home == 'Arsenal' else 'A'
    opp = away if home == 'Arsenal' else home
    
    # Expected points
    exp_pts = probs.get('W', 0) * 3 + probs.get('D', 0) * 1
    if home != 'Arsenal':  # Arsenal is away, so W/L are flipped
        exp_pts = probs.get('L', 0) * 3 + probs.get('D', 0) * 1
    
    print(f"  {loc} vs {opp:15s} | W: {probs.get('W', 0)*100:4.1f}%  D: {probs.get('D', 0)*100:4.1f}%  L: {probs.get('L', 0)*100:4.1f}% | Exp pts: {exp_pts:.2f}")

print(f"\nMan City match-by-match probabilities:")

for _, match in city_upcoming.iterrows():
    home = match['Home Team']
    away = match['Away Team']
    probs = predict_match_probs(home, away, model, team_strength)
    
    loc = 'H' if home == 'Man City' else 'A'
    opp = away if home == 'Man City' else home
    
    exp_pts = probs.get('W', 0) * 3 + probs.get('D', 0) * 1
    if home != 'Man City':
        exp_pts = probs.get('L', 0) * 3 + probs.get('D', 0) * 1
    
    print(f"  {loc} vs {opp:15s} | W: {probs.get('W', 0)*100:4.1f}%  D: {probs.get('D', 0)*100:4.1f}%  L: {probs.get('L', 0)*100:4.1f}% | Exp pts: {exp_pts:.2f}")

    
# Monte Carlo simulation
N_SIMS = 10000
arsenal_wins = 0
city_wins = 0
ties = 0

arsenal_points_dist = []
city_points_dist = []

for sim in range(N_SIMS):
    arsenal_points = ARSENAL_CURRENT
    city_points = CITY_CURRENT
    
    # Simulate Arsenal fixtures
    for _, match in arsenal_upcoming.iterrows():
        home = match['Home Team']
        away = match['Away Team']
        
        probs = predict_match_probs(home, away, model, team_strength)
        outcome = np.random.choice(['D', 'L', 'W'], p=[probs.get('D', 0), probs.get('L', 0), probs.get('W', 0)])
        
        if outcome == 'W':
            pts_home = 3
        elif outcome == 'D':
            pts_home = 1
        else:
            pts_home = 0
        
        if home == 'Arsenal':
            arsenal_points += pts_home
        else:
            arsenal_points += (3 - pts_home)
    
    # Simulate Man City fixtures
    for _, match in city_upcoming.iterrows():
        home = match['Home Team']
        away = match['Away Team']
        
        probs = predict_match_probs(home, away, model, team_strength)
        outcome = np.random.choice(['D', 'L', 'W'], p=[probs.get('D', 0), probs.get('L', 0), probs.get('W', 0)])
        
        if outcome == 'W':
            pts_home = 3
        elif outcome == 'D':
            pts_home = 1
        else:
            pts_home = 0
        
        if home == 'Man City':
            city_points += pts_home
        else:
            city_points += (3 - pts_home)
    
    arsenal_points_dist.append(arsenal_points)
    city_points_dist.append(city_points)
    
    # Determine winner
    if arsenal_points > city_points:
        arsenal_wins += 1
    elif city_points > arsenal_points:
        city_wins += 1
    else:
        ties += 1

# Results
print("\n" + "-"*60)
print("SIMULATION RESULTS (10,000 runs)")
print("-"*60)
print(f"\nArsenal wins title: {arsenal_wins/N_SIMS*100:.1f}%")
print(f"Man City wins title: {city_wins/N_SIMS*100:.1f}%")
print(f"Finish level on points: {ties/N_SIMS*100:.1f}%")

print(f"\nExpected final points:")
print(f"Arsenal:  {np.mean(arsenal_points_dist):.1f} ± {np.std(arsenal_points_dist):.1f}")
print(f"Man City: {np.mean(city_points_dist):.1f} ± {np.std(city_points_dist):.1f}")