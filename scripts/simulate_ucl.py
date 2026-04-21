import pandas as pd
import numpy as np
import joblib

# Load model and current season strength
model = joblib.load('outputs/models/match_predictor.pkl')
team_strength = pd.read_csv('outputs/models/current_season_strength.csv', index_col=0)

# Load Arsenal UCL fixtures
ucl_fixtures = pd.read_csv('data/raw/arsenal_ucl_2025_26.csv')
ucl_fixtures.columns = ucl_fixtures.columns.str.strip()

# Load Atletico UCL fixtures
atleti_fixtures = pd.read_csv('data/raw/atleti_ucl_2025_26.csv')
atleti_fixtures.columns = atleti_fixtures.columns.str.strip()

# Calculate UCL strength from completed matches
def calculate_ucl_strength(fixtures, team_name):
    """Calculate team strength from UCL completed matches"""
    completed = fixtures[fixtures['Result'].notna()].copy()
    
    if len(completed) == 0:
        return 1.5  # Default if no matches
    
    def parse_result(row):
        if pd.isna(row['Result']):
            return None, None
        try:
            home_goals, away_goals = map(int, row['Result'].split(' - '))
            return home_goals, away_goals
        except:
            return None, None
    
    completed[['home_goals', 'away_goals']] = completed.apply(
        lambda row: pd.Series(parse_result(row)), axis=1
    )
    
    completed = completed.dropna(subset=['home_goals', 'away_goals'])
    
    total_points = 0
    for _, match in completed.iterrows():
        if match['Home Team'] == team_name:
            if match['home_goals'] > match['away_goals']:
                total_points += 3
            elif match['home_goals'] == match['away_goals']:
                total_points += 1
        else:  # Away
            if match['away_goals'] > match['home_goals']:
                total_points += 3
            elif match['home_goals'] == match['away_goals']:
                total_points += 1
    
    return total_points / len(completed) if len(completed) > 0 else 1.5

ARSENAL_UCL_STRENGTH = calculate_ucl_strength(ucl_fixtures, 'Arsenal')
ATLETICO_UCL_STRENGTH = calculate_ucl_strength(atleti_fixtures, 'Atleti')
ARSENAL_EPL_STRENGTH = team_strength.loc['Arsenal', 'avg_points']

print("-"*60)
print("UEFA CHAMPIONS LEAGUE KNOCKOUT SIMULATION")
print("-"*60)

# Get semi-final fixtures
semi_final = ucl_fixtures[ucl_fixtures['Result'].isna()].copy()

if len(semi_final) == 0:
    print("\nNo upcoming UCL matches found!")
    exit()

print(f"\nSemi-Final: Arsenal vs Atletico Madrid (Two Legs)")
print(semi_final[['Date', 'Home Team', 'Away Team']])

print(f"\nTeam Strength:")
print(f"Arsenal  - EPL: {ARSENAL_EPL_STRENGTH:.2f} pts/game | UCL: {ARSENAL_UCL_STRENGTH:.2f} pts/game")
print(f"Atletico - UCL: {ATLETICO_UCL_STRENGTH:.2f} pts/game")

# Predict match probabilities function
def predict_match_probs(home_team, away_team, home_str, away_str, model):
    """Predict match probabilities"""
    features = pd.DataFrame([{
        'home_strength': home_str,
        'away_strength': away_str,
        'strength_diff': home_str - away_str,
        'home_form': home_str,
        'away_form': away_str,
        'form_diff': home_str - away_str,
        'home_advantage': 1.0
    }])
    
    probs = model.predict_proba(features)[0]
    return dict(zip(model.classes_, probs))

# Get probabilities for each leg
leg1 = semi_final.iloc[0]  # First leg
leg2 = semi_final.iloc[1]  # Second leg

# Leg 1 probabilities
leg1_home = leg1['Home Team']
leg1_away = leg1['Away Team']

if leg1_home == 'Arsenal':
    leg1_probs = predict_match_probs(leg1_home, leg1_away, ARSENAL_UCL_STRENGTH, ATLETICO_UCL_STRENGTH, model)
else:
    leg1_probs = predict_match_probs(leg1_home, leg1_away, ATLETICO_UCL_STRENGTH, ARSENAL_UCL_STRENGTH, model)

# Leg 2 probabilities
leg2_home = leg2['Home Team']
leg2_away = leg2['Away Team']

if leg2_home == 'Arsenal':
    leg2_probs = predict_match_probs(leg2_home, leg2_away, ARSENAL_UCL_STRENGTH, ATLETICO_UCL_STRENGTH, model)
else:
    leg2_probs = predict_match_probs(leg2_home, leg2_away, ATLETICO_UCL_STRENGTH, ARSENAL_UCL_STRENGTH, model)

print(f"\nLeg 1 ({leg1_home} vs {leg1_away}):")
if leg1_home == 'Arsenal':
    print(f"  Arsenal Win: {leg1_probs['W']*100:.1f}%  Draw: {leg1_probs['D']*100:.1f}%  Atletico Win: {leg1_probs['L']*100:.1f}%")
else:
    print(f"  Atletico Win: {leg1_probs['W']*100:.1f}%  Draw: {leg1_probs['D']*100:.1f}%  Arsenal Win: {leg1_probs['L']*100:.1f}%")

print(f"\nLeg 2 ({leg2_home} vs {leg2_away}):")
if leg2_home == 'Arsenal':
    print(f"  Arsenal Win: {leg2_probs['W']*100:.1f}%  Draw: {leg2_probs['D']*100:.1f}%  Atletico Win: {leg2_probs['L']*100:.1f}%")
else:
    print(f"  Atletico Win: {leg2_probs['W']*100:.1f}%  Draw: {leg2_probs['D']*100:.1f}%  Arsenal Win: {leg2_probs['L']*100:.1f}%")

# Simulate aggregate score over two legs
def simulate_two_leg_tie(leg1_home, leg1_away, leg2_home, leg2_away, 
                         arsenal_str, atletico_str, model, n_sims=10000):
    """Simulate two-legged knockout tie"""
    
    arsenal_advance = 0
    atletico_advance = 0
    
    for _ in range(n_sims):
        arsenal_goals = 0
        atletico_goals = 0
        
        # Simulate Leg 1
        if leg1_home == 'Arsenal':
            probs = predict_match_probs(leg1_home, leg1_away, arsenal_str, atletico_str, model)
            outcome = np.random.choice(['D', 'L', 'W'], p=[probs['D'], probs['L'], probs['W']])
            
            # Simple goal estimation based on outcome
            if outcome == 'W':  # Arsenal win
                arsenal_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                atletico_goals += np.random.choice([0, 1], p=[0.6, 0.4])
            elif outcome == 'D':
                arsenal_goals += np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                atletico_goals += arsenal_goals  # Draw
            else:  # Arsenal loss
                atletico_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                arsenal_goals += np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            probs = predict_match_probs(leg1_home, leg1_away, atletico_str, arsenal_str, model)
            outcome = np.random.choice(['D', 'L', 'W'], p=[probs['D'], probs['L'], probs['W']])
            
            if outcome == 'W':  # Atletico win
                atletico_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                arsenal_goals += np.random.choice([0, 1], p=[0.6, 0.4])
            elif outcome == 'D':
                arsenal_goals += np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                atletico_goals += arsenal_goals
            else:
                arsenal_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                atletico_goals += np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Simulate Leg 2
        if leg2_home == 'Arsenal':
            probs = predict_match_probs(leg2_home, leg2_away, arsenal_str, atletico_str, model)
            outcome = np.random.choice(['D', 'L', 'W'], p=[probs['D'], probs['L'], probs['W']])
            
            if outcome == 'W':
                arsenal_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                atletico_goals += np.random.choice([0, 1], p=[0.6, 0.4])
            elif outcome == 'D':
                leg2_home_goals = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                arsenal_goals += leg2_home_goals
                atletico_goals += leg2_home_goals
            else:
                atletico_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                arsenal_goals += np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            probs = predict_match_probs(leg2_home, leg2_away, atletico_str, arsenal_str, model)
            outcome = np.random.choice(['D', 'L', 'W'], p=[probs['D'], probs['L'], probs['W']])
            
            if outcome == 'W':
                atletico_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                arsenal_goals += np.random.choice([0, 1], p=[0.6, 0.4])
            elif outcome == 'D':
                leg2_home_goals = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                atletico_goals += leg2_home_goals
                arsenal_goals += leg2_home_goals
            else:
                arsenal_goals += np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                atletico_goals += np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Determine winner (aggregate score, away goals not modeled for simplicity)
        if arsenal_goals > atletico_goals:
            arsenal_advance += 1
        elif atletico_goals > arsenal_goals:
            atletico_advance += 1
        else:
            # If tied on aggregate, 50-50 (simplified - ignores away goals rule)
            if np.random.random() < 0.5:
                arsenal_advance += 1
            else:
                atletico_advance += 1
    
    return arsenal_advance, atletico_advance

# Run simulation
arsenal_advance, atletico_advance = simulate_two_leg_tie(
    leg1_home, leg1_away, leg2_home, leg2_away,
    ARSENAL_UCL_STRENGTH, ATLETICO_UCL_STRENGTH, model
)

print("\n" + "-"*60)
print("SEMI-FINAL SIMULATION (10,000 runs)")
print("-"*60)
print(f"\nArsenal advance to final: {arsenal_advance/10000*100:.1f}%")
print(f"Atletico advance to final: {atletico_advance/10000*100:.1f}%")

# Assume 50% chance to win final if Arsenal advances
ucl_win_probability = (arsenal_advance / 10000) * 0.5

print(f"\n" + "-"*60)
print("CHAMPIONS LEAGUE TROPHY PROBABILITY")
print("-"*60)
print(f"\nAssuming 50% chance to win final if Arsenal advances:")
print(f"Arsenal wins UCL: {ucl_win_probability*100:.1f}%")

# Save result
with open('outputs/simulations/ucl_probability.txt', 'w') as f:
    f.write(f"Arsenal UCL Win Probability: {ucl_win_probability*100:.1f}%\n")
    f.write(f"Arsenal advance from semi-final: {arsenal_advance/10000*100:.1f}%\n")

print("\nResults saved to outputs/simulations/ucl_probability.txt")