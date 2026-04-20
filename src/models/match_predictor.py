import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def add_rolling_form(df, window=5):
    """Add recent form for each team"""
    df = df.sort_values('date').reset_index(drop=True)
    
    team_form = {}
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get recent form (last 5 matches)
        if home_team not in team_form:
            team_form[home_team] = []
        if away_team not in team_form:
            team_form[away_team] = []
        
        # Calculate current form before this match
        df.at[idx, 'home_form'] = sum(team_form[home_team][-window:]) / max(len(team_form[home_team][-window:]), 1) if team_form[home_team] else 1.5
        df.at[idx, 'away_form'] = sum(team_form[away_team][-window:]) / max(len(team_form[away_team][-window:]), 1) if team_form[away_team] else 1.5
        
        # Update form after match
        home_pts = 3 if row['home_goals'] > row['away_goals'] else (1 if row['home_goals'] == row['away_goals'] else 0)
        away_pts = 3 if row['away_goals'] > row['home_goals'] else (1 if row['away_goals'] == row['home_goals'] else 0)
        
        team_form[home_team].append(home_pts)
        team_form[away_team].append(away_pts)
    
    return df

def prepare_match_features(df, team_strength):
    """Engineer features for each match"""
    df = df.copy()
    
    # Add rolling form FIRST
    df = add_rolling_form(df)
    
    # Add team strength
    df['home_strength'] = df['home_team'].map(team_strength['avg_points'])
    df['away_strength'] = df['away_team'].map(team_strength['avg_points'])
    
    # Strength differential
    df['strength_diff'] = df['home_strength'] - df['away_strength']
    
    # Form differential
    df['form_diff'] = df['home_form'] - df['away_form']
    
    # Home advantage
    df['home_advantage'] = 1.0
    
    # Create target
    df['result'] = df.apply(
        lambda x: 'W' if x['home_goals'] > x['away_goals']
                  else ('D' if x['home_goals'] == x['away_goals'] else 'L'),
        axis=1
    )
    
    return df
def train_match_model(df, team_strength):
    """Train logistic regression model for match outcomes"""
    
    # Prepare features
    df = prepare_match_features(df, team_strength)
    
    # Drop rows with missing data
    df = df.dropna(subset=['home_strength', 'away_strength', 'home_form', 'away_form'])
    
    # Features and target
    features = ['home_strength', 'away_strength', 'strength_diff', 
                'home_form', 'away_form', 'form_diff', 'home_advantage']
    X = df[features]
    y = df['result']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train logistic regression with class balancing
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, df

if __name__ == "__main__":
    from src.data.prepare_training_data import prepare_historical_data
    from src.features.team_strength import calculate_team_strength
    
    # Load data
    df = prepare_historical_data()
    team_strength = calculate_team_strength(df)
    
    # Train model
    model, prepared_df = train_match_model(df, team_strength)
    
    # Save model and team strength
    joblib.dump(model, 'outputs/models/match_predictor.pkl')
    team_strength.to_csv('outputs/models/team_strength.csv')
    
    print("\nModel saved to outputs/models/match_predictor.pkl")
    print("Team strength saved to outputs/models/team_strength.csv")