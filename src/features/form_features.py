import pandas as pd

def add_form_features(df, window=5):
    """Add rolling form over last N matches"""
    df = df.copy()
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    # Rolling averages
    df['goals_for_last5'] = df['goals_for'].rolling(window=window, min_periods=1).mean()
    df['goals_against_last5'] = df['goals_against'].rolling(window=window, min_periods=1).mean()
    df['points_last5'] = df['points'].rolling(window=window, min_periods=1).mean()
    
    return df

# Test
from src.data.loaders import load_arsenal_matches
from src.features.basic_features import add_basic_features

matches = load_arsenal_matches()
matches = add_basic_features(matches)
matches = add_form_features(matches)

print(matches[['MatchDate', 'result', 'points', 'points_last5']].tail(10))