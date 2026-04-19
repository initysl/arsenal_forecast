import pandas as pd

def add_basic_features(df):
    """Add win/loss, points, goal difference"""
    df = df.copy()

    # Arsenal goals for/against
    df['goals_for'] = df.apply(
        lambda x: x['FullTimeHomeGoals'] if x['is_home'] else x['FullTimeAwayGoals'], axis=1
    )
    df['goals_against'] = df.apply(
        lambda x: x['FullTimeAwayGoals'] if x['is_home'] else x['FullTimeHomeGoals'], axis=1
    )
    
    # Result
    df['result'] = df.apply(
        lambda x: 'W' if x['goals_for'] > x['goals_against']
                  else ('D' if x['goals_for'] == x['goals_against'] else 'L'),
        axis=1
    )
    # Points
    df['points'] = df['result'].map({'W': 3, 'D': 1, 'L': 0})

    return df

# Test
from src.data.loaders import load_arsenal_matches
matches = load_arsenal_matches()
matches = add_basic_features(matches)
print(matches[['MatchDate', 'is_home', 'goals_for', 'goals_against', 'result', 'points']].head())