import pandas as pd

def add_opponent_strength(df):
    """Calculate opponent's average points per game"""
    df = df.copy()
    
    # Get all teams' average points
    team_strength = {}
    
    for team in df['HomeTeam'].unique():
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        
        home_points = home_games.apply(
            lambda x: 3 if x['FullTimeHomeGoals'] > x['FullTimeAwayGoals']
                     else (1 if x['FullTimeHomeGoals'] == x['FullTimeAwayGoals'] else 0),
            axis=1
        ).mean()
        
        away_points = away_games.apply(
            lambda x: 3 if x['FullTimeAwayGoals'] > x['FullTimeHomeGoals']
                     else (1 if x['FullTimeAwayGoals'] == x['FullTimeHomeGoals'] else 0),
            axis=1
        ).mean()
        
        team_strength[team] = (home_points + away_points) / 2
    
    # Add opponent strength for Arsenal matches
    df['opponent'] = df.apply(
        lambda x: x['AwayTeam'] if x['is_home'] else x['HomeTeam'], axis=1
    )
    df['opponent_strength'] = df['opponent'].map(team_strength)
    
    return df