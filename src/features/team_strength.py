import pandas as pd

def calculate_team_strength(df):
    """Calculate each team's average points per game"""
    team_stats = {}
    
    for team in set(list(df['home_team'].unique()) + list(df['away_team'].unique())):
        home_matches = df[df['home_team'] == team]
        away_matches = df[df['away_team'] == team]
        
        total_points = home_matches['home_points'].sum() + away_matches['away_points'].sum()
        total_matches = len(home_matches) + len(away_matches)
        
        avg_points = total_points / total_matches if total_matches > 0 else 1.0
        
        team_stats[team] = {
            'avg_points': avg_points,
            'matches': total_matches,
            'goals_scored': (home_matches['home_goals'].sum() + away_matches['away_goals'].sum()) / total_matches if total_matches > 0 else 0,
            'goals_conceded': (home_matches['away_goals'].sum() + away_matches['home_goals'].sum()) / total_matches if total_matches > 0 else 0
        }
    
    return pd.DataFrame(team_stats).T

if __name__ == "__main__":
    from src.data.prepare_training_data import prepare_historical_data
    
    df = prepare_historical_data()
    team_strength = calculate_team_strength(df)
    
    team_strength.to_csv('data/processed/team_strength.csv')
    print("\nTop 10 teams by strength:")
    print(team_strength.sort_values('avg_points', ascending=False).head(10))