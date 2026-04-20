import pandas as pd

def calculate_current_season_strength(arsenal_file, city_file, full_season_file=None):
    """
    Calculate team strength from 2025/26 completed matches only
    
    Args:
        arsenal_file: Arsenal's 2025/26 fixtures
        city_file: Man City's 2025/26 fixtures
        full_season_file: Optional - full EPL 2025/26 data for all teams
    """
    
    # Load Arsenal and City data
    arsenal = pd.read_csv(arsenal_file)
    city = pd.read_csv(city_file)
    
    # Clean column names
    arsenal.columns = arsenal.columns.str.strip()
    city.columns = city.columns.str.strip()
    
    # Combine into one dataframe
    all_matches = pd.concat([arsenal, city], ignore_index=True)
    
    # If full EPL dataset, use it instead
    if full_season_file:
        all_matches = pd.read_csv(full_season_file)
        all_matches.columns = all_matches.columns.str.strip()
    
    # Filter completed matches only
    completed = all_matches[all_matches['Result'].notna()].copy()
    
    # Parse results
    def parse_result(row):
        """Extract goals from result string"""
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
    
    # Drop unparseable results
    completed = completed.dropna(subset=['home_goals', 'away_goals'])
    
    # Calculate points
    completed['home_points'] = completed.apply(
        lambda x: 3 if x['home_goals'] > x['away_goals']
                  else (1 if x['home_goals'] == x['away_goals'] else 0),
        axis=1
    )
    completed['away_points'] = completed.apply(
        lambda x: 3 if x['away_goals'] > x['home_goals']
                  else (1 if x['away_goals'] == x['home_goals'] else 0),
        axis=1
    )
    
    # Calculate team statistics
    team_stats = {}
    
    all_teams = set(list(completed['Home Team'].unique()) + list(completed['Away Team'].unique()))
    
    for team in all_teams:
        home_matches = completed[completed['Home Team'] == team]
        away_matches = completed[completed['Away Team'] == team]
        
        total_points = home_matches['home_points'].sum() + away_matches['away_points'].sum()
        total_matches = len(home_matches) + len(away_matches)
        
        goals_scored = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
        goals_conceded = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
        
        if total_matches > 0:
            team_stats[team] = {
                'avg_points': total_points / total_matches,
                'matches': total_matches,
                'goals_scored': goals_scored / total_matches,
                'goals_conceded': goals_conceded / total_matches,
                'total_points': total_points
            }
    
    # Convert to DataFrame
    strength_df = pd.DataFrame(team_stats).T
    strength_df = strength_df.sort_values('avg_points', ascending=False)
    
    return strength_df

if __name__ == "__main__":
    # Calculate current season strength from FULL EPL dataset
    strength = calculate_current_season_strength(
        'data/raw/arsenal_epl_2025_26.csv',
        'data/raw/mancity_epl_2025_26.csv',
        full_season_file='data/raw/epl_full_2025_26.csv'  # Use this
    )
    
    # Save
    strength.to_csv('outputs/models/current_season_strength.csv')
    
    print("Current 2025/26 Season Strength (Top 10):")
    print(strength.head(10))
    
    print("\nKey teams:")
    for team in ['Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Tottenham']:
        if team in strength.index:
            print(f"{team}: {strength.loc[team, 'avg_points']:.2f} pts/game, {strength.loc[team, 'total_points']:.0f} total pts")