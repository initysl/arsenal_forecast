import pandas as pd

def prepare_historical_data(filepath='data/raw/epl_historical_2022_2024.csv'):
    """Clean and engineer features from historical data"""
    df = pd.read_csv(filepath)
    
    # Basic result features
    df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
    df['draw'] = (df['home_goals'] == df['away_goals']).astype(int)
    df['away_win'] = (df['home_goals'] < df['away_goals']).astype(int)
    
    # Points
    df['home_points'] = df.apply(
        lambda x: 3 if x['home_goals'] > x['away_goals'] 
                  else (1 if x['home_goals'] == x['away_goals'] else 0),
        axis=1
    )
    df['away_points'] = df.apply(
        lambda x: 3 if x['away_goals'] > x['home_goals']
                  else (1 if x['home_goals'] == x['away_goals'] else 0),
        axis=1
    )
    
    # Goal difference
    df['home_gd'] = df['home_goals'] - df['away_goals']
    df['away_gd'] = df['away_goals'] - df['home_goals']
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    df = prepare_historical_data()
    df.to_csv('data/processed/epl_historical_clean.csv', index=False)
    print(f"Processed {len(df)} historical matches")
    print(df.head())