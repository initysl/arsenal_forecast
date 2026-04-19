import pandas as pd

def load_arsenal_matches(filepath='data/raw/epl_final.csv'):
    """Load and filter Arsenal matches"""
    df = pd.read_csv(filepath)

    # Filter Arsenal games -Get last 3 seasons
    seasons = df['Season'].unique()
    # print(f"Available seasons: {sorted(seasons)}")

    last_3_seasons = sorted(seasons)[-3:]  
    print(f"Using: {last_3_seasons}")

    # Filter
    df_filtered = df[df['Season'].isin(last_3_seasons)]

    arsenal = df_filtered[
        (df_filtered['HomeTeam'] == 'Arsenal') | 
        (df_filtered['AwayTeam'] == 'Arsenal')
    ].copy()

    # is_home flag - Home advatage
    arsenal['is_home'] = arsenal['HomeTeam'] == 'Arsenal'

    return arsenal


if __name__ == "__main__":
    matches = load_arsenal_matches()
    print(f"Loaded {len(matches)} Arsenal matches")
    matches.to_csv('data/processed/arsenal_matches.csv', index=False)