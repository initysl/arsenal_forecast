def train_test_split_temporal(df, test_season='2023/24'):
    """Split by season - train on old, test on recent"""
    train =  df[df['Season'] != test_season]
    test = df[df['Season'] == test_season]

    print(f"Train: {len(train)} matches")
    print(f"Test: {len(test)} matches")

    return train, test