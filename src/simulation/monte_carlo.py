import numpy as np
import pandas as pd

def simulate_season(match_probs, n_simulations=10000):
    """
    Simulate season outcomes based on match probabilities
    
    Args:
        match_probs: DataFrame with columns ['D', 'L', 'W'] for each match
        n_simulations: Number of Monte Carlo runs
    
    Returns:
        Dictionary with outcome probabilities
    """
    results = []
    
    for _ in range(n_simulations):
        season_points = 0
        
        for idx, row in match_probs.iterrows():
            # Sample outcome based on probabilities
            outcome = np.random.choice(
                ['D', 'L', 'W'],
                p=[row.get('D', 0), row.get('L', 0), row.get('W', 0)]
            )
            
            # Add points
            if outcome == 'W':
                season_points += 3
            elif outcome == 'D':
                season_points += 1
        
        results.append(season_points)
    
    results = np.array(results)
    
    return {
        'mean_points': results.mean(),
        'std_points': results.std(),
        'min_points': results.min(),
        'max_points': results.max(),
        'percentile_25': np.percentile(results, 25),
        'percentile_75': np.percentile(results, 75),
        'distribution': results
    }