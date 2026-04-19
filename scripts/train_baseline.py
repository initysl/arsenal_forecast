from src.data.loaders import load_arsenal_matches
from src.features.basic_features import add_basic_features
from src.features.form_features import add_form_features
from src.features.opponent_features import add_opponent_strength
from src.data.split import train_test_split_temporal
from src.models.baseline import train_baseline_model, evaluate_model

# Load & engineer 
matches = load_arsenal_matches()
matches = add_basic_features(matches)
matches = add_opponent_strength(matches)  
matches = add_form_features(matches)

# THEN split
train, test = train_test_split_temporal(matches, test_season='2024/25')

# Train
model = train_baseline_model(train)

# Evaluate
evaluate_model(model, test)