from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_baseline_model(train_df):
    """Logistic regression on home + form"""
    features = ['is_home', 'goals_for_last5', 'goals_against_last5', 'opponent_strength']
    X_train = train_df[features]
    y_train = train_df['result']

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train,y_train)

    return model

def evaluate_model(model, test_df):
    """Check accuracy"""
    features = ['is_home', 'goals_for_last5', 'goals_against_last5', 'opponent_strength']
    X_test = test_df[features]
    y_test = test_df['result']
    
    predictions = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\n", classification_report(y_test, predictions))
    
    return predictions