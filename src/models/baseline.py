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


def predict_probabilities(model, match_df):
    """Get probabilities for each outcome"""
    features = ['is_home', 'goals_for_last5', 'goals_against_last5', 'opponent_strength']
    X = match_df[features]
    
    # Get probabilities for each class
    probs = model.predict_proba(X)
    
    # Convert to dataframe
    prob_df = pd.DataFrame(
        probs,
        columns=model.classes_,  # ['D', 'L', 'W']
        index=match_df.index
    )
    
    return prob_df


def evaluate_model(model, test_df):
    features = ['is_home', 'goals_for_last5', 'goals_against_last5', 'opponent_strength']
    X_test = test_df[features]
    y_test = test_df['result']
    
    # Hard predictions
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\n", classification_report(y_test, predictions))
    
    # Probabilities
    probs = predict_probabilities(model, test_df)
    print("\nSample probabilities:")
    print(probs.head(10))
    
    return predictions, probs