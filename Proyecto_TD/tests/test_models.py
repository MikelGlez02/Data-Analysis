# tests/test_models.py
from models.regression import train_regression_model, evaluate_regression_model

def test_model_training_and_evaluation():
    # Mock data for testing
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    data = pd.DataFrame({
        "desc": ["This is a sample recipe", "Another example recipe"],
        "rating": [4.5, 3.0]
    })
    train_regression_model(vectorizer="tfidf", epochs=1, batch_size=1, learning_rate=0.01)
    evaluate_regression_model(metric="mae")
    print("Model training and evaluation test passed.")
