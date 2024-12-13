# models/regression.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def train_regression_model(vectorizer="tfidf", epochs=20, batch_size=32, learning_rate=0.001):
    # Load and preprocess data
    data = pd.read_json("data/processed/recipes_cleaned.json")
    if vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data["desc"].fillna("")).toarray()
    elif vectorizer == "word2vec":
        # Word2Vec implementation placeholder
        pass
    else:
        raise ValueError("Unsupported vectorizer type.")

    y = data["rating"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and test data for evaluation
    global saved_model, X_test_global, y_test_global
    saved_model = model
    X_test_global = X_test
    y_test_global = y_test

def evaluate_regression_model(metric="mae"):
    global saved_model, X_test_global, y_test_global
    predictions = saved_model.predict(X_test_global)

    if metric == "mae":
        result = mean_absolute_error(y_test_global, predictions)
    elif metric == "mse":
        result = mean_squared_error(y_test_global, predictions)
    elif metric == "r2":
        result = r2_score(y_test_global, predictions)
    else:
        raise ValueError("Unsupported evaluation metric.")

    print(f"Evaluation result ({metric}): {result}")
