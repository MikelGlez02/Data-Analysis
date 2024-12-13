# models/regression.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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

    # Evaluate model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
