# preprocessing/data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_relationships(data):
    """Analyzes the relationship between 'rating' and other variables."""
    # Plot categories vs rating
    if "categories" in data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="categories", y="rating", data=data)
        plt.xticks(rotation=45, ha="right")
        plt.title("Rating vs Categories")
        plt.show()
    # Additional analysis: rating distribution
    if "rating" in data.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data["rating"], kde=True, bins=20)
        plt.title("Distribution of Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.show()
    # Analyze correlation with numerical variables
    numerical_columns = [col for col in data.columns if data[col].dtype in ["int64", "float64"] and col != "rating"]
    if numerical_columns:
        correlation_matrix = data[["rating"] + numerical_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()
    # Top categories by average rating
    if "categories" in data.columns:
        top_categories = data.groupby("categories")["rating"].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
        plt.title("Top Categories by Average Rating")
        plt.xlabel("Average Rating")
        plt.ylabel("Categories")
        plt.show()

