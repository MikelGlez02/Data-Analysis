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
    # Word frequency in descriptions
    if "desc" in data.columns:
        from collections import Counter
        from nltk.tokenize import word_tokenize
        all_words = " ".join(data["desc"].dropna()).lower()
        tokens = word_tokenize(all_words)
        word_freq = Counter(tokens)
        common_words = word_freq.most_common(20)
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words), palette="mako")
        plt.title("Most Common Words in Descriptions")
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.show()
    # Relationship between sodium and rating
    if "sodium" in data.columns and "rating" in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="sodium", y="rating", data=data)
        plt.title("Sodium vs Rating")
        plt.xlabel("Sodium (mg)")
        plt.ylabel("Rating")
        plt.show()
    # Temporal analysis of ratings by publication date
    if "date" in data.columns and "rating" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if data["date"].notna().any():
            avg_rating_by_date = data.groupby(data["date"].dt.to_period("M"))["rating"].mean()
            plt.figure(figsize=(12, 6))
            avg_rating_by_date.plot()
            plt.title("Average Rating Over Time")
            plt.xlabel("Date")
            plt.ylabel("Average Rating")
            plt.show()


