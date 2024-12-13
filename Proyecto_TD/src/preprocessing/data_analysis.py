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

    # Additional analysis can be added here
