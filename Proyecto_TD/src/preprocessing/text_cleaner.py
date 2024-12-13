# preprocessing/text_cleaner.py
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize stop words
nltk_stopwords = set(stopwords.words("english"))

def clean_text(text):
    """Cleans text by removing special characters, stopwords, and tokenizing."""
    # Lowercase and remove special characters
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in nltk_stopwords]

    return " ".join(tokens)
