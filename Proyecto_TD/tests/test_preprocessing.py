# tests/test_preprocessing.py
from preprocessing.text_cleaner import clean_text

def test_clean_text():
    input_text = "This is a TEST text with punctuation!"
    expected_output = "test text punctuation"
    cleaned_text = clean_text(input_text)
    assert cleaned_text == expected_output, f"Expected {expected_output}, but got {cleaned_text}"
    print("Preprocessing test passed.")
