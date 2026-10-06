"""
Simple Text Preprocessor
Practices: strings, sets, lists, loops, functions,
conditions, list comprehension, and user input.
"""

import string

def preprocess_text(text):
    """Clean text and return a list of useful tokens."""

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation
    cleaned_text = ""

    for char in text:
        if char not in string.punctuation:
            cleaned_text += char

    # 3. Convert text into words
    tokens = cleaned_text.split()

    # 4. Define stop words
    stop_words = {
        "is", "the", "a", "an", "and",
        "for", "to", "of", "in", "are"
    }

    # 5. Remove stop words
    cleaned_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return cleaned_tokens

# Take paragraphs from the user

print("Enter paragraph:\n")

text1 = input("Paragraph : ")

# Process and display the results

print("\n--- Cleaned Results ---")

print("Paragraph:", preprocess_text(text1))