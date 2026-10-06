import string

def clean_text(text):
    text = text.lower()
    cleaned = ""

    for char in text:
        if char not in string.punctuation:
            cleaned += char
    return cleaned


def tokenize(text):
    return text.split()

def count_frequency(words):
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency


def pipeline(*functions):
    def run(data):
        for function in functions:
            data = function(data)
        return data
    return run


# Create pipeline
text_pipeline = pipeline(
    clean_text,
    tokenize,
    count_frequency
)

# Get input
text = input("Enter text: ")

# Run pipeline
result = text_pipeline(text)

print("Word frequency:")
print(result)