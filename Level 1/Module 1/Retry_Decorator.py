# A retry decorator in Python is a design pattern used to automatically re-execute a function if it fails due to an exception.
# An exponential backoff retry decorator automatically retries a failing function by exponentially increasing the delay between each attempt.
import time
import random

def retry(max_attempts=4):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):

                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    print(f"Attempt {attempt} failed: {error}")
                    if attempt == max_attempts:
                        raise
                    delay = 2 ** (attempt - 1)
                    print(f"Waiting {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=4)
def call_api():

    if random.random() < 0.7:
        raise ConnectionError("API failed")
    return "API successful!"

# Testing
result = call_api()
print(result)