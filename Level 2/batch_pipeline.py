#(TOPIC: Structured Outputs)

import os
import time
import random
import logging
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MAX_RETRIES = 3
REQUEST_DELAY = 0.5  # Seconds between processing attempts

# Configure logging to save errors in a file.
logging.basicConfig(
    filename="pipeline_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# 2. SENTIMENT ANALYZER
# --------------------------------------------------

def analyze_sentiment(text):
    """
    Simple local sentiment analyzer.
    Returns sentiment and confidence.
    Replace this function with an API call if needed.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text is empty or invalid.")

    # Simple keyword lists for demonstration.
    positive_words = {
        "good", "great", "excellent", "happy",
        "love", "wonderful", "best", "amazing"
    }

    negative_words = {
        "bad", "terrible", "poor", "sad",
        "hate", "worst", "awful", "disappointed"
    }

    words = text.lower().split()

    positive_count = sum(word.strip(".,!?") in positive_words
                         for word in words)
    negative_count = sum(word.strip(".,!?") in negative_words
                         for word in words)

    # Determine sentiment and illustrative confidence.
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(
            0.60 + 0.10 * (positive_count - negative_count), 0.95
        )
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(
            0.60 + 0.10 * (negative_count - positive_count), 0.95
        )
    else:
        sentiment = "neutral"
        confidence = 0.60

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }


# --------------------------------------------------
# 3. RETRY WITH EXPONENTIAL BACKOFF
# --------------------------------------------------

def process_with_retry(text):
    """
    Retry failed processing up to MAX_RETRIES times.
    Wait time increases after every failed attempt.
    """

    for attempt in range(MAX_RETRIES):
        try:
            # Limit processing frequency.
            time.sleep(REQUEST_DELAY)

            # Analyze the text.
            result = analyze_sentiment(text)

            return result

        except Exception as error:
            logging.error(
                "Attempt %d failed for text %.100r: %s",
                attempt + 1, text, error
            )

            # Do not wait after the final attempt.
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)

    # Return failure after all attempts are exhausted.
    return None


# --------------------------------------------------
# 4. BATCH PROCESSING PIPELINE
# --------------------------------------------------

def run_pipeline():
    file_path = input("Enter the CSV file path: ").strip().strip('"')

    # Check whether the file exists.
    if not os.path.isfile(file_path):
        print("Error: CSV file not found.")
        return

    try:
        # Load the CSV file.
        df = pd.read_csv(file_path)

        if df.empty:
            print("Error: CSV file is empty.")
            return

        # Expect a column named 'text'.
        if "text" not in df.columns:
            print("Error: CSV must contain a 'text' column.")
            return

    except Exception as error:
        print("Could not read CSV:", error)
        return

    results = []
    success_count = 0
    failure_count = 0
    confidence_scores = []

    print(f"\nProcessing {len(df)} text entries...\n")

    # Progress bar displays completed rows.
    for row_number, text in enumerate(
        tqdm(df["text"], desc="Processing", unit="text"),
        start=1
    ):
        result = process_with_retry(text)

        if result is not None:
            success_count += 1
            confidence_scores.append(result["confidence"])

            results.append({
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "status": "success"
            })

        else:
            failure_count += 1

            results.append({
                "text": text,
                "sentiment": None,
                "confidence": None,
                "status": "failed"
            })

            logging.error(
                "Row %d failed after all retries.", row_number
            )

    # --------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------

    output_path = os.path.splitext(file_path)[0] + "_results.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # --------------------------------------------------
    # 6. FINAL SUMMARY REPORT
    # --------------------------------------------------

    average_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0
    )

    print("\n========== BATCH SUMMARY ==========")
    print("Total entries:       ", len(df))
    print("Successful entries:  ", success_count)
    print("Failed entries:      ", failure_count)
    print(f"Success rate:         {success_count / len(df) * 100:.2f}%")
    print(f"Average confidence:   {average_confidence:.2f}")
    print("Results saved to:    ", output_path)
    print("Error log:            pipeline_errors.log")
    print("===================================")


# --------------------------------------------------
# 7. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

# python -m pip install pandas tqdm

"""
                  START
                    |
                    v
          Ask user for CSV path
                    |
                    v
          Does the file exist?
             /             \
           No               Yes
           |                 |
           v                 v
      Show error       Read CSV file
           |                 |
           v                 v
          END          Check text column
                             |
                             v
                    Process each text row
                             |
                             v
                    Apply rate limiting
                             |
                             v
                    Analyze sentiment
                             |
                             v
                    Did processing succeed?
                       /           \
                     Yes            No
                      |              |
                      v              v
               Save result     Retry with backoff
                      |              |
                      |       Still failing?
                      |          /      \
                      |        No        Yes
                      |         |         |
                      |         v         v
                      |    Save result  Log failure
                      |                   |
                      +---------+---------+
                                |
                                v
                         Update progress bar
                                |
                                v
                         More rows remaining?
                           /          \
                         Yes           No
                          |             |
                          +-- Repeat    v
                                  Save results CSV
                                        |
                                        v
                                 Print summary report
                                        |
                                        v
                                       END
"""
