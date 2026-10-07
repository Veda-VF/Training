"""
loader.load("data.csv")
        ↓
Path("data.csv")
        ↓
Does file exist?
    ↓              ↓
   YES             NO
    ↓              ↓
check .suffix    return []
    ↓
   ".csv"
    ↓
csv.DictReader
    ↓
list of dictionaries
    ↓
loader.filter(...)
    ↓
condition(record)
    ↓
score > 0.8
    ↓
filtered records"""
import csv
import json
from pathlib import Path


class DatasetLoader:

    def load(self, filepath):

        path = Path(filepath)

        if not path.exists():
            print(f"File not found: {filepath}")
            return []

        if path.suffix == ".csv":

            with open(path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                return list(reader)

        elif path.suffix == ".json":

            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)

        elif path.suffix == ".jsonl":

            records = []

            with open(path, "r", encoding="utf-8") as file:

                for line in file:

                    if line.strip():
                        records.append(json.loads(line))

            return records

        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}"
            )

    def filter(self, records, condition):

        filtered_records = []

        for record in records:

            if condition(record):
                filtered_records.append(record)

        return filtered_records


# -----------------------------
# Create sample CSV
# -----------------------------

dataset = [
    {"name": "Alice", "score": 0.95},
    {"name": "Bob", "score": 0.72},
    {"name": "Charlie", "score": 0.88},
    {"name": "David", "score": 0.65}
]

with open("data.csv", "w", newline="", encoding="utf-8") as file:

    fieldnames = ["name", "score"]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(dataset)


# -----------------------------
# Test DatasetLoader
# -----------------------------

loader = DatasetLoader()

records = loader.load("data.csv")

print("All records:")
print(records)


filtered_records = loader.filter(
    records,
    lambda record: float(record["score"]) > 0.8
)

print("\nRecords with score greater than 0.8:")
print(filtered_records)