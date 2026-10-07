def validate_response(response, schema):
    errors = []

    for key, rules in schema.items():
        # Check if key exists
        if key not in response:
            errors.append(f"Missing key: {key}")
            continue
        value = response[key]

        # Check type
        if not isinstance(value, rules["type"]):
            errors.append(
                f"{key} must be of type {rules['type'].__name__}"
            )
            continue

        # Check minimum value
        if "min" in rules and value < rules["min"]:
            errors.append(
                f"{key} must be at least {rules['min']}"
            )

        # Check maximum value
        if "max" in rules and value > rules["max"]:
            errors.append(
                f"{key} must be at most {rules['max']}"
            )
    return errors

schema = {
    "name": {
        "type": str
    },
    "confidence": {
        "type": float,
        "min": 0,
        "max": 1
    },
    "score": {
        "type": int,
        "min": 1,
        "max": 10
    }
}

response = {
    "name": "AI Model",
    "confidence": 1.5,
    "score": 15
}

errors = validate_response(response, schema)
if errors:
    print("Validation failed:")

    for error in errors:
        print("-", error)
else:
    print("Response is valid.")