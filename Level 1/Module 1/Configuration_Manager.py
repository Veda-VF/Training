"""
Flow of Code Logic:
.env
   ↓
config.json
   ↓
ConfigurationManager
   ↓
merge settings
   ↓
.env overrides JSON
   ↓
validate required keys
   ↓
get(key, default)
"""
import json
from pathlib import Path


class ConfigurationManager:

    def __init__(self, env_file=".env", config_file="config.json"):
        self.settings = {}

        self.load_json(config_file)
        self.load_env(env_file)

    def load_json(self, filepath):

        path = Path(filepath)

        if not path.exists():
            print(f"Config file not found: {filepath}")
            return

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.settings.update(data)

    def load_env(self, filepath):

        path = Path(filepath)

        if not path.exists():
            print(f"Environment file not found: {filepath}")
            return

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                # Ignore empty lines and comments
                if not line or line.startswith("#"):
                    continue

                if "=" in line:

                    key, value = line.split("=", 1)

                    key = key.strip()
                    value = value.strip()

                    self.settings[key] = value

    def validate(self, required_keys):

        missing_keys = []

        for key in required_keys:

            if key not in self.settings:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {missing_keys}"
            )

    def get(self, key, default=None):

        return self.settings.get(key, default)
    
# --------------------------------
# Create config.json
# --------------------------------

config_data = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "debug": False
}

with open("config.json", "w", encoding="utf-8") as file:
    json.dump(config_data, file, indent=4)

# --------------------------------
# Create .env
# --------------------------------

with open(".env", "w", encoding="utf-8") as file:

    file.write("API_KEY=abc123\n")
    file.write("MODEL=claude\n")

# --------------------------------
# Test ConfigurationManager
# --------------------------------

config = ConfigurationManager()

print("Model:", config.get("MODEL"))
print("API Key:", config.get("API_KEY"))
print("Temperature:", config.get("temperature"))
print("Max Tokens:", config.get("max_tokens"))

print("Unknown value:", config.get("database", "localhost"))


# --------------------------------
# Validate required keys
# --------------------------------

config.validate([
    "API_KEY",
    "MODEL",
    "temperature"
])

print("Configuration is valid!")