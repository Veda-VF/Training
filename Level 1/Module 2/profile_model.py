"""
FLOWCHART:

 START
   |
   v
User sends POST request to /v1/profiles
   |
   v
Validate request data using ProfileModel (Pydantic)
   |
   v
Is the input data valid?
   |
   |---- NO ----> Return validation error (HTTP 422)
   |
  YES
   |
   v
Convert profile into a dictionary using
model_dump(exclude_none=True)
   |
   v
Store the profile in saved_profiles list
   |
   v
Return success message and created profile
   |
   v
User sends GET request to /v1/profiles
   |
   v
 Return all saved profiles
   |
   v
  END
"""
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI()

# Available AI models.
class ModelEnum(str, Enum):
    gpt = "gpt"
    claude = "claude"
    gemini = "gemini"


# Settings that control how the model responds.
class ModelParameters(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=500, gt=0)


# Simple safety preferences.
class SafetySettings(BaseModel):
    content_filter: bool = True
    pii_protection: bool = True


# Structure of a saved profile.
class ProfileModel(BaseModel):
    name: str = Field(min_length=3, max_length=50)
    default_model: ModelEnum
    default_parameters: ModelParameters
    safety_settings: SafetySettings
    description: str | None = None


# Temporary storage for profiles.
saved_profiles = []


# Create and save a profile.
@app.post("/v1/profiles")
def create_profile(profile: ProfileModel):
    # Convert the validated profile into a dictionary.
    # Fields with None values are excluded.
    profile_data = profile.model_dump(exclude_none=True)

    # Store the dictionary in our list.
    saved_profiles.append(profile_data)

    return {
        "message": "Profile created successfully",
        "profile": profile_data
    }


# Return all saved profiles.
@app.get("/v1/profiles")
def list_profiles():
    return saved_profiles

# uvicorn main:app --reload
# http://127.0.0.1:8000/docs
"""
Example POST request to create a profile:
{
  "name": "Study Assistant",
  "default_model": "gpt",
  "default_parameters": {
    "temperature": 0.7,
    "max_tokens": 500
  },
  "safety_settings": {
    "content_filter": true,
    "pii_protection": true
  },
  "description": "Profile for study and coding"
}
"""