"""
Configuration settings for the LLM-based mobility simulation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# DeepBricks API configuration
DEEPBRICKS_API_KEY = os.getenv("DEEPBRICKS_API_KEY")
DEEPBRICKS_BASE_URL = os.getenv("DEEPBRICKS_BASE_URL")
USE_DEEPBRICKS_API = True  # Always use DeepBricks API

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 700

# Simulation Parameters
NUM_DAYS_TO_SIMULATE = 3
SIMULATION_START_DATE = "2025-02-03"
USE_GOOGLE_MAPS = True  # If False, will use OSM

# File Paths
PERSONA_DATA_PATH = "data/personas.json"
RESULTS_DIR = "data/results/"

# Activity Types
ACTIVITY_TYPES = [
    "sleep",
    "work",
    "shopping",
    "dining",
    "recreation",
    "healthcare",
    "social",
    "education",
    "leisure",
    "errands"
]

# Transportation Modes
TRANSPORT_MODES = [
    "walking",
    "driving",
    "public_transit",
    "cycling",
    "rideshare"
]

# Environmental Exposure Factors
ENVIRONMENTAL_FACTORS = [
    "air_quality",
    "noise_level",
    "green_space",
    "urban_density",
    "traffic_density"
]

# Prompt Templates
ACTIVITY_GENERATION_PROMPT = """
You are simulating the daily activity schedule for a person with the following characteristics:
- Gender: {gender}
- Age: {age}
- Income level: ${income} annually
- Consumption habits: {consumption}
- Education: {education}
- Day of week: {day_of_week}
- Date: {date}
- Home location: {home_location}
- Work location: {work_location}

Based on this information, generate a daily schedule for this person, from morning to evening. Include at least 4-6 activities throughout the day, including mandatory activities (like work), discretionary activities and travel activities.  

GEOGRAPHIC RULES: considering the geographical distance between activities, such as from home to workplace.
DEMOGRAPHIC CONSIDERATIONS: considering the demographic profile of the person, such as age, gender, income level, consumption habits, education level, etc.

IMPORTANT RULES FOR ACTIVITY TYPES:
- "sleep": ONLY for sleeping activities (night sleep, naps)
- "work": work-related activities (office work, meetings, etc.)
- "travel": travel activities (travel to a new location, usually from home to workplace or other locations)
- "shopping": purchasing goods (groceries, clothes, etc.)
- "dining": eating meals outside or at home (restaurants, cafes, home cooking)
- "recreation": leisure activities (sports, exercise, etc.)
- "healthcare": medical appointments, therapy, etc.)
- "social": meeting friends, parties, etc.
- "education": classes, studying, etc.
- "leisure": relaxing activities (reading, watching TV, etc.)
- "errands": short tasks (bank, post office, etc.)

TRANSPORTATION MODES:
For activities that require travel to a new location, specify a transportation mode from this list:
- "walking": on foot (for short distances, usually <1km, or for exercise)
- "cycling": using a bicycle (for medium distances, usually <5km, or for exercise)
- "public_transit": using bus, subway, train (common for medium-long distances in urban areas)
- "driving": using a private car (common for longer distances or when convenience is needed)
- "rideshare": using taxi, Uber, Lyft, etc. (for special occasions or when other options aren't available)

For each activity, specify:
1. Activity type (MUST be one from the list above)
2. Start time
3. End time
4. Brief description of the activity
5. Location type (e.g., "home", "work", "restaurant", "gym", "park", etc.)
6. Transport mode (ONLY include this for activities requiring travel to a new location, using one of the transportation modes from the list above)

Format your response as a JSON array of activities:
[
  {{
    "activity_type": "...",
    "start_time": "...",
    "end_time": "...",
    "description": "...",
    "location_type": "...",
    "transport_mode": "..."  // Only include this when the person is traveling to a new location
  }},
  ...
]
"""

DESTINATION_SELECTION_PROMPT = """
You are helping to select a specific destination for a person with the following characteristics:
- Gender: {gender}
- Age: {age}
- Income level: ${income} annually
- Consumption habits: {consumption}
- Education: {education}

They want to engage in the following activity: {activity_description}
Activity type: {activity_type}
Time of day: {time}
Day of week: {day_of_week}

Their current location is at coordinates: {current_location}

Given these parameters, what specific type of place would this person likely visit for this activity?
Consider their demographic profile and consumption habits.

Format your response as a single JSON object:
{{
  "place_type": "type of place (e.g., 'restaurant'). Google Places API supported types",
  "search_query": "keyword for place (e.g., 'Chinese restaurant')",
  "distance_preference": "preferred travel distance in kilometers (1-10, where 10 means can be very far)",
  "price_level": "price level preference (1-4, where 4 is most expensive)"
}}
"""

# Add batch processing configuration
BATCH_PROCESSING = True  # Enable batch processing
BATCH_SIZE = 5  # Number of activities to process at once

# Add cache configuration
ENABLE_CACHING = True  # Enable caching
CACHE_EXPIRY = 3600  # Cache expiration time (seconds) 