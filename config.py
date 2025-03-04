"""
Configuration settings for the LLM-based mobility simulation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# DeepBricks API配置
DEEPBRICKS_API_KEY = os.getenv("DEEPBRICKS_API_KEY")
DEEPBRICKS_BASE_URL = os.getenv("DEEPBRICKS_BASE_URL")
USE_DEEPBRICKS_API = True  # Always use DeepBricks API

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

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

Based on this information, generate a realistic daily schedule for this person, from morning to evening.
Include at least 4-6 activities throughout the day, including mandatory activities (like work) and discretionary activities.

IMPORTANT RULES FOR ACTIVITY TYPES:
- "sleep": ONLY for sleeping activities (night sleep, naps)
- "work": work-related activities (office work, meetings, etc.)
- "shopping": purchasing goods (groceries, clothes, etc.)
- "dining": eating meals outside or at home (restaurants, cafes, home cooking)
- "recreation": leisure activities (sports, exercise, etc.)
- "healthcare": medical appointments, therapy, etc.)
- "social": meeting friends, parties, etc.
- "education": classes, studying, etc.
- "leisure": relaxing activities (reading, watching TV, etc.)
- "errands": short tasks (bank, post office, etc.)

IMPORTANT RULES FOR TRANSPORT MODES:
- ONLY specify a transport mode when the person is ACTUALLY TRAVELING from one location to another
- "walking": for short distances on foot
- "driving": using a personal car
- "public_transit": using buses, trains, subway, etc.
- "cycling": using a bicycle
- "rideshare": using taxis, Uber, Lyft, etc.
- If the activity happens at the same location as the previous one, do NOT specify a transport mode

For each activity, specify:
1. Activity type (MUST be one from the list above)
2. Start time
3. End time
4. Brief description of the activity
5. Mode of transportation ONLY if traveling to a new location

Format your response as a JSON array of activities:
[
  {{
    "activity_type": "...",
    "start_time": "...",
    "end_time": "...",
    "description": "...",
    "transport_mode": "..." // ONLY include if traveling to a new location
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
  "distance_preference": "preferred travel distance in kilometers",
  "price_level": "price level preference (1-4, where 4 is most expensive)"
}}
""" 