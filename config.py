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
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800

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
    "errands",
    "travel",
    # Sub-activity types
    "commuting",
    "warm_up",
    "main_exercise",
    "cool_down",
    "meeting",
    "break",
    "meal",
    "preparation",
    "relaxation"
]

# Transportation Modes
TRANSPORT_MODES = [
    "walking",
    "driving",
    "public_transit",
    "cycling",
    "rideshare"
]

# Location Types
LOCATION_TYPES = [
    "home",
    "workplace",
    "restaurant",
    "cafe",
    "grocery_store",
    "shopping_mall",
    "retail_store",
    "gym",
    "park",
    "hospital",
    "clinic",
    "school",
    "university",
    "library",
    "theater",
    "cinema",
    "museum",
    "bar",
    "friend_home",
    "family_home",
    "bank",
    "post_office",
    "transit_station",
    "hotel",
    "religious_place"
]

# # Environmental Exposure Factors
# ENVIRONMENTAL_FACTORS = [
#     "air_quality",
#     "noise_level",
#     "green_space",
#     "urban_density",
#     "traffic_density"
# ]

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

Based on this information, generate a daily schedule for this person, from morning to evening. Include at least 4-6 activities throughout the day, including mandatory activities (like work) and discretionary activities. MAKE SURE the time is continuous and there is no blank window.

GEOGRAPHIC RULES: considering the geographical distance between activities, such as from home to workplace.
DEMOGRAPHIC CONSIDERATIONS: considering the demographic profile of the person, such as age, gender, income level, consumption habits, education level, etc.

IMPORTANT RULES FOR ACTIVITY TYPES:
- "sleep": ONLY for sleeping activities (night sleep, naps)
- "work": work-related activities (office work, meetings, etc.)
- "shopping": purchasing goods (groceries, clothes, etc.)
- "commuting": travel between home and work
- "dining": eating meals outside or at home (restaurants, cafes, home cooking)
- "recreation": leisure activities (sports, exercise, etc.)
- "healthcare": medical appointments, therapy, etc.)
- "social": meeting friends, parties, etc.
- "education": classes, studying, etc.
- "leisure": relaxing activities (reading, watching TV, etc.)
- "errands": short tasks (bank, post office, etc.)

IMPORTANT RULES FOR LOCATION TYPES:
- "home": person's residence
- "workplace": office, work site, or primary work location
- "restaurant": places primarily for dining
- "cafe": coffee shops, tea houses
- "grocery_store": supermarkets, food stores
- "shopping_mall": large shopping centers
- "retail_store": individual shops, boutiques
- "gym": fitness centers, sports facilities
- "park": public parks, gardens, outdoor recreational spaces
- "hospital": large medical facilities
- "clinic": small medical offices, dental offices
- "school": K-12 educational institutions
- "university": higher education institutions
- "library": public or private libraries
- "theater": performing arts venues
- "cinema": movie theaters
- "museum": art galleries, museums, exhibitions
- "bar": pubs, nightclubs, drinking establishments
- "friend_home": another person's residence (not family)
- "family_home": residence of family members
- "bank": financial institutions
- "post_office": mail services
- "transit_station": bus stops, train stations
- "hotel": accommodation facilities
- "religious_place": churches, temples, mosques, etc.

For each activity, specify:
1. Activity type (MUST be one from the list above)
2. Start time (MUST be the same as the previous activity's end time)
3. End time (Next activity MUST start at this time)
4. Detailed description of the activity (limited to 20 words)
5. Location type (MUST be one from the location types list above)

Format your response as a JSON array of activities:
[
  {{
    "activity_type": "...",
    "start_time": "... (HH:MM)",
    "end_time": "... (HH:MM)",
    "description": "...",
    "location_type": "..."
  }},
  ...// ... more activities with continuous times ...
]
"""

# Add new prompt for refining activities
ACTIVITY_REFINEMENT_PROMPT = """
You are helping to refine a person's activity schedule. Break them down into sub-activities with specific times, when activity need location change based on the previous and current location, you need to add a travel activity.

Person Information:
- Gender: {gender}
- Age: {age}
- Income level: ${income} annually
- Consumption habits: {consumption}
- Education: {education}

Activity Information:
- Date: {date}
- Day of week: {day_of_week}
- Activity description: {activity_description}
- Location type: {location_type} (MUST be one from the defined location types list)
- Start time: {start_time}
- End time: {end_time}
- Previous activity type: {previous_activity_type}
- Previous location: {previous_location}
- Previous end time: {previous_end_time}
- Requires transportation: {requires_transportation}

ACTIVITY TYPES:
"sleep",
"work",
"shopping",
"dining",
"recreation",
"healthcare",
"social",
"education",
"leisure",
"errands",
"travel",
"exercise",
"meeting",
"break",
"meal"

LOCATION TYPES:
"home", "workplace", "restaurant", "cafe", "grocery_store", "shopping_mall", "retail_store", 
"gym", "park", "hospital", "clinic", "school", "university", "library", "theater", "cinema", 
"museum", "bar", "friend_home", "family_home", "bank", "post_office", "transit_station", "hotel", "religious_place"

TRANSPORTATION MODES:
If the activity requires travel to a new location (i.e., requires_transportation is true), select a transportation mode from this list:
- "walking": on foot (suitable for trips under 15 minutes, or for exercise)
- "cycling": using a bicycle (suitable for trips under 15-30 minutes, or for exercise)
- "public_transit": using bus, subway, train (common for 30-60 minute trips in urban areas)
- "driving": using a private car (suitable for trips over 30 minutes or when convenience is needed)
- "rideshare": using taxi, Uber, Lyft, etc. (for special occasions or when other options aren't available)

Please provide descriptions and time arrangements for the activities based on the following information:
1. More precise start and end times, including the time of travel to the activity location
2. More detailed activity content descriptions (limited to 20 words)
3. The specific transport mode used for the travle activity (ONLY include this for activities requiring travel to a new location, using one of the transportation modes from the list above)

Return in JSON format:
{{
  "activity_type": "MUST be one from the list above",
  "start_time": "More precise start time (HH:MM)",
  "end_time": "More precise end time (HH:MM)",
  "description": "More detailed activity description (limited to 20 words)",
  "transport_mode": "ONLY include this IF requires_transportation is true, and it MUST be one from the list above"
}}
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

LOCATION TYPES:
"home", "workplace", "restaurant", "cafe", "grocery_store", "shopping_mall", "retail_store", 
"gym", "park", "hospital", "clinic", "school", "university", "library", "theater", "cinema", 
"museum", "bar", "friend_home", "family_home", "bank", "post_office", "transit_station", "hotel", "religious_place"

Format your response as a single JSON object:
{{
  "place_type": "type of place (MUST be one from the location types list above)",
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