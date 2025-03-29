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
LLM_MAX_TOKENS = 400

# Simulation Parameters
NUM_DAYS_TO_SIMULATE = 7
SIMULATION_START_DATE = "2025-03-10"
USE_GOOGLE_MAPS = True  # If False, will use OSM
MEMORY_DAYS = 2  # Number of days to keep in memory

# File Paths
RESULTS_DIR = "data/results/"

# CSV Data Files
PERSON_CSV_PATH = "data/person.csv"
LOCATION_CSV_PATH = "data/location.csv"
GPS_PLACE_CSV_PATH = "data/gps_place.csv"
HOUSEHOLD_CSV_PATH = "data/household.csv"

# # Processing Options
BATCH_PROCESSING = False  # Whether to use batch processing
BATCH_SIZE = 10  # Number of items to process in a batch
ENABLE_CACHING = True  # Whether to enable function result caching
CACHE_EXPIRY = 86400  # Cache expiry time in seconds (24 hours)

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
    "travel"
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

# Prompt Templates
ACTIVITY_GENERATION_PROMPT = """
You are simulating the daily activity schedule for a person with the following characteristics:
- Gender: {gender}
- Age: {age}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
- Race/ethnicity: {race}
- Day of week: {day_of_week}
- Date: {date}
- Home location: {home_location}
- Work location: {work_location}
- Memory patterns: {memory_patterns}

Based on this information, generate a realistic daily schedule for this person, from morning to evening. Include at least 4-6 activities throughout the day start at home, including mandatory activities (like working, eating, sleeping) and discretionary activities. MUST consider the memory patterns, MAKE SURE the time is continuous and there is no blank window.

IMPORTANT RULES FOR ACTIVITY TYPES:
- "sleep": ONLY for sleeping activities (night sleep, naps)
- "work": work-related activities (office work, meetings, etc.)
- "shopping": purchasing goods (groceries, clothes, etc.)
- "commuting": travel between home and work
- "travel": travel between locations
- "dining": eating meals outside or at home (restaurants, cafes, home cooking)
- "recreation": leisure activities (sports, exercise, etc.)
- "healthcare": medical appointments, therapy, etc.)
- "social": meeting friends, parties, etc.
- "education": classes, studying, etc.)
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
4. Detailed description of the activity (limited to 30 words)
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

# Destination Selection Prompt
DESTINATION_SELECTION_PROMPT = """
Generate a specific destination type and preferences for a person with the following characteristics:
- Gender: {gender}
- Age: {age}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
- Activity: {activity_type}
- Description: {activity_description}
- Time of day: {time}
- Day of week: {day_of_week}

{memory_context}

Based on this information, especially considering the historical behavior patterns if available, provide:
1. A specific type of destination suitable for this activity
2. Price preference (budget, mid-range, upscale, or a numeric value 1-4)
3. Distance preference (on a scale of 1-10, where 1 means very close to current location and 10 means can be quite far)
4. Any specific features or amenities this person would look for

Format your response as a JSON object:
{{
  "place_type": "specific type of place",
  "search_query": "search terms for this type of place",
  "price_level": 2,
  "distance_preference": 5,
  "features": ["feature1", "feature2", ...]
}}
"""

# Activity Refinement Prompt
ACTIVITY_REFINEMENT_PROMPT = """
You are helping to refine a person's activity schedule. Break them down into sub-activities with specific times, when activity need location change based on the previous and current location, you need to add a travel activity.

Person Information:
- Gender: {gender}
- Age: {age}
- Education: {education}
- Household income: ${household_income}
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

IMPORTANT: Consider the person's demographic profile when refining activities:
- Their age ({age}) and gender ({gender}) may affect activity choices
- Their education level ({education}) might influence activity preferences
- Consider any mobility limitations if the person has disabilities
- Consider cultural preferences based on their racial/ethnic background

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
"travel"

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

# Transportation and Distance Refinement Prompt
TRANSPORTATION_REFINEMENT_PROMPT = """
You are finalizing a person's activity and travel schedule with accurate transportation information. Given the calculated distances and optimal transportation modes, generate the final activity log with precise transportation details.

Person Information:
- Gender: {gender}
- Age: {age}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
Activity Details:
- Activity type: {activity_type}
- Description: {description}
- From location: {from_location_type}
- To location: {to_location_type}
- Start time: {start_time}
- End time: {end_time}
- Calculated distance: {distance} km
- Calculated travel time: {travel_time} minutes
- Optimal transport mode: {transport_mode}

TRANSPORT MODE CODES:
- "walking": 200 (on foot)
- "cycling": 201 (bicycle)
- "public_transit": 202 (bus, subway, train)
- "driving": 203 (private car)
- "rideshare": 204 (taxi, Uber, Lyft)

Based on this information, please finalize the activity entry with exact arrival time, departure time, activity duration, and appropriate transportation details.

Return ONLY a JSON object with the following format:
{{
  "activity_type": "{activity_type}",
  "description": "refined description with transportation details if needed",
  "location_type": "{to_location_type}",
  "arrtime": "HH:MM",
  "deptime": "HH:MM",
  "travtime": {travel_time},
  "actdur": calculated activity duration in minutes,
  "distance": {distance},
  "mode": transport mode code
}}
"""

# Transport Mode Selection Prompt
TRANSPORT_MODE_PROMPT = """
As a transportation expert, please select the most suitable transport mode based on the following information:

Person Information:
- Gender: {gender}
- Age: {age}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}

Trip Conditions:
- Activity Type: {activity}
- Available Time: {minutes} minutes
- Distance: {distance} kilometers
- Historical Preferences: {patterns}

Please select ONE most suitable transport mode from the following options:
Walk
My own bike
Divvy bike
Zagster bike
Motorcycle/moped
Auto / van / truck (as the driver) 
Auto / van / truck (as the passenger) 
Carpool/vanpool
School bus
Bus (CTA, PACE, Huskie Line, Indiana)
Dial-a-Ride
Call-n-Ride
Paratransit
Train (CTA, METRA, South Shore Line)
Local transit (NIRPC region)
Private shuttle bus
Taxi
Private limo
Private car
Uber/Lyft
Via/Uber Pool/Lyft Line (shared ride)
Airplane

Return ONLY the transport mode without any explanation.""" 
# CSV Data Files
PERSON_CSV_PATH = "data/person.csv"
LOCATION_CSV_PATH = "data/location.csv"
GPS_PLACE_CSV_PATH = "data/gps_place.csv"
HOUSEHOLD_CSV_PATH = "data/household.csv"
