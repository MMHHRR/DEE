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

# 活动生成模块使用的模型
ACTIVITY_LLM_MODEL = "gpt-4o-mini"  
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 600

# 用于基础数据总结与分析的模型
BASIC_LLM_MODEL = "gpt-4o-mini"

# Simulation Parameters
NUM_DAYS_TO_SIMULATE = 3
SIMULATION_START_DATE = "2025-03-10"
USE_GOOGLE_MAPS = False  # 设置为False，使用OSM而不是Google Maps
USE_OSM = False  # 禁用OpenStreetMap
USE_LOCAL_POI = True  # 启用本地POI数据
MEMORY_DAYS = 2  # Number of days to keep in memory

# File Paths
RESULTS_DIR = "data/results/"

# CSV Data Files
PERSON_CSV_PATH = "data/person.csv"
LOCATION_CSV_PATH = "data/location_new.csv"
GPS_PLACE_CSV_PATH = "data/gps_place.csv"
HOUSEHOLD_CSV_PATH = "data/household.csv"
POI_CSV_PATH = "D:/A_Research/A_doing_research/20250228_LLM+green exposure/data/chicago_poi/meta-Illinois_selected.csv"  # POI数据文件路径

# POI Data Source Configuration
USE_GOOGLE_MAPS = False  # 禁用Google Maps API
USE_OSM = False  # 禁用OpenStreetMap
USE_LOCAL_POI = True  # 启用本地POI数据
POI_SEARCH_RADIUS = 50.0  # 默认搜索半径（公里）

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
- Race/ethnicity: {race}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
- Household vehicles: {household_vehicles}

- Day of week: {day_of_week}
- Date: {date}
- Home location: {home_location}
- Work location: {work_location}
- Memory patterns: {memory_patterns}

Based on this information, generate a realistic daily schedule for this person, MUST START from 00:00 to END at 23:59 for ONE DAY. MAKE SURE the time is continuous and there is no blank window and MUST consider the Memory patterns, especially the 'activity duration'.

CRITICAL: Your schedule must follow a heavy-tailed distribution for activity durations, with SIGNIFICANT emphasis on 2-3 very long activities (such as work) rather than many short ones.

IMPORTANT - FOLLOW THESE DURATION GUIDELINES STRICTLY:
1. Super long activities (>= 480 minutes): At least 1-2 activities MUST be this long (especially sleep and work)
2. Long activities (240-480 minutes): At least 1 activity MUST be in this range
3. Medium activities (120-240 minutes): Only 1-2 activities maximum in this range
4. Short activities (< 120 minutes): Maximum of 1 such activity, ONLY if necessary

The TOTAL daily activities MUST be limited to 4-6 maximum. Over 70 percent of the day MUST be spent on activities longer than 240 minutes.

IMPORTANT RULES FOR ACTIVITY TYPES:
- "sleep": ONLY for sleeping activities (night sleep, naps)
- "work": work-related activities (office work, meetings, etc)
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
4. Detailed description of the activity (limited to 25 words)
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
- Race: {race}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
- Household vehicles: {household_vehicles}
- Activity: {activity_type}
- Description: {activity_description}
- Time of day: {time}
- Day of week: {day_of_week}
{memory_context}

Based on this information, especially considering the historical behavior patterns if available, provide:
1. A specific type of destination suitable for {activity_type}
2. Distance preference (on a scale of 1-10, where 1 means very close to current location and 10 means can be quite far)
3. Rating preference (on a scale of 1-5, where 1 means doesn't care about ratings, 5 means strongly prefers high-rated places)
4. Popularity preference (on a scale of 1-5, where 1 means prefers quiet, less-visited places, 5 means strongly prefers busy, popular places with many reviews)

IMPORTANT DIVERSE DISTANCES PREFERENCE FOR {activity_type} (BASED ON Memory Patterns: 'distances'):
- Short distance travel (≤2 km): 1-3 score
- Medium-short distance travel (2-5 km): 4-6 score
- Medium distance travel (5-10 km): 7-8 score
- Medium-long distance travel (10-15 km): 9 score
- Long distance travel (≥15 km): 10 score

Format your response as a JSON object:
{{
  "place_type": "specific type of place",
  "search_query": "search terms for this type of place",
  "distance_preference": 7,
  "rating_preference": 3,
  "popularity_preference": 3
}}
"""

# Transport Mode Selection Prompt
TRANSPORT_MODE_PROMPT = """
As a transportation expert, please select the most suitable transport mode based on the following information:

Person Information:
- Gender: {gender}
- Age: {age}
- Race: {race}
- Education: {education}
- Occupation: {occupation}
- Household income: ${household_income}
- Household vehicles: {household_vehicles}

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
