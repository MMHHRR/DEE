"""
Destination module for the LLM-based mobility simulation.
Handles location retrieval using Google Maps API or OpenStreetMap.
"""

import json
import random
import requests
import openai
import numpy as np
import re
import pandas as pd
import math
from config import (
    GOOGLE_MAPS_API_KEY,
    BASIC_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    USE_GOOGLE_MAPS,
    USE_OSM,
    USE_LOCAL_POI,
    POI_CSV_PATH,
    POI_SEARCH_RADIUS,
    DESTINATION_SELECTION_PROMPT,
    TRANSPORT_MODE_PROMPT,
    DEEPBRICKS_API_KEY,
    DEEPBRICKS_BASE_URL,
    TRANSPORT_MODES,
    ENABLE_CACHING
)
from utils import (
    calculate_distance, 
    generate_random_location_near, 
    cached, 
    normalize_transport_mode,
    estimate_travel_time, 
    cache,
    llm_manager
)

# Create OpenAI client - 弃用的代码，改用LLMManager
# client = openai.OpenAI(
#     api_key=DEEPBRICKS_API_KEY,
#     base_url=DEEPBRICKS_BASE_URL,
# )

class Destination:
    """
    Handles destination selection and location retrieval.
    """
    
    def __init__(self, config=None):
        """Initialize the Destination selector."""
        self.google_maps_api_key = GOOGLE_MAPS_API_KEY
        self.model = BASIC_LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.use_google_maps = USE_GOOGLE_MAPS
        self.use_osm = USE_OSM
        self.use_local_poi = USE_LOCAL_POI
        self.config = config or {}
        
        # Initialize caches
        self.location_cache = {}
        self.transport_mode_cache = {}
        self.google_maps_cache = {}
        
        # Load POI data if using local POI
        if self.use_local_poi:
            try:
                self.poi_data = pd.read_csv(POI_CSV_PATH, sep=',', encoding='utf-8')
                # 创建空间索引
                self.poi_data['distance'] = 0.0  # use for temporary storage of distance
                # print(f"Successfully loaded {len(self.poi_data)} POIs from {POI_CSV_PATH}")
            except Exception as e:
                print(f"Error loading POI data: {e}")
                self.poi_data = None
    
    @cached
    def select_destination(self, persona, current_location, activity_type, time, day_of_week, available_minutes, memory_patterns=None, location_type_override=None, search_query_override=None):
        """
        Select an appropriate destination for the given activity
        
        Args:
            persona: Persona object
            current_location: Current location coordinates
            activity_type: Activity type
            time: Time string (HH:MM)
            day_of_week: Day of week string
            available_minutes: Available time (minutes)
            memory_patterns: Historical memory patterns for LLM analysis (optional)
            location_type_override: Location type information extracted from activity description (optional)
            search_query_override: Search query information extracted from activity description (optional)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Step 1: Determine appropriate destination type
            if location_type_override:

                # Get normal destination type information first
                destination_type = self._determine_destination_type(
                    persona, 
                    {"activity_type": activity_type, "description": f"{activity_type} activity"}, 
                    time, 
                    day_of_week,
                    memory_patterns
                )

                # Then replace specific fields with override values
                destination_type['place_type'] = location_type_override
                if search_query_override:
                    destination_type['search_query'] = search_query_override
            else:
                destination_type = self._determine_destination_type(
                    persona, 
                    {"activity_type": activity_type, "description": f"{activity_type} activity"}, 
                    time, 
                    day_of_week,
                    memory_patterns
                )
                
                # Use _process_search_query method to process the query result returned by LLM
                destination_type = self._process_search_query(
                    destination_type,
                    f"{activity_type} activity"
                )

            # Step 2: Calculate search parameters
            time_window = self._calculate_time_window(available_minutes)
            max_radius = self._calculate_search_radius(persona, activity_type, time_window, destination_type)

            # Step 3: Find appropriate destination location
            location, details = self._retrieve_location(
                current_location, 
                destination_type, 
                max_radius, 
                available_minutes
            )

            # Step 4: Ensure distance is calculated correctly
            if 'distance' not in details or not isinstance(details['distance'], (int, float)):
                details['distance'] = calculate_distance(current_location, location)

            # Step 5: Determine transportation details 
            transport_mode = self._determine_transport_mode(
                persona, 
                activity_type,
                available_minutes,
                details['distance'],
                memory_patterns
            )
            
            # Step 6: Calculate travel time for selected transport mode
            travel_time, _ = estimate_travel_time(current_location, location, transport_mode)
                
            # Step 7: Update location details with transport information
            details.update({
                'name': details['name'],
                'transport_mode': transport_mode,
                'travel_time': travel_time,
                'distance': details['distance']  # Ensure distance information exists
            })
            
            return location, details
            
        except Exception as e:
            print(f"Error occurred during destination selection: {e}")
            # Generate random location as backup
            radius = min(5, available_minutes / 60)  # Simple estimate: 1 hour range 5 kilometers
            random_location = generate_random_location_near(current_location, max_distance_km=radius, validate=False)
            return random_location, {"name": f"{activity_type} Location", "address": "Generated Location"}
    
    @cached
    def _calculate_time_window(self, available_minutes):
        """
        Calculate available time window for activity
        
        Args:
            available_minutes: Available time (minutes)
        
        Returns:
            int: Available time (minutes)
        """
        # Return default value if no time information
        if not available_minutes:
            return 60
        
        # Handle crossing midnight
        if available_minutes < 0:
            available_minutes += 24 * 60
            
        return available_minutes
    
    @cached
    def _calculate_max_radius(self, persona, activity_type, time_window, destination_type=None):
        """
        Calculate the maximum search radius based on available time.
        
        Args:
            persona: persona object
            activity_type: activity type
            time_window: available time (minutes)
            destination_type: destination type information, including distance preference
        
        Returns:
            float: maximum radius (kilometers)
        """
        # 1. Based on available time, the basic radius
        if time_window < 30:
            # Very short activity, very small range
            base_radius = 0.5  # 500 meters range
        elif time_window < 60:
            # Short activity, small range
            base_radius = 1.2  # 1.2 kilometers range
        elif time_window < 120:
            # Medium activity, medium range
            base_radius = 2.5  # 2.5 kilometers range
        else:
            # Long activity, larger range
            base_radius = 4.0  # 4 kilometers range
        
        # 2. Apply distance preference from destination_type if available
        final_radius = base_radius
        if destination_type and 'distance_preference' in destination_type:
            # Distance preference value range 1-10, 1 means very close, 10 means can be very far
            distance_pref = destination_type.get('distance_preference', 5)
            # Convert distance preference to radius adjustment factor (0.5-1.5)
            distance_factor = 0.5 + (distance_pref / 10)
            # Adjust base radius
            final_radius = base_radius * distance_factor
        
        # 3. Apply upper limit - Ensure maximum search radius is 50 kilometers
        MAX_ALLOWED_RADIUS = 50.0  # Maximum allowed search radius is 50 kilometers
        final_radius = min(final_radius, MAX_ALLOWED_RADIUS)
        
        # Ensure minimum radius
        return max(0.3, final_radius)
    
    @cached
    def _determine_destination_type(self, persona, activity, time, day_of_week, memory_patterns=None):
        """
        Determine destination type based on persona, activity, and context.
        
        Args:
            persona: Character object
            activity: Activity dictionary
            time: Time string (HH:MM)
            day_of_week: Day of week string
            memory_patterns: Historical memory patterns (optional)
            
        Returns:
            dict: Destination type information with preferences
        """
        try:
            # Try to determine destination type using LLM
            llm_result = self._determine_destination_type_with_llm(
                persona, 
                activity, 
                time, 
                day_of_week, 
                memory_patterns
            )
            
            if llm_result and all(key in llm_result for key in ['place_type', 'search_query', 'distance_preference']):
                return llm_result
                
            # Fall back to rule-based approach if LLM fails
            return self._generate_default_destination(persona, activity, time, day_of_week, memory_patterns)
            
        except Exception as e:
            print(f"Error determining destination type: {e}")
            # Return simple default value
            return {
                'place_type': 'point_of_interest',
                'search_query': activity['activity_type'] if activity['activity_type'] else 'place',
                'distance_preference': 5,
                'rating_preference': 3,
                'popularity_preference': 3
            }
    
    def _determine_destination_type_with_llm(self, persona, activity, time, day_of_week, memory_patterns):
        """Use LLM to determine appropriate destination type"""
        try:
            # Format prompt with all contextual information
            # Get persona attributes safely with defaults
            gender = getattr(persona, 'gender', 'unknown')
            age = getattr(persona, 'age', 30)
            race = getattr(persona, 'race', 'unknown')
            occupation = getattr(persona, 'occupation', 'unknown')
            education = getattr(persona, 'education', 'unknown')
            current_location = getattr(persona, 'current_location', 'unknown')
            
            # Get household income safely
            try:
                household_income = persona.get_household_income()
            except Exception:
                household_income = 50000  # Default value

            # Get household vehicles safely
            try:
                household_vehicles = persona.get_household_vehicles()
            except Exception:
                household_vehicles = 0  # Default value     
                
            # Get activity details safely
            activity_description = activity.get('description', 'unknown activity')
            activity_type = activity.get('activity_type', 'unknown')
            
            # Format memory context
            memory_context = str(memory_patterns) if memory_patterns else "No memory patterns available"
            
            # Format the prompt
            enhanced_prompt = DESTINATION_SELECTION_PROMPT.format(
                gender=gender,
                age=age,
                race=race,
                household_income=household_income,
                household_vehicles=household_vehicles,
                education=education,
                occupation=occupation,
                activity_description=activity_description,
                activity_type=activity_type,
                time=time,
                day_of_week=day_of_week,
                memory_context=memory_context
            )
            
            # 使用LLMManager替代直接调用client
            response = llm_manager.completion_basic(
                enhanced_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=50
            )
            
            # Extract structured data from LLM response
            result = self._extract_destination_from_text(response.choices[0].message.content)
            
            # Ensure required fields exist
            if not result:
                result = {}
                
            if 'place_type' not in result:
                result['place_type'] = 'point_of_interest'
                
            if 'search_query' not in result:
                result['search_query'] = activity_type
                
            if 'distance_preference' not in result:
                result['distance_preference'] = 5
                
            if 'rating_preference' not in result:
                result['rating_preference'] = 3
                
            if 'popularity_preference' not in result:
                result['popularity_preference'] = 3
                
            # 使用关键词映射处理搜索查询
            result = self._process_search_query(result, activity_description)
                
            return result
            
        except Exception as llm_error:
            print(f"LLM destination type determination failed: {str(llm_error)}")
            return None
    
    def _process_search_query(self, destination_type, description):
        """
        Process search query, use keyword mapping to replace incorrect queries
        
        Args:
            destination_type: Target location type dictionary
            description: Activity description
            
        Returns:
            dict: Processed target location type dictionary
        """
        try:
            # If there is no search_query, initialize it as an empty string
            search_query = destination_type.get('search_query', '')
            
            # Ensure query is a string and remove leading/trailing spaces
            if not isinstance(search_query, str):
                search_query = str(search_query)
            search_query = search_query.strip().lower()
            
            # Keyword mapping list
            place_keywords = {
                # Financial Services
                'bank': {'place_type': 'bank', 'search_query': 'bank'},
                'atm': {'place_type': 'atm', 'search_query': 'atm'},
                
                # Food Shopping
                'grocery': {'place_type': 'grocery_or_supermarket', 'search_query': 'grocery'},
                'supermarket': {'place_type': 'grocery_or_supermarket', 'search_query': 'supermarket'},
                'bakery': {'place_type': 'bakery', 'search_query': 'bakery'},
                'butcher': {'place_type': 'butcher', 'search_query': 'butcher'},
                'deli': {'place_type': 'delicatessen', 'search_query': 'deli'},
                'seafood': {'place_type': 'fishmonger', 'search_query': 'seafood'},
                'greengrocer': {'place_type': 'greengrocer', 'search_query': 'greengrocer'},
                
                # Food & Drink
                'restaurant': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dining': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dinner': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'lunch': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'breakfast': {'place_type': 'cafe', 'search_query': 'cafe'},
                'cafe': {'place_type': 'cafe', 'search_query': 'cafe'},
                'café': {'place_type': 'cafe', 'search_query': 'cafe'},
                'coffee': {'place_type': 'cafe', 'search_query': 'cafe'},
                'fast food': {'place_type': 'fast_food', 'search_query': 'fast_food'},
                'food court': {'place_type': 'food_court', 'search_query': 'food_court'},
                'pub': {'place_type': 'pub', 'search_query': 'pub'},
                'bar': {'place_type': 'bar', 'search_query': 'bar'},
                'ice cream': {'place_type': 'ice_cream', 'search_query': 'ice_cream'},
                
                # Leisure & Entertainment
                'gym': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'fitness': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'workout': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'exercise': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'park': {'place_type': 'park', 'search_query': 'park'},
                'cinema': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'movie': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'theatre': {'place_type': 'theatre', 'search_query': 'theatre'},
                'sports': {'place_type': 'stadium', 'search_query': 'sports_centre'},
                'swimming': {'place_type': 'swimming_pool', 'search_query': 'swimming_pool'},
                'nightclub': {'place_type': 'night_club', 'search_query': 'nightclub'},
                
                # Culture & Education
                'library': {'place_type': 'library', 'search_query': 'library'},
                'museum': {'place_type': 'museum', 'search_query': 'museum'},
                'art gallery': {'place_type': 'art_gallery', 'search_query': 'gallery'},
                'gallery': {'place_type': 'art_gallery', 'search_query': 'gallery'},
                'school': {'place_type': 'school', 'search_query': 'school'},
                'university': {'place_type': 'university', 'search_query': 'university'},
                'college': {'place_type': 'university', 'search_query': 'college'},
                
                # Healthcare
                'hospital': {'place_type': 'hospital', 'search_query': 'hospital'},
                'doctor': {'place_type': 'doctor', 'search_query': 'clinic'},
                'medical': {'place_type': 'medical_center', 'search_query': 'hospital'},
                'clinic': {'place_type': 'doctor', 'search_query': 'clinic'},
                'dentist': {'place_type': 'dentist', 'search_query': 'dentist'},
                'pharmacy': {'place_type': 'pharmacy', 'search_query': 'chemist'},
                'drugstore': {'place_type': 'pharmacy', 'search_query': 'chemist'},
                'veterinary': {'place_type': 'veterinary_care', 'search_query': 'veterinary'},
                
                # Shopping
                'shop': {'place_type': 'store', 'search_query': 'mall'},
                'shopping': {'place_type': 'store', 'search_query': 'mall'},
                'shopping mall': {'place_type': 'store', 'search_query': 'mall'},
                'mall': {'place_type': 'shopping_mall', 'search_query': 'mall'},
                'clothing': {'place_type': 'clothing_store', 'search_query': 'mall'},
                'shoes': {'place_type': 'shoe_store', 'search_query': 'shoes'},
                'electronics': {'place_type': 'electronics_store', 'search_query': 'electronics'},
                'hardware': {'place_type': 'hardware_store', 'search_query': 'hardware'},
                'furniture': {'place_type': 'furniture_store', 'search_query': 'furniture'},
                'bookstore': {'place_type': 'book_store', 'search_query': 'books'},
                'jewelry': {'place_type': 'jewelry_store', 'search_query': 'jewelry'},
                'toys': {'place_type': 'toy_store', 'search_query': 'toys'},
                
                # Personal Services
                'hair': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'salon': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'barber': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'spa': {'place_type': 'spa', 'search_query': 'beauty'},
                'laundry': {'place_type': 'laundry', 'search_query': 'laundry'},
                'dry cleaning': {'place_type': 'dry_cleaning', 'search_query': 'dry_cleaning'},
                
                # Accommodation
                'hotel': {'place_type': 'lodging', 'search_query': 'hotel'},
                'motel': {'place_type': 'lodging', 'search_query': 'motel'},
                'hostel': {'place_type': 'lodging', 'search_query': 'hostel'},
                'guest house': {'place_type': 'lodging', 'search_query': 'guest_house'},
                
                # Transportation
                'gas station': {'place_type': 'gas_station', 'search_query': 'gas'},
                'car repair': {'place_type': 'car_repair', 'search_query': 'car_repair'},
                'car wash': {'place_type': 'car_wash', 'search_query': 'car_wash'},
                'parking': {'place_type': 'parking', 'search_query': 'parking'},
                'bike parking': {'place_type': 'bicycle_parking', 'search_query': 'bicycle_parking'},
                'bus station': {'place_type': 'bus_station', 'search_query': 'bus_station'},
                'train station': {'place_type': 'train_station', 'search_query': 'station'},
                'subway': {'place_type': 'subway_station', 'search_query': 'subway'},
                'airport': {'place_type': 'airport', 'search_query': 'aerodrome'},
                
                # Convenience
                'grocery store': {'place_type': 'grocery_or_supermarket', 'search_query': 'retail'},
                'convenience store': {'place_type': 'convenience_store', 'search_query': 'convenience'},
                'post office': {'place_type': 'post_office', 'search_query': 'post_office'},
                
                # Public Services
                'police': {'place_type': 'police', 'search_query': 'police'},
                'fire station': {'place_type': 'fire_station', 'search_query': 'fire_station'},
                'town hall': {'place_type': 'city_hall', 'search_query': 'government'},
                'courthouse': {'place_type': 'courthouse', 'search_query': 'courthouse'},
                'embassy': {'place_type': 'embassy', 'search_query': 'diplomatic'},
                
                # Work Places
                'office': {'place_type': 'point_of_interest', 'search_query': 'office_building'},
                'workplace': {'place_type': 'point_of_interest', 'search_query': 'office_building'},
            }
            
            # Extended problematic pattern list
            problematic_patterns = [
                "near", "close to", "around", "spaces", "located", 
                "looking for", "find", "searching", "seeking",
                "next to", "nearby", "within", "by the", "available",
                "nearest", "area", "places", "in the", "center",
                "at the", "with", "for", "me", "near me"
            ]
            
            # 1. First check if there is a predefined exact match
            if search_query in place_keywords:
                destination_type['place_type'] = place_keywords[search_query]['place_type']
                destination_type['search_query'] = place_keywords[search_query]['search_query']
                return destination_type
                
            # 2. Check if the search query contains problematic patterns or is too long
            is_problematic = any(pattern in search_query for pattern in problematic_patterns)
            
            # Remove common problematic words (e.g. "near me") to try to get keywords
            cleaned_query = search_query
            for pattern in problematic_patterns:
                cleaned_query = cleaned_query.replace(pattern, ' ')
            cleaned_query = ' '.join(cleaned_query.split())  # Process extra spaces
            
            # If the cleaned query exists in the keywords, use it
            if cleaned_query in place_keywords:
                destination_type['place_type'] = place_keywords[cleaned_query]['place_type']
                destination_type['search_query'] = place_keywords[cleaned_query]['search_query']
                return destination_type
            
            # If the search query has problems or is more than 3 words
            if is_problematic or len(search_query.split()) > 3 or not search_query:
                description = description.lower() if description else ""
                
                # Try to match multi-word keywords
                multi_word_keywords = sorted([k for k in place_keywords.keys() if ' ' in k], 
                                           key=len, reverse=True)
                
                for keyword in multi_word_keywords:
                    if keyword in description:
                        destination_type['place_type'] = place_keywords[keyword]['place_type']
                        destination_type['search_query'] = place_keywords[keyword]['search_query']
                        return destination_type
                
                # If no multi-word phrase matches, check word matches
                for keyword, place_info in place_keywords.items():
                    if ' ' not in keyword and keyword in description:
                        # Avoid partial matches (e.g. avoid matching "background" as "back")
                        word_boundaries = [' ', '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '\n', '\t']
                        
                        # Check if the keyword is at the beginning/end of a word or string
                        keyword_positions = [m.start() for m in re.finditer(re.escape(keyword), description)]
                        for pos in keyword_positions:
                            # Check the left boundary
                            before_ok = (pos == 0 or description[pos-1] in word_boundaries)
                            # Check the right boundary
                            after_pos = pos + len(keyword)
                            after_ok = (after_pos >= len(description) or description[after_pos] in word_boundaries)
                            
                            if before_ok and after_ok:
                                destination_type['place_type'] = place_info['place_type']
                                destination_type['search_query'] = place_info['search_query']
                                return destination_type
                    
                # If still no match, try to extract keywords from the search query 
                if cleaned_query:
                    words = cleaned_query.split()
                    for word in words:
                        if word in place_keywords:
                            destination_type['place_type'] = place_keywords[word]['place_type']
                            destination_type['search_query'] = place_keywords[word]['search_query']
                            return destination_type
            
            # If the query has problems but no better match is found, use the default value
            if is_problematic or len(search_query.split()) > 3:
                # Default to "point_of_interest" as a fallback option
                destination_type['place_type'] = 'point_of_interest'
                destination_type['search_query'] = 'place'
            else:
                # Clean the original query to the minimum extent (remove spaces, normalize)
                destination_type['search_query'] = search_query.strip()
            
            return destination_type
            
        except Exception as e:
            print(f"Error processing search query: {e}")
            # Use safe default values when there is an error
            destination_type['place_type'] = destination_type.get('place_type', 'point_of_interest')
            destination_type['search_query'] = 'place'
            return destination_type
    
    def _generate_default_destination(self, persona, activity, time, day_of_week, memory_patterns=None):
        """
        Generate default destination type based on predefined mappings when LLM analysis fails.
        
        Args:
            persona: Character object
            activity: Activity dictionary, containing activity_type and description
            time: Time string
            day_of_week: Day of week
            memory_patterns: Historical memory patterns (optional)
            
        Returns:
            dict: Default destination type information
        """
        try:
            # Activity type to destination type basic mapping
            default_destinations = {
                'shopping': {
                    'place_type': 'store',
                    'search_query': 'shopping center',
                    'distance_preference': 4,
                    'rating_preference': 3,
                    'popularity_preference': 4
                },
                'dining': {
                    'place_type': 'restaurant',
                    'search_query': 'restaurant',
                    'distance_preference': 3,
                    'rating_preference': 4,
                    'popularity_preference': 3
                },
                'recreation': {
                    'place_type': 'park',
                    'search_query': 'park recreation area',
                    'distance_preference': 4,
                    'rating_preference': 3,
                    'popularity_preference': 2
                },
                'leisure': {
                    'place_type': 'point_of_interest',
                    'search_query': 'leisure entertainment',
                    'distance_preference': 5,
                    'rating_preference': 3,
                    'popularity_preference': 3
                },
                'healthcare': {
                    'place_type': 'health',
                    'search_query': 'medical center',
                    'distance_preference': 6,
                    'rating_preference': 5,
                    'popularity_preference': 2
                },
                'education': {
                    'place_type': 'school',
                    'search_query': 'education center',
                    'distance_preference': 5,
                    'rating_preference': 4,
                    'popularity_preference': 3
                },
                'social': {
                    'place_type': 'bar',
                    'search_query': 'social venue cafe',
                    'distance_preference': 4,
                    'rating_preference': 3,
                    'popularity_preference': 5
                },
                'errands': {
                    'place_type': 'store',
                    'search_query': 'convenience services',
                    'distance_preference': 2,
                    'rating_preference': 2,
                    'popularity_preference': 2
                }
            }
            
            # Apply memory-based preferences if available
            if memory_patterns and 'distances' in memory_patterns and activity['activity_type'] in memory_patterns['distances']:
                distances = memory_patterns['distances'][activity['activity_type']]
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    # Convert average distance to preference scale (1-10)
                    # 0-1km: 1-2, 1-3km: 3-4, 3-5km: 5-6, 5-8km: 7-8, 8+km: 9-10
                    if avg_distance < 1:
                        distance_pref = min(2, max(1, int(avg_distance * 2)))
                    elif avg_distance < 3:
                        distance_pref = min(4, max(3, int(2 + (avg_distance - 1))))
                    elif avg_distance < 5:
                        distance_pref = min(6, max(5, int(4 + (avg_distance - 3) / 2)))
                    elif avg_distance < 8:
                        distance_pref = min(8, max(7, int(6 + (avg_distance - 5) / 3)))
                    else:
                        distance_pref = min(10, max(9, int(8 + (avg_distance - 8) / 2)))
                        
                    # Update default preference with memory-based value
                    if activity['activity_type'] in default_destinations:
                        default_destinations[activity['activity_type']]['distance_preference'] = distance_pref
            
            # Keyword mapping to optimize destination type
            keywords = {
                'grocery': {'place_type': 'supermarket', 'search_query': 'grocery store'},
                'gym': {'place_type': 'gym', 'search_query': 'fitness center'},
                'coffee': {'place_type': 'cafe', 'search_query': 'coffee shop'},
                'movie': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'library': {'place_type': 'library', 'search_query': 'library'},
                'restaurant': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'cafe': {'place_type': 'cafe', 'search_query': 'cafe'},
                'book': {'place_type': 'book_store', 'search_query': 'bookstore'},
                'mall': {'place_type': 'shopping_mall', 'search_query': 'shopping mall'},
                'clothing': {'place_type': 'clothing_store', 'search_query': 'clothing store'},
                'doctor': {'place_type': 'doctor', 'search_query': 'doctor clinic'},
                'hospital': {'place_type': 'hospital', 'search_query': 'hospital'},
                'bar': {'place_type': 'bar', 'search_query': 'bar pub'},
                'pharmacy': {'place_type': 'pharmacy', 'search_query': 'pharmacy'},
                'school': {'place_type': 'school', 'search_query': 'school'},
                'university': {'place_type': 'university', 'search_query': 'university'},
                'museum': {'place_type': 'museum', 'search_query': 'museum'},
                'park': {'place_type': 'park', 'search_query': 'park'},
                'hair': {'place_type': 'beauty_salon', 'search_query': 'hair salon'}
            }
            
            # Based on activity type, select destination
            destination = default_destinations.get(activity['activity_type'], {
                'place_type': 'point_of_interest',
                'search_query': activity['activity_type'],
                'distance_preference': 5,
                'rating_preference': 3,
                'popularity_preference': 3
            }).copy()
            
            # Based on description keyword optimization
            for keyword, mapping in keywords.items():
                if keyword in activity['description'].lower():
                    destination['place_type'] = mapping['place_type']
                    destination['search_query'] = mapping['search_query']
                    break
            
            hour = int(time.split(':')[0])
            if 17 <= hour <= 20:
                # Dinner time may prefer slightly further place
                destination['distance_preference'] = min(10, destination['distance_preference'] + 1)
            
            return destination
            
        except Exception as e:
            print(f"Error generating default destination: {e}")
            # Return basic default value
            return {
                'place_type': 'point_of_interest',
                'search_query': activity['activity_type'] if activity['activity_type'] else 'place',
                'distance_preference': 5,
                'rating_preference': 3,
                'popularity_preference': 3
            }
    
    def _retrieve_location(self, current_location, destination_type, max_radius, available_minutes):
        """
        Retrieve a suitable location based on the given parameters.
        This is a wrapper that uses Google Maps API or falls back to alternatives.
        
        Args:
            current_location: Current location (latitude, longitude)
            destination_type: Destination type information
            max_radius: Maximum search radius (kilometers)
            available_minutes: Available time (minutes)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Validate inputs
            if not current_location or not isinstance(current_location, (tuple, list)) or len(current_location) != 2:
                print(f"Invalid current_location format: {current_location}, using fallback")
                return self._generate_fallback_location(max_radius)
            
            # Convert list to tuple if needed for consistency
            if isinstance(current_location, list):
                current_location = tuple(current_location)
                
            # Verify destination_type is a valid dictionary
            if not destination_type or not isinstance(destination_type, dict):
                print(f"Invalid destination_type: {destination_type}, using default")
                destination_type = {
                    'place_type': 'point_of_interest',
                    'search_query': 'place',
                    'distance_preference': 5,
                }
                
            # Validate and correct max_radius if necessary
            if not isinstance(max_radius, (int, float)) or max_radius <= 0:
                max_radius = POI_SEARCH_RADIUS
                
            # Select method based on configuration
            if self.use_local_poi and self.poi_data is not None:
                return self._retrieve_location_local_poi(
                    current_location, 
                    destination_type, 
                    max_radius, 
                    available_minutes
                )
            elif self.use_google_maps:
                return self._retrieve_location_google_maps(
                    current_location, 
                    destination_type, 
                    max_radius, 
                    available_minutes
                )
            elif self.use_osm:
                return self._retrieve_location_osm(
                    current_location,
                    destination_type,
                    max_radius,
                    available_minutes
                )
            else:
                # If all methods are disabled, use random location generation
                return self._generate_fallback_location(max_radius, current_location)
        
        except Exception as e:
            print(f"Error in _retrieve_location: {str(e)}")
            return self._generate_fallback_location(max_radius, current_location)

    def _retrieve_location_local_poi(self, current_location, destination_type, max_radius, available_minutes):
        """
        Use local POI data to get destination location
        
        Args:
            current_location: Current location (latitude, longitude)
            destination_type: Destination type information
            max_radius: Maximum search radius (kilometers)
            available_minutes: Available time (minutes)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Use spatial index for initial screening
            lat_min = current_location[0] - (max_radius / 111.32)  # 1度约111.32公里
            lat_max = current_location[0] + (max_radius / 111.32)
            lon_min = current_location[1] - (max_radius / (111.32 * math.cos(math.radians(current_location[0]))))
            lon_max = current_location[1] + (max_radius / (111.32 * math.cos(math.radians(current_location[0]))))

            # Initial screening of POIs within the rectangular range
            filtered_pois = self.poi_data[
                (self.poi_data['latitude'] >= lat_min) &
                (self.poi_data['latitude'] <= lat_max) &
                (self.poi_data['longitude'] >= lon_min) &
                (self.poi_data['longitude'] <= lon_max)
            ]
            
            if len(filtered_pois) == 0:
                # print(f"No POIs found in initial spatial filter")
                random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
                return random_location, {
                    'name': f'Location for {destination_type.get("search_query", "activity")}',
                    'address': f'Generated Location',
                    'is_fallback': True
                }
            
            # Create a clear copy
            filtered_pois = filtered_pois.copy()
            
            # Calculate exact distance
            filtered_pois['distance'] = filtered_pois.apply(
                lambda row: calculate_distance(
                    current_location, 
                    (row['latitude'], row['longitude'])
                ),
                axis=1
            )
            
            # Handle missing values
            filtered_pois['avg_rating'] = filtered_pois['avg_rating'].fillna(3.0)  # Default rating 3.0
            filtered_pois['num_of_reviews'] = filtered_pois['num_of_reviews'].fillna(1)  # Default number of reviews 1
            filtered_pois['category'] = filtered_pois['category'].fillna('unknown')  # Default category
            filtered_pois['name'] = filtered_pois['name'].fillna('')  # 默认空名称
            filtered_pois['address'] = filtered_pois['address'].fillna('')  # 默认空地址
            filtered_pois['description'] = filtered_pois['description'].fillna('')  # 默认空描述
            
            # Further筛选符合条件的POI
            filtered_pois = filtered_pois[
                (filtered_pois['distance'] <= max_radius)
            ]
            
            # Try up to 3 times with increasing radius if no POIs found
            original_max_radius = max_radius
            retry_count = 0
            max_retry = 3
            
            while len(filtered_pois) == 0 and retry_count < max_retry:
                if retry_count > 0:  # Skip the first time as we've already searched
                    # Increase search radius by 50% each time
                    max_radius = original_max_radius * (1 + 0.5 * retry_count)
                    print(f"Attempt {retry_count+1}: Increasing search radius to {max_radius:.1f}km")
                    
                    # Re-filter POIs with new radius
                    filtered_pois = self.poi_data[
                        (self.poi_data['distance'] <= max_radius)
                    ]
                    
                    # Apply category filter on expanded results
                    filtered_pois['category'] = filtered_pois['category'].fillna('unknown')
                    filtered_pois['name'] = filtered_pois['name'].fillna('')
                    filtered_pois['address'] = filtered_pois['address'].fillna('')
                    filtered_pois['description'] = filtered_pois['description'].fillna('')
                
                retry_count += 1
            
            if len(filtered_pois) == 0:
                print(f"No suitable POIs found after {retry_count} attempts with max radius {max_radius:.1f}km")
                random_location = generate_random_location_near(current_location, original_max_radius * 0.5, validate=False)
                return random_location, {
                    'name': f'Location for {destination_type.get("search_query", "activity")}',
                    'address': f'Generated Location',
                    'is_fallback': True
                }
            
            # Filter based on activity type and search query
            search_query = destination_type.get('search_query', '').lower()
            
            # 1. Directly process the search query to get more accurate search terms
            if search_query:
                category_match = filtered_pois[
                    filtered_pois['category'].str.lower().str.contains(search_query, na=False) |
                    filtered_pois['name'].str.lower().str.contains(search_query, na=False)
                ]
                
                if len(category_match) > 0:
                    filtered_pois = category_match

            if len(filtered_pois) == 0:
                print(f"No POIs match the search query: {search_query}")
                random_location = generate_random_location_near(current_location, original_max_radius * 0.5, validate=False)
                return random_location, {
                    'name': f'Location for {destination_type.get("search_query", "activity")}',
                    'address': f'Generated Location',
                    'is_fallback': True
                }
            
            # Get preference parameters
            rating_preference = destination_type.get('rating_preference', 3)
            popularity_preference = destination_type.get('popularity_preference', 3)
            
            # Calculate comprehensive score, using the modified function
            filtered_pois['score'] = filtered_pois.apply(
                lambda row: self._calculate_place_score(
                    rating=row['avg_rating'],
                    distance=row['distance'],
                    max_distance=max_radius,
                    num_reviews=row['num_of_reviews'],
                    rating_preference=rating_preference,
                    popularity_preference=popularity_preference
                ),
                axis=1
            )
            
            # Select POI based on comprehensive score
            top_pois = filtered_pois.nlargest(5, 'score')
            selected_poi = top_pois.iloc[random.randint(0, len(top_pois)-1)]
            
            location = (selected_poi['latitude'], selected_poi['longitude'])
            details = {
                'name': selected_poi['name'],
                'address': selected_poi['address'],
                'description': selected_poi['description'],
                'category': selected_poi['category'],
                'rating': selected_poi['avg_rating'],
                'reviews': selected_poi['num_of_reviews'],
                'distance': selected_poi['distance'],
                'score': selected_poi['score']
            }
            
            return location, details
            
        except Exception as e:
            print(f"Error in _retrieve_location_local_poi: {str(e)}")
            random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
            return random_location, {
                'name': f'Location for {destination_type.get("search_query", "activity")}',
                'address': f'Generated Location',
                'is_fallback': True
            }

    def _retrieve_location_google_maps(self, current_location, destination_type, max_radius, available_minutes):
        """
        Use Google Maps API to get destination location
        
        Args:
            current_location: Current location (latitude, longitude)
            destination_type: Destination type information
            max_radius: Maximum search radius (kilometers)
            available_minutes: Available time (minutes)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Check cache
            cache_key = self._build_location_cache_key(current_location, destination_type, max_radius)
            cached_result = cache.get(cache_key)
            if cached_result and ENABLE_CACHING:
                return cached_result
            
            # Get places from Google Maps API
            candidates = self._get_places_from_google_maps(
                current_location, 
                destination_type, 
                max_radius, 
                available_minutes
            )
            
            # Select best candidate and cache result
            best_candidate = self._select_best_candidate(candidates)
            result = (best_candidate['coords'], best_candidate['details'])
            
            if ENABLE_CACHING:
                cache.set(cache_key, result)
            
            return result
                
        except Exception as e:
            print(f"Error occurred retrieving location: {str(e)}")
            # Return random location in case of error
            random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
            return random_location, {'name': f'Random Location (Error)', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}
            
    def _build_location_cache_key(self, current_location, destination_type, max_radius):
        """Build cache key for location retrieval"""
        cache_key = f"google_maps_{current_location}_{destination_type.get('search_query')}_{max_radius}"
        return cache_key.replace(" ", "_").replace(",", "_")
        
    def _get_places_from_google_maps(self, current_location, destination_type, max_radius, available_minutes):
        """
        Query Google Maps API and process results into a list of candidate locations
        
        Returns:
            list: List of candidate locations with their details and scores
        """
        # Prepare parameters for Google Maps API
        search_query = destination_type.get('search_query', 'point of interest')
        place_type = destination_type.get('place_type', 'point_of_interest')
        
        # 确保搜索半径不超过50公里(50000米)
        MAX_RADIUS_METERS = 50000  # 最大搜索半径50公里
        radius_meters = min(MAX_RADIUS_METERS, int(max_radius * 1000))
        
        # Prepare API request parameters
        params = {
            'key': self.google_maps_api_key,
            'location': f"{current_location[0]},{current_location[1]}",
            'radius': radius_meters,
            'type': place_type,
            'keyword': search_query,
            'rankby': 'prominence'  # Default sort by popularity
        }
        
        # Call Places API
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK' or not data.get('results'):
            return []
            
        # Process only the first 5 results
        results = data.get('results')[:5]
        candidates = []
        
        for result in results:
            try:
                # Get location coordinates
                location = result.get('geometry', {}).get('location', {})
                lat = location.get('lat')
                lng = location.get('lng')
                
                # Skip invalid coordinates
                if lat is None or lng is None:
                    continue
                    
                dest_coords = (lat, lng)
                
                # Calculate distance
                distance = calculate_distance(current_location, dest_coords)
                
                # Skip if location is too far
                estimated_time_minutes = (distance / 30) * 60
                if available_minutes and estimated_time_minutes > available_minutes * 0.8:
                    continue
                
                # Get rating and price level
                rating = result.get('rating', 3.0) or 3.0
                
                # 使用优化后的评分函数，传递所有需要的参数
                score = self._calculate_place_score(
                    rating=rating, 
                    distance=distance, 
                    max_distance=max_radius
                )
                
                # Add to candidate list
                candidates.append({
                    'coords': dest_coords,
                    'details': {
                        'name': result.get('name', 'Unknown Location'),
                        'address': result.get('vicinity', 'Unknown Address'),
                        'place_id': result.get('place_id', ''),
                        'rating': rating,
                        'distance': distance
                    },
                    'score': score
                })
            except Exception as e:
                print(f"Error processing place result: {e}")
                continue
            
        return candidates
        
    def _select_best_candidate(self, candidates):
        """Select the best candidate from a list based on score"""
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]

    def _retrieve_location_osm(self, current_location, destination_type, max_radius, available_minutes):
        """
        Use OpenStreetMap's Overpass API to retrieve destination locations
        
        Args:
            current_location: Current location (latitude, longitude)
            destination_type: Destination type information
            max_radius: Maximum search radius (kilometers)
            available_minutes: Available time (minutes)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # 生成缓存键
            search_query = destination_type.get('search_query', 'point of interest')
            place_type = destination_type.get('place_type', 'amenity')
            cache_key = f"osm_{current_location}_{search_query}_{max_radius}"
            cache_key = cache_key.replace(" ", "_").replace(",", "_")
            
            # 检查缓存
            cached_result = cache.get(cache_key)
            if cached_result and ENABLE_CACHING:
                return cached_result
            
            # 为防止API失败，准备备用位置
            backup_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
            backup_details = {
                'name': f"{search_query.capitalize()}",
                'address': f"Distance: {round(calculate_distance(current_location, backup_location), 1)}km",
                'rating': 3.0,
                'distance': calculate_distance(current_location, backup_location)
            }
            
            # Limit search radius to avoid Bad Request errors - 确保最大搜索半径为50公里(50000米)
            MAX_RADIUS_METERS = 50000  # 最大搜索半径50公里
            radius_meters = min(MAX_RADIUS_METERS, int(max_radius * 1000))
            
            # Google Places type to OSM tag mapping
            osm_tag_mapping = {
                # 金融服务 (Financial Services)
                'bank': 'amenity=bank',
                'atm': 'amenity=atm',
                
                # 食品购物 (Food Shopping)
                'grocery_store': 'shop=supermarket',
                'grocery_or_supermarket': 'shop=supermarket',
                'bakery': 'shop=bakery',
                'butcher': 'shop=butcher',
                'delicatessen': 'shop=deli',
                'fishmonger': 'shop=seafood',
                'greengrocer': 'shop=greengrocer',
                
                # 餐饮场所 (Food & Drink)
                'restaurant': 'amenity=restaurant',
                'cafe': 'amenity=cafe',
                'bar': 'amenity=bar',
                'fast_food': 'amenity=fast_food',
                'food_court': 'amenity=food_court',
                'pub': 'amenity=pub',
                'ice_cream': 'amenity=ice_cream',
                
                # 休闲娱乐 (Leisure & Entertainment)
                'gym': 'leisure=fitness_centre',
                'park': 'leisure=park',
                'cinema': 'amenity=cinema',
                'movie_theater': 'amenity=cinema',
                'theatre': 'amenity=theatre',
                'stadium': 'leisure=stadium',
                'swimming_pool': 'leisure=swimming_pool',
                'night_club': 'amenity=nightclub',
                
                # 文化与教育 (Culture & Education)
                'library': 'amenity=library',
                'museum': 'tourism=museum',
                'art_gallery': 'tourism=gallery',
                'school': 'amenity=school',
                'university': 'amenity=university',
                
                # 医疗健康 (Healthcare)
                'hospital': 'amenity=hospital',
                'doctor': 'amenity=doctors',
                'medical_center': 'amenity=hospital',
                'dentist': 'amenity=dentist',
                'pharmacy': 'amenity=pharmacy',
                'veterinary_care': 'amenity=veterinary',
                'health': 'amenity=hospital',
                
                # 购物场所 (Shopping)
                'store': 'shop=mall',
                'shopping_mall': 'shop=mall',
                'clothing_store': 'shop=clothes',
                'shoe_store': 'shop=shoes',
                'electronics_store': 'shop=electronics',
                'hardware_store': 'shop=hardware',
                'furniture_store': 'shop=furniture',
                'book_store': 'shop=books',
                'jewelry_store': 'shop=jewelry',
                'toy_store': 'shop=toys',
                
                # 个人服务 (Personal Services)
                'beauty_salon': 'shop=hairdresser',
                'laundry': 'shop=laundry',
                'dry_cleaning': 'shop=dry_cleaning',
                
                # 住宿 (Accommodation)
                'hotel': 'tourism=hotel',
                'lodging': 'tourism=hotel',
                
                # 交通服务 (Transportation)
                'gas': 'shop=gas',
                'car_repair': 'shop=car_repair',
                'car_wash': 'amenity=wash',
                'parking': 'amenity=parking',
                'bicycle_parking': 'amenity=bicycle_parking',
                'bus_station': 'amenity=bus_station',
                'train_station': 'railway=station',
                'subway': 'railway=subway',
                'airport': 'aeroway=aerodrome',
                
                # 日常便利设施 (Convenience)
                'convenience_store': 'shop=convenience',
                'post_office': 'amenity=post_office',
                
                # 公共服务 (Public Services)
                'police': 'amenity=police',
                'fire_station': 'amenity=fire_station',
                'city_hall': 'amenity=townhall',
                'courthouse': 'amenity=courthouse',
                'embassy': 'amenity=embassy',
            }
            
            main_tag = osm_tag_mapping.get(place_type)
            
            # Build a single integrated Overpass query - includes main tag and fallback tag
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json];
            (
              // Main search tags
              node[{main_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
              way[{main_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
            );
            out center;
            """
            
            # Send request
            response = requests.post(overpass_url, data={"data": overpass_query})
            response.raise_for_status()
            data = response.json()
            
            # Process results
            candidates = []
            
            if "elements" in data and len(data["elements"]) > 0:
                # 处理最多5个结果
                results = data["elements"][:5]
                
                for result in results:
                    try:
                        # Get location coordinates
                        if "center" in result:
                            lat = result["center"]["lat"]
                            lng = result["center"]["lon"]
                        elif "lat" in result and "lon" in result:
                            lat = result["lat"]
                            lng = result["lon"]
                        else:
                            continue
                            
                        dest_coords = (lat, lng)
                        
                        # Calculate distance
                        distance = calculate_distance(current_location, dest_coords)
                        
                        # Skip if location is too far
                        estimated_time_minutes = (distance / 30) * 60
                        if available_minutes and estimated_time_minutes > available_minutes * 0.75:
                            continue
                        
                        # Get name and address
                        tags = result.get("tags", {})

                        name = tags.get("name", "Unnamed Location")
                        address = tags.get("addr:street", "") + " " + tags.get("addr:housenumber", "")
                        address = address.strip() or f"Distance: {round(distance, 1)}km"
                        
                        # 使用固定默认值替代推断的评分和价格
                        rating = 4.5
                        
                        # Calculate score
                        score = self._calculate_place_score(
                            rating=rating,
                            distance=distance,
                            max_distance=max_radius
                        )
                        
                        # Add to candidate list
                        candidates.append({
                            'coords': dest_coords,
                            'details': {
                                'name': name,
                                'address': address,
                                'place_id': str(result.get("id", "")),
                                'rating': rating,
                                'distance': distance
                            },
                            'score': score
                        })
                    except Exception as e:
                        print(f"Error processing place result: {e}")
                        continue
            
            # If no suitable locations found, try alternative search
            if not candidates:
                if ENABLE_CACHING:
                    cache.set(cache_key, (backup_location, backup_details))
                return backup_location, backup_details
            
            # Select best candidate and cache result
            best_candidate = self._select_best_candidate(candidates)
            result = (best_candidate['coords'], best_candidate['details'])
            
            if ENABLE_CACHING:
                cache.set(cache_key, result)
            
            return result
                
        except Exception as e:
            print(f"Error retrieving location via OSM: {str(e)}")
            # Return random location in case of error
            random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=True)
            return random_location, {'name': f'Random Location (OSM Error)', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}

    def _calculate_place_score(self, rating, distance, max_distance, num_reviews=1, rating_preference=3, popularity_preference=3):
        """
        计算POI的综合评分
        
        Args:
            rating: POI评分 (1-5)
            distance: 到当前位置的距离（公里）
            max_distance: 最大搜索半径（公里）
            num_reviews: 评论数量
            rating_preference: 评分偏好 (1-5)，值越高表示越重视高评分
            popularity_preference: 热度偏好 (1-5)，值越高表示越重视高人气(评论数量)
            
        Returns:
            float: 综合评分 (0-1)
        """
        # 确保参数有效
        if not isinstance(rating, (int, float)):
            rating = 3.0
        if not isinstance(distance, (int, float)):
            distance = max_distance / 2  # 默认使用搜索半径的一半
        if not isinstance(max_distance, (int, float)) or max_distance <= 0:
            max_distance = 50.0  # 默认最大距离为50公里
        if not isinstance(num_reviews, (int, float)):
            num_reviews = 1
        if not isinstance(rating_preference, (int, float)) or rating_preference < 1 or rating_preference > 5:
            rating_preference = 3
        if not isinstance(popularity_preference, (int, float)) or popularity_preference < 1 or popularity_preference > 5:
            popularity_preference = 3
        
        # 评分因素 (30%) - 根据评分偏好调整权重
        rating_weight = 0.3 * (rating_preference / 3)  # 评分偏好标准为3时权重为0.3
        rating_factor = (min(5.0, max(1.0, rating)) / 5.0) * rating_weight
        
        # 距离因素 (40%) - 距离越近分数越高
        distance_factor = max(0, 1 - (distance / max_distance)) * 0.4
        
        # 评论数量因素 (30%) - 根据热度偏好调整权重
        popularity_weight = 0.3 * (popularity_preference / 3)  # 热度偏好标准为3时权重为0.3
        reviews_factor = min(1.0, math.log(num_reviews + 1) / math.log(100)) * popularity_weight
        
        # 确保总权重为1
        remaining_weight = 1.0 - (rating_weight + popularity_weight)
        distance_factor = distance_factor * (remaining_weight / 0.4)
        
        return rating_factor + distance_factor + reviews_factor

    def _generate_random_point_geometrically(self, center, max_distance_km):
        """Generate a random point at a given distance from center point"""
        # Earth radius in kilometers
        earth_radius = 6371.0
        
        # Random distance within the maximum
        distance = random.uniform(0, max_distance_km)
        
        # Random angle (bearing) in radians
        bearing = random.uniform(0, 2 * math.pi)
        
        # Convert latitude and longitude to radians
        lat1 = math.radians(center[0])
        lon1 = math.radians(center[1])
        
        # Calculate target latitude
        lat2 = math.asin(math.sin(lat1) * math.cos(distance / earth_radius) +
                       math.cos(lat1) * math.sin(distance / earth_radius) * math.cos(bearing))
        
        # Calculate target longitude
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / earth_radius) * math.cos(lat1),
                                math.cos(distance / earth_radius) - math.sin(lat1) * math.sin(lat2))
        
        # Convert back to degrees
        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)
        
        return (lat2, lon2)
    
    def _extract_destination_from_text(self, text):
        """
        Extract destination information from text when JSON parsing fails
        
        Args:
            text: Text containing destination information
            
        Returns:
            dict: Extracted destination information
        """
        try:
            # First try to parse as JSON
            if text.strip().startswith('{') and text.strip().endswith('}'):
                try:
                    json_data = json.loads(text)
                    # Validate and normalize JSON result
                    result = {}
                    if 'place_type' in json_data:
                        result['place_type'] = str(json_data['place_type'])
                    if 'search_query' in json_data:
                        result['search_query'] = str(json_data['search_query'])
                    if 'distance_preference' in json_data:
                        try:
                            result['distance_preference'] = float(json_data['distance_preference'])
                        except (ValueError, TypeError):
                            result['distance_preference'] = 5.0
                    
                    return result
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to text parsing
                    pass
        except Exception as e:
            print(f"Error parsing destination as JSON: {e}")
        
        # Fall back to text parsing
        destination = {}
        lines = text.split('\n')
        
        # Define keys to look for
        keys = {
            'place_type': ['place type', 'type of place', 'destination type', 'place_type'],
            'search_query': ['search query', 'search term', 'search for', 'keyword', 'search_query'],
            'distance_preference': ['distance', 'how far', 'kilometers', 'miles', 'distance_preference'],
        }
        
        # Extract information from each line
        current_key = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are likely not relevant
            if line.startswith('```') or line.startswith('#'):
                continue
                
            # Check if this line contains any keywords
            found_key = False
            for key, patterns in keys.items():
                if any(pattern.lower() in line.lower() for pattern in patterns):
                    current_key = key
                    found_key = True
                    # Try to extract value from this line
                    if ':' in line:
                        value_part = line.split(':', 1)[1].strip()
                        value = value_part.strip('",\'').strip()
                        if value:
                            destination[key] = self._process_destination_value(key, value)
                    elif '=' in line:
                        value_part = line.split('=', 1)[1].strip()
                        value = value_part.strip('",\'').strip()
                        if value:
                            destination[key] = self._process_destination_value(key, value)
                    break
            
            # If no keyword found but we have a current key, treat as continuation
            if not found_key and current_key and current_key not in destination:
                destination[current_key] = self._process_destination_value(current_key, line)
        
        # Ensure we have all required fields with default values if missing
        if 'place_type' not in destination:
            destination['place_type'] = 'point_of_interest'
        if 'search_query' not in destination:
            destination['search_query'] = 'place'
        if 'distance_preference' not in destination:
            destination['distance_preference'] = 5.0
            
        return destination
    
    def _process_destination_value(self, key, value):
        """
        Process extracted destination values based on field type
        
        Args:
            key: Field name
            value: Extracted value
            
        Returns:
            Processed value
        """
        if key == 'distance_preference':
            # Try to extract numbers
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                return float(numbers[0])
            return 5.0  # Default
        
        return value

    def _determine_transport_mode(self, persona, activity_type, available_minutes, distance, memory_patterns=None):
        """
        Determine suitable transportation mode based on persona, activity, and distance.
        
        Args:
            persona: Character object
            activity_type: Activity type
            available_minutes: Available time (minutes)
            distance: Distance in kilometers
            memory_patterns: Historical memory patterns (optional)
            
        Returns:
            str: Transportation mode ('walking', 'driving', 'public_transit', 'cycling', 'rideshare')
        """
        # Try LLM-based determination
        transport_mode = self._determine_transport_mode_with_llm(
            persona, 
            activity_type, 
            available_minutes, 
            distance, 
            memory_patterns
        )
        return transport_mode
    
    def _determine_transport_mode_with_llm(self, persona, activity_type, available_minutes, distance, memory_patterns):
        """Use LLM to determine the most appropriate transport mode"""
        try:
            # For very short distances, use heuristic directly without calling LLM
            if distance < 0.3:  # Less than 300 meters
                return 'walking'
                
            # Prepare prompt inputs
            prompt = self._prepare_transport_mode_prompt(
                persona, 
                activity_type, 
                available_minutes, 
                distance, 
                memory_patterns
            )
            
            # 使用LLMManager替代直接调用client
            response = llm_manager.completion_basic(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=20  # Only need a short answer
            )
            
            # Get and normalize response
            transport_mode = self._normalize_transport_mode(response.choices[0].message.content)
            return transport_mode
            
        except Exception as e:
            print(f"LLM transport mode determination failed: {str(e)}")
            # Fall back to simple heuristic determination
            return self._determine_transport_mode_heuristic(distance)
    
    def _prepare_transport_mode_prompt(self, persona, activity_type, available_minutes, distance, memory_patterns):
        """Prepare the prompt for transport mode determination"""
        try:
            # Prepare historical preference data
            pattern_str = "No historical data"
            if memory_patterns and 'transport_modes' in memory_patterns:
                pattern_items = []
                for mode, count in memory_patterns['transport_modes'].items():
                    if mode and mode != 'unknown':
                        pattern_items.append(f"{mode}({count} times)")
                if pattern_items:
                    pattern_str = ", ".join(pattern_items)
            
            # Prepare persona traits
            traits = []
            if hasattr(persona, 'traits'):
                traits.extend(persona.traits)
            traits_str = ", ".join(traits) if traits else "No special traits"
            
            # Ensure all parameters are valid
            gender = getattr(persona, 'gender', 'unknown')
            age = getattr(persona, 'age', 30)
            race = getattr(persona, 'race', 'unknown')
            education = getattr(persona, 'education', 'unknown')
            occupation = getattr(persona, 'occupation', 'unknown')
            
            # Get household income safely
            try:
                household_income = persona.get_household_income()
            except Exception:
                household_income = 50000  # Default value

            # Get household vehicles safely
            try:
                household_vehicles = persona.get_household_vehicles()
            except Exception:
                household_vehicles = 0  # Default value
            
            # Format distance value
            distance_str = f"{float(distance):.1f}" if isinstance(distance, (int, float)) else "1.0"
            
            # Build and return prompt
            return TRANSPORT_MODE_PROMPT.format(
                gender=gender,
                age=age,
                race=race,
                education=education,
                occupation=occupation,
                household_income=household_income,
                household_vehicles=household_vehicles,
                traits=traits_str,
                activity=activity_type,
                minutes=available_minutes,
                distance=distance_str,
                patterns=pattern_str
            )
        except Exception as e:
            print(f"Error preparing transport mode prompt: {str(e)}")
            # Return simplified prompt
            return f"What transportation mode should be used for a {activity_type} activity that is {distance:.1f}km away?"
    
    def _normalize_transport_mode(self, transport_mode_text):
        """Normalize transport mode text to standard categories"""
        transport_mode = transport_mode_text.strip().lower()
        
        if 'walk' in transport_mode:
            return 'walking'
        elif 'cycl' in transport_mode or 'bike' in transport_mode:
            return 'cycling'
        elif 'bus' in transport_mode or 'train' in transport_mode or 'transit' in transport_mode:
            return 'public_transit'
        elif 'car' in transport_mode or 'driv' in transport_mode:
            return 'driving'
        elif 'taxi' in transport_mode or 'uber' in transport_mode or 'lyft' in transport_mode or 'ride' in transport_mode:
            return 'rideshare'
        else:
            return 'walking'  # Default to walking
    
    def _determine_transport_mode_heuristic(self, distance):
        """Determine transport mode based on simple distance heuristics"""
        if distance < 1:
            return 'walking'
        elif distance < 3:
            return 'cycling'
        elif distance < 10:
            return 'public_transit'
        else:
            return 'driving'
        
    def _calculate_search_radius(self, persona, activity_type, time_window, destination_type):
        """
        Calculate the maximum search radius based on available time and preferences.
        
        Args:
            persona: Persona object
            activity_type: Activity type
            time_window: Available time window in minutes
            destination_type: Destination type information with preferences
            
        Returns:
            float: Maximum search radius in kilometers
        """
        # Get base radius from time window, passing destination_type
        max_radius = self._calculate_max_radius(persona, activity_type, time_window, destination_type)
        
        return max_radius

    def _generate_fallback_location(self, max_radius, current_location=None, destination_type=None):
        """Generate a fallback location when normal methods fail"""
        try:
            # Use a random location near the current location if available
            if current_location and isinstance(current_location, tuple) and len(current_location) == 2:
                random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
                name = f'Location for {destination_type.get("search_query", "activity")}' if destination_type else 'Fallback Location'
                return random_location, {
                    'name': name, 
                    'address': f'Generated Location',
                    'is_fallback': True
                }
            else:
                # Complete fallback with default values
                return (0.0, 0.0), {
                    'name': 'Default Location',
                    'address': 'No valid location data available',
                    'is_fallback': True
                }
        except Exception as e:
            print(f"Error generating fallback location: {str(e)}")
            return (0.0, 0.0), {'name': 'Error Location', 'address': 'Error generating location'}