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
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    USE_GOOGLE_MAPS,
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
    cache
)

# Create OpenAI client
client = openai.OpenAI(
    api_key=DEEPBRICKS_API_KEY,
    base_url=DEEPBRICKS_BASE_URL,
)

class Destination:
    """
    Handles destination selection and location retrieval.
    """
    
    def __init__(self, config=None):
        """Initialize the Destination selector."""
        self.google_maps_api_key = GOOGLE_MAPS_API_KEY
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.use_google_maps = USE_GOOGLE_MAPS
        self.config = config or {}
        
        # Initialize caches
        self.location_cache = {}
        self.transport_mode_cache = {}
        self.google_maps_cache = {}
    
    @cached
    def select_destination(self, persona, current_location, activity_type, time, day_of_week, available_minutes, memory_patterns=None, location_type_override=None):
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
            location_type_override: 根据活动描述提取的位置类型信息 (optional)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Step 1: Determine appropriate destination type
            if location_type_override:
                # 如果提供了特定地点类型覆盖信息，直接使用
                destination_type = location_type_override
            else:
                # 否则使用正常逻辑确定目的地类型
                destination_type = self._determine_destination_type(
                    persona, 
                    {"activity_type": activity_type, "description": f"{activity_type} activity"}, 
                    time, 
                    day_of_week,
                    memory_patterns
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

            print('---------------------------------------')    
            print(location, details)
            print('---------------------------------------')

            # Step 4: 确保距离被正确计算
            if 'distance' not in details or not isinstance(details['distance'], (int, float)):
                details['distance'] = calculate_distance(current_location, location)

            # Step 5: Determine transportation details - 仅在目的地确定后才确定交通方式
            transport_mode = self._determine_transport_mode(
                persona, 
                activity_type,
                available_minutes,
                details['distance'],
                memory_patterns
            )
            
            # Step 6: 计算选定交通方式下的旅行时间
            travel_time, _ = estimate_travel_time(current_location, location, transport_mode)
                
            # Step 7: Update location details with transport information
            details.update({
                'name': details['name'],
                'transport_mode': transport_mode,
                'travel_time': travel_time,
                'distance': details['distance']  # 确保距离信息存在
            })
            
            return location, details
            
        except Exception as e:
            print(f"Error occurred during destination selection: {e}")
            # Generate random location as backup
            radius = min(5, available_minutes / 60)  # Simple estimate: 1 hour range 5 kilometers
            random_location = generate_random_location_near(current_location, max_distance_km=radius)
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
    def _calculate_max_radius(self, persona, activity_type, time_window):
        """
        Calculate the maximum search radius based on available time.
        
        Args:
            available_minutes: 可用时间（分钟）
            activity_type: 活动类型
        
        Returns:
            float: 最大半径（公里）
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
        
        # 2. Activity type correction
        activity_modifiers = {
            'shopping': 0.8,    # Shopping preference close
            'dining': 0.7,      # Dining usually close
            'recreation': 1.1,  # Entertainment venue medium
            'healthcare': 1.2,  # Medical facilities may require specific location
            'education': 1.2,   # Education venue may require specific institution
            'social': 0.9,      # Social activity usually choose convenient location
            'errands': 0.5      # Daily affairs try to be close
        }
        activity_modifier = activity_modifiers.get(activity_type.lower(), 1.0)
        
        # 3. Calculate final radius and apply upper limit
        final_radius = min(base_radius * activity_modifier, 8.0)
        
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
                'price_level': 2
            }
    
    def _determine_destination_type_with_llm(self, persona, activity, time, day_of_week, memory_patterns):
        """Use LLM to determine appropriate destination type"""
        try:
            # Format prompt with all contextual information
            # Get persona attributes safely with defaults
            gender = getattr(persona, 'gender', 'unknown')
            age = getattr(persona, 'age', 30)
            occupation = getattr(persona, 'occupation', 'unknown')
            education = getattr(persona, 'education', 'unknown')
            current_location = getattr(persona, 'current_location', 'unknown')
            
            # Get household income safely
            try:
                household_income = persona.get_household_income()
            except Exception:
                household_income = 50000  # Default value
                
            # Get activity details safely
            activity_description = activity.get('description', 'unknown activity')
            activity_type = activity.get('activity_type', 'unknown')
            
            # Format memory context
            memory_context = str(memory_patterns) if memory_patterns else "No memory patterns available"
            
            # Format the prompt
            enhanced_prompt = DESTINATION_SELECTION_PROMPT.format(
                gender=gender,
                age=age,
                household_income=household_income,
                education=education,
                occupation=occupation,
                activity_description=activity_description,
                activity_type=activity_type,
                time=time,
                day_of_week=day_of_week,
                memory_context=memory_context
            )
            
            # Call LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
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
                
            if 'price_level' not in result:
                result['price_level'] = 2
                
            return result
            
        except Exception as llm_error:
            print(f"LLM destination type determination failed: {str(llm_error)}")
            return None
    
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
                    'price_level': min(3, max(1, int(persona.get_household_income() / 25000)))
                },
                'dining': {
                    'place_type': 'restaurant',
                    'search_query': 'restaurant',
                    'distance_preference': 3,
                    'price_level': min(4, max(1, int(persona.get_household_income() / 20000)))
                },
                'recreation': {
                    'place_type': 'park',
                    'search_query': 'park recreation area',
                    'distance_preference': 4,
                    'price_level': 1
                },
                'leisure': {
                    'place_type': 'point_of_interest',
                    'search_query': 'leisure entertainment',
                    'distance_preference': 5,
                    'price_level': min(3, max(1, int(persona.get_household_income() / 30000)))
                },
                'healthcare': {
                    'place_type': 'health',
                    'search_query': 'medical center',
                    'distance_preference': 6,
                    'price_level': 2
                },
                'education': {
                    'place_type': 'school',
                    'search_query': 'education center',
                    'distance_preference': 5,
                    'price_level': 2
                },
                'social': {
                    'place_type': 'bar',
                    'search_query': 'social venue cafe',
                    'distance_preference': 4,
                    'price_level': min(3, max(1, int(persona.get_household_income() / 25000)))
                },
                'errands': {
                    'place_type': 'store',
                    'search_query': 'convenience services',
                    'distance_preference': 2,
                    'price_level': 1
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
                'bank': {'place_type': 'bank', 'search_query': 'bank'},
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
                'price_level': 2
            }).copy()
            
            # Based on description keyword optimization
            for keyword, mapping in keywords.items():
                if keyword in activity['description'].lower():
                    destination['place_type'] = mapping['place_type']
                    destination['search_query'] = mapping['search_query']
                    break
            
            # Further adjust based on day of week and time
            if day_of_week in ['Saturday', 'Sunday']:
                # Weekend may prefer better venue
                destination['price_level'] = min(4, destination['price_level'] + 1)
            
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
                'price_level': 2
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
            if not current_location or not isinstance(current_location, tuple) or len(current_location) != 2:
                print(f"Invalid current_location format: {current_location}, using fallback")
                # Use a default location if the input is invalid
                return self._generate_fallback_location(max_radius)
                
            # Verify destination_type is a valid dictionary
            if not destination_type or not isinstance(destination_type, dict):
                print(f"Invalid destination_type: {destination_type}, using default")
                destination_type = {
                    'place_type': 'point_of_interest',
                    'search_query': 'place',
                    'distance_preference': 5,
                    'price_level': 2
                }
                
            # Validate and correct max_radius if necessary
            if not isinstance(max_radius, (int, float)) or max_radius <= 0:
                max_radius = 2.0  # Default 2km
                
            # Select method based on configuration
            if self.use_google_maps:
                return self._retrieve_location_google_maps(
                    current_location, 
                    destination_type, 
                    max_radius, 
                    available_minutes
                )
            else:
                # 使用OpenStreetMap作为替代
                return self._retrieve_location_osm(
                    current_location,
                    destination_type,
                    max_radius,
                    available_minutes
                )
        
        except Exception as e:
            print(f"Error in _retrieve_location: {str(e)}")
            return self._generate_fallback_location(max_radius, current_location)
            
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
            
            # If no suitable location found, try alternative approach
            if not candidates:
                return self._try_alternative_search(current_location, destination_type, max_radius)
            
            # Select best candidate and cache result
            best_candidate = self._select_best_candidate(candidates)
            result = (best_candidate['coords'], best_candidate['details'])
            
            if ENABLE_CACHING:
                cache.set(cache_key, result)
            
            return result
                
        except Exception as e:
            print(f"Error occurred retrieving location: {str(e)}")
            # Return random location in case of error
            random_location = generate_random_location_near(current_location, max_radius * 0.5)
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
        radius_meters = int(max_radius * 1000)
        
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
        
        # 获取价格偏好
        price_preference = destination_type.get('price_level', 2)
        
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
                
                # Skip if location is too far (rough estimate based on distance)
                # Assuming average speed of 30km/h, estimating if we can reach in time
                estimated_time_minutes = (distance / 30) * 60
                if available_minutes and estimated_time_minutes > available_minutes * 0.4:
                    continue
                
                # Get rating and price level
                rating = result.get('rating', 3.0) or 3.0
                place_price_level = result.get('price_level', 2)
                
                # 使用优化后的评分函数，传递所有需要的参数
                score = self._calculate_place_score(
                    rating=rating, 
                    distance=distance, 
                    max_distance=max_radius, 
                    price_level=place_price_level, 
                    price_preference=price_preference
                )
                
                # Add to candidate list
                candidates.append({
                    'coords': dest_coords,
                    'details': {
                        'name': result.get('name', 'Unknown Location'),
                        'address': result.get('vicinity', 'Unknown Address'),
                        'place_id': result.get('place_id', ''),
                        'rating': rating,
                        'price_level': place_price_level,
                        'distance': distance
                    },
                    'score': score
                })
            except Exception as e:
                print(f"Error processing place result: {e}")
                continue
            
        return candidates
        
    def _calculate_place_score(self, rating, distance, max_distance, price_level, price_preference):
        """
        Calculate a comprehensive score for a place based on rating, distance and price match
        
        Args:
            rating: Place rating (1-5)
            distance: Distance in kilometers
            max_distance: Maximum search radius
            price_level: Place price level (1-4)
            price_preference: User price preference (1-4)
            
        Returns:
            float: Score value
        """
        # Ensure parameters are valid
        if not isinstance(rating, (int, float)):
            rating = 3.0
        if not isinstance(distance, (int, float)):
            distance = max_distance / 2  # Default to middle of search radius
        if not isinstance(max_distance, (int, float)) or max_distance <= 0:
            max_distance = 5.0
        if not isinstance(price_level, (int, float)):
            price_level = 2
        if not isinstance(price_preference, (int, float)):
            price_preference = 2
            
        # Normalize rating (40%)
        rating_factor = (min(5.0, max(1.0, rating)) / 5.0) * 0.4
        
        # Distance factor (40%) - 越近越好
        distance_factor = max(0, 1 - (distance / max_distance)) * 0.3
        
        # Price match (20%) - 价格等级与偏好匹配程度
        price_match = 1 - (abs(price_level - price_preference) / 4)
        price_factor = max(0, price_match) * 0.3
        
        return rating_factor + distance_factor + price_factor
        
    def _select_best_candidate(self, candidates):
        """Select the best candidate from a list based on score"""
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]
    
    def _try_alternative_search(self, current_location, destination_type, max_radius):
        """
        Try alternative search approaches when primary search fails
        
        Args:
            current_location: Current location coordinates
            destination_type: Destination type information
            max_radius: Maximum search radius in kilometers
            
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        # Try with generic keywords using OSM first
        generic_result = self._try_alternative_osm_search(current_location, destination_type, max_radius)
        if generic_result:
            return generic_result
        
        # Fall back to random location as last resort
        random_location = generate_random_location_near(current_location, max_radius, validate=False)
        return random_location, {
            'name': f'Nearby location', 
            'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'
        }
    
    def _try_alternative_osm_search(self, current_location, destination_type, max_radius):
        """
        Try OSM search with more generic keywords when specific search fails
        
        Args:
            current_location: Current location coordinates
            destination_type: Destination type information
            max_radius: Maximum search radius in kilometers
            
        Returns:
            tuple: ((latitude, longitude), location_details) or None if search fails
        """
        # Map of generic keywords for different activity types
        generic_keywords = {
            'restaurant': ['amenity=restaurant', 'amenity=food'],
            'cafe': ['amenity=cafe', 'amenity=restaurant'],
            'shopping': ['shop', 'shop=mall', 'shop=supermarket'],
            'recreation': ['leisure=park', 'leisure'],
            'work': ['office', 'building=office'],
            'education': ['amenity=school', 'amenity=library'],
            'leisure': ['leisure', 'amenity=cinema', 'amenity=theatre'],
            'exercise': ['leisure=fitness_centre', 'leisure=sports_centre'],
            'errand': ['shop=convenience', 'amenity=pharmacy']
        }
        
        # Extract possible activity type from search query
        search_query = destination_type.get('search_query', '').lower()
        activity_type = None
        
        # Try to infer activity type from search query
        for key in generic_keywords.keys():
            if key in search_query:
                activity_type = key
                break
                
        if not activity_type:
            # If activity type still not found, use a very generic query
            try:
                # Expanded search radius
                radius_meters = int(max_radius * 1000 * 1.5)
                
                # Generic OSM query
                overpass_url = "https://overpass-api.de/api/interpreter"
                overpass_query = f"""
                [out:json];
                (
                  node[amenity](around:{radius_meters},{current_location[0]},{current_location[1]});
                  way[amenity](around:{radius_meters},{current_location[0]},{current_location[1]});
                );
                out center;
                """
                
                response = requests.post(overpass_url, data={"data": overpass_query})
                data = response.json()
                
                if "elements" in data and len(data["elements"]) > 0:
                    # Process first result
                    result = data["elements"][0]
                    
                    # Get coordinates
                    if "center" in result:
                        lat = result["center"]["lat"]
                        lng = result["center"]["lon"]
                    elif "lat" in result and "lon" in result:
                        lat = result["lat"]
                        lng = result["lon"]
                    else:
                        return None
                        
                    dest_coords = (lat, lng)
                    
                    # Calculate distance
                    distance = calculate_distance(current_location, dest_coords)
                    
                    # Get name and address
                    tags = result.get("tags", {})
                    name = tags.get("name", "Alternative location")
                    
                    details = {
                        'name': name,
                        'address': tags.get("addr:street", "Unknown address"),
                        'rating': 3.0,
                        'distance': distance
                    }
                    
                    return dest_coords, details
                    
            except Exception as e:
                print(f"Alternative OSM search error: {str(e)}")
                return None
        else:
            # If we have identified an activity type, use OSM with generic keywords
            try:
                # Get generic OSM tags for this activity type
                osm_tags = generic_keywords.get(activity_type, ['amenity'])
                osm_tag = osm_tags[0]  # Use first (most generic) tag
                
                # Expanded search radius
                radius_meters = int(max_radius * 1000 * 1.5)
                
                # OSM query
                overpass_url = "https://overpass-api.de/api/interpreter"
                overpass_query = f"""
                [out:json];
                (
                  node[{osm_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
                  way[{osm_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
                );
                out center;
                """
                
                response = requests.post(overpass_url, data={"data": overpass_query})
                data = response.json()
                
                if "elements" in data and len(data["elements"]) > 0:
                    # Process first result
                    result = data["elements"][0]
                    
                    # Get coordinates
                    if "center" in result:
                        lat = result["center"]["lat"]
                        lng = result["center"]["lon"]
                    elif "lat" in result and "lon" in result:
                        lat = result["lat"]
                        lng = result["lon"]
                    else:
                        return None
                        
                    dest_coords = (lat, lng)
                    
                    # Calculate distance
                    distance = calculate_distance(current_location, dest_coords)
                    
                    # Get name and address
                    tags = result.get("tags", {})
                    name = tags.get("name", "Alternative location")
                    
                    details = {
                        'name': name,
                        'address': tags.get("addr:street", "Unknown address"),
                        'rating': self._get_venue_rating(tags, activity_type, name),
                        'price_level': self._infer_price_level(tags, dest_coords, activity_type),
                        'distance': distance
                    }
                    
                    return dest_coords, details
                    
            except Exception as e:
                print(f"Alternative OSM search error: {str(e)}")
                
        return None

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
        # Get base radius from time window
        max_radius = self._calculate_max_radius(persona, activity_type, time_window)
        
        # Apply distance preference if available
        if 'distance_preference' in destination_type:
            # Distance preference value range 1-10, 1 means very close, 10 means can be very far
            distance_pref = destination_type.get('distance_preference', 5)
            # Convert distance preference to radius adjustment factor (0.5-1.5)
            distance_factor = 0.5 + (distance_pref / 10)
            # Adjust maximum radius
            max_radius = max_radius * distance_factor
            
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
            # Check cache
            cache_key = f"osm_{current_location}_{destination_type.get('search_query')}_{max_radius}"
            cache_key = cache_key.replace(" ", "_").replace(",", "_")
            cached_result = cache.get(cache_key)
            if cached_result and ENABLE_CACHING:
                return cached_result
            
            # Prepare Overpass API query
            search_query = destination_type.get('search_query', 'point of interest')
            place_type = destination_type.get('place_type', 'amenity')
            radius_meters = int(max_radius * 1000)
            
            # Map Google Places API location types to OSM tags
            osm_tag_mapping = {
                'restaurant': 'amenity=restaurant',
                'cafe': 'amenity=cafe',
                'bar': 'amenity=bar',
                'shopping_mall': 'shop=mall',
                'grocery_store': 'shop=supermarket',
                'park': 'leisure=park',
                'gym': 'leisure=fitness_centre',
                'school': 'amenity=school',
                'university': 'amenity=university',
                'hospital': 'amenity=hospital',
                'cinema': 'amenity=cinema',
                'theater': 'amenity=theatre',
                'museum': 'tourism=museum',
                'bank': 'amenity=bank',
                'post_office': 'amenity=post_office',
                'hotel': 'tourism=hotel',
                'point_of_interest': 'amenity'
            }
            
            # Get corresponding OSM tag
            osm_tag = osm_tag_mapping.get(place_type, f"name~'{search_query}'")
            if "=" not in osm_tag:
                # If no exact match found, use generic query
                osm_tag = f"name~'{search_query}'"
            
            # Build Overpass query
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json];
            (
              node[{osm_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
              way[{osm_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
              relation[{osm_tag}](around:{radius_meters},{current_location[0]},{current_location[1]});
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
                # Process up to 5 results
                results = data["elements"][:5]
                price_preference = destination_type.get('price_level', 2)
                
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
                        if available_minutes and estimated_time_minutes > available_minutes * 0.4:
                            continue
                        
                        # Get name and address
                        tags = result.get("tags", {})
                        name = tags.get("name", "Unnamed Location")
                        address = tags.get("addr:street", "") + " " + tags.get("addr:housenumber", "")
                        address = address.strip() or f"Distance: {round(distance, 1)}km"
                        
                        # Use intelligent inference instead of default values
                        # Infer rating based on venue type
                        rating = self._get_venue_rating(tags, place_type, name)
                        
                        # Infer price level
                        price_level = self._infer_price_level(tags, dest_coords, place_type)
                        
                        # Calculate score
                        score = self._calculate_place_score(
                            rating=rating,
                            distance=distance,
                            max_distance=max_radius,
                            price_level=price_level,
                            price_preference=price_preference
                        )
                        
                        # Add to candidate list
                        candidates.append({
                            'coords': dest_coords,
                            'details': {
                                'name': name,
                                'address': address,
                                'place_id': str(result.get("id", "")),
                                'rating': rating,
                                'price_level': price_level,
                                'distance': distance
                            },
                            'score': score
                        })
                    except Exception as e:
                        print(f"Error processing place result: {e}")
                        continue
            
            # If no suitable locations found, try alternative search
            if not candidates:
                return self._try_alternative_osm_search(current_location, destination_type, max_radius)
            
            # Select best candidate and cache result
            best_candidate = self._select_best_candidate(candidates)
            result = (best_candidate['coords'], best_candidate['details'])
            
            if ENABLE_CACHING:
                cache.set(cache_key, result)
            
            return result
                
        except Exception as e:
            print(f"Error retrieving location via OSM: {str(e)}")
            # Return random location in case of error
            random_location = generate_random_location_near(current_location, max_radius * 0.5, validate=False)
            return random_location, {'name': f'Random Location (OSM Error)', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}

    def _infer_price_level(self, tags, location, place_type):
        """
        Infer price level based on tags and geographic location
        
        Args:
            tags: OSM tags
            location: Venue location
            place_type: Place type
            
        Returns:
            int: Price level (1-4)
        """
        # Infer price based on place type
        amenity = tags.get("amenity")
        shop_type = tags.get("shop")
        tourism = tags.get("tourism")
        
        # Restaurants are generally more expensive
        if amenity in ["restaurant"]:
            # If it's an upscale restaurant, return higher price level
            if "cuisine" in tags and tags["cuisine"] in ["french", "japanese", "italian"]:
                return 3
            return 2
        elif amenity in ["cafe", "bar"]:
            return 2
        elif amenity in ["fast_food"] or shop_type in ["convenience"]:
            return 1
        elif shop_type in ["mall", "department_store"]:
            return 3
        elif tourism in ["hotel"]:
            if tags.get("stars") in ["4", "5"]:
                return 4
            return 3
        
        # Based on place type mapping (from Google Places conversion to OSM)
        price_mapping = {
            'restaurant': 2,
            'cafe': 2,
            'bar': 2,
            'shopping_mall': 3,
            'grocery_store': 1,
            'retail_store': 2,
            'cinema': 2,
            'theater': 3,
            'museum': 2,
            'hotel': 3
        }
        
        return price_mapping.get(place_type, 2)
    
    def _get_venue_rating(self, tags, place_type, name):
        """
        Infer rating based on venue type and name
        
        Args:
            tags: OSM tags
            place_type: Place type
            name: Venue name
            
        Returns:
            float: Rating (1.0-5.0)
        """
        # Infer venue type based on OSM tags
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        leisure = tags.get("leisure")
        tourism = tags.get("tourism")
        
        # Default ratings based on venue type
        default_ratings = {
            'restaurant': 3.8,
            'cafe': 3.7,
            'park': 4.2,
            'shopping_mall': 3.6,
            'grocery_store': 3.5,
            'gym': 4.0,
            'hospital': 3.9,
            'school': 3.8,
            'university': 4.1,
            'cinema': 3.9,
            'theater': 4.0,
            'museum': 4.3,
            'bar': 3.6,
            'hotel': 3.8,
            'bank': 3.0,
            'post_office': 3.0
        }
        
        # Higher ratings for certain amenity types
        if leisure == "park":
            return 4.2
        elif tourism in ["museum", "attraction"]:
            return 4.3
        
        # Check for keywords in name that might indicate chains or known places
        name_lower = name.lower()
        chain_keywords = ["mcdonald", "starbucks", "kfc", "walmart", "subway", "carrefour"]
        if any(keyword in name_lower for keyword in chain_keywords):
            return 3.9
        
        # Return default rating
        if amenity:
            return default_ratings.get(amenity, 3.0)
        elif shop:
            return default_ratings.get(shop, 3.0)
        
        # Return rating based on Google Places type mapping
        return default_ratings.get(place_type, 3.0)

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
            
            # Call LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=20,  # Only need a short answer
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
            education = getattr(persona, 'education', 'unknown')
            occupation = getattr(persona, 'occupation', 'unknown')
            
            # Get household income safely
            try:
                household_income = persona.get_household_income()
            except Exception:
                household_income = 50000  # Default value
            
            # Format distance value
            distance_str = f"{float(distance):.1f}" if isinstance(distance, (int, float)) else "1.0"
            
            # Build and return prompt
            return TRANSPORT_MODE_PROMPT.format(
                gender=gender,
                age=age,
                education=education,
                occupation=occupation,
                household_income=household_income,
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