"""
Destination module for the LLM-based mobility simulation.
Handles location retrieval using Google Maps API or OpenStreetMap.
"""

import json
import random
import requests
import openai
import osmnx as ox
import numpy as np
from config import (
    GOOGLE_MAPS_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    USE_GOOGLE_MAPS,
    DESTINATION_SELECTION_PROMPT,
    DEEPBRICKS_API_KEY,
    DEEPBRICKS_BASE_URL,
    TRANSPORT_MODES,
    ENABLE_CACHING
)
from utils import calculate_distance, generate_random_location_near, cached, normalize_transport_mode, estimate_travel_time, cache
import re
import pandas as pd

# Create OpenAI client
client = openai.OpenAI(
    api_key = DEEPBRICKS_API_KEY,
    base_url = DEEPBRICKS_BASE_URL,
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
        # Add location cache
        self.location_cache = {}
        self.transport_mode_cache = {}  # Transportation mode cache
        self.google_maps_cache = {}  # Google Maps API results cache
        self.config = config
    
    @cached
    def select_destination(self, persona, current_location, activity_type, time, day_of_week, available_minutes, transport_mode=None):
        """
        Select an appropriate destination for the given activity
        
        Args:
            persona: Persona object
            current_location: Current location coordinates
            activity_type: Activity type
            time: Time string (HH:MM)
            day_of_week: Day of week string
            available_minutes: Available time (minutes)
            transport_mode: Transportation mode (obtained from activity plan)
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Generate cache key
            cache_key = f"dest_{persona.id}_{activity_type}_{time}_{day_of_week}_{available_minutes}_{transport_mode}"
            cache_key = cache_key.replace(" ", "_")
            
            # Check cache
            if cache_key in self.location_cache:
                print(f"Using cached destination: {cache_key}")
                return self.location_cache[cache_key]
            
            # If no transportation mode specified, determine default transportation mode
            if not transport_mode:
                transport_mode = self._determine_transport_mode(persona, activity_type, available_minutes, current_location)
            else:
                # Standardize transportation mode
                transport_mode = normalize_transport_mode(transport_mode)
            
            # 1. Use LLM to determine suitable destination type, price preference, and distance preference based on activity and character features
            destination_type = self._determine_destination_type(
                persona, 
                {"activity_type": activity_type, "description": f"{activity_type} activity"}, 
                time, 
                day_of_week, 
                transport_mode
            )
            
            # 2. Calculate available time window (excluding round trip travel time)
            time_window = self._calculate_time_window(available_minutes, transport_mode=transport_mode)
            
            # 3. Determine maximum search radius based on available time and transportation mode
            max_radius = self._calculate_max_radius(
                persona, 
                activity_type, 
                time_window, 
                transport_mode=transport_mode
            )
            
            # 4. Apply distance preference adjustment factor based on LLM provided distance preference
            if 'distance_preference' in destination_type:
                # Distance preference value range 1-10, 1 means very close, 10 means can be very far
                distance_pref = destination_type.get('distance_preference', 5)
                # Convert distance preference to radius adjustment factor (0.5-1.5)
                distance_factor = 0.5 + (distance_pref / 10)
                # Adjust maximum radius
                max_radius = max_radius * distance_factor
                print(f"  Distance preference: {distance_pref}, adjusted radius: {max_radius:.2f}km")
            
            # 5. Consider all factors to get destination location
            location, details = self._retrieve_location_google_maps(
                current_location, 
                destination_type, 
                max_radius, 
                available_minutes,
                transport_mode
            )
            
            # 6. Cache result
            self.location_cache[cache_key] = (location, details)
            
            return location, details
            
        except Exception as e:
            print(f"Error occurred during destination selection: {e}")
            # Generate random location as backup
            radius = min(5, available_minutes / 60)  # Simple estimate: 1 hour range 5 kilometers
            random_location = generate_random_location_near(current_location, max_distance_km=radius)
            return random_location, {"name": f"{activity_type}(Backup Location)", "address": "Generated Location"}
    
    @cached
    def _calculate_time_window(self, available_minutes, transport_mode=None):
        """
        Calculate available time window for activity
        
        Args:
            available_minutes: Available time (minutes)
            transport_mode: Transportation mode
        
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
    def _calculate_max_radius(self, persona, activity_type, time_window, transport_mode=None):
        """
        Calculate the maximum search radius based on available time and transportation mode.
        
        Args:
            available_minutes: 可用时间（分钟）
            transport_mode: 交通方式
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
        
        # 2. Transportation mode correction
        transport_modifiers = {
            'walking': 0.5,      # Walking range minimum
            'cycling': 1.0,      # Bicycle range medium
            'public_transit': 1.5, # Public transportation larger range
            'driving': 2.0      # Car range maximum
        }
        transport_modifier = transport_modifiers.get(transport_mode.lower(), 1.0)
        
        # 3. Activity type correction
        activity_modifiers = {
            'work': 1.3,        # Work location can be slightly further
            'shopping': 0.8,    # Shopping preference close
            'dining': 0.7,      # Dining usually close
            'recreation': 1.1,  # Entertainment venue medium
            'healthcare': 1.2,  # Medical facilities may require specific location
            'education': 1.2,   # Education venue may require specific institution
            'social': 0.9,      # Social activity usually choose convenient location
            'errands': 0.5      # Daily affairs try to be close
        }
        activity_modifier = activity_modifiers.get(activity_type.lower(), 1.0)
        
        # 4. Calculate final radius and apply upper limit
        final_radius = min(base_radius * transport_modifier * activity_modifier, 8.0)
        
        # Ensure minimum radius
        return max(0.3, final_radius)
    
    @cached
    def _determine_destination_type(self, persona, activity, time, day_of_week, transport_mode):
        """
        Determine destination type based on character features, activity details, time, and transportation mode
        
        Args:
            persona: Character object
            activity: Activity dictionary, containing activity_type and description
            time: Time string
            day_of_week: Day of week
            transport_mode: Transportation mode
        
        Returns:
            dict: Destination type information
        """
        try:
            # Use LLM for judgment first
            prompt = DESTINATION_SELECTION_PROMPT.format(
                gender=persona.gender,
                age=persona.age,
                income=persona.income,
                consumption=persona.consumption,
                education=persona.education,
                activity_description=activity['description'],
                activity_type=activity['activity_type'],
                time=time,
                day_of_week=day_of_week,
                current_location=persona.current_location
            )
            
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                llm_result = self._extract_destination_from_text(response.choices[0].message.content)
                if llm_result and all(key in llm_result for key in ['place_type', 'search_query', 'distance_preference']):
                    return llm_result
            except Exception as llm_error:
                print(f"LLM call failed, use alternative: {llm_error}")
            
            # Alternative: Pre-defined mapping
            # Activity type to destination type basic mapping
            default_destinations = {
                'shopping': {
                    'place_type': 'store',
                    'search_query': 'shopping center',
                    'distance_preference': 4,
                    'price_level': min(3, max(1, int(persona.income / 25000)))
                },
                'dining': {
                    'place_type': 'restaurant',
                    'search_query': 'restaurant',
                    'distance_preference': 3,
                    'price_level': min(4, max(1, int(persona.income / 20000)))
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
                    'price_level': min(3, max(1, int(persona.income / 30000)))
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
                    'price_level': min(3, max(1, int(persona.income / 25000)))
                },
                'errands': {
                    'place_type': 'store',
                    'search_query': 'convenience services',
                    'distance_preference': 2,
                    'price_level': 1
                }
            }
            
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
                if keyword in activity['description']:
                    destination['place_type'] = mapping['place_type']
                    destination['search_query'] = mapping['search_query']
                    break
            
            # Adjust distance preference based on transportation mode
            if transport_mode == 'walking':
                destination['distance_preference'] = min(3, destination['distance_preference'])
            elif transport_mode == 'cycling':
                destination['distance_preference'] = min(5, destination['distance_preference'])
            elif transport_mode == 'public_transit':
                destination['distance_preference'] = min(7, destination['distance_preference'])
            
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
            print(f"Destination type judgment error: {e}")
            # Return default value
            return {
                'place_type': 'point_of_interest',
                'search_query': activity['activity_type'] if activity['activity_type'] else 'place',
                'distance_preference': 5,
                'price_level': 2
            }
    
    def _retrieve_location_google_maps(self, current_location, destination_type, max_radius, available_minutes, transport_mode):
        """
        Use Google Maps API to get destination location
        
        Args:
            current_location: Current location (latitude, longitude)
            destination_type: Destination type information
            max_radius: Maximum search radius (kilometers)
            available_minutes: Available time (minutes)
            transport_mode: Transportation mode
        
        Returns:
            tuple: ((latitude, longitude), location_details)
        """
        try:
            # Build cache key
            cache_key = f"google_maps_{current_location}_{destination_type.get('search_query')}_{max_radius}_{transport_mode}"
            cache_key = cache_key.replace(" ", "_").replace(",", "_")
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result and ENABLE_CACHING:
                return cached_result
                
            # Prepare parameters for Google Maps API
            search_query = destination_type.get('search_query', 'point of interest')
            place_type = destination_type.get('place_type', 'point_of_interest')
            
            # Convert maximum radius from kilometers to meters
            radius_meters = int(max_radius * 1000)
            
            # Prepare API request parameters
            params = {
                'key': GOOGLE_MAPS_API_KEY,
                'location': f"{current_location[0]},{current_location[1]}",
                'radius': radius_meters,
                'type': place_type,
                'keyword': search_query,
                'language': 'zh-CN',
                'rankby': 'prominence'  # Default sort by popularity
            }
            
            # Call Places API
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                # Process only the first 10 results
                results = data.get('results')[:10]
                
                # Create candidate location list
                candidates = []
                
                for result in results:
                    # Get location coordinates
                    location = result.get('geometry', {}).get('location', {})
                    dest_coords = (location.get('lat'), location.get('lng'))
                    
                    # Calculate distance from current location
                    distance = calculate_distance(current_location, dest_coords)
                    
                    # Estimate travel time
                    travel_time, actual_transport_mode = estimate_travel_time(
                        current_location, 
                        dest_coords, 
                        transport_mode,
                        persona=None
                    )
                    
                    # Skip if travel time exceeds 80% of available time
                    if travel_time > available_minutes * 0.8:
                        continue
                    
                    # Get rating (default 3.0)
                    rating = result.get('rating', 3.0) or 3.0
                    
                    # Calculate simple comprehensive score
                    score = (
                        (available_minutes - travel_time) / available_minutes * 0.4 +  # Time window score
                        (1.0 if actual_transport_mode == transport_mode else 0.5) * 0.3 +  # Transportation mode match
                        (rating / 5.0) * 0.3  # Rating
                    )
                    
                    # Add to candidate list
                    candidates.append({
                        'coords': dest_coords,
                        'details': {
                            'name': result.get('name', 'Unknown Location'),
                            'address': result.get('vicinity', 'Unknown Address'),
                            'place_id': result.get('place_id', ''),
                            'rating': rating,
                            'distance': distance,
                            'travel_time': travel_time,
                            'transport_mode': actual_transport_mode
                        },
                        'score': score
                    })
                
                # If no suitable location found, try alternative approach
                if not candidates:
                    # print(f"Google Places API returned no results: {data.get('status')}")
                    
                    # Try approach 1: Use more generic search keywords
                    generic_keywords = {
                        'restaurant': ['restaurant', 'dining', 'food'],
                        'cafe': ['cafe', 'coffee', 'tea'],
                        'shopping': ['shopping', 'mall', 'store', 'shop'],
                        'recreation': ['park', 'recreation', 'leisure'],
                        'work': ['office', 'business', 'work'],
                        'education': ['school', 'library', 'bookstore'],
                        'leisure': ['entertainment', 'cinema', 'theater'],
                        'exercise': ['gym', 'fitness', 'sports'],
                        'errand': ['convenience', 'pharmacy', 'service']
                    }
                    
                    # Extract possible activity type from search query
                    activity_type = None
                    search_query = destination_type.get('search_query', '')
                    for key, values in generic_keywords.items():
                        if any(keyword in search_query.lower() for keyword in values):
                            activity_type = key
                            break
                    
                    if activity_type:
                        # Search with more generic keywords
                        generic_params = {
                            'key': GOOGLE_MAPS_API_KEY,
                            'location': f"{current_location[0]},{current_location[1]}",
                            'radius': int(max_radius * 1000 * 1.5),  # Expand search range
                            'keyword': generic_keywords.get(activity_type, ['place'])[0]
                        }
                        
                        try:
                            generic_response = requests.get(url, params=generic_params)
                            generic_data = generic_response.json()
                            
                            if generic_data.get('status') == 'OK' and generic_data.get('results'):
                                # Process found alternative location
                                result = generic_data['results'][0]  # Use first result directly
                                location = result.get('geometry', {}).get('location', {})
                                dest_coords = (location.get('lat'), location.get('lng'))
                                
                                # Calculate distance and travel time
                                distance = calculate_distance(current_location, dest_coords)
                                details = {
                                    'name': result.get('name', 'Alternative Location'),
                                    'address': result.get('vicinity', 'Unknown address'),
                                    'rating': result.get('rating', 3.0),
                                    'distance': distance
                                }
                                
                                # print(f"Found alternative location: {details['name']} ({details['address']})")
                                return dest_coords, details
                        except Exception as e:
                            print(f"Alternative search error: {str(e)}")
                    
                    # Last resort use random location
                    random_location = generate_random_location_near(current_location, max_radius)
                    return random_location, {'name': f'Nearby Location', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}
                
                # Sort by score and select best location
                candidates.sort(key=lambda x: x['score'], reverse=True)
                best_candidate = candidates[0]
                
                # Save to cache
                result = (best_candidate['coords'], best_candidate['details'])
                if ENABLE_CACHING:
                    cache.set(cache_key, result)
                
                return result
                
            else:
                # If API returns no results, try alternative approaches
                # print(f"Google Places API returned no results: {data.get('status')}")
                pass
                
                # Approach 1: Use more generic search keywords
                generic_keywords = {
                    'restaurant': ['restaurant', 'dining', 'food'],
                    'cafe': ['cafe', 'coffee', 'tea'],
                    'shopping': ['shopping', 'mall', 'store', 'shop'],
                    'recreation': ['park', 'recreation', 'leisure'],
                    'work': ['office', 'business', 'work'],
                    'education': ['school', 'library', 'bookstore'],
                    'leisure': ['entertainment', 'cinema', 'theater'],
                    'exercise': ['gym', 'fitness', 'sports'],
                    'errand': ['convenience', 'pharmacy', 'service']
                }
                
                # Extract possible activity type from search query
                activity_type = None
                search_query = destination_type.get('search_query', '')
                for key, values in generic_keywords.items():
                    if any(keyword in search_query.lower() for keyword in values):
                        activity_type = key
                        break
                
                if activity_type:
                    # Search with more generic keywords
                    generic_params = {
                        'key': GOOGLE_MAPS_API_KEY,
                        'location': f"{current_location[0]},{current_location[1]}",
                        'radius': int(max_radius * 1000 * 1.5),  # Expand search radius
                        'keyword': generic_keywords.get(activity_type, ['place'])[0]
                    }
                    
                    try:
                        generic_response = requests.get(url, params=generic_params)
                        generic_data = generic_response.json()
                        
                        if generic_data.get('status') == 'OK' and generic_data.get('results'):
                            # Process alternative location found
                            result = generic_data['results'][0]  # Use first result
                            location = result.get('geometry', {}).get('location', {})
                            dest_coords = (location.get('lat'), location.get('lng'))
                            
                            # Calculate distance and travel time
                            distance = calculate_distance(current_location, dest_coords)
                            details = {
                                'name': result.get('name', 'Alternative location'),
                                'address': result.get('vicinity', 'Unknown address'),
                                'rating': result.get('rating', 3.0),
                                'distance': distance
                            }
                            
                            # print(f"    Found alternative location: {details['name']} ({details['address']})")
                            return dest_coords, details
                    except Exception as e:
                        print(f"Alternative search error: {str(e)}")
                
                # Only use random location as last resort
                random_location = generate_random_location_near(current_location, max_radius)
                return random_location, {'name': f'Nearby location', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}
                
        except Exception as e:
            print(f"Error occurred retrieving location: {str(e)}")
            # Return random location in case of error
            random_location = generate_random_location_near(current_location, max_radius * 0.5)
            return random_location, {'name': f'Random Location (Error)', 'address': f'Distance: {round(calculate_distance(current_location, random_location), 1)}km'}
    
    def _extract_destination_from_text(self, text):
        """
        Extract destination information from text when JSON parsing fails
        
        Args:
            text: Text containing destination information
            
        Returns:
            dict: Extracted destination information
        """
        destination = {}
        lines = text.split('\n')
        
        # Define keys to look for
        keys = {
            'place_type': ['place type', 'type of place', 'destination type'],
            'search_query': ['search query', 'search term', 'search for'],
            'distance_preference': ['distance', 'how far', 'kilometers', 'miles'],
            'price_level': ['price', 'cost', 'expensive', 'budget']
        }
        
        # Extract information from each line
        current_key = None
        for line in lines:
            line = line.strip()
            if not line:
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
                    break
            
            # If no keyword found but we have a current key, treat as continuation
            if not found_key and current_key and current_key not in destination:
                destination[current_key] = self._process_destination_value(current_key, line)
        
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
        
        elif key == 'price_level':
            # Try to map price descriptions to levels
            value = value.lower()
            if 'high' in value or 'expensive' in value or 'premium' in value:
                return 4
            elif 'mid-high' in value or 'above average' in value:
                return 3
            elif 'mid' in value or 'average' in value or 'moderate' in value:
                return 2
            elif 'low' in value or 'budget' in value or 'cheap' in value:
                return 1
            # Try to extract numbers
            numbers = re.findall(r'\d+', value)
            if numbers:
                return min(4, max(1, int(numbers[0])))
            return 2  # Default
        
        return value
    
    def _determine_transport_mode(self, persona, activity_type, available_minutes, current_location):
        """
        Determine suitable transportation mode based on character features, activity, and available time
        
        Args:
            persona: Character object
            activity_type: An activity type (e.g. "dinner", "shopping")
            available_minutes: Available time (minutes)
            current_location: Current location
            
        Returns:
            str: Transportation mode ('walking', 'driving', 'public_transit', 'cycling')
        """
        # Cache key
        cache_key = f"transport_{persona.id}_{activity_type}_{available_minutes}"
        if cache_key in self.transport_mode_cache:
            return self.transport_mode_cache[cache_key]
        
        # Default transportation mode
        transport_mode = 'walking'
        
        # Based on available time, preliminary judgment
        if available_minutes < 30:
            transport_mode = 'walking'  # Short time, walking is the most convenient
        elif 30 <= available_minutes < 90:
            # Short to medium distance, possibly walking, cycling, or public transit
            options = ['walking', 'cycling', 'public_transit']
            transport_mode = random.choice(options)
        else:
            # Long activity, may use any transportation mode
            options = ['walking', 'cycling', 'public_transit', 'driving']
            transport_mode = random.choice(options)
        
        # Based on personal preference adjustment
        if hasattr(persona, 'transportation_preference') and persona.transportation_preference:
            # 70% probability use preferred transportation mode
            if random.random() < 0.7:
                transport_mode = persona.transportation_preference
        
        # Based on activity type adjustment
        shopping_activities = ['shopping', 'grocery shopping', 'mall']
        if any(act in activity_type.lower() for act in shopping_activities) and available_minutes > 60:
            # Shopping may require carrying items, tend to drive or public transit
            if random.random() < 0.7:
                transport_mode = random.choice(['driving', 'public_transit'])
        
        # Cache result
        self.transport_mode_cache[cache_key] = transport_mode
        return transport_mode