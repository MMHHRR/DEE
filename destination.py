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
    TRANSPORT_MODES
)
from utils import calculate_distance, generate_random_location_near
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
    
    def __init__(self):
        """Initialize the Destination selector."""
        self.google_maps_api_key = GOOGLE_MAPS_API_KEY
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.use_google_maps = USE_GOOGLE_MAPS
    
    def select_destination(self, persona, activity, date, time, day_of_week):
        """
        Select a destination for an activity based on persona characteristics.
        
        Args:
            persona: Persona object
            activity: Activity dictionary
            date: Date string (YYYY-MM-DD)
            time: Time string (HH:MM)
            day_of_week: Day of week string
        
        Returns:
            tuple: (latitude, longitude) of the selected destination
        """
        # The activity location is fixed
        if activity['location_type'].lower() in ['home']:
            activity['location_name'] = "Home"
            return persona.home
        
        if activity['location_type'].lower() in ['work', 'workplace']:
            activity['location_name'] = "Workplace"
            return persona.work
        
        # Determine the appropriate destination type using LLM
        destination_type = self._determine_destination_type(
            persona, 
            activity, 
            time, 
            day_of_week
        )
        
        # Retrieve the actual location based on the destination type
        if self.use_google_maps and self.google_maps_api_key:
            location_coords, location_details = self._retrieve_location_google_maps(
                persona.current_location, 
                destination_type
            )
            # Store location details in the activity
            activity['location_name'] = location_details.get('name', 'Unknown location')
            activity['location_details'] = location_details
            return location_coords
        else:
            location_coords, location_details = self._retrieve_location_osm(
                persona.current_location, 
                destination_type
            )
            # Store location details in the activity
            activity['location_name'] = location_details.get('name', 'Unknown location (OSM)')
            activity['location_details'] = location_details
            return location_coords
    
    def _determine_destination_type(self, persona, activity, time, day_of_week):
        """
        Use LLM to determine the appropriate destination type for an activity.
        
        Args:
            persona: Persona object
            activity: Activity dictionary
            time: Time string (HH:MM)
            day_of_week: Day of week string
        
        Returns:
            dict: Destination type information
        """
        # Prepare the prompt with persona and activity information
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
                messages=[
                    {"role": "system", "content": "You are a destination selector for human activities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            # print(f"Raw destination response (first 100 chars): {generated_text[:100]}...")
            # Clean the text
            cleaned_text = generated_text.strip()
            
            # Enhanced JSON parsing logic
            destination_type = {}
            
            # Step 1: Try direct JSON parsing
            try:
                destination_type = json.loads(cleaned_text)
                print("Successfully parsed destination response as JSON")
            except json.JSONDecodeError as e:
                # print(f"Direct destination JSON parsing failed: {e}")
                pass
                
                # Step 2: Try to extract a JSON object
                try:
                    # Look for object pattern with non-greedy matching
                    object_pattern = r'\{\s*".*?"\s*:.*?\}'
                    object_match = re.search(object_pattern, cleaned_text, re.DOTALL)
                    
                    if object_match:
                        object_content = object_match.group(0)
                        # print(f"Found potential JSON object: {object_content[:50]}...")
                        
                        try:
                            # Try to parse the extracted object
                            destination_type = json.loads(object_content)
                            print("Successfully parsed extracted JSON object")
                        except json.JSONDecodeError as inner_e:
                            print(f"Parsing extracted object failed: {inner_e}")
                            
                            # Try to fix common JSON formatting issues
                            fixed_object = self._fix_json_object(object_content)
                            try:
                                destination_type = json.loads(fixed_object)
                                print("Successfully parsed fixed JSON object")
                            except json.JSONDecodeError:
                                print("Failed to fix and parse JSON object")
                    else:
                        print("No JSON object pattern found")
                        
                        # Try an alternate approach - extract key-value pairs
                        destination_type = self._extract_destination_from_text(cleaned_text)
                except Exception as ex:
                    print(f"Error extracting JSON object: {ex}")
            
            # If all attempts failed, use default values
            if not destination_type:
                print("No valid destination type extracted, using defaults")
                destination_type = self._generate_default_destination_type(activity)
            
            # Ensure all required fields exist
            required_fields = ['place_type', 'search_query', 'distance_preference', 'price_level']
            for field in required_fields:
                if field not in destination_type:
                    # Use default values for missing fields
                    if field == 'place_type':
                        destination_type[field] = activity['activity_type']
                    elif field == 'search_query':
                        destination_type[field] = activity['activity_type']
                    elif field == 'distance_preference':
                        destination_type[field] = 5.0
                    elif field == 'price_level':
                        destination_type[field] = 2
            
            return destination_type
                
        except Exception as e:
            print(f"Error determining destination type: {e}")
            # Return a default destination type
            return self._generate_default_destination_type(activity)
    
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
    
    def _retrieve_location_google_maps(self, current_location, destination_type):
        """
        Retrieve a location using Google Maps API.
        
        Args:
            current_location: (latitude, longitude)
            destination_type: Destination type information
        
        Returns:
            tuple: ((latitude, longitude), location_details) of the selected destination
        """
        if not self.google_maps_api_key:
            print("Google Maps API key not provided. Using random location.")
            random_location = generate_random_location_near(current_location)
            return random_location, {"name": "Unknown location", "address": "Unknown address"}
        
        try:
            # Prepare the request
            search_query = destination_type.get('search_query', 'restaurant')
            
            # Ensure search_query is a string and not too long
            if not isinstance(search_query, str):
                search_query = str(search_query)
            search_query = search_query[:100]  # Limit length
            
            # Safety check for distance preference
            try:
                distance_preference = float(destination_type.get('distance_preference', 5))
                if distance_preference <= 0 or distance_preference > 50:
                    distance_preference = 5.0
            except (ValueError, TypeError):
                distance_preference = 5.0
                
            # Convert km to meters with safe bounds
            radius = min(50000, max(1000, int(distance_preference * 1000)))
            
            # Call Google Places API
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': f"{current_location[0]},{current_location[1]}",
                'radius': radius,
                'key': self.google_maps_api_key
            }
            
            # Add optional parameters if available
            place_type = destination_type.get('place_type', '').lower().replace(' ', '_')
            if place_type and len(place_type) < 50:  # Safety check
                params['type'] = place_type
                
            if search_query:
                params['keyword'] = search_query

            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if results:
                        # 简化评分逻辑，只考虑距离和评分
                        scored_results = []
                        for result in results[:10]:  # 只处理前10个结果
                            try:
                                location = result.get('geometry', {}).get('location', {})
                                lat, lng = location.get('lat'), location.get('lng')
                                
                                if not lat or not lng:
                                    continue
                                    
                                try:
                                    lat = float(lat)
                                    lng = float(lng)
                                except (ValueError, TypeError):
                                    continue
                                    
                                # 计算距离
                                dest_coords = (lat, lng)
                                distance = calculate_distance(current_location, dest_coords)
                                
                                # 获取评分和价格
                                rating = float(result.get('rating', 0))
                                try:
                                    price_level = int(result.get('price_level', 2))
                                    if price_level < 1 or price_level > 4:
                                        price_level = 2
                                except (ValueError, TypeError):
                                    price_level = 2
                                
                                distance_score = min(1.0, distance / 5.0)  # 标准化距离(最大5km)
                                rating_score = (5 - min(5, rating)) / 5  # 转换评分使得较低分数更好
                                
                                # 组合评分
                                score = (distance_score * 0.7) + (rating_score * 0.3)
                                score = score + (price_level / 100.0)  # Very small influence 

                                # 存储基本信息
                                location_details = {
                                    "name": str(result.get('name', 'Unknown location'))[:100],
                                    "address": str(result.get('vicinity', 'Unknown address'))[:200],
                                    "rating": rating,
                                    "distance": distance,
                                    "price_level": price_level
                                }
                                
                                scored_results.append((score, dest_coords, location_details))
                            except Exception as item_e:
                                print(f"Error processing place result: {item_e}")
                                continue

                        if scored_results:
                            # 按评分排序
                            scored_results.sort(key=lambda x: x[0])
                            
                            # 直接选择排序后的第一个结果
                            selected = scored_results[0]
                            
                            return selected[1], selected[2]
                        
                except Exception as parse_e:
                    print(f"Error parsing Google Places API response: {parse_e}. Using random location.")
                
                # If we get here, we couldn't find or process any valid results
                print(f"No suitable places found for {search_query}. Using random location.")
                random_location = generate_random_location_near(current_location)
                return random_location, {"name": "Unknown location", "address": "Unknown address"}
            else:
                print(f"Error calling Google Places API: {response.status_code} - {response.text}. Using random location.")
                random_location = generate_random_location_near(current_location)
                return random_location, {"name": "Unknown location", "address": "Unknown address"}
                
        except Exception as e:
            print(f"Error retrieving location from Google Maps: {e}. Using random location.")
            random_location = generate_random_location_near(current_location)
            return random_location, {"name": "Unknown location", "address": "Unknown address"}
    
    def _retrieve_location_osm(self, current_location, destination_type):
        """
        Retrieve a location using OpenStreetMap.
        
        Args:
            current_location: (latitude, longitude)
            destination_type: Destination type information
        
        Returns:
            tuple: ((latitude, longitude), location_details) of the selected destination
        """
        try:
            # Get the search radius (in km)
            search_radius = min(10, destination_type.get('distance_preference', 3))
            
            # Get the bounding box for the search
            bbox = self._calculate_bounding_box(current_location, search_radius)
            
            # Map the place type to OSM tags
            tags = self._map_to_osm_tags(destination_type.get('place_type', ''))
            
            # If we couldn't map to specific tags, try to use the activity type
            if not tags:
                activity_type = destination_type.get('activity_type', '')
                tags = self._activity_type_to_amenity(activity_type)
            
            # If we still don't have tags, use a generic tag
            if not tags:
                tags = {'amenity': 'restaurant'}
            
            # Search for places with the specified tags
            try:
                # Safely handle import errors - osmnx may not be available
                try:
                    import osmnx as ox
                    import pandas as pd
                    
                    pois = ox.geometries_from_bbox(
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        tags
                    )
                    
                    if not pois.empty:
                        # Convert to standard format
                        poi_list = []
                        for idx, poi in pois.iterrows():
                            try:
                                # Get the centroid of the geometry
                                centroid = poi.geometry.centroid
                                coords = (centroid.y, centroid.x)
                                
                                # Calculate distance
                                distance = calculate_distance(current_location, coords)
                                
                                # Get name (if available)
                                name = poi.get('name', 'Unknown location (OSM)')
                                if pd.isna(name):
                                    name = f"Place at {coords[0]:.6f}, {coords[1]:.6f}"
                                
                                # Store the POI with its distance
                                poi_list.append((distance, coords, {'name': name, 'address': 'OSM location', 'types': [tags.get('amenity', 'place')]}))
                            except Exception as detail_e:
                                print(f"Error processing POI: {detail_e}")
                                continue
                        
                        if poi_list:
                            # Sort by distance
                            poi_list.sort(key=lambda x: x[0])
                            
                            # Select from top 3 if available
                            if len(poi_list) >= 3:
                                selected = random.choice(poi_list[:3])
                            else:
                                selected = poi_list[0]
                            
                            return selected[1], selected[2]
                    
                    # Fall back to random location if no POIs found
                    print(f"No OSM data found. Using random location.")
                    random_location = generate_random_location_near(current_location)
                    return random_location, {'name': f"Random {tags.get('amenity', 'place')} (OSM)", 'address': 'OSM location', 'types': [tags.get('amenity', 'place')]}
                    
                except ImportError:
                    print("OSM libraries not available. Using random location.")
                    random_location = generate_random_location_near(current_location)
                    return random_location, {'name': 'Random location (OSM not available)', 'address': 'OSM location', 'types': ['place']}
                    
            except Exception as e:
                print(f"OSM search error: {e}. Using random location.")
                random_location = generate_random_location_near(current_location)
                return random_location, {'name': 'Random location (OSM error)', 'address': 'OSM location', 'types': ['place']}
            
        except Exception as e:
            print(f"Error retrieving location from OSM: {e}. Using random location.")
            random_location = generate_random_location_near(current_location)
            return random_location, {'name': 'Random location (OSM general error)', 'address': 'OSM location', 'types': ['place']}
    
    def _map_to_osm_tags(self, place_type):
        """
        Map a place type to OpenStreetMap tags.
        
        Args:
            place_type: String describing the place type
        
        Returns:
            dict: Dictionary of OSM tags
        """
        place_type = place_type.lower()
        
        # Common mappings
        if 'restaurant' in place_type or 'dining' in place_type:
            return {'amenity': 'restaurant'}
        elif 'cafe' in place_type or 'coffee' in place_type:
            return {'amenity': 'cafe'}
        elif 'bar' in place_type or 'pub' in place_type:
            return {'amenity': 'bar'}
        elif 'shop' in place_type or 'store' in place_type or 'mall' in place_type:
            return {'shop': 'mall'}
        elif 'supermarket' in place_type or 'grocery' in place_type:
            return {'shop': 'supermarket'}
        elif 'park' in place_type:
            return {'leisure': 'park'}
        elif 'gym' in place_type or 'fitness' in place_type:
            return {'leisure': 'fitness_centre'}
        elif 'hospital' in place_type or 'clinic' in place_type:
            return {'amenity': 'hospital'}
        elif 'school' in place_type or 'education' in place_type:
            return {'amenity': 'school'}
        elif 'university' in place_type or 'college' in place_type:
            return {'amenity': 'university'}
        elif 'library' in place_type:
            return {'amenity': 'library'}
        elif 'cinema' in place_type or 'movie' in place_type:
            return {'amenity': 'cinema'}
        elif 'theatre' in place_type or 'theater' in place_type:
            return {'amenity': 'theatre'}
        elif 'museum' in place_type:
            return {'tourism': 'museum'}
        elif 'hotel' in place_type:
            return {'tourism': 'hotel'}
        
        # Default to a general amenity tag
        return {'amenity': 'restaurant'}
    
    def _activity_type_to_amenity(self, activity_type):
        """
        Map an activity type to an OSM amenity value.
        
        Args:
            activity_type: Activity type string
        
        Returns:
            str: OSM amenity value
        """
        activity_type = activity_type.lower()
        
        if 'shop' in activity_type or 'shopping' in activity_type:
            return 'shopping_mall'
        elif 'din' in activity_type or 'eat' in activity_type or 'food' in activity_type:
            return 'restaurant'
        elif 'recreation' in activity_type or 'leisure' in activity_type:
            return 'park'
        elif 'health' in activity_type:
            return 'hospital'
        elif 'social' in activity_type:
            return 'cafe'
        elif 'education' in activity_type:
            return 'school'
        elif 'errand' in activity_type:
            return 'marketplace'
        
        # Default
        return 'restaurant'
    
    def _calculate_bounding_box(self, center, distance_km):
        """
        Calculate the bounding box for a given center point and distance.
        
        Args:
            center: (latitude, longitude)
            distance_km: Distance in kilometers
        
        Returns:
            tuple: (north, south, east, west) coordinates
        """
        # Earth's radius in kilometers
        R = 6371.0
        
        lat = np.radians(center[0])
        lon = np.radians(center[1])
        
        # Angular distance
        angular_distance = distance_km / R
        
        # Calculate the bounding box
        north = np.degrees(lat + angular_distance)
        south = np.degrees(lat - angular_distance)
        
        # Longitude is affected by latitude
        delta_lon = np.arcsin(np.sin(angular_distance) / np.cos(lat))
        east = np.degrees(lon + delta_lon)
        west = np.degrees(lon - delta_lon)
        
        return north, south, east, west 