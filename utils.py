"""
Utility functions for the LLM-based mobility simulation.
"""

import json
import os
import datetime
import random
import numpy as np
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt
from config import RESULTS_DIR, TRANSPORT_MODES, ENABLE_CACHING, CACHE_EXPIRY
import requests
import math
import hashlib
import time
import functools

# Add caching system
class Cache:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.enabled = ENABLE_CACHING
        self.expiry = CACHE_EXPIRY
        
        # Ensure cache directory exists
        os.makedirs(os.path.join(RESULTS_DIR, 'cache'), exist_ok=True)
        self.cache_file = os.path.join(RESULTS_DIR, 'cache', 'function_cache.json')
        self.load_cache()
    
    def get(self, key):
        """Get value from cache"""
        if not self.enabled:
            return None
            
        if key in self.cache:
            value, timestamp = self.cache[key]
            # Check if expired
            if time.time() - timestamp < self.expiry:
                self.cache_hits += 1
                return value
        self.cache_misses += 1
        return None
    
    def set(self, key, value):
        """Set cache value"""
        if not self.enabled:
            return
            
        self.cache[key] = (value, time.time())
        # Save every 100 cache operations
        if (self.cache_hits + self.cache_misses) % 100 == 0:
            self.save_cache()
    
    def load_cache(self):
        """Load cache from file"""
        if not self.enabled:
            return
            
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                    self.cache = {k: (v[0], v[1]) for k, v in loaded_cache.items()}
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to file"""
        if not self.enabled:
            return
            
        try:
            # Create a JSON serializable version of the cache
            json_safe_cache = {}
            for k, v in self.cache.items():
                # Convert keys to strings if they're not already JSON serializable
                if isinstance(k, (str, int, float, bool)) or k is None:
                    json_key = k
                else:
                    json_key = str(k)
                json_safe_cache[json_key] = v
                
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
    
    def stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_ratio": hit_ratio,
            "size": len(self.cache)
        }

# Create global cache instance
cache = Cache()

def cached(func):
    """Cache decorator, used to cache function call results"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not cache.enabled:
            return func(*args, **kwargs)
            
        # Create cache key
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        
        # Try to get from cache
        cached_result = cache.get(key)
        if cached_result is not None:
            return cached_result
        
        # Calculate result and cache
        result = func(*args, **kwargs)
        cache.set(key, result)
        return result
    
    return wrapper

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def calculate_distance(point1, point2):
    """Calculate distance between two points in kilometers."""
    return geodesic(point1, point2).kilometers

def format_time(hour, minute=0):
    """Format time as HH:MM."""
    return f"{hour:02d}:{minute:02d}"

def parse_time(time_str):
    """Parse time string (HH:MM) to hours and minutes."""
    hours, minutes = map(int, time_str.split(':'))
    return hours, minutes

def time_difference_minutes(start_time, end_time):
    """Calculate the difference between two times in minutes."""
    start_h, start_m = parse_time(start_time)
    end_h, end_m = parse_time(end_time)
    
    # Handle crossing midnight
    if end_h < start_h:
        end_h += 24
        
    return (end_h - start_h) * 60 + (end_m - start_m)

def generate_date_range(start_date, num_days):
    """Generate a list of dates starting from start_date."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

def get_day_of_week(date_str):
    """Get the day of week from a date string."""
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%A")

def visualize_trajectory(trajectory_data, output_file):
    """
    Visualize a day's trajectory on a map.
    
    Args:
        trajectory_data: List of trajectory points
        output_file: Output HTML file path
    """
    if not trajectory_data:
        return
    
    # Calculate center point
    lats = [point['location'][0] for point in trajectory_data]
    lons = [point['location'][1] for point in trajectory_data]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Create location dictionary to track repeated locations
    location_counts = {}
    
    # Add time line connection points
    coordinates = []
    
    # Define different activity type icons
    icons = {
        'travel': 'car',
        'work': 'briefcase',
        'leisure': 'leaf',
        'shopping': 'shopping-cart',
        'dining': 'cutlery',
        'home': 'home',
        'sleep': 'bed'
    }
    
    # Add markers for each point
    for point in trajectory_data:
        lat, lon = point['location']
        loc_key = f"{lat:.5f},{lon:.5f}"  # Use 5 decimal precision as location key
        
        # Check if there are repeated locations, if so, offset
        if loc_key in location_counts:
            location_counts[loc_key] += 1
            # Calculate offset based on repeated times (approximately 20 meters each time)
            offset = location_counts[loc_key] * 0.0002
            lat += offset
            lon += offset
        else:
            location_counts[loc_key] = 0
            
        # Record coordinates for drawing time line
        coordinates.append([lat, lon])
        
        # Set marker color and icon based on activity type
        if point['activity_type'] == 'travel':
            color = 'gray'
        elif point['activity_type'] == 'work':
            color = 'red'
        elif point['activity_type'] == 'leisure':
            color = 'blue'
        elif point['activity_type'] == 'shopping':
            color = 'green'
        elif point['activity_type'] == 'dining':
            color = 'orange'
        elif point['activity_type'] == 'home':
            color = 'purple'
        elif point['activity_type'] == 'sleep':
            color = 'darkblue'
        else:
            color = 'purple'
            
        # Get activity type corresponding icon
        icon_name = icons.get(point['activity_type'].lower(), 'info-sign')
            
        # Create popup content
        popup_content = f"Activity: {point['activity_type']}<br>"
        popup_content += f"Time: {point['timestamp']}<br>"
        popup_content += f"Description: {point['description']}"
        
        # Add marker with custom icon
        folium.Marker(
            location=[lat, lon],
            popup=popup_content,
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
        ).add_to(m)
        
        # If there are route coordinates, add route
        if 'route_coordinates' in point and point['route_coordinates']:
            route_coords = point['route_coordinates']
            # Select route style based on transportation mode
            if 'transport_mode' in point:
                if point['transport_mode'] == 'walking':
                    color = 'green'
                    weight = 2
                    dash_array = '5,10'
                elif point['transport_mode'] == 'cycling':
                    color = 'blue'
                    weight = 2
                    dash_array = '5,5'
                elif point['transport_mode'] == 'driving':
                    color = 'red'
                    weight = 3
                    dash_array = None
                else:  # public transit or other
                    color = 'purple'
                    weight = 3
                    dash_array = '10,10'
            else:
                color = 'gray'
                weight = 2
                dash_array = None
                
            folium.PolyLine(
                route_coords,
                weight=weight,
                color=color,
                opacity=0.8,
                dash_array=dash_array
            ).add_to(m)
    
    # Add time line connection all points
    folium.PolyLine(
        coordinates,
        weight=1,
        color='black',
        opacity=0.5,
        dash_array='3,6'
    ).add_to(m)
    
    # Save map
    m.save(output_file)

def plot_activity_distribution(memory_data, output_file=None):
    """
    Plot the distribution of activities based on duration.
    
    Args:
        memory_data: Memory data containing activities
        output_file: File path to save the plot
    """
    activity_durations = {}
    
    for day_data in memory_data.get('days', []):
        for activity in day_data.get('activities', []):
            activity_type = activity.get('activity_type', 'unknown')
            start_time = activity.get('start_time', '00:00')
            end_time = activity.get('end_time', '23:59')
            
            # Calculate activity duration (minutes)
            start_minutes = time_to_minutes(start_time)
            end_minutes = time_to_minutes(end_time)
            
            # Special handling for activities that cross midnight
            if end_minutes < start_minutes:
                # For activities crossing midnight, calculate time until midnight plus time after midnight
                duration = (24 * 60 - start_minutes) + end_minutes
                
                # For sleep activities, ensure no duplicate counting
                if activity_type == 'sleep' and end_time == '08:00' and start_time == '22:30':
                    # Typical night sleep pattern, calculate normally
                    duration = (24 * 60 - start_minutes) + end_minutes
                elif activity_type == 'sleep' and end_time == '08:00' and start_time == '22:00':
                    # Another sleep pattern
                    duration = (24 * 60 - start_minutes) + end_minutes
                else:
                    # Other activities crossing midnight
                    duration = (24 * 60 - start_minutes) + end_minutes
            else:
                # Activities within the same day
                duration = end_minutes - start_minutes
            
            # Add to the appropriate activity type
            activity_durations[activity_type] = activity_durations.get(activity_type, 0) + duration
    
    # Sort by duration
    sorted_activities = sorted(activity_durations.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_activities]
    durations = [item[1] for item in sorted_activities]
    
    # Convert minutes to hours for display
    durations_hours = [duration/60 for duration in durations]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, durations_hours, color='skyblue')
    plt.xlabel('Activity Type')
    plt.ylabel('Total Duration (hours)')
    plt.title('Activity Distribution by Time Spent')
    plt.xticks(rotation=45, ha='right')
    
    # Add specific values on top of the bars
    for bar in bars:
        height = bar.get_height()
        hours = int(height)
        minutes = int((height - hours) * 60)
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{hours}h {minutes}m',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def generate_random_location_near(center, max_distance_km=5.0):
    """
    Generate a random location within max_distance_km of center.
    
    Args:
        center: [lat, lon]
        max_distance_km: Maximum distance in kilometers
    
    Returns:
        [lat, lon]
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Random distance within max_distance_km
    distance = random.uniform(0, max_distance_km)
    
    # Random angle
    angle = random.uniform(0, 2 * np.pi)
    
    # Calculate new coordinates
    lat1 = np.radians(center[0])
    lon1 = np.radians(center[1])
    
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance/R) +
                     np.cos(lat1) * np.sin(distance/R) * np.cos(angle))
    lon2 = lon1 + np.arctan2(np.sin(angle) * np.sin(distance/R) * np.cos(lat1),
                            np.cos(distance/R) - np.sin(lat1) * np.sin(lat2))
    
    return [np.degrees(lat2), np.degrees(lon2)]

def normalize_transport_mode(mode):
    """
    Standardize transportation mode, converting any transportation string to the system supported standard format.
    
    Args:
        mode: Original transportation string
        
    Returns:
        str: Standardized transportation mode
    """
    # Default transportation mode
    DEFAULT_MODE = 'walking'
    
    # Check for null or special cases
    if not mode or not isinstance(mode, str):
        return DEFAULT_MODE
        
    mode = mode.lower().strip()
    
    # Handle special values
    invalid_values = ['n/a', 'none', 'null', 'nan', '']
    if mode in invalid_values:
        return DEFAULT_MODE
        
    # If already standardized transportation mode, return directly
    if mode in TRANSPORT_MODES:
        return mode
        
    # Simple mapping table, map common non-standard transportation modes to standard modes
    mode_mapping = {
        # Walking related
        'foot': 'walking',
        'pedestrian': 'walking',
        'stroll': 'walking',
        'walk': 'walking',
        'on foot': 'walking',
        'by foot': 'walking',
        'strolling': 'walking',
        'hiking': 'walking',
        
        # Driving related
        'car': 'driving',
        'automobile': 'driving',
        'vehicle': 'driving',
        'drive': 'driving',
        'by car': 'driving',
        'personal car': 'driving',
        'private car': 'driving',
        'own car': 'driving',
        
        # Public transit related
        'bus': 'public_transit',
        'train': 'public_transit',
        'subway': 'public_transit',
        'metro': 'public_transit',
        'tram': 'public_transit',
        'public transport': 'public_transit',
        'public transportation': 'public_transit',
        'transit': 'public_transit',
        'rail': 'public_transit',
        'light rail': 'public_transit',
        'commuter rail': 'public_transit',
        'ferry': 'public_transit',
        
        # Cycling related
        'bike': 'cycling',
        'bicycle': 'cycling',
        'cycle': 'cycling',
        'biking': 'cycling',
        'by bike': 'cycling',
        'by bicycle': 'cycling',
        
        # Rideshare related
        'cab': 'rideshare',
        'taxi': 'rideshare',
        'uber': 'rideshare',
        'lyft': 'rideshare',
        'didi': 'rideshare',
        'ride-sharing': 'rideshare',
        'ridesharing': 'rideshare',
        'ride sharing': 'rideshare',
        'ride hailing': 'rideshare',
        'ride-hailing': 'rideshare',
        'shared ride': 'rideshare'
    }
    
    # Quickly match using mapping table
    if mode in mode_mapping:
        return mode_mapping[mode]
    
    # If direct match failed, try fuzzy matching by checking if mode contains key words
    mode_keywords = {
        'walking': ['walk', 'foot', 'pedestrian', 'stroll', 'hike'],
        'driving': ['drive', 'car', 'auto', 'vehicle'],
        'public_transit': ['bus', 'train', 'subway', 'metro', 'transit', 'public', 'tram', 'rail'],
        'cycling': ['bike', 'cycle', 'bicycle'],
        'rideshare': ['taxi', 'cab', 'uber', 'lyft', 'ride', 'sharing', 'hail']
    }
    
    for std_mode, keywords in mode_keywords.items():
        if any(keyword in mode for keyword in keywords):
            return std_mode
    
    # If no match, return default transportation mode
    return DEFAULT_MODE

def get_route_coordinates(start_location, end_location, transport_mode='driving'):
    """
    Get coordinates for a route between two locations using OSRM API.
    
    Args:
        start_location: (latitude, longitude)
        end_location: (latitude, longitude)
        transport_mode: Mode of transportation
    
    Returns:
        list: List of coordinates along the route [(lat1, lon1), (lat2, lon2), ...]
    """
    # OSRM service base URL (using demo server)
    base_url = "http://router.project-osrm.org"
    
    # Build API request URL
    url = f"{base_url}/route/v1/{transport_mode}/{start_location[1]},{start_location[0]};{end_location[1]},{end_location[0]}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "routes" in data and len(data["routes"]) > 0:
            # Extract route coordinates
            coordinates = data["routes"][0]["geometry"]["coordinates"]
            # OSRM returns coordinates as [lon, lat], we need to convert to [lat, lon]
            return [(coord[1], coord[0]) for coord in coordinates]
        else:
            print(f"Could not find route from {start_location} to {end_location}")
            return []
            
    except Exception as e:
        print(f"Error getting route: {str(e)}")
        return []

@cached
def estimate_travel_time(start_location, end_location, transport_mode, persona=None):
    """
    Estimate travel time between two locations based on distance and transport mode.
    Added caching to avoid redundant calculations.
    
    Args:
        start_location: (latitude, longitude)
        end_location: (latitude, longitude)
        transport_mode: Mode of transportation
        persona: Person object containing transportation preferences
    
    Returns:
        tuple: (travel_time_minutes, selected_transport_mode)
    """
    distance_km = calculate_distance(start_location, end_location)
    
    # If no transport mode specified, select appropriate mode based on distance and person characteristics
    if not transport_mode:
        # Default transport mode selection logic
        if persona:
            # Consider person characteristics and preferences
            if distance_km < 0.5:
                # Very short distance, typically walking
                transport_mode = 'walking'
            elif distance_km < 3:
                # Short distance, decide based on bike availability and preferences
                if persona.has_bike and (persona.preferred_transport == 'cycling' or random.random() < 0.7):
                    transport_mode = 'cycling'
                else:
                    transport_mode = 'walking'
            elif distance_km < 10:
                # Medium distance
                if persona.has_bike and persona.preferred_transport == 'cycling':
                    transport_mode = 'cycling'
                elif persona.has_car and persona.preferred_transport == 'driving':
                    transport_mode = 'driving'
                elif persona.preferred_transport == 'public_transit':
                    transport_mode = 'public_transit'
                else:
                    # Choose based on availability
                    if persona.has_car:
                        transport_mode = 'driving'
                    elif persona.has_bike:
                        transport_mode = 'cycling'
                    else:
                        transport_mode = 'public_transit'
            else:
                # Long distance
                if persona.has_car:
                    transport_mode = 'driving'
                elif persona.preferred_transport == 'public_transit':
                    transport_mode = 'public_transit'
                else:
                    transport_mode = 'rideshare'
        else:
            # If no persona information, use simple distance-based logic
            if distance_km < 1:
                transport_mode = 'walking'
            elif distance_km < 5:
                transport_mode = 'cycling'
            elif distance_km < 15:
                transport_mode = 'public_transit'
            else:
                transport_mode = 'driving'
    
    # Normalize transport mode
    transport_mode = normalize_transport_mode(transport_mode)
    
    # Average speeds in km/h
    speeds = {
        'walking': 5,
        'cycling': 15,
        'driving': 30,
        'public_transit': 20,
        'rideshare': 25
    }
    
    # Get speed for the transport mode, default to walking if not found
    speed_kmh = speeds.get(transport_mode, speeds['walking'])
    
    # Calculate time in minutes
    time_hours = distance_km / speed_kmh
    time_minutes = int(time_hours * 60)
    
    # Add some randomness to account for traffic, waiting times, etc.
    time_minutes = int(time_minutes * random.uniform(0.8, 1.2))
    
    # Minimum travel time
    return max(5, time_minutes), transport_mode

def time_to_minutes(time_str):
    """
    Convert time string (HH:MM) to minutes
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        int: Total minutes
    """
    hours, minutes = parse_time(time_str)
    return hours * 60 + minutes 