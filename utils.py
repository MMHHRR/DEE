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
from config import RESULTS_DIR, TRANSPORT_MODES, ENABLE_CACHING, CACHE_EXPIRY, USE_LOCAL_POI, POI_CSV_PATH
import requests
import math
import hashlib
import time
import functools
import gzip
import shutil
import pandas as pd
import threading

# Caching system implementation
class Cache:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.enabled = ENABLE_CACHING
        self.expiry = CACHE_EXPIRY
        self._lock = threading.Lock()  # 添加线程锁
        
        # 确保缓存目录存在
        try:
            cache_dir = os.path.join(RESULTS_DIR, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, 'function_cache.json')
            self.load_cache()
        except Exception as e:
            print(f"Warning: Could not initialize cache directory: {e}")
            self.enabled = False
    
    def get(self, key):
        """Get value from cache"""
        if not self.enabled:
            return None
            
        with self._lock:  # 使用线程锁
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
            
        with self._lock:  # 使用线程锁
            try:
                self.cache[key] = (value, time.time())
                # Save every 100 cache operations
                if (self.cache_hits + self.cache_misses) % 100 == 0:
                    self.save_cache()
            except Exception as e:
                print(f"Warning: Could not set cache value: {e}")
    
    def load_cache(self):
        """Load cache from file"""
        if not self.enabled:
            return
            
        if os.path.exists(self.cache_file):
            try:
                with self._lock:  # 使用线程锁
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        loaded_cache = json.load(f)
                        self.cache = {k: (v[0], v[1]) for k, v in loaded_cache.items()}
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to file"""
        if not self.enabled:
            return
            
        try:
            with self._lock:  # 使用线程锁
                # 创建临时文件
                temp_file = f"{self.cache_file}.tmp"
                
                # 创建JSON可序列化的缓存版本
                json_safe_cache = {}
                for k, v in self.cache.items():
                    try:
                        # 转换键为字符串
                        if isinstance(k, (str, int, float, bool)) or k is None:
                            json_key = k
                        else:
                            json_key = str(k)
                        json_safe_cache[json_key] = v
                    except Exception as e:
                        print(f"Warning: Could not serialize cache key {k}: {e}")
                        continue
                
                # 先写入临时文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(json_safe_cache, f)
                
                # 如果临时文件写入成功，替换原文件
                if os.path.exists(temp_file):
                    if os.path.exists(self.cache_file):
                        os.replace(temp_file, self.cache_file)
                    else:
                        os.rename(temp_file, self.cache_file)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def clear(self):
        """Clear cache"""
        with self._lock:  # 使用线程锁
            self.cache = {}
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except Exception as e:
                    print(f"Warning: Could not remove cache file: {e}")
    
    def stats(self):
        """Return cache statistics"""
        with self._lock:  # 使用线程锁
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
    """
    Cache decorator to cache function call results
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function that uses caching
    """
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
    """
    Load data from a JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 自定义JSON编码器，处理NumPy数据类型
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，可处理NumPy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_json(data, file_path):
    """
    Save data to a JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

def calculate_distance(point1, point2):
    """
    Calculate distance between two points in kilometers
    
    Args:
        point1: First point (lat, lon)
        point2: Second point (lat, lon)
        
    Returns:
        float: Distance in kilometers
    """
    return geodesic(point1, point2).kilometers

def format_time(hour, minute=0):
    """
    Format time as HH:MM
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        
    Returns:
        str: Formatted time string HH:MM
    """
    return f"{hour:02d}:{minute:02d}"

def parse_time(time_str):
    """
    Parse time string (HH:MM) to hours and minutes
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        tuple: (hours, minutes)
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours, minutes

def time_difference_minutes(start_time, end_time):
    """
    Calculate the difference between two times in minutes
    
    Args:
        start_time: Start time (HH:MM)
        end_time: End time (HH:MM)
        
    Returns:
        int: Minutes difference
    """
    start_h, start_m = parse_time(start_time)
    end_h, end_m = parse_time(end_time)
    
    # Handle crossing midnight
    if end_h < start_h:
        end_h += 24
        
    return (end_h - start_h) * 60 + (end_m - start_m)

def generate_date_range(start_date, num_days):
    """
    Generate a list of dates starting from start_date
    
    Args:
        start_date: Starting date string (YYYY-MM-DD)
        num_days: Number of days to generate
        
    Returns:
        list: List of date strings
    """
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

def get_day_of_week(date_str):
    """
    Get the day of week from a date string
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        
    Returns:
        str: Day of week (Monday, Tuesday, etc.)
    """
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%A")

def visualize_trajectory(trajectory_data, output_file):
    """
    Visualize a day's trajectory on a map
    
    Args:
        trajectory_data: List of trajectory points
        output_file: Output HTML file path
    """
    if not trajectory_data:
        return
    
    # Calculate center point
    lats = [point['location'][0] for point in trajectory_data if 'location' in point and point['location']]
    lons = [point['location'][1] for point in trajectory_data if 'location' in point and point['location']]
    
    if not lats or not lons:
        print("No valid location data for visualization")
        return
        
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add markers for each point
    for i, point in enumerate(trajectory_data):
        if 'location' not in point or not point['location']:
            continue
            
        lat, lon = point['location']
        time = point.get('time', '')
        activity = point.get('activity_type', 'Unknown')
        description = point.get('description', '')
        transport = point.get('transport_mode', '')
        
        # Create popup content
        popup_content = f"""
        <b>Time:</b> {time}<br>
        <b>Activity:</b> {activity}<br>
        <b>Description:</b> {description}<br>
        """
        if transport:
            popup_content += f"<b>Transport:</b> {transport}<br>"
        
        # Use different colors based on activity type
        icon_color = get_activity_color(activity)
        
        # Add marker
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(m)
        
        # Add lines connecting points in sequence
        if i > 0 and 'location' in trajectory_data[i-1] and trajectory_data[i-1]['location']:
            prev_lat, prev_lon = trajectory_data[i-1]['location']
            
            # Use transport mode color if available
            line_color = get_transport_color(transport) if transport else 'gray'
            
            folium.PolyLine(
                [[prev_lat, prev_lon], [lat, lon]],
                color=line_color,
                weight=3,
                opacity=0.7
            ).add_to(m)
    
    # Save the map
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        print(f"Map saved to {output_file}")
    except Exception as e:
        print(f"Error saving map: {e}")

def get_activity_color(activity_type):
    """
    Get color for activity type
    
    Args:
        activity_type: Type of activity
        
    Returns:
        str: Color name
    """
    colors = {
        'home': 'green',
        'work': 'blue',
        'shopping': 'purple',
        'dining': 'orange',
        'recreation': 'cadetblue',
        'education': 'darkblue',
        'healthcare': 'red',
        'social': 'pink',
        'leisure': 'darkgreen',
        'errands': 'gray',
        'travel': 'black'
    }
    return colors.get(activity_type.lower(), 'darkpurple')

def get_transport_color(transport_mode):
    """
    Get color for transport mode
    
    Args:
        transport_mode: Mode of transport
        
    Returns:
        str: Color code
    """
    colors = {
        'walking': '#66c2a5',
        'cycling': '#fc8d62',
        'driving': '#8da0cb',
        'public_transit': '#e78ac3',
        'rideshare': '#a6d854'
    }
    return colors.get(transport_mode.lower(), '#999999')

def normalize_transport_mode(mode):
    """
    Normalize transportation mode string to standard values
    
    Args:
        mode: Transport mode string
        
    Returns:
        str: Normalized transport mode
    """
    if not mode:
        return 'driving'
        
    mode = str(mode).lower().strip()
    
    # Direct matches
    if mode in TRANSPORT_MODES:
        return mode
        
    # Walking variations
    if mode in ['walk', 'walking', 'on foot', 'foot', 'pedestrian']:
        return 'walking'
        
    # Cycling variations
    if mode in ['cycle', 'cycling', 'bicycle', 'bike', 'biking']:
        return 'cycling'
        
    # Driving variations
    if mode in ['drive', 'driving', 'car', 'auto', 'automobile']:
        return 'driving'
        
    # Public transit variations
    if mode in ['transit', 'public transit', 'bus', 'subway', 'train', 'metro', 'public_transport', 'public transport']:
        return 'public_transit'
        
    # Rideshare variations
    if mode in ['taxi', 'uber', 'lyft', 'rideshare', 'ride_share', 'ride share', 'ride-share']:
        return 'rideshare'
        
    # Default to driving if not recognized
    return 'driving'

@cached
def format_time_after_minutes(start_time, minutes):
    """
    Calculate a new time after adding minutes to a start time
    
    Args:
        start_time: Time string (HH:MM)
        minutes: Minutes to add
    
    Returns:
        str: New time string (HH:MM)
    """
    start_h, start_m = parse_time(start_time)
    
    # Add minutes
    total_minutes = start_h * 60 + start_m + minutes
    
    # Convert back to hours and minutes
    new_h = (total_minutes // 60) % 24
    new_m = total_minutes % 60
    
    return f"{new_h:02d}:{new_m:02d}"

@cached
def estimate_travel_time(origin, destination, transport_mode=None, persona=None):
    """
    Estimate travel time between origin and destination based on distance and transport mode
    
    Args:
        origin: Origin coordinates (latitude, longitude)
        destination: Destination coordinates (latitude, longitude)
        transport_mode: Transportation mode (walking, driving, public_transit, cycling)
        persona: Optional persona object with additional context
        
    Returns:
        tuple: (travel_time_minutes, actual_transport_mode)
    """
    # Calculate distance
    distance_km = calculate_distance(origin, destination)
    
    # Default speeds (km/h) for different transportation modes
    speeds = {
        'walking': 5.0,       # Average walking speed: 5 km/h
        'cycling': 15.0,      # Average cycling speed: 15 km/h
        'public_transit': 20.0, # Average public transit speed: 20 km/h
        'driving': 30.0,      # Average driving speed in urban areas: 30 km/h
        'rideshare': 25.0     # Average rideshare speed: 25 km/h
    }
    
    # Default transport mode if none provided
    if not transport_mode:
        # Select transport mode based on distance
        if distance_km < 1.0:
            transport_mode = 'walking'
        elif distance_km < 3.0:
            transport_mode = 'cycling'
        elif distance_km < 10.0:
            transport_mode = 'public_transit'
        else:
            transport_mode = 'driving'
    
    # Normalize transport mode
    transport_mode = normalize_transport_mode(transport_mode)
    
    # Get speed for the selected transport mode
    speed = speeds.get(transport_mode, 15.0)  # Default to 15 km/h if unknown mode
    
    # Adjust speed based on personal attributes if available
    if persona:
        if hasattr(persona, 'age'):
            # Elderly people might move slower
            if persona.age > 65 and transport_mode in ['walking', 'cycling']:
                speed = speed * 0.8
            # Young adults might move faster
            elif 18 <= persona.age <= 35 and transport_mode in ['walking', 'cycling']:
                speed = speed * 1.2
            
        if hasattr(persona, 'disability') and persona.disability:
            # People with disabilities might move slower
            speed = speed * 0.7
    
    # Calculate travel time (hours) = distance / speed
    travel_time_hours = distance_km / speed
    
    # Convert to minutes and round to nearest minute
    travel_time_minutes = round(travel_time_hours * 60)
    
    # Add waiting time for public transit and rideshare
    if transport_mode == 'public_transit':
        travel_time_minutes += 5  # Average 5 min waiting time
    elif transport_mode == 'rideshare':
        travel_time_minutes += 3  # Average 3 min waiting time
    
    # Add 2 min buffer time for any mode
    travel_time_minutes += 2
    
    # Ensure minimum travel time of 1 minute
    travel_time_minutes = max(1, travel_time_minutes)
    
    return travel_time_minutes, transport_mode

def time_to_minutes(time_str):
    """
    Convert time string to minutes since midnight
    
    Args:
        time_str: Time string (HH:MM)
        
    Returns:
        int: Minutes since midnight
    """
    try:
        hours, minutes = parse_time(time_str)
        return hours * 60 + minutes
    except:
        return 0

@cached
def generate_random_location_near(center, max_distance_km=50.0, max_attempts=10, validate=True, search_query=None):
    """
    Generate a random location within a specified distance from a center point.
    If local POI data is available, use it to generate more realistic locations.
    
    Args:
        center: Center point (lat, lon)
        max_distance_km: Maximum distance in kilometers
        max_attempts: Maximum number of attempts
        validate: Whether to validate the location is valid
        search_query: Search query for filtering POIs
        
    Returns:
        tuple: (lat, lon) of random location
    """
    # 限制最大距离为50公里
    max_distance_km = min(max_distance_km, 50.0)
    
    # 如果不需要验证，直接使用几何算法生成随机点
    if not validate:
        return _generate_random_point_geometrically(center, max_distance_km)
    
    # 尝试从本地POI数据生成位置
    try:
        if USE_LOCAL_POI:
            # 使用制表符分隔符读取CSV文件
            poi_data = pd.read_csv(POI_CSV_PATH, sep=',')
            
            # 使用空间索引进行初步筛选
            lat_min = center[0] - (max_distance_km / 111.32)  # 1度约111.32公里
            lat_max = center[0] + (max_distance_km / 111.32)
            lon_min = center[1] - (max_distance_km / (111.32 * math.cos(math.radians(center[0]))))
            lon_max = center[1] + (max_distance_km / (111.32 * math.cos(math.radians(center[0]))))
            
            # 初步筛选在矩形范围内的POI
            filtered_pois = poi_data[
                (poi_data['latitude'] >= lat_min) &
                (poi_data['latitude'] <= lat_max) &
                (poi_data['longitude'] >= lon_min) &
                (poi_data['longitude'] <= lon_max)
            ]
            
            if len(filtered_pois) == 0:
                return _generate_random_point_geometrically(center, max_distance_km)
            
            # 创建明确的副本并计算精确距离
            filtered_pois = filtered_pois.copy()
            filtered_pois['distance'] = filtered_pois.apply(
                lambda row: calculate_distance(
                    center, 
                    (row['latitude'], row['longitude'])
                ),
                axis=1
            )
            
            # 处理缺失值
            filtered_pois['category'] = filtered_pois['category'].fillna('unknown')
            
            # 进一步筛选符合条件的POI
            filtered_pois = filtered_pois[
                (filtered_pois['distance'] <= max_distance_km)
            ]
            
            # 如果有搜索查询，进行进一步筛选
            if search_query and len(filtered_pois) > 0:
                search_query = search_query.lower()
                
                # 尝试直接匹配类别
                category_match = filtered_pois[
                    filtered_pois['category'].str.lower().str.contains(search_query, na=False)
                ]
                
                if len(category_match) > 0:
                    filtered_pois = category_match
                else:
                    # 如果没有找到匹配，尝试拆分关键词
                    search_keywords = search_query.split()
                    if len(search_keywords) > 1:
                        mask = pd.Series(False, index=filtered_pois.index)
                        for keyword in search_keywords:
                            if len(keyword) > 2:  # 忽略太短的关键词
                                keyword_mask = (
                                    filtered_pois['category'].str.lower().str.contains(keyword, na=False)
                                )
                                mask = mask | keyword_mask
                        
                        keyword_matches = filtered_pois[mask]
                        if len(keyword_matches) > 0:
                            filtered_pois = keyword_matches
            
            if len(filtered_pois) > 0:
                # 随机选择一个POI
                selected_poi = filtered_pois.sample(n=1).iloc[0]
                return (selected_poi['latitude'], selected_poi['longitude'])
    except Exception as e:
        print(f"Error using local POI data for random location: {e}")
    
    # 如果上述过程失败，使用几何算法生成随机点
    return _generate_random_point_geometrically(center, max_distance_km)

def _generate_random_point_geometrically(center, max_distance_km):
    """
    Generate a random point using geometric method
    
    Args:
        center: Center point coordinates (lat, lon)
        max_distance_km: Maximum distance (kilometers)
        
    Returns:
        tuple: Random location coordinates (lat, lon)
    """
    import random
    import math
    
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    # Convert maximum distance to radians
    max_distance_radians = max_distance_km / earth_radius
    
    # Generate random distance and angle
    # Using square root to ensure uniform distribution
    random_distance = max_distance_radians * math.sqrt(random.random())
    random_angle = random.random() * 2 * math.pi
    
    # Current point in radians
    lat1 = math.radians(center[0])
    lon1 = math.radians(center[1])
    
    # Calculate new point
    lat2 = math.asin(math.sin(lat1) * math.cos(random_distance) + 
                    math.cos(lat1) * math.sin(random_distance) * math.cos(random_angle))
    
    lon2 = lon1 + math.atan2(math.sin(random_angle) * math.sin(random_distance) * math.cos(lat1),
                           math.cos(random_distance) - math.sin(lat1) * math.sin(lat2))
    
    # Convert back to degrees
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    
    return (lat2, lon2)

def compress_trajectory_data(file_path, method='gzip'):
    """
    Compress JSON trajectory data file to save disk space.
    
    Args:
        file_path: Path to the JSON file
        method: Compression method ('gzip' is currently supported)
    
    Returns:
        str: Path to the compressed file
    """
    if method != 'gzip':
        print(f"Warning: Compression method {method} not supported, using gzip")
        
    try:
        # Create compressed file path
        compressed_file = f"{file_path}.gz"
        
        # Compress file
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        # Return path to compressed file
        return compressed_file
    except Exception as e:
        print(f"Error compressing file {file_path}: {e}")
        return None

def batch_save_trajectories(results, output_dir, format='json'):
    """
    Save all activity data in batch for efficient storage.
    Only saves activity data, not trajectory data.
    
    Args:
        results: Dictionary of Memory objects
        output_dir: Directory to save the activities
        format: Output format ('json' or 'merged')
    
    Returns:
        str: Path to saved file or directory
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if format == 'merged':
            # Save all activities in a single JSON file with links to daily files
            merged_data = {}
            
            for household_id, memory in results.items():
                if hasattr(memory, 'days'):
                    # For each household, create a summary with dates
                    merged_data[str(household_id)] = {
                        'persona_info': memory.persona_info,
                        'days_summary': [{'date': day['date'], 'day_of_week': day['day_of_week']} for day in memory.days]
                    }
            
            # Save merged data
            merged_file = os.path.join(output_dir, 'all_activities_summary.json')
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2)
                
            print(f"Saved merged activities summary to {merged_file}")
            return merged_file
            
        else:  # Individual JSON files
            for household_id, memory in results.items():
                if hasattr(memory, 'days'):
                    # Create a directory for this household
                    household_dir = os.path.join(output_dir, f"household_{household_id}")
                    os.makedirs(household_dir, exist_ok=True)
                    
                    # Save summary file
                    summary_file = os.path.join(household_dir, "summary.json")
                    summary_data = {
                        'persona_id': household_id,
                        'persona_info': memory.persona_info,
                        'days_summary': [{'date': day['date'], 'day_of_week': day['day_of_week']} for day in memory.days]
                    }
                    
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary_data, f, indent=2)
                    
                    # Save each day to a separate file
                    for day in memory.days:
                        date = day['date']
                        date_filename = date.replace('-', '')  # Remove hyphens for filename
                        day_file = os.path.join(household_dir, f"{date_filename}.json")
                        
                        # Create a copy of the day data with only activities, without trajectory
                        day_data = {
                            'date': day['date'],
                            'day_of_week': day['day_of_week'],
                            'activities': day['activities']
                        }
                        
                        with open(day_file, 'w', encoding='utf-8') as f:
                            json.dump(day_data, f, indent=2)
            
            print(f"Saved individual activities to {output_dir}")
            return output_dir
            
    except Exception as e:
        print(f"Error saving activities: {e}")
        return None

def generate_summary_report(results, output_dir):
    """
    Generate a summary report of the simulation results.
    
    Args:
        results: Dictionary of Memory objects
        output_dir: Directory to save the summary report
    
    Returns:
        str: Path to the summary report file
    """
    try:
        # Prepare summary data
        summary = {
            'total_households': len(results),
            'total_activities': 0,
            'activity_types': {},
            'transport_modes': {},
            'location_types': {}
        }
        
        # Calculate statistics
        for household_id, memory in results.items():
            if not hasattr(memory, 'days'):
                continue
                
            for day in memory.days:
                for activity in day['activities']:
                    # Count activities
                    summary['total_activities'] += 1
                    
                    # Count activity types
                    activity_type = activity.get('activity_type', 'unknown')
                    summary['activity_types'][activity_type] = summary['activity_types'].get(activity_type, 0) + 1
                    
                    # Count transport modes for travel activities
                    if activity_type == 'travel' and 'transport_mode' in activity:
                        mode = activity['transport_mode']
                        summary['transport_modes'][mode] = summary['transport_modes'].get(mode, 0) + 1
                    
                    # Count location types
                    location_type = activity.get('location_type', 'unknown')
                    summary['location_types'][location_type] = summary['location_types'].get(location_type, 0) + 1
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'simulation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Generated summary report with {summary['total_activities']} activities")
        return summary_file
        
    except Exception as e:
        print(f"Error generating summary report: {e}")
        return None