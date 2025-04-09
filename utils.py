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
import pickle
from collections import OrderedDict
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# 增强版缓存系统实现
class Cache:
    def __init__(self):
        # 基本缓存设置
        self.enabled = ENABLE_CACHING
        self.expiry = CACHE_EXPIRY
        self._lock = threading.Lock()  # 缓存访问锁
        
        # 多层缓存系统
        self._memory_cache = OrderedDict()  # 内存LRU缓存
        self._memory_cache_max_size = 10000  # 最大内存缓存条目数
        
        # 分区缓存，按功能分区减少锁竞争
        self._cache_partitions = {
            'location': {},  # 位置相关缓存
            'activity': {},  # 活动生成相关缓存
            'transportation': {},  # 交通模式相关缓存
            'general': {}   # 通用缓存
        }
        
        # 分布式缓存客户端
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_host = os.environ.get('REDIS_HOST', 'localhost')
                redis_port = int(os.environ.get('REDIS_PORT', 6379))
                redis_db = int(os.environ.get('REDIS_DB', 0))
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    db=redis_db,
                    socket_timeout=5
                )
                # 测试连接
                self.redis_client.ping()
                print("Successfully connected to Redis cache server")
            except Exception as e:
                print(f"Warning: Could not initialize Redis client: {e}")
                self.redis_client = None
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.redis_hits = 0
        
        # 确保缓存目录存在
        try:
            self.cache_dir = os.path.join(RESULTS_DIR, 'cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, 'function_cache.pkl')
            
            # 分区文件缓存
            for partition in self._cache_partitions:
                partition_dir = os.path.join(self.cache_dir, partition)
                os.makedirs(partition_dir, exist_ok=True)
                
            self.load_cache()
        except Exception as e:
            print(f"Warning: Could not initialize cache directory: {e}")
            self.enabled = False
    
    def get(self, key, partition='general'):
        """获取缓存值，采用多层缓存查询策略"""
        if not self.enabled:
            return None
            
        # 首先在内存LRU缓存中查找（无需加锁，提高并发性能）
        memory_key = f"{partition}:{key}"
        cached_value = self._memory_cache.get(memory_key)
        if cached_value is not None:
            value, timestamp = cached_value
            # 检查是否过期
            if time.time() - timestamp < self.expiry:
                self.cache_hits += 1
                self.memory_hits += 1
                # 将此条目移动到OrderedDict的末尾（表示最近使用）
                self._memory_cache.move_to_end(memory_key)
                return value
                
        # 检查Redis分布式缓存（如果可用）
        if self.redis_client:
            try:
                redis_value = self.redis_client.get(memory_key)
                if redis_value:
                    # Redis数据需要反序列化
                    value, timestamp = pickle.loads(redis_value)
                    if time.time() - timestamp < self.expiry:
                        # 命中Redis缓存，同时更新内存缓存
                        self._add_to_memory_cache(memory_key, value, timestamp)
                        self.cache_hits += 1
                        self.redis_hits += 1
                        return value
            except Exception as e:
                # Redis错误不应阻止继续查找其他缓存
                print(f"Redis cache error: {e}")
        
        # 最后检查分区内存缓存
        with self._lock:
            cache_partition = self._cache_partitions.get(partition, self._cache_partitions['general'])
            if key in cache_partition:
                value, timestamp = cache_partition[key]
                # 检查是否过期
                if time.time() - timestamp < self.expiry:
                    self.cache_hits += 1
                    self.disk_hits += 1
                    # 同时更新内存LRU缓存
                    self._add_to_memory_cache(memory_key, value, timestamp)
                    return value
        
        # 缓存未命中
        self.cache_misses += 1
        return None
    
    def set(self, key, value, partition='general'):
        """设置缓存值到多层缓存系统"""
        if not self.enabled:
            return
            
        current_time = time.time()
        memory_key = f"{partition}:{key}"
        
        # 更新内存LRU缓存
        self._add_to_memory_cache(memory_key, value, current_time)
        
        # 更新Redis分布式缓存（如果可用）
        if self.redis_client:
            try:
                # 序列化数据并设置生存时间 (TTL)
                redis_value = pickle.dumps((value, current_time))
                self.redis_client.setex(
                    memory_key, 
                    int(self.expiry), 
                    redis_value
                )
            except Exception as e:
                # Redis错误只打印警告
                print(f"Redis cache set error: {e}")
        
        # 更新分区缓存
        with self._lock:
            try:
                # 确保分区存在
                if partition not in self._cache_partitions:
                    self._cache_partitions[partition] = {}
                
                # 更新分区缓存
                self._cache_partitions[partition][key] = (value, current_time)
                
                # 每100次缓存操作保存一次，或缓存大小达到阈值
                operations = self.cache_hits + self.cache_misses
                total_items = sum(len(p) for p in self._cache_partitions.values())
                
                if operations % 100 == 0 or total_items > self._memory_cache_max_size * 2:
                    self.save_cache()
            except Exception as e:
                print(f"Warning: Could not set cache value: {e}")
    
    def _add_to_memory_cache(self, key, value, timestamp):
        """添加值到内存LRU缓存，并在必要时清理旧条目"""
        # 如果存在，先删除旧值并移动到末尾
        if key in self._memory_cache:
            del self._memory_cache[key]
            
        # 添加新值
        self._memory_cache[key] = (value, timestamp)
        
        # 如果超过大小限制，删除最旧的条目（OrderedDict的开头）
        if len(self._memory_cache) > self._memory_cache_max_size:
            self._memory_cache.popitem(last=False)
    
    def load_cache(self):
        """从磁盘加载缓存到内存"""
        if not self.enabled:
            return
        
        # 首先尝试加载主缓存文件
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    
                    # 将加载的数据赋值给分区缓存
                    if isinstance(loaded_cache, dict):
                        for partition, partition_data in loaded_cache.items():
                            if partition in self._cache_partitions:
                                self._cache_partitions[partition] = partition_data
                                
                                # 同时更新LRU内存缓存
                                for key, (value, timestamp) in partition_data.items():
                                    memory_key = f"{partition}:{key}"
                                    self._add_to_memory_cache(memory_key, value, timestamp)
        except Exception as e:
            print(f"Warning: Could not load main cache file: {e}")
            
        # 然后尝试加载分区缓存文件（增量方式）
        for partition in self._cache_partitions:
            partition_file = os.path.join(self.cache_dir, partition, 'cache.pkl')
            try:
                if os.path.exists(partition_file):
                    with open(partition_file, 'rb') as f:
                        partition_data = pickle.load(f)
                        self._cache_partitions[partition].update(partition_data)
                        
                        # 更新内存缓存
                        for key, (value, timestamp) in partition_data.items():
                            memory_key = f"{partition}:{key}"
                            self._add_to_memory_cache(memory_key, value, timestamp)
            except Exception as e:
                print(f"Warning: Could not load partition cache file for {partition}: {e}")
    
    def save_cache(self):
        """将缓存保存到磁盘，使用分区和临时文件策略确保数据完整性"""
        if not self.enabled:
            return
            
        with self._lock:
            try:
                # 1. 首先保存主缓存文件
                temp_file = f"{self.cache_file}.tmp"
                with open(temp_file, 'wb') as f:
                    pickle.dump(self._cache_partitions, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                if os.path.exists(temp_file):
                    if os.path.exists(self.cache_file):
                        os.replace(temp_file, self.cache_file)
                    else:
                        os.rename(temp_file, self.cache_file)
                        
                # 2. 然后保存每个分区的缓存文件
                for partition, partition_data in self._cache_partitions.items():
                    partition_dir = os.path.join(self.cache_dir, partition)
                    os.makedirs(partition_dir, exist_ok=True)
                    
                    partition_file = os.path.join(partition_dir, 'cache.pkl')
                    temp_partition_file = f"{partition_file}.tmp"
                    
                    with open(temp_partition_file, 'wb') as f:
                        pickle.dump(partition_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    if os.path.exists(temp_partition_file):
                        if os.path.exists(partition_file):
                            os.replace(temp_partition_file, partition_file)
                        else:
                            os.rename(temp_partition_file, partition_file)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
                # 清理临时文件
                for temp_file in [f"{self.cache_file}.tmp"] + [f"{os.path.join(self.cache_dir, p, 'cache.pkl')}.tmp" for p in self._cache_partitions]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
    
    def clear(self):
        """清除所有缓存，包括内存缓存、文件缓存和Redis缓存"""
        with self._lock:
            # 清除内存缓存
            self._memory_cache.clear()
            
            # 清除分区缓存
            for partition in self._cache_partitions:
                self._cache_partitions[partition] = {}
            
            # 清除Redis缓存
            if self.redis_client:
                try:
                    # 只清除与当前应用相关的键
                    for partition in self._cache_partitions:
                        pattern = f"{partition}:*"
                        keys = self.redis_client.keys(pattern)
                        if keys:
                            self.redis_client.delete(*keys)
                except Exception as e:
                    print(f"Warning: Could not clear Redis cache: {e}")
            
            # 清除文件缓存
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except Exception as e:
                    print(f"Warning: Could not remove main cache file: {e}")
            
            # 清除分区文件缓存
            for partition in self._cache_partitions:
                partition_file = os.path.join(self.cache_dir, partition, 'cache.pkl')
                if os.path.exists(partition_file):
                    try:
                        os.remove(partition_file)
                    except Exception as e:
                        print(f"Warning: Could not remove partition cache file for {partition}: {e}")
    
    def stats(self):
        """返回缓存统计信息"""
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0
            
            # 计算每个分区的缓存条目数
            partition_sizes = {partition: len(data) for partition, data in self._cache_partitions.items()}
            
            # 计算内存缓存大小
            memory_size = len(self._memory_cache)
            
            # Redis状态
            redis_status = "Connected" if self.redis_client else "Not connected"
            
            return {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "memory_hits": self.memory_hits,
                "disk_hits": self.disk_hits,
                "redis_hits": self.redis_hits,
                "total_requests": total_requests,
                "hit_ratio": hit_ratio,
                "memory_cache_size": memory_size,
                "partition_sizes": partition_sizes,
                "redis_status": redis_status,
                "enabled": self.enabled,
                "expiry": self.expiry
            }
            
    def get_partition_stats(self):
        """获取每个分区的缓存命中统计信息"""
        # 在实际实现中，此方法可以追踪每个分区的命中率
        return {partition: {"size": len(data)} for partition, data in self._cache_partitions.items()}

# 创建全局缓存实例
cache = Cache()

def cached(func=None, partition='general', ttl=None):
    """
    改进的缓存装饰器，支持分区和自定义TTL
    
    Args:
        func: 要缓存的函数
        partition: 缓存分区名称
        ttl: 自定义TTL（秒），None使用默认的CACHE_EXPIRY
        
    Returns:
        经过缓存包装的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cache.enabled:
                return func(*args, **kwargs)
                
            # 创建缓存键
            key_parts = [func.__name__]
            # 为对象方法处理self参数
            arg_values = list(args)
            if args and hasattr(args[0], '__class__'):
                arg_values[0] = args[0].__class__.__name__
            key_parts.extend([str(arg) for arg in arg_values])
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_result = cache.get(key, partition)
            if cached_result is not None:
                return cached_result
            
            # 计算结果并缓存
            result = func(*args, **kwargs)
            cache.set(key, result, partition)
            return result
        
        return wrapper
    
    if func is None:
        # 被调用为 @cached(partition='xxx')
        return decorator
    else:
        # 被调用为简单的 @cached
        return decorator(func)

# 保留原始cached函数的兼容性
def original_cached(func):
    """
    原始缓存装饰器（保留向后兼容性）
    """
    return cached(func, partition='general')

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

# LLM Management Class
class LLMManager:
    """
    A class to manage multiple LLM clients, providing a unified interface to access different LLM services.
    """
    
    def __init__(self):
        """initialize LLM manager, create all needed LLM clients."""
        from config import (
            DEEPBRICKS_API_KEY, DEEPBRICKS_BASE_URL, BASIC_LLM_MODEL,
            ZETA_API_KEY, ZETA_BASE_URL, ACTIVITY_LLM_MODEL,
            LLM_TEMPERATURE, LLM_MAX_TOKENS
        )
        import openai
        import threading
        import time
        
        # create basic LLM client (DeepBricks)
        self.basic_client = openai.OpenAI(
            api_key=DEEPBRICKS_API_KEY,
            base_url=DEEPBRICKS_BASE_URL,
        )
        self.basic_model = BASIC_LLM_MODEL
        
        # create activity LLM client (Zeta)
        self.activity_client = openai.OpenAI(
            api_key=ZETA_API_KEY,
            base_url=ZETA_BASE_URL,
        )
        self.activity_model = ACTIVITY_LLM_MODEL
        
        # default parameters
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        
        # basic LLM's rate limit and concurrency control
        self.basic_rate_limit = 0.2  # default request interval (seconds)
        self.basic_max_concurrent = 5  # default maximum concurrent requests
        self.basic_semaphore = threading.Semaphore(self.basic_max_concurrent)
        self.basic_rate_lock = threading.Lock()
        self.basic_last_request_time = time.time() - self.basic_rate_limit
        
        # activity LLM's rate limit and concurrency control
        self.activity_rate_limit = 0.2  # default request interval (seconds)
        self.activity_max_concurrent = 5  # default maximum concurrent requests
        self.activity_semaphore = threading.Semaphore(self.activity_max_concurrent)
        self.activity_rate_lock = threading.Lock()
        self.activity_last_request_time = time.time() - self.activity_rate_limit
    
    def set_basic_rate_limit(self, rate_limit):
        """set basic LLM request rate limit"""
        if rate_limit >= 0:
            self.basic_rate_limit = rate_limit
    
    def set_activity_rate_limit(self, rate_limit):
        """set activity LLM request rate limit"""
        if rate_limit >= 0:
            self.activity_rate_limit = rate_limit
    
    def set_basic_concurrency_limit(self, max_concurrent):
        """set basic LLM request maximum concurrent requests"""
        if max_concurrent > 0:
            self.basic_max_concurrent = max_concurrent
            self.basic_semaphore = threading.Semaphore(max_concurrent)
    
    def set_activity_concurrency_limit(self, max_concurrent):
        """set activity LLM request maximum concurrent requests"""
        if max_concurrent > 0:
            self.activity_max_concurrent = max_concurrent
            self.activity_semaphore = threading.Semaphore(max_concurrent)
    
    def _apply_basic_rate_limit(self):
        """apply basic LLM's rate limit"""
        if self.basic_rate_limit <= 0:
            return
            
        with self.basic_rate_lock:
            current_time = time.time()
            elapsed = current_time - self.basic_last_request_time
            
            if elapsed < self.basic_rate_limit:
                time.sleep(self.basic_rate_limit - elapsed)
                
            self.basic_last_request_time = time.time()
    
    def _apply_activity_rate_limit(self):
        """apply activity LLM's rate limit"""
        if self.activity_rate_limit <= 0:
            return
            
        with self.activity_rate_lock:
            current_time = time.time()
            elapsed = current_time - self.activity_last_request_time
            
            if elapsed < self.activity_rate_limit:
                time.sleep(self.activity_rate_limit - elapsed)
                
            self.activity_last_request_time = time.time()
    
    def completion_basic(self, prompt, model=None, temperature=None, max_tokens=None):
        """use basic LLM client to create a completion request"""
        # use basic LLM's semaphore to control concurrency
        with self.basic_semaphore:
            # apply basic LLM's rate limit
            self._apply_basic_rate_limit()
            
            # send actual request
            return self.basic_client.chat.completions.create(
                model=model or self.basic_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
    
    def completion_activity(self, prompt, model=None, temperature=None, max_tokens=None):
        """use activity LLM client to create a completion request"""
        # use activity LLM's semaphore to control concurrency
        with self.activity_semaphore:
            # apply activity LLM's rate limit
            self._apply_activity_rate_limit()
            
            # send actual request
            return self.activity_client.chat.completions.create(
                model=model or self.activity_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )

# create a global LLM manager instance
llm_manager = LLMManager()