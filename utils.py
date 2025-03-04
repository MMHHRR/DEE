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
from config import RESULTS_DIR, TRANSPORT_MODES
import requests
import math

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
    
    # 创建位置字典来跟踪重复位置
    location_counts = {}
    
    # 添加时间线连接点
    coordinates = []
    
    # 定义不同活动类型的图标
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
        loc_key = f"{lat:.5f},{lon:.5f}"  # 使用5位小数精度作为位置键
        
        # 检查是否有重复位置，如果有则偏移
        if loc_key in location_counts:
            location_counts[loc_key] += 1
            # 根据重复次数计算偏移量（每次偏移约20米）
            offset = location_counts[loc_key] * 0.0002
            lat += offset
            lon += offset
        else:
            location_counts[loc_key] = 0
            
        # 记录坐标用于绘制时间线
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
            
        # 获取活动类型对应的图标
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
        
        # 如果有路线坐标，添加路线
        if 'route_coordinates' in point and point['route_coordinates']:
            route_coords = point['route_coordinates']
            # 根据交通方式选择路线样式
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
    
    # 添加时间线连接所有点
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
            
            # Calculate duration in minutes
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            # Handle activities that cross midnight
            if end_hour < start_hour:
                end_hour += 24
            
            duration = (end_hour - start_hour) * 60 + (end_minute - start_minute)
            
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
    标准化交通方式，将任何交通方式转换为系统支持的标准格式。
    
    Args:
        mode: 原始交通方式字符串
        
    Returns:
        str: 标准化的交通方式
    """
    # 默认交通方式
    DEFAULT_MODE = 'walking'
    
    # 检查空值或特殊情况
    if not mode or not isinstance(mode, str):
        return DEFAULT_MODE
        
    mode = mode.lower().strip()
    
    # 处理特殊值
    invalid_values = ['n/a', 'none', 'null', 'nan', '']
    if mode in invalid_values:
        return DEFAULT_MODE
        
    # 如果已经是标准交通方式，直接返回
    if mode in TRANSPORT_MODES:
        return mode
        
    # 简单的映射表，将常见的非标准交通方式映射到标准方式
    mode_mapping = {
        'foot': 'walking',
        'pedestrian': 'walking',
        'stroll': 'walking',
        'car': 'driving',
        'automobile': 'driving',
        'vehicle': 'driving',
        'bus': 'public_transit',
        'train': 'public_transit',
        'subway': 'public_transit',
        'metro': 'public_transit',
        'tram': 'public_transit',
        'bike': 'cycling',
        'bicycle': 'cycling',
        'cab': 'rideshare',
        'taxi': 'rideshare',
        'uber': 'rideshare',
        'lyft': 'rideshare',
        'didi': 'rideshare'
    }
    
    # 通过映射表快速匹配
    if mode in mode_mapping:
        return mode_mapping[mode]
    
    # 如果无法匹配，返回默认交通方式
    return DEFAULT_MODE 

def get_route_coordinates(start_location, end_location, transport_mode='driving'):
    """
    使用OSRM获取两点之间的路线坐标
    
    Args:
        start_location: (latitude, longitude) 起点坐标
        end_location: (latitude, longitude) 终点坐标
        transport_mode: 交通方式 ('driving', 'walking', 'cycling')
    
    Returns:
        list: 路线上的坐标点列表 [(lat1, lon1), (lat2, lon2), ...]
    """
    # OSRM服务的基础URL（使用演示服务器）
    base_url = "http://router.project-osrm.org"
    
    # 构建API请求URL
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
            # 提取路线坐标
            coordinates = data["routes"][0]["geometry"]["coordinates"]
            # OSRM返回的坐标格式是[lon, lat]，我们需要转换为[lat, lon]
            return [(coord[1], coord[0]) for coord in coordinates]
        else:
            print(f"无法找到从 {start_location} 到 {end_location} 的路线")
            return []
            
    except Exception as e:
        print(f"获取路线时出错: {str(e)}")
        return [] 