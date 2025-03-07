"""
Memory module for the LLM-based mobility simulation.
Records daily mobility trajectories and activities.
"""

import os
import json
import datetime
import numpy as np
import csv
from config import RESULTS_DIR
from utils import (
    visualize_trajectory, 
    plot_activity_distribution, 
    save_json, 
    normalize_transport_mode, 
    get_route_coordinates
)

class Memory:
    """
    Records and manages the memory of daily mobility trajectories.
    """
    
    def __init__(self, persona_id):
        """
        Initialize a Memory instance.
        
        Args:
            persona_id: Identifier for the persona
        """
        self.persona_id = persona_id
        self.persona_info = None
        self.days = []
        self.current_day = None
        self.location_frequency = {}
        self.activity_type_frequency = {}
        self.time_preference = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def initialize_persona(self, persona):
        """
        Initialize the memory with persona information.
        
        Args:
            persona: Persona object
        """
        self.persona_info = persona.to_dict()
    
    def start_new_day(self, date):
        """
        Start recording a new day.
        
        Args:
            date: Date string (YYYY-MM-DD)
        """
        self.current_day = {
            'date': date,
            'day_of_week': datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%A"),
            'activities': [],
            'trajectory': []
        }
    
    def record_activity(self, activity, location, timestamp):
        """
        Record activity to memory
        
        Args:
            activity: Activity dictionary
            location: Location coordinates (latitude, longitude)
            timestamp: Current timestamp string
        """
        if self.current_day is None:
            print("No current day to record activity")
            return
            
        # Make a copy of the activity to avoid modifying the original
        activity_record = activity.copy()
        
        # Add location and timestamp
        activity_record['location'] = location
        activity_record['timestamp'] = timestamp
        
        # Process location name if not present
        if 'location_name' not in activity_record:
            activity_record['location_name'] = self._get_location_name_from_activity(activity_record)
            
        # Add to activities list
        self.current_day['activities'].append(activity_record)
        
        # Create trajectory point
        self._add_trajectory_point(activity_record, location, timestamp)
        
        # Update statistics
        self._update_statistics(activity_record, location, timestamp)
    
    def _get_location_name_from_activity(self, activity):
        """
        Extract a meaningful location name from an activity
        
        Args:
            activity: Activity dictionary
            
        Returns:
            str: Location name
        """
        # If activity already has a location name, use it
        if 'location_name' in activity and activity['location_name']:
            return activity['location_name']
            
        # For dining activities, try to extract restaurant name
        if activity.get('activity_type') == 'dining' and 'description' in activity:
            description = activity['description']
            # Look for phrases like "at [Restaurant]" or "eating at [Restaurant]"
            if 'at ' in description:
                restaurant_name = description.split('at ')[-1].strip()
                # Remove trailing punctuation and extra text
                for suffix in ['.', ',', ' with', ' for']:
                    if restaurant_name.endswith(suffix):
                        restaurant_name = restaurant_name[:-len(suffix)]
                # If name looks good, use it
                if restaurant_name and len(restaurant_name) > 2:
                    return restaurant_name.strip()
        
        # Try to get a name from location type
        location_type = activity.get('location_type', '')
        if location_type:
            # Common places map
            if location_type == 'home':
                return 'Home'
            elif location_type == 'workplace':
                return 'Work'
            elif location_type == 'restaurant':
                # Try to get specific restaurant name from description
                if 'description' in activity:
                    # Look for restaurant name patterns
                    description = activity['description'].lower()
                    for pattern in ['at ', 'visiting ', 'dining at ', 'lunch at ', 'dinner at ']:
                        if pattern in description:
                            parts = description.split(pattern)
                            if len(parts) > 1:
                                name_part = parts[1].strip()
                                # Clean up the extracted name
                                for ending in ['.', ',', ' with', ' and', ' for']:
                                    if name_part.endswith(ending):
                                        name_part = name_part[:-len(ending)]
                                # If it's a reasonable name length, return it
                                if len(name_part) > 2 and len(name_part) < 30:
                                    return name_part.strip().title()
                return 'Restaurant'
            else:
                # Try to make a nicer name from the location type
                location_name = location_type.replace('_', ' ').title()
                return location_name
        
        # If no other options, use activity type
        return activity.get('activity_type', 'Location').title()
    
    def _update_statistics(self, activity, location, timestamp):
        """
        Update frequency statistics for location and activity.
        
        Args:
            activity: Activity dictionary
            location: Location coordinates
            timestamp: Time of activity
        """
        # Update location frequency statistics
        location_key = f"{location[0]:.5f},{location[1]:.5f}"
        if 'location_name' in activity:
            location_key = activity['location_name']
        
        self.location_frequency[location_key] = self.location_frequency.get(location_key, 0) + 1
        
        # Update activity type frequency statistics
        activity_type = activity['activity_type']
        self.activity_type_frequency[activity_type] = self.activity_type_frequency.get(activity_type, 0) + 1
        
        # Record time preference
        hour = timestamp.split(':')[0]
        time_key = f"{activity_type}_{hour}"
        self.time_preference[time_key] = self.time_preference.get(time_key, 0) + 1
    
    def record_travel(self, start_location, end_location, start_time, end_time, transport_mode):
        """
        Record a travel segment.
        
        Args:
            start_location: (latitude, longitude) of starting point
            end_location: (latitude, longitude) of ending point
            start_time: Time string (HH:MM) when travel starts
            end_time: Time string (HH:MM) when travel ends
            transport_mode: Mode of transportation
        """
        if self.current_day is None:
            return
            
        # Validate location format
        if not self._is_valid_location(start_location) or not self._is_valid_location(end_location):
            print(f"Invalid location format: start {start_location}, end {end_location}")
            return
        
        # Get location names
        start_location_name = self._get_start_location_name()
        
        # 直接在活动列表中查找下一个在此位置的非旅行活动
        end_location_name = None
        for activity in self.current_day['activities']:
            # 找到在end_location发生的、不是旅行类型的活动
            if (activity.get('activity_type') not in ['travel', 'commuting'] and 
                self._is_same_location(activity.get('location'), end_location)):
                
                # 优先使用已设置的location_name
                if 'location_name' in activity and activity['location_name']:
                    end_location_name = activity['location_name']
                    break
                
                # 其次查找描述中"at [PLACE]"格式的地点名称
                if 'description' in activity:
                    description = activity['description']
                    if ' at ' in description:
                        place_name = description.split(' at ')[1].strip()
                        # 清理名称
                        for suffix in ['.', ',', ' with', ' for']:
                            if place_name.endswith(suffix):
                                place_name = place_name[:-len(suffix)]
                        if place_name and len(place_name) > 2:
                            end_location_name = place_name
                            break
        
        # 如果没有找到明确的地点名称，再使用辅助方法
        if not end_location_name:
            end_location_name = self._get_end_location_name(end_location, end_time)
        
        # Normalize transport mode
        normalized_transport_mode = normalize_transport_mode(transport_mode)
        
        # Calculate route coordinates
        route_coords = get_route_coordinates(
            start_location,
            end_location,
            normalized_transport_mode
        )
        
        # Create and record travel activity
        travel_record = {
            'activity_type': 'travel',
            'start_time': start_time,
            'end_time': end_time,
            'start_location': start_location,
            'end_location': end_location,
            'start_location_name': start_location_name,
            'end_location_name': end_location_name,
            'transport_mode': normalized_transport_mode,
            'description': f"从 {start_location_name} 到 {end_location_name} ({normalized_transport_mode})"
        }
        
        self.current_day['activities'].append(travel_record)
        
        # Create and add trajectory points
        self._add_travel_trajectory_points(
            start_location, end_location, 
            start_time, end_time, 
            start_location_name, end_location_name,
            normalized_transport_mode, route_coords
        )
    
    def _get_start_location_name(self):
        """Get the name of the starting location from the last trajectory point."""
        if not self.current_day['trajectory']:
            return "Starting point"
            
        last_point = self.current_day['trajectory'][-1]
        return last_point.get('location_name', "Starting point")
    
    def _get_end_location_name(self, end_location, end_time):
        """
        确定终点位置的名称，直接使用时间上下一个活动的location_name
        
        Args:
            end_location: 终点位置坐标
            end_time: 到达时间
            
        Returns:
            str: 终点位置名称
        """
        # 1. 匹配已知位置（家、工作地点）
        if self.persona_info:
            if 'home' in self.persona_info and self._is_same_location(end_location, self.persona_info['home']):
                return "Home"
            if 'work' in self.persona_info and self._is_same_location(end_location, self.persona_info['work']):
                return "Work"
        
        # 2. 简单查找时间上紧接着的下一个活动
        all_activities = sorted(self.current_day['activities'], 
                               key=lambda x: (x.get('start_time', ''), x.get('end_time', '')))
        
        # 找到所有开始时间等于end_time的非旅行活动
        next_activities = [act for act in all_activities 
                          if act.get('start_time', '') == end_time and 
                             act.get('activity_type') not in ['travel', 'commuting']]
        
        # 如果找到了直接使用时间相同的活动
        if next_activities:
            next_act = next_activities[0]
            if 'location_name' in next_act and next_act['location_name']:
                return next_act['location_name']
        
        # 3. 如果没有找到开始时间相同的活动，找时间上最近的下一个活动
        next_activities = [act for act in all_activities 
                          if act.get('start_time', '') > end_time and 
                             act.get('activity_type') not in ['travel', 'commuting']]
        
        if next_activities:
            next_act = next_activities[0]  # 时间上最近的下一个活动
            if 'location_name' in next_act and next_act['location_name']:
                return next_act['location_name']
        
        # 4. 如果所有尝试都失败，返回默认名称
        return "Destination"
    
    def _find_matching_activity(self, location, time):
        """Find activity matching the given location and time."""
        min_time_diff = float('inf')
        matching_activity = None

        for activity in self.current_day['activities']:
            # Check if the activity is after this time
            if activity.get('start_time', '') >= time:
                # Check if the location matches
                if self._is_same_location(activity.get('location'), location):
                    # Calculate time difference
                    time_diff = self._calculate_time_diff(time, activity.get('start_time', ''))
                    # If this activity is closer
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        matching_activity = activity
        
        return matching_activity
    
    def _add_travel_trajectory_points(self, start_location, end_location, 
                                      start_time, end_time, 
                                      start_location_name, end_location_name,
                                      transport_mode, route_coords):
        """Add departure and arrival trajectory points for travel."""
        # Create departure point
        departure_point = {
            'location': start_location,
            'timestamp': start_time,
            'activity_type': 'travel',
            'transport_mode': transport_mode,
            'description': f"Starting {transport_mode} journey",
            'location_name': start_location_name,
            'route_coordinates': route_coords if route_coords else []
        }
        self.current_day['trajectory'].append(departure_point)
        
        # Create arrival point
        arrival_point = {
            'location': end_location,
            'timestamp': end_time,
            'activity_type': 'travel',
            'transport_mode': transport_mode,
            'description': f"Ending {transport_mode} journey",
            'location_name': end_location_name,
            'route_coordinates': route_coords if route_coords else []
        }
        self.current_day['trajectory'].append(arrival_point)
    
    def _calculate_time_diff(self, time1, time2):
        """
        Calculate the time difference in minutes between two time strings in HH:MM format.
        
        Args:
            time1: First time string (HH:MM)
            time2: Second time string (HH:MM)
            
        Returns:
            float: Absolute time difference in minutes
        """
        try:
            h1, m1 = map(int, time1.split(':'))
            h2, m2 = map(int, time2.split(':'))
            
            minutes1 = h1 * 60 + m1
            minutes2 = h2 * 60 + m2
            
            return abs(minutes2 - minutes1)
        except:
            return float('inf')
    
    def _is_valid_location(self, location):
        """
        Validate location format.
        
        Args:
            location: Location coordinates (latitude, longitude)
            
        Returns:
            bool: Whether the location is valid
        """
        if not location or not isinstance(location, (list, tuple)):
            return False
        if len(location) != 2:
            return False
        try:
            lat, lon = location
            return isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        except:
            return False
    
    def _is_same_location(self, loc1, loc2):
        """
        Compare two locations.
        
        Args:
            loc1: First location coordinates
            loc2: Second location coordinates
            
        Returns:
            bool: Whether the two locations are the same
        """
        if not self._is_valid_location(loc1) or not self._is_valid_location(loc2):
            return False
        return loc1[0] == loc2[0] and loc1[1] == loc2[1]
    
    def end_day(self):
        """
        Finish recording the current day and add it to the days list.
        """
        if self.current_day is not None:
            self.days.append(self.current_day)
            self.current_day = None
    
    def update_patterns(self, patterns):
        """
        Update activity patterns in memory.
        
        Args:
            patterns: Dictionary containing activity pattern analysis results
        """
        if not hasattr(self, 'patterns'):
            self.patterns = {
                'frequent_locations': {},
                'time_preferences': {},
                'pattern_strength': {}  # Used to track the strength of patterns
            }
        
        # Update location frequency
        for location, count in patterns['frequent_locations'].items():
            if location in self.patterns['frequent_locations']:
                self.patterns['frequent_locations'][location] += count
            else:
                self.patterns['frequent_locations'][location] = count
        
        # Update time preferences
        for activity_type, times in patterns['time_preferences'].items():
            if activity_type not in self.patterns['time_preferences']:
                self.patterns['time_preferences'][activity_type] = []
            self.patterns['time_preferences'][activity_type].extend(times)
            
            # Calculate time stability for this activity type
            if len(self.patterns['time_preferences'][activity_type]) > 1:
                times_int = [int(t.split(':')[0]) for t in self.patterns['time_preferences'][activity_type]]
                std_dev = np.std(times_int)
                self.patterns['pattern_strength'][f'{activity_type}_time'] = 1 / (1 + std_dev)
    
    def save_memory(self):
        """
        Save the memory to JSON files.
        
        Returns:
            str: Path to the saved memory file
        """
        memory_data = {
            'persona_id': self.persona_id,
            'persona_info': self.persona_info,
            'days': self.days
        }
        
        # Save complete memory
        memory_file = os.path.join(RESULTS_DIR, f"persona_{self.persona_id}_memory.json")
        save_json(memory_data, memory_file)
        
        # Create visualizations
        self.create_visualizations()
        
        # Export to CSV
        self.export_to_csv()
        
        return memory_file
    
    def create_visualizations(self):
        """
        Create visualizations of the mobility patterns.
        """
        # Create trajectory visualizations for each day
        for i, day in enumerate(self.days):
            trajectory = day.get('trajectory', [])
            if trajectory:
                map_file = os.path.join(RESULTS_DIR, f"persona_{self.persona_id}_day_{i+1}_trajectory.html")
                visualize_trajectory(trajectory, map_file)
        
        # Create activity distribution plot
        memory_data = {
            'persona_id': self.persona_id,
            'persona_info': self.persona_info,
            'days': self.days
        }
        plot_file = os.path.join(RESULTS_DIR, f"persona_{self.persona_id}_activity_distribution.png")
        plot_activity_distribution(memory_data, plot_file)
        
    def export_to_csv(self):
        """
        Export trajectory data to CSV format.
        
        The CSV will contain: person id, day, time, activity type, transport mode,
        location name, location type, location coordinates, and route coordinates.
        """
        csv_file = os.path.join(RESULTS_DIR, f"persona_{self.persona_id}_trajectory.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'person_id', 'date', 'time', 'activity_type', 'transport_mode',
                'location_name', 'location_type', 'location_coordinates', 'route_coordinates'
            ])
            
            # Write data for each day
            for day in self.days:
                date = day['date']
                for point in day['trajectory']:
                    # Prepare row data
                    row = [
                        self.persona_id,
                        date,
                        point['timestamp'],
                        point['activity_type'],
                        point.get('transport_mode', ''),  # May not exist for non-travel activities
                        point.get('location_name', ''),
                        point.get('location_type', ''),
                        str(point['location']),  # Convert coordinates to string
                        str(point.get('route_coordinates', []))  # Convert route coordinates to string
                    ]
                    writer.writerow(row)
    
    def to_dict(self):
        """
        Convert memory to dictionary representation.
        
        Returns:
            dict: Dictionary representation of the memory
        """
        return {
            'persona_id': self.persona_id,
            'persona_info': self.persona_info,
            'days': self.days
        }

    def _validate_activities(self, activities):
        """
        验证和清理活动数据。
        处理分解的子活动，确保时间连续性和正确的活动类型。
        
        Args:
            activities: 活动字典列表（包括普通活动和分解的子活动）
                
        Returns:
            list: 验证后的活动
        """
        if not activities:
            return []
        
        # 首先按开始时间排序
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # 合并相似的活动（例如 travel 和 commuting）
        merged_activities = []
        i = 0
        while i < len(sorted_activities):
            current = sorted_activities[i]
            
            # 确保基本字段存在
            if not all(key in current for key in ['activity_type', 'start_time', 'end_time']):
                i += 1
                continue
            
            # 格式化时间
            current['start_time'] = self._format_time(current['start_time'])
            current['end_time'] = self._format_time(current['end_time'])
            
            # 检查下一个活动是否与当前活动可以合并
            if i + 1 < len(sorted_activities):
                next_activity = sorted_activities[i + 1]
                if all(key in next_activity for key in ['activity_type', 'start_time', 'end_time']):
                    next_start = self._format_time(next_activity['start_time'])
                    next_end = self._format_time(next_activity['end_time'])
                    
                    # 如果是相关的交通活动（travel 和 commuting）
                    if (current['activity_type'] in ['travel', 'commuting'] and 
                        next_activity['activity_type'] in ['travel', 'commuting']):
                        # 合并这两个活动
                        current['end_time'] = max(current['end_time'], next_end)
                        current['activity_type'] = 'travel'  # 统一使用 travel 类型
                        if 'description' in next_activity:
                            current['description'] = f"{current.get('description', '')}; {next_activity['description']}"
                        i += 2
                        merged_activities.append(current)
                        continue
            
            merged_activities.append(current)
            i += 1
        
        # 处理时间重叠和间隙
        final_activities = []
        last_end_time = "00:00"
        
        for activity in merged_activities:
            start_time = activity['start_time']
            end_time = activity['end_time']
            
            # 如果当前活动开始时间早于上一个活动结束时间
            if start_time < last_end_time:
                # 调整当前活动的开始时间
                activity['start_time'] = last_end_time
                
                # 如果调整后开始时间大于等于结束时间，跳过此活动
                if activity['start_time'] >= activity['end_time']:
                    continue
            
            # 如果存在时间间隙（超过1分钟）
            elif self._calculate_time_diff(last_end_time, start_time) > 1:
                # 添加过渡活动（如果间隙超过5分钟）
                if self._calculate_time_diff(last_end_time, start_time) > 5:
                    transition_activity = {
                        'activity_type': 'transition',
                        'start_time': last_end_time,
                        'end_time': start_time,
                        'description': 'Transition period',
                        'location': activity.get('location', final_activities[-1].get('location') if final_activities else None),
                        'location_type': activity.get('location_type', final_activities[-1].get('location_type') if final_activities else None)
                    }
                    final_activities.append(transition_activity)
            
            final_activities.append(activity)
            last_end_time = end_time
        
        # 确保一天24小时都有覆盖
        if final_activities:
            # 处理午夜前的时间
            if final_activities[-1]['end_time'] < "23:59":
                final_activities.append({
                    'activity_type': 'sleep',
                    'start_time': final_activities[-1]['end_time'],
                    'end_time': "23:59",
                    'description': "Sleeping at home",
                    'location': final_activities[0].get('location'),  # 使用第一个活动的位置（通常是家）
                    'location_type': 'home'
                })
            
            # 处理午夜后的时间
            if final_activities[0]['start_time'] > "00:00":
                final_activities.insert(0, {
                    'activity_type': 'sleep',
                    'start_time': "00:00",
                    'end_time': final_activities[0]['start_time'],
                    'description': "Sleeping at home",
                    'location': final_activities[0].get('location'),  # 使用第一个活动的位置
                    'location_type': 'home'
                })
        
        # 最后按开始时间重新排序
        return sorted(final_activities, key=lambda x: x['start_time'])

    def _add_trajectory_point(self, activity, location, timestamp):
        """
        Create and add a trajectory point for an activity
        
        Args:
            activity: Activity dictionary
            location: Location coordinates (latitude, longitude)
            timestamp: Timestamp string
        """
        # Create trajectory point
        trajectory_point = {
            'location': location,
            'timestamp': timestamp,
            'activity_type': activity.get('activity_type', ''),
            'description': activity.get('description', '')
        }
        
        # Copy location name if available
        if 'location_name' in activity:
            trajectory_point['location_name'] = activity['location_name']
        else:
            # Use activity type as fallback
            trajectory_point['location_name'] = activity.get('activity_type', 'Location').title()
        
        # Copy other relevant fields
        for field in ['transport_mode', 'location_type', 'route_coordinates']:
            if field in activity:
                trajectory_point[field] = activity[field]
        
        # Add to trajectory list
        self.current_day['trajectory'].append(trajectory_point) 