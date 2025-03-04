"""
Memory module for the LLM-based mobility simulation.
Records daily mobility trajectories and activities.
"""

import os
import json
import datetime
import numpy as np
from config import RESULTS_DIR
from utils import visualize_trajectory, plot_activity_distribution, save_json, normalize_transport_mode

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
        Record an activity.
        
        Args:
            activity: Activity dictionary
            location: (latitude, longitude)
            timestamp: Time string (HH:MM)
        """
        if self.current_day is None:
            return
        
        # Create activity record with only essential fields
        activity_record = {
            'activity_type': activity['activity_type'],
            'start_time': activity['start_time'],
            'end_time': activity['end_time'],
            'description': activity['description'],
            'location': location
        }
        
        # Add location name if available
        if 'location_name' in activity:
            activity_record['location_name'] = activity['location_name']
        
        # Only add transport mode if it's actually needed
        if 'transport_mode' in activity and activity['transport_mode']:
            # Normalize transport mode
            transport_mode = normalize_transport_mode(activity['transport_mode'])
            activity_record['transport_mode'] = transport_mode
        
        # Add only essential additional details
        if 'detailed_description' in activity:
            activity_record['detailed_description'] = activity['detailed_description']
        
        if 'location_type' in activity:
            activity_record['location_type'] = activity['location_type']
        
        # Remove other detailed information to keep memory records lightweight
        
        self.current_day['activities'].append(activity_record)
        
        # Record simplified trajectory point
        trajectory_point = {
            'location': location,
            'timestamp': timestamp,
            'activity_type': activity['activity_type'],
            'description': activity['description']
        }
        
        # Add location name to trajectory point if available
        if 'location_name' in activity:
            trajectory_point['location_name'] = activity['location_name']
        
        self.current_day['trajectory'].append(trajectory_point)
    
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
        
        # Get location name (if available)
        start_location_name = "Starting point"
        end_location_name = None  # Initialize as None to track if we find a name
        
        # Try to get start location name from previous trajectory points
        if self.current_day['trajectory']:
            last_point = self.current_day['trajectory'][-1]
            if 'location_name' in last_point:
                start_location_name = last_point['location_name']
        
        # Try to get destination name from activity records
        # 首先检查紧接着的活动（通过时间和位置匹配）
        next_activity = None
        min_time_diff = float('inf')
        matching_activity = None

        for activity in self.current_day['activities']:
            # 检查活动是否在这次旅程之后
            if activity.get('start_time', '') >= end_time:
                # 检查位置是否匹配
                if self._is_same_location(activity.get('location'), end_location):
                    # 计算时间差（分钟）
                    time_diff = self._calculate_time_diff(end_time, activity.get('start_time', ''))
                    # 如果这个活动更接近旅程结束时间
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        matching_activity = activity

        # 如果找到匹配的下一个活动
        if matching_activity and 'location_name' in matching_activity:
            end_location_name = matching_activity['location_name']
        else:
            # 如果找不到下一个活动，查找之前的活动记录
            for activity in reversed(self.current_day['activities']):
                if self._is_same_location(activity.get('location'), end_location) and 'location_name' in activity:
                    end_location_name = activity['location_name']
                    break
        
        # 如果仍然没有找到名称，检查是否是家或工作地点
        if end_location_name is None:
            if self.persona_info and 'home' in self.persona_info and self._is_same_location(end_location, self.persona_info['home']):
                end_location_name = "Home"
            elif self.persona_info and 'work' in self.persona_info and self._is_same_location(end_location, self.persona_info['work']):
                end_location_name = "Workplace"
            else:
                # 如果实在找不到名称，尝试从下一个活动的类型推断
                if matching_activity:
                    activity_type = matching_activity.get('activity_type', '').lower()
                    location_type = matching_activity.get('location_type', '').lower()
                    
                    if activity_type == 'dining' or 'restaurant' in location_type:
                        end_location_name = "Restaurant"
                    elif activity_type == 'shopping' or 'shop' in location_type or 'mall' in location_type:
                        end_location_name = "Shopping Center"
                    elif activity_type == 'recreation' or 'gym' in location_type:
                        end_location_name = "Fitness Center"
                    elif activity_type == 'leisure' or 'park' in location_type:
                        end_location_name = "Park"
                    else:
                        end_location_name = "Destination"  # 使用更通用的词替代 "Unknown location"
        
        # Normalize transport mode
        normalized_transport_mode = normalize_transport_mode(transport_mode)
        
        # Create travel record
        travel_record = {
            'activity_type': 'travel',
            'start_time': start_time,
            'end_time': end_time,
            'start_location': start_location,
            'end_location': end_location,
            'start_location_name': start_location_name,
            'end_location_name': end_location_name,
            'transport_mode': normalized_transport_mode,
            'description': f"Travel from {start_location_name} to {end_location_name} by {normalized_transport_mode}"
        }
        
        self.current_day['activities'].append(travel_record)
        
        # Create intermediate trajectory points for visualization
        # First point - departure
        self.current_day['trajectory'].append({
            'location': start_location,
            'timestamp': start_time,
            'activity_type': 'travel',
            'transport_mode': normalized_transport_mode,
            'description': f"Starting {normalized_transport_mode} journey",
            'location_name': start_location_name
        })
        
        # Last point - arrival
        self.current_day['trajectory'].append({
            'location': end_location,
            'timestamp': end_time,
            'activity_type': 'travel',
            'transport_mode': normalized_transport_mode,
            'description': f"Ending {normalized_transport_mode} journey",
            'location_name': end_location_name
        })

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
    
    def get_pattern_strength(self, pattern_type):
        """
        Get the strength of a specific pattern type.
        
        Args:
            pattern_type: Pattern type (e.g., 'location', 'time', 'sequence', 'environmental')
            
        Returns:
            float: Pattern strength (0-1)
        """
        if not hasattr(self, 'patterns') or 'pattern_strength' not in self.patterns:
            return 0.0
            
        relevant_strengths = []
        for key, value in self.patterns['pattern_strength'].items():
            if key.startswith(pattern_type):
                relevant_strengths.append(value)
                
        return np.mean(relevant_strengths) if relevant_strengths else 0.0
    
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