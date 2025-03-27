"""
Memory module for the LLM-based mobility simulation.
Records daily mobility trajectories and activities.
"""

import os
import json
import datetime
from config import RESULTS_DIR
from utils import visualize_trajectory, get_day_of_week

class Memory:
    """
    Records and manages the memory of daily mobility trajectories.
    限制为仅保留最近2天的活动记录，减轻LLM处理压力
    """
    
    def __init__(self, persona_id, memory_days=2):
        """
        Initialize a Memory instance.
        
        Args:
            persona_id: Identifier for the persona
            memory_days: Number of days to retain in memory (default: 2)
        """
        self.persona_id = persona_id
        self.persona_info = None
        self.days = []
        self.current_day = None
        self.memory_days = memory_days
        self.location_frequency = {}
        self.activity_type_frequency = {}
        self.time_preference = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 为高效处理创建索引
        self.location_index = {}  # 位置索引 - 提高位置查询效率
        self.latest_activities = {}  # 最近的活动类型 - 用于更快地检索历史行为
    
    def initialize_persona(self, persona):
        """
        Initialize memory with persona information
        
        Args:
            persona: Persona object
        """
        self.persona_info = {
            'id': persona.id,
            'name': persona.name,
            'gender': persona.gender,
            'age': persona.age,
            'income': persona.get_household_income(),
            'education': persona.education,
            'home': persona.home,
            'work': persona.work
        }
    
    def start_new_day(self, date):
        """
        Start recording a new day.
        
        Args:
            date: Date string (YYYY-MM-DD)
        """
        day_of_week = get_day_of_week(date)
        self.current_day = {
            'date': date,
            'day_of_week': day_of_week,
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
            activity_record['location_name'] = self._get_location_name(activity_record)
            
        # Add to activities list
        self.current_day['activities'].append(activity_record)
        
        # Create trajectory point
        self._add_trajectory_point(activity_record, location, timestamp)
        
        # Update statistics
        self._update_statistics(activity_record, location, timestamp)
        
        # 更新最近活动索引
        activity_type = activity_record.get('activity_type')
        if activity_type:
            self.latest_activities[activity_type] = {
                'timestamp': timestamp,
                'location': location,
                'location_name': activity_record.get('location_name', '')
            }
            
        # 更新位置索引 - 使用位置名作为键
        location_name = activity_record.get('location_name')
        if location_name:
            self.location_index[location_name] = location
    
    def _get_location_name(self, activity):
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
            
        # Try to get a name from location type
        location_type = activity.get('location_type', '')
        if location_type:
            if location_type == 'home':
                return 'Home'
            elif location_type == 'workplace':
                return 'Work'
            else:
                # Format location type as a title
                return location_type.replace('_', ' ').title()
        
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
            print("Invalid location format for travel record")
            return
            
        # Get location names
        start_location_name = self._find_location_name(start_location)
        end_location_name = self._find_location_name(end_location)
        
        # Create travel activity
        travel_activity = {
            'activity_type': 'travel',
            'start_time': start_time,
            'end_time': end_time,
            'location': end_location,  # End location is the destination
            'description': f"Traveling from {start_location_name} to {end_location_name}",
            'transport_mode': transport_mode,
            'from_location': start_location,
            'to_location': end_location,
            'from_name': start_location_name,
            'to_name': end_location_name
        }
        
        # Add to activities and create trajectory points
        self.current_day['activities'].append(travel_activity)
        
        # Add trajectory points for start and end of travel
        self._add_trajectory_point(
            travel_activity,
            start_location,
            start_time,
            is_travel=True,
            travel_position='start'
        )
        
        self._add_trajectory_point(
            travel_activity,
            end_location, 
            end_time,
            is_travel=True,
            travel_position='end'
        )
        
        # 更新位置索引
        if start_location_name:
            self.location_index[start_location_name] = start_location
        if end_location_name:
            self.location_index[end_location_name] = end_location
            
        # 更新最近活动
        self.latest_activities['travel'] = {
            'timestamp': end_time,
            'transport_mode': transport_mode,
            'from': start_location_name,
            'to': end_location_name
        }
    
    def _find_location_name(self, location):
        """
        Find location name from previous activities at same location
        
        Args:
            location: Location coordinates
            
        Returns:
            str: Location name if found, otherwise "Unknown Location"
        """
        # 先检查位置索引，这比遍历活动更快
        for name, coords in self.location_index.items():
            if self._is_same_location(coords, location):
                return name
                
        # 如果索引中没有找到，再检查当前日的活动
        if not self.current_day or not self.current_day['activities']:
            return "Unknown Location"
            
        # Look for activities at the same location
        for activity in reversed(self.current_day['activities']):
            if 'location' in activity and self._is_same_location(activity['location'], location):
                if 'location_name' in activity:
                    return activity['location_name']
        
        return "Unknown Location"
    
    def _is_valid_location(self, location):
        """
        Check if a location is in valid format.
        
        Args:
            location: Location to check
            
        Returns:
            bool: True if location is valid, False otherwise
        """
        try:
            if not location or len(location) != 2:
                return False
                
            lat, lon = location
            return isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        except:
            return False
    
    def _is_same_location(self, loc1, loc2, threshold=0.001):
        """
        Check if two locations are close enough to be considered the same.
        
        Args:
            loc1: First location
            loc2: Second location
            threshold: Distance threshold in degrees
            
        Returns:
            bool: True if same location, False otherwise
        """
        if not loc1 or not loc2:
            return False
            
        return (abs(loc1[0] - loc2[0]) < threshold and
                abs(loc1[1] - loc2[1]) < threshold)
    
    def end_day(self):
        """
        End current day recording and add to days list.
        维持只有最近的memory_days天的记录
        """
        if self.current_day:
            # 添加当前日到历史记录
            self.days.append(self.current_day)
            
            # 如果超过记忆限制，删除最早的日记录
            while len(self.days) > self.memory_days:
                oldest_day = self.days.pop(0)
                print(f"Removed oldest day ({oldest_day.get('date', 'unknown')}) from memory")
                
            self.current_day = None
    
    def get_recent_activities(self, activity_type=None, days=None):
        """
        获取最近的活动，可以按类型过滤
        
        Args:
            activity_type: 可选的活动类型过滤
            days: 要获取的天数，默认为所有记忆中的天
            
        Returns:
            list: 符合条件的活动列表
        """
        if days is None:
            days = self.memory_days
            
        days = min(days, len(self.days))
        
        # 只获取最近的n天
        recent_days = self.days[-days:]
        
        activities = []
        for day in recent_days:
            for activity in day['activities']:
                if activity_type is None or activity['activity_type'] == activity_type:
                    activities.append(activity)
                    
        return activities
    
    def get_mobility_patterns(self):
        """
        提取基本的移动模式统计信息，仅基于存储的天数
        
        Returns:
            dict: 移动模式信息
        """
        patterns = {
            'frequent_locations': {},
            'transport_preferences': {},
            'activity_time_patterns': {},
            'daily_routines': {}
        }
        
        # 收集所有活动
        all_activities = []
        for day in self.days:
            all_activities.extend(day['activities'])
            
        # 频繁位置
        patterns['frequent_locations'] = self.location_frequency
        
        # 交通偏好
        transport_counts = {}
        for activity in all_activities:
            if activity['activity_type'] == 'travel':
                mode = activity.get('transport_mode', 'unknown')
                transport_counts[mode] = transport_counts.get(mode, 0) + 1
                
        patterns['transport_preferences'] = transport_counts
        
        # 活动时间模式
        activity_times = {}
        for activity in all_activities:
            act_type = activity['activity_type']
            if act_type not in activity_times:
                activity_times[act_type] = []
                
            start_time = activity.get('start_time', '')
            if start_time:
                activity_times[act_type].append(start_time)
                
        for act_type, times in activity_times.items():
            if times:
                # 将时间转换为分钟
                minutes = []
                for t in times:
                    h, m = map(int, t.split(':'))
                    minutes.append(h * 60 + m)
                    
                # 计算平均时间
                avg_minutes = sum(minutes) / len(minutes)
                avg_hour = int(avg_minutes / 60)
                avg_min = int(avg_minutes % 60)
                
                patterns['activity_time_patterns'][act_type] = f"{avg_hour:02d}:{avg_min:02d}"
        
        # 日常规律
        for day in self.days:
            day_type = 'weekday'
            if day['day_of_week'] in ['Saturday', 'Sunday']:
                day_type = 'weekend'
                
            if day_type not in patterns['daily_routines']:
                patterns['daily_routines'][day_type] = []
                
            # 提取当天的活动序列
            activity_sequence = [a['activity_type'] for a in day['activities'] if a['activity_type'] != 'travel']
            if activity_sequence:
                patterns['daily_routines'][day_type].append(activity_sequence)
        
        return patterns
    
    def save_to_file(self, file_path):
        """
        Save memory data to a JSON file.
        
        Args:
            file_path: Path to save the file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data - 减少数据量，只保存必要信息
            data = {
                'persona_id': self.persona_id,
                'persona_info': self.persona_info,
                'days': self.days,
                'statistics': {
                    'location_frequency': self.location_frequency,
                    'activity_type_frequency': self.activity_type_frequency,
                    'time_preference': self.time_preference
                },
                'mobility_patterns': self.get_mobility_patterns()
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            print(f"Memory data saved to {file_path}")
            
            # 生成可视化
            self.create_visualizations(os.path.dirname(file_path))
            
        except Exception as e:
            print(f"Error saving memory data: {e}")
    
    def create_visualizations(self, output_dir):
        """
        Create visualizations for each day's trajectory.
        
        Args:
            output_dir: Directory to save visualizations
        """
        for i, day in enumerate(self.days):
            if not day.get('trajectory'):
                continue
                
            # Create trajectory map
            try:
                date_str = day.get('date', f"day_{i+1}")
                map_file = os.path.join(output_dir, f"trajectory_{self.persona_id}_{date_str}.html")
                visualize_trajectory(
                    day['trajectory'], 
                    map_file,
                    title=f"Trajectory for Persona {self.persona_id} on {date_str}"
                )
            except Exception as e:
                print(f"Error creating visualization: {e}")
    
    def _add_trajectory_point(self, activity, location, timestamp, is_travel=False, travel_position=None):
        """
        Add a point to the trajectory.
        
        Args:
            activity: Activity dictionary
            location: Location coordinates
            timestamp: Time string
            is_travel: Whether this is a travel activity
            travel_position: Position in travel (start or end)
        """
        if self.current_day is None or not self._is_valid_location(location):
            return
            
        point = {
            'activity_type': activity.get('activity_type', 'unknown'),
            'time': timestamp,
            'location': location,
            'description': activity.get('description', '')
        }
        
        # Add transport mode for travel activities
        if 'transport_mode' in activity:
            point['transport_mode'] = activity['transport_mode']
        
        # Add travel-specific information
        if is_travel and travel_position:
            point['is_travel'] = True
            point['travel_position'] = travel_position
            
            if travel_position == 'start' and 'from_name' in activity:
                point['location_name'] = activity['from_name']
            elif travel_position == 'end' and 'to_name' in activity:
                point['location_name'] = activity['to_name']
        elif 'location_name' in activity:
            point['location_name'] = activity['location_name']
            
        self.current_day['trajectory'].append(point) 