"""
Activity module for the LLM-based mobility simulation.
Handles activity generation and refinement using LLM.
"""

import json
import re
import openai
import random
from datetime import datetime, timedelta
from config import (
    BASIC_LLM_MODEL,
    ACTIVITY_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    ACTIVITY_GENERATION_PROMPT,
    DEEPBRICKS_API_KEY,
    DEEPBRICKS_BASE_URL
)
from utils import cached, get_day_of_week, calculate_distance, estimate_travel_time,generate_random_location_near

# Create OpenAI client
client = openai.OpenAI(
    api_key = DEEPBRICKS_API_KEY,
    base_url = DEEPBRICKS_BASE_URL,
)

class Activity:
    """
    Manages the generation and processing of daily activity plans.
    """
    
    def __init__(self, config=None):
        """Initialize the Activity generator."""
        self.model = BASIC_LLM_MODEL
        self.act_model = ACTIVITY_LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.activity_queue = []  # For batch processing of activities
        self.config = config  # Store config for destination selector
    
    def analyze_memory_patterns(self, memory, persona=None):
        """
        Analyze the activity patterns in the historical memory.
        Try to use LLM summary first, fallback to basic statistics if LLM fails.
        Only analyze the most recent memory_days days of data, not all historical data.
        
        Args:
            memory: Memory object, containing historical activity records
            persona: Optional Persona object for additional context
            
        Returns:
            dict: Dictionary containing activity pattern analysis results
        """
        # Only analyze the most recent memory_days days of data, not all historical data.
        memory_days_limit = memory.memory_days
        recent_days = memory.days[-memory_days_limit:] if len(memory.days) > memory_days_limit else memory.days
        
        patterns = {
            'summaries': [],  # LLM generated summaries
            'frequent_locations': {},  # Frequent locations (fallback)
            'time_preferences': {},    # Time preferences (fallback)
            'travel_times': {},        # Travel times for different activities
            'distances': {},           # Travel distances
            'activity_durations': {},  # Duration of different activities
            'transport_modes': {}     # Transport mode preferences
        }
        
        # Try LLM summary for each day in the recent days only
        for day_index, day in enumerate(recent_days):
            # Record the analyzed date
            date = day.get('date', 'unknown')
                
            filtered_activities = []
            
            for activity in day['activities']:
                # Skip 3 AM data and 24-hour activities
                actdur = activity.get('actdur', 0)
                arrtime = activity.get('arrtime', '')
                start_time = activity.get('start_time', '')
                
                if (actdur != 1440 and arrtime != '03:00') and (start_time != '03:00'):
                    filtered_activities.append(activity)
            
            day['activities'] = filtered_activities
            try:
                summary = self._generate_activities_summary(filtered_activities)
                if summary and not summary.startswith("Unable to generate"):
                    patterns['summaries'].append(summary)
            except Exception as e:
                print(f"LLM summary generation failed: {e}")

            # Collect statistics regardless of whether LLM summary succeeded
            for activity in filtered_activities:
                # Collect location frequency
                location_type = activity.get('location_type', 'unknown')
                if location_type in patterns['frequent_locations']:
                    patterns['frequent_locations'][location_type] += 1
                else:
                    patterns['frequent_locations'][location_type] = 1
                
                # Get activity type - handle both formats (CSV and LLM generated)
                activity_type = activity.get('activity_type', activity.get('actype', 'unknown'))
                
                # Get start time - handle both formats
                start_time = activity.get('start_time', activity.get('arrtime', ''))
                
                # Collect time preferences
                if activity_type not in patterns['time_preferences']:
                    patterns['time_preferences'][activity_type] = []
                if start_time:
                    patterns['time_preferences'][activity_type].append(start_time)
                
                # Collect travel times - handle both formats
                travel_time = activity.get('travel_time', activity.get('travtime', 0))
                if activity_type not in patterns['travel_times']:
                    patterns['travel_times'][activity_type] = []
                patterns['travel_times'][activity_type].append(travel_time)
                
                # Collect activity durations - handle both formats
                duration = activity.get('activity_duration', activity.get('actdur', 0))
                if not duration and 'start_time' in activity and 'end_time' in activity:
                    # Calculate duration if start and end times are available
                    try:
                        from utils import time_difference_minutes
                        duration = time_difference_minutes(activity['start_time'], activity['end_time'])
                    except:
                        duration = 0
                
                if activity_type not in patterns['activity_durations']:
                    patterns['activity_durations'][activity_type] = []
                patterns['activity_durations'][activity_type].append(duration)
                
                # Collect distance information - handle both formats
                distance = activity.get('distance', 0)
                if activity_type not in patterns['distances']:
                    patterns['distances'][activity_type] = []
                # Only add non-negative distance values
                if distance >= 0:
                    # 优先记录长距离活动，给予更高权重
                    patterns['distances'][activity_type].append(distance)
                
                # Collect transport modes - handle both formats
                mode = activity.get('transport_mode', activity.get('transportmode', 'unknown'))
                # Only record non-None transport modes
                if mode is not None and mode != 'unknown':
                    if mode not in patterns['transport_modes']:
                        patterns['transport_modes'][mode] = 1
                    else:
                        patterns['transport_modes'][mode] += 1
        
        # Clean the memory patterns to remove empty values and zero values before returning
        return self._clean_memory_patterns(patterns)
        
    def _clean_memory_patterns(self, patterns):
        """
        Clean memory patterns by removing empty values and zero values.
        Round decimal values to 3 decimal places for better readability.
        Sort numerical values from largest to smallest.
        
        Args:
            patterns: Dictionary containing memory patterns
            
        Returns:
            dict: Cleaned memory patterns dictionary
        """
        cleaned_patterns = {}
        
        # Process each key in the patterns dictionary
        for key, value in patterns.items():
            if isinstance(value, dict):
                # Clean nested dictionaries
                cleaned_dict = {}
                for k, v in value.items():
                    # Skip empty values and zero values
                    if v and v != 0:
                        cleaned_dict[k] = v
                if cleaned_dict:
                    cleaned_patterns[key] = cleaned_dict
            elif isinstance(value, list):
                # Clean lists based on the key type
                if key == 'summaries':
                    # For summaries, keep non-empty strings
                    cleaned_list = [item for item in value if item]
                else:
                    # For other lists, remove zeros and empty values
                    cleaned_list = []
                    for item in value:
                        if item and item != 0:
                            # Round float values to 3 decimal places if it's in the distances dictionary
                            if key == 'distances' and isinstance(item, float):
                                item = round(item, 3)
                            cleaned_list.append(item)
                
                if cleaned_list:
                    # 对数值型列表进行从大到小排序
                    if key not in ['summaries'] and all(isinstance(item, (int, float)) for item in cleaned_list):
                        cleaned_list = sorted(cleaned_list, reverse=True)
                    cleaned_patterns[key] = cleaned_list
            elif value and value != 0:
                # Keep non-empty and non-zero scalar values
                cleaned_patterns[key] = value
        
        # Special handling for nested dictionaries with lists that might contain zeros
        if 'distances' in cleaned_patterns:
            distances = cleaned_patterns['distances']
            for activity_type, distance_list in list(distances.items()):
                # 处理新格式的距离数据（带权重的字典）
                if isinstance(distance_list[0], dict):
                    # 筛选掉零值
                    filtered_distances = [d for d in distance_list if d['value'] > 0]
                    # 按照value从大到小排序，优先长距离活动
                    sorted_distances = sorted(filtered_distances, key=lambda x: x['value'], reverse=True)
                    if sorted_distances:
                        distances[activity_type] = sorted_distances
                    else:
                        del distances[activity_type]
                else:
                    # 处理旧格式的距离数据（纯数值）
                    # Round all values to 3 decimal places and remove zeros
                    rounded_distances = [round(d, 3) for d in distance_list if d > 0]
                    # 从大到小排序
                    rounded_distances = sorted(rounded_distances, reverse=True)
                    if rounded_distances:
                        distances[activity_type] = rounded_distances
                    else:
                        del distances[activity_type]
                    
        if 'travel_times' in cleaned_patterns:
            travel_times = cleaned_patterns['travel_times']
            for activity_type, time_list in list(travel_times.items()):
                # Remove all zero values
                filtered_times = [t for t in time_list if t > 0]
                # 从大到小排序
                filtered_times = sorted(filtered_times, reverse=True)
                if filtered_times:
                    travel_times[activity_type] = filtered_times
                else:
                    del travel_times[activity_type]
        
        # 处理活动持续时间，从大到小排序
        if 'activity_durations' in cleaned_patterns:
            durations = cleaned_patterns['activity_durations']
            stats_durations = {}
            for activity_type, duration_list in list(durations.items()):
                filtered_durations = [d for d in duration_list if d > 0]
                # 从大到小排序
                filtered_durations = sorted(filtered_durations, reverse=True)
                if filtered_durations:
                    # 计算统计值
                    max_val = max(filtered_durations)
                    min_val = min(filtered_durations)
                    # 计算中位数
                    n = len(filtered_durations)
                    if n % 2 == 0:
                        median_val = (filtered_durations[n//2-1] + filtered_durations[n//2]) / 2
                    else:
                        median_val = filtered_durations[n//2]
                    # 计算平均值
                    mean_val = sum(filtered_durations) / n
                    
                    # 保存原始数据
                    durations[activity_type] = filtered_durations
                    
                    # 保存统计数据
                    stats_durations[activity_type] = {
                        'max': max_val,
                        'min': min_val,
                        'median': round(median_val, 3),
                        'mean': round(mean_val, 3),
                        'values': filtered_durations
                    }
                else:
                    del durations[activity_type]
            
            # 添加统计数据到清理后的模式中
            if stats_durations:
                cleaned_patterns['activity_durations'] = stats_durations
        
        # Make sure we're not storing any activity types with empty lists
        for dict_key in ['time_preferences', 'travel_times', 'distances', 'activity_durations', 'transport_modes']:
            if dict_key in cleaned_patterns:
                nested_dict = cleaned_patterns[dict_key]
                for activity_type in list(nested_dict.keys()):
                    if not nested_dict[activity_type]:
                        del nested_dict[activity_type]
        
        return cleaned_patterns

    def _generate_activities_summary(self, activities, persona=None):
        """
        Generate a summary of activities using LLM.
        
        Args:
            activities: List of activities
            persona: Optional Persona object for additional context
            
        Returns:
            str: Summary text in English
        """
        # Convert activities to JSON string
        activities_json = json.dumps(activities, ensure_ascii=False)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Please summarize the following activities for the day in a concise, coherent paragraph, MUST NOT to exceed 100 words: {activities_json}"}
                ],
                max_tokens=100,  # Reduced for faster response
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return "Unable to generate activity summary."
    
    def generate_daily_schedule(self, persona, date):
        """
        Generate a daily schedule for a given persona using a three-stage approach:
        1. Generate basic daily activities
        2. Determine actual destinations and calculate distances
        3. Refine with accurate travel and transportation details
        
        Args:
            persona: Persona object
            date: Date string (YYYY-MM-DD)
        
        Returns:
            list: List of activities for the day with complete travel details
        """
        
        # Analyze historical memory patterns (if any)
        memory_patterns = None
        if hasattr(persona, 'memory') and persona.memory and persona.memory.days:
            memory_patterns = self.analyze_memory_patterns(persona.memory, persona)
        
        # Check if it's a weekend, get weekend/weekday activity preferences from history
        day_of_week = get_day_of_week(date)
        is_weekend = day_of_week in ['Saturday', 'Sunday']
        
        # Stage 1: Generate basic activities with LLM
        basic_activities = self._generate_activities_with_llm(persona, date, day_of_week, memory_patterns, is_weekend)
        
        print('--------------------------')
        print(basic_activities)
        print('--------------------------')

        # Initialize destination selector if needed
        if not hasattr(self, 'destination_selector'):
            try:
                from destination import Destination
                self.destination_selector = Destination(config=self.config)
            except Exception as e:
                print(f"Error initializing destination selector: {e}")
                # Create a fallback destination selector with empty config if needed
                from destination import Destination
                self.destination_selector = Destination()

        # Stage 2: Calculate actual destinations and distances
        activities_with_destinations = self._add_destinations_and_distances(persona, basic_activities, date, day_of_week, memory_patterns)

        return activities_with_destinations
        
    def _add_destinations_and_distances(self, persona, activities, date, day_of_week, memory_patterns=None):
        """
        Add actual destinations and calculate distances between activities,
        passing memory patterns to the destination selector for LLM analysis
        
        Args:
            persona: Persona object
            activities: List of basic activities
            date: Date string (YYYY-MM-DD)
            day_of_week: Day of week string
            
        Returns:
            list: Activities with destination details and distances
        """
        enhanced_activities = []
        current_location = persona.home  # Start at home
        
        # Process each activity
        for i, activity in enumerate(activities):
            # 1. 处理在家中的睡眠活动
            if activity['activity_type'] == 'sleep' and activity['location_type'] == 'home':
                activity['coordinates'] = persona.home
                activity['distance'] = 0
                activity['travel_time'] = 0
                activity['transport_mode'] = 'No need to travel'
                activity['location_name'] = 'Home'
                enhanced_activities.append(activity)
                continue
            
            # 2. 处理通勤和旅行活动 - 需要分析描述来确定起点和终点
            if activity['activity_type'] in ['commuting', 'travel']:
                # 分析描述来确定起点和终点
                desc = activity['description'].lower()
                
                # 检查描述中是否包含更丰富的信息
                home_indicators = ['home', 'house', 'apartment', 'residence']
                work_indicators = ['work', 'office', 'workplace', 'job', 'company', 'business']
                
                # 确定描述中是否提到了从家到工作的通勤
                from_home_to_work = False
                # 只有当用户有工作时才考虑通勤
                if getattr(persona, 'has_job', True):
                    if any(f"from {hi}" in desc for hi in home_indicators) and any(f"to {wi}" in desc for wi in work_indicators):
                        from_home_to_work = True
                    # 处理"去上班"这种形式的描述
                    elif any(wi in desc for wi in work_indicators) and not any(f"from {wi}" in desc for wi in work_indicators):
                        from_home_to_work = True
                
                # 确定描述中是否提到了从工作到家的通勤
                from_work_to_home = False
                # 只有当用户有工作时才考虑通勤
                if getattr(persona, 'has_job', True):
                    if any(f"from {wi}" in desc for wi in work_indicators) and any(f"to {hi}" in desc for hi in home_indicators):
                        from_work_to_home = True
                    # 处理"回家"这种形式的描述
                    elif any(hi in desc for hi in home_indicators) and not any(f"from {hi}" in desc for hi in home_indicators):
                        from_work_to_home = True
                
                # 从家到工作场所
                if from_home_to_work:
                    distance = calculate_distance(persona.home, persona.work)
                    
                    # 使用destination模块中的方法确定交通方式，保持一致性
                    transport_mode = self.destination_selector._determine_transport_mode(
                        persona,
                        activity['activity_type'],
                        available_minutes=self._calculate_available_time(activities, i),
                        distance=distance,
                        memory_patterns=memory_patterns
                    )
                    
                    activity['coordinates'] = persona.work  # 终点是工作地点
                    activity['distance'] = distance
                    activity['travel_time'] = estimate_travel_time(persona.home, persona.work, transport_mode)[0]
                    activity['from_location'] = persona.home
                    activity['to_location'] = persona.work
                    activity['from_location_name'] = 'Home'
                    activity['to_location_name'] = 'Workplace'
                    activity['location_name'] = 'Workplace'
                    activity['transport_mode'] = transport_mode
                    current_location = persona.work
                
                # 从工作场所到家
                elif from_work_to_home:
                    distance = calculate_distance(persona.work, persona.home)
                    
                    # 使用destination模块中的方法确定交通方式，保持一致性
                    transport_mode = self.destination_selector._determine_transport_mode(
                        persona,
                        activity['activity_type'],
                        available_minutes=self._calculate_available_time(activities, i),
                        distance=distance,
                        memory_patterns=memory_patterns
                    )
                    
                    activity['coordinates'] = persona.home  # 终点是家
                    activity['distance'] = distance
                    activity['travel_time'] = estimate_travel_time(persona.work, persona.home, transport_mode)[0]
                    activity['from_location'] = persona.work
                    activity['to_location'] = persona.home
                    activity['from_location_name'] = 'Workplace'
                    activity['to_location_name'] = 'Home'
                    activity['location_name'] = 'Home'
                    activity['transport_mode'] = transport_mode
                    current_location = persona.home
                
                # 其他类型的旅行，保持原有逻辑，但提供起点和终点
                else:
                    # 使用目标选择器获取目的地信息
                    location, details = self.destination_selector.select_destination(
                        persona,
                        current_location,
                        activity['activity_type'],
                        activity['start_time'],
                        day_of_week,
                        self._calculate_available_time(activities, i),
                        memory_patterns,
                        None,  # location_type_override
                        None   # search_query_override
                    )
                    
                    activity['coordinates'] = location
                    activity['distance'] = details.get('distance', 0)
                    activity['travel_time'] = details.get('travel_time', 0)
                    activity['transport_mode'] = details.get('transport_mode', 'walking')
                    activity['location_name'] = details.get('name', '')
                    activity['from_location'] = current_location
                    activity['to_location'] = location
                    # 为起点和终点添加名称
                    if current_location == persona.home:
                        activity['from_location_name'] = 'Home'
                    elif current_location == persona.work:
                        activity['from_location_name'] = 'Workplace'
                    else:
                        activity['from_location_name'] = 'Previous Location'
                    activity['to_location_name'] = details.get('name', 'Destination')
                    current_location = location
                
                enhanced_activities.append(activity)
                continue
            
            # 3. 处理在工作场所的工作活动 - 直接使用工作场所坐标
            if activity['activity_type'] == 'work' and (
                    activity['location_type'] == 'workplace' or 
                    'at work' in activity['description'].lower() or 
                    'at office' in activity['description'].lower() or
                    'at the office' in activity['description'].lower()) and getattr(persona, 'has_job', True):  # 确保用户有工作
                activity['coordinates'] = persona.work
                
                # 计算从当前位置到工作地点的距离
                distance = 0 if current_location == persona.work else calculate_distance(current_location, persona.work)
                
                # 如果不在同一地点，使用destination模块的方法确定交通方式
                if distance > 0:
                    transport_mode = self.destination_selector._determine_transport_mode(
                        persona,
                        'commuting',  # 使用通勤活动类型来确定交通方式
                        available_minutes=self._calculate_available_time(activities, i),
                        distance=distance,
                        memory_patterns=memory_patterns
                    )
                else:
                    transport_mode = 'No need to travel'
                
                activity['distance'] = distance
                activity['travel_time'] = 0 if distance == 0 else estimate_travel_time(current_location, persona.work, transport_mode)[0]
                activity['transport_mode'] = transport_mode
                activity['location_name'] = 'Workplace'  # 给工作地点一个名称
                current_location = persona.work
                enhanced_activities.append(activity)
                continue
            
            # 4. 处理在家中的活动
            if (activity['location_type'] == 'home' or 
                'at home' in activity['description'].lower() or 
                'home' in activity['description'].lower()):
                activity['coordinates'] = persona.home
                
                # 计算从当前位置到家的距离
                distance = 0 if current_location == persona.home else calculate_distance(current_location, persona.home)
                
                # 如果不在同一地点，使用destination模块的方法确定交通方式
                if distance > 0:
                    transport_mode = self.destination_selector._determine_transport_mode(
                        persona,
                        'commuting',  # 使用通勤活动类型来确定交通方式
                        available_minutes=self._calculate_available_time(activities, i),
                        distance=distance,
                        memory_patterns=memory_patterns
                    )
                else:
                    transport_mode = 'No need to travel'
                
                activity['distance'] = distance
                activity['travel_time'] = 0 if distance == 0 else estimate_travel_time(current_location, persona.home, transport_mode)[0]
                activity['transport_mode'] = transport_mode
                activity['location_name'] = 'Home'
                current_location = persona.home
                enhanced_activities.append(activity)
                continue
            
            # 4.5. 特殊处理步行/散步等户外活动
            if ('walk' in activity['description'].lower() or 
                'stroll' in activity['description'].lower() or 
                'walking' in activity['description'].lower() or
                'jog' in activity['description'].lower() or
                'run' in activity['description'].lower() or
                ('exercise' in activity['description'].lower() and 'neighborhood' in activity['description'].lower())):
                
                # 为散步活动生成一个在周边的步行道路上的位置，而不是在家里
                if current_location == persona.home:
                    # 使用改进的函数生成在家附近0.5-1公里范围内的道路上的随机位置
                    neighborhood_location = generate_random_location_near(persona.home, max_distance_km=0.8, validate=False)
                    activity['coordinates'] = neighborhood_location
                    activity['distance'] = calculate_distance(current_location, neighborhood_location)
                    activity['travel_time'] = estimate_travel_time(current_location, neighborhood_location, 'walking')[0]
                    activity['transport_mode'] = 'walking'  # 这种活动通常是步行的
                    activity['location_name'] = 'Neighborhood Walking Path'
                    current_location = neighborhood_location
                elif current_location == persona.work:
                    # 使用改进的函数生成在工作地附近0.3-0.8公里范围内的道路上的随机位置
                    workplace_area_location = generate_random_location_near(persona.work, max_distance_km=0.6, validate=False)
                    activity['coordinates'] = workplace_area_location
                    activity['distance'] = calculate_distance(current_location, workplace_area_location)
                    activity['travel_time'] = estimate_travel_time(current_location, workplace_area_location, 'walking')[0]
                    activity['transport_mode'] = 'walking'  # 这种活动通常是步行的
                    activity['location_name'] = 'Work Area Walking Path'
                    current_location = workplace_area_location
                else:
                    # 如果当前不在家也不在工作地，就在当前位置附近生成一个道路上的随机位置
                    nearby_location = generate_random_location_near(current_location, max_distance_km=0.5, validate=False)
                    activity['coordinates'] = nearby_location
                    activity['distance'] = calculate_distance(current_location, nearby_location)
                    activity['travel_time'] = estimate_travel_time(current_location, nearby_location, 'walking')[0]
                    activity['transport_mode'] = 'walking'  # 这种活动通常是步行的
                    activity['location_name'] = 'Nearby Walking Path'
                    current_location = nearby_location
                
                enhanced_activities.append(activity)
                continue
                
            # 5. 处理其他活动，通过描述确定更精确的地点类型
            # 提取描述中的关键地点信息
            place_keywords = {
                # 金融服务 (Financial Services)
                'bank': {'place_type': 'bank', 'search_query': 'bank'},
                'atm': {'place_type': 'atm', 'search_query': 'atm'},
                
                # 食品购物 (Food Shopping)
                'grocery': {'place_type': 'grocery_or_supermarket', 'search_query': 'grocery'},
                'supermarket': {'place_type': 'grocery_or_supermarket', 'search_query': 'supermarket'},
                'bakery': {'place_type': 'bakery', 'search_query': 'bakery'},
                'butcher': {'place_type': 'butcher', 'search_query': 'butcher'},
                'deli': {'place_type': 'delicatessen', 'search_query': 'deli'},
                'seafood': {'place_type': 'fishmonger', 'search_query': 'seafood'},
                'greengrocer': {'place_type': 'greengrocer', 'search_query': 'greengrocer'},
                
                # 餐饮场所 (Food & Drink)
                'restaurant': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dining': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dinner': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'lunch': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'breakfast': {'place_type': 'cafe', 'search_query': 'cafe'},
                'cafe': {'place_type': 'cafe', 'search_query': 'cafe'},
                'coffee': {'place_type': 'cafe', 'search_query': 'cafe'},
                'fast food': {'place_type': 'fast_food', 'search_query': 'fast_food'},
                'food court': {'place_type': 'food_court', 'search_query': 'food_court'},
                'pub': {'place_type': 'pub', 'search_query': 'pub'},
                'bar': {'place_type': 'bar', 'search_query': 'bar'},
                'ice cream': {'place_type': 'ice_cream', 'search_query': 'ice_cream'},
                
                # 休闲娱乐 (Leisure & Entertainment)
                'gym': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'fitness': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'workout': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'exercise': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                'park': {'place_type': 'park', 'search_query': 'park'},
                'cinema': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'movie': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'theatre': {'place_type': 'theatre', 'search_query': 'theatre'},
                'sports': {'place_type': 'stadium', 'search_query': 'sports_centre'},
                'swimming': {'place_type': 'swimming_pool', 'search_query': 'swimming_pool'},
                'nightclub': {'place_type': 'night_club', 'search_query': 'nightclub'},
                
                # 文化与教育 (Culture & Education)
                'library': {'place_type': 'library', 'search_query': 'library'},
                'museum': {'place_type': 'museum', 'search_query': 'museum'},
                'art gallery': {'place_type': 'art_gallery', 'search_query': 'gallery'},
                'gallery': {'place_type': 'art_gallery', 'search_query': 'gallery'},
                'school': {'place_type': 'school', 'search_query': 'school'},
                'university': {'place_type': 'university', 'search_query': 'university'},
                'college': {'place_type': 'university', 'search_query': 'college'},
                
                # 医疗健康 (Healthcare)
                'hospital': {'place_type': 'hospital', 'search_query': 'hospital'},
                'doctor': {'place_type': 'doctor', 'search_query': 'clinic'},
                'medical': {'place_type': 'medical_center', 'search_query': 'hospital'},
                'clinic': {'place_type': 'doctor', 'search_query': 'clinic'},
                'dentist': {'place_type': 'dentist', 'search_query': 'dentist'},
                'pharmacy': {'place_type': 'pharmacy', 'search_query': 'chemist'},
                'drugstore': {'place_type': 'pharmacy', 'search_query': 'chemist'},
                'veterinary': {'place_type': 'veterinary_care', 'search_query': 'veterinary'},
                
                # 购物场所 (Shopping)
                'shop': {'place_type': 'store', 'search_query': 'mall'},
                'shopping': {'place_type': 'store', 'search_query': 'mall'},
                'shopping mall': {'place_type': 'store', 'search_query': 'mall'},
                'mall': {'place_type': 'shopping_mall', 'search_query': 'mall'},
                'clothing': {'place_type': 'clothing_store', 'search_query': ' mall'},
                'shoes': {'place_type': 'shoe_store', 'search_query': 'shoes'},
                'electronics': {'place_type': 'electronics_store', 'search_query': 'electronics'},
                'hardware': {'place_type': 'hardware_store', 'search_query': 'hardware'},
                'furniture': {'place_type': 'furniture_store', 'search_query': 'furniture'},
                'bookstore': {'place_type': 'book_store', 'search_query': 'books'},
                'jewelry': {'place_type': 'jewelry_store', 'search_query': 'jewelry'},
                'toys': {'place_type': 'toy_store', 'search_query': 'toys'},
                
                # 个人服务 (Personal Services)
                'hair': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'salon': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'barber': {'place_type': 'beauty_salon', 'search_query': 'beauty'},
                'spa': {'place_type': 'spa', 'search_query': 'beauty'},
                'laundry': {'place_type': 'laundry', 'search_query': 'laundry'},
                'dry cleaning': {'place_type': 'dry_cleaning', 'search_query': 'dry_cleaning'},
                
                # 住宿 (Accommodation)
                'hotel': {'place_type': 'lodging', 'search_query': 'hotel'},
                'motel': {'place_type': 'lodging', 'search_query': 'motel'},
                'hostel': {'place_type': 'lodging', 'search_query': 'hostel'},
                'guest house': {'place_type': 'lodging', 'search_query': 'guest_house'},
                
                # 交通服务 (Transportation)
                'gas station': {'place_type': 'gas_station', 'search_query': 'gas'},
                'car repair': {'place_type': 'car_repair', 'search_query': 'car_repair'},
                'car wash': {'place_type': 'car_wash', 'search_query': 'car_wash'},
                'parking': {'place_type': 'parking', 'search_query': 'parking'},
                'bike parking': {'place_type': 'bicycle_parking', 'search_query': 'bicycle_parking'},
                'bus station': {'place_type': 'bus_station', 'search_query': 'bus_station'},
                'train station': {'place_type': 'train_station', 'search_query': 'station'},
                'subway': {'place_type': 'subway_station', 'search_query': 'subway'},
                'airport': {'place_type': 'airport', 'search_query': 'aerodrome'},
                
                # 日常便利设施 (Convenience)
                'grocery store': {'place_type': 'grocery_or_supermarket', 'search_query': 'retail'},
                'convenience store': {'place_type': 'convenience_store', 'search_query': 'convenience'},
                'post office': {'place_type': 'post_office', 'search_query': 'post_office'},
                
                # 公共服务 (Public Services)
                'police': {'place_type': 'police', 'search_query': 'police'},
                'fire station': {'place_type': 'fire_station', 'search_query': 'fire_station'},
                'town hall': {'place_type': 'city_hall', 'search_query': 'government'},
                'courthouse': {'place_type': 'courthouse', 'search_query': 'courthouse'},
                'embassy': {'place_type': 'embassy', 'search_query': 'diplomatic'},
            }
            
            # 检查描述中是否包含关键场所
            description = activity['description'].lower()
            location_type_override = None
            search_query_override = None
            
            # 首先尝试匹配更具体的多词短语
            multi_word_keywords = [k for k in place_keywords.keys() if ' ' in k]
            for keyword in multi_word_keywords:
                if keyword in description:
                    location_type_override = place_keywords[keyword]['place_type']
                    search_query_override = place_keywords[keyword]['search_query']
                    break
            
            # 如果没有匹配到多词短语，再检查单词匹配
            if not location_type_override:
                for keyword, place_info in place_keywords.items():
                    if ' ' not in keyword and keyword in description:
                        # 避免部分匹配（例如，避免将"background"匹配为"back"）
                        # 检查是否是单词边界
                        word_boundaries = [' ', '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '\n', '\t']
                        
                        # 检查关键词前后是否是单词边界或字符串的开始/结束
                        keyword_positions = [m.start() for m in re.finditer(keyword, description)]
                        for pos in keyword_positions:
                            # 检查前边界
                            before_ok = (pos == 0 or description[pos-1] in word_boundaries)
                            # 检查后边界
                            after_pos = pos + len(keyword)
                            after_ok = (after_pos >= len(description) or description[after_pos] in word_boundaries)
                            
                            if before_ok and after_ok:
                                location_type_override = place_info['place_type']
                                search_query_override = place_info['search_query']
                                break
                                
                    if location_type_override:
                        break
            
            # 如果仍未匹配到关键词，尝试从活动类型和描述进一步分析
            if not location_type_override:
                # 活动类型到可能场所的映射
                activity_to_place_mapping = {
                    'shopping': {'place_type': 'store', 'search_query': 'shop'},
                    'dining': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                    'socializing': {'place_type': 'cafe', 'search_query': 'cafe'},
                    'exercise': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                    'education': {'place_type': 'library', 'search_query': 'library'},
                    'health': {'place_type': 'doctor', 'search_query': 'clinic'},
                    'leisure': {'place_type': 'park', 'search_query': 'park'},
                    'entertainment': {'place_type': 'movie_theater', 'search_query': 'cinema'}
                }
                
                # 从活动类型推断
                act_type = activity.get('activity_type', '').lower()
                if act_type in activity_to_place_mapping:
                    location_type_override = activity_to_place_mapping[act_type]['place_type']
                    search_query_override = activity_to_place_mapping[act_type]['search_query']
                
                # 进一步分析描述中的词语，捕获可能的场所信息
                if not location_type_override:
                    # 提取描述中的名词短语，可能是场所名称
                    place_indicators = [
                        ('store', 'shop'), ('market', 'market'), ('center', 'center'),
                        ('place', 'place'), ('area', 'area'), ('building', 'building'),
                        ('complex', 'complex'), ('venue', 'venue'), ('facility', 'facility')
                    ]
                    
                    # 检查是否有指示地点的词语
                    for indicator, query in place_indicators:
                        if indicator in description:
                            # 提取包含指示词的短语
                            phrases = re.findall(r'\b\w+\s+' + indicator + r'\b|\b' + indicator + r'\s+\w+\b', description)
                            if phrases:
                                # 使用发现的短语作为搜索查询
                                location_type_override = 'point_of_interest'
                                search_query_override = phrases[0]
                                break
            
            # 如果所有尝试都失败，才使用默认值
            if not location_type_override or not search_query_override:
                # 根据描述中的动词推断可能的场所类型
                activity_verbs = {
                    'eat': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                    'dine': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                    'drink': {'place_type': 'cafe', 'search_query': 'cafe'},
                    'shop': {'place_type': 'store', 'search_query': 'shop'},
                    'buy': {'place_type': 'store', 'search_query': 'shop'},
                    'purchase': {'place_type': 'store', 'search_query': 'shop'},
                    'workout': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                    'exercise': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                    'train': {'place_type': 'gym', 'search_query': 'fitness_centre'},
                    'visit': {'place_type': 'point_of_interest', 'search_query': 'attraction'},
                    'meet': {'place_type': 'cafe', 'search_query': 'cafe'},
                    'study': {'place_type': 'library', 'search_query': 'library'},
                    'watch': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                    'play': {'place_type': 'park', 'search_query': 'park'}
                }
                
                # 检查描述中是否包含动词
                for verb, place_info in activity_verbs.items():
                    if re.search(r'\b' + verb + r'(?:ing|s|ed)?\b', description):
                        location_type_override = place_info['place_type']
                        search_query_override = place_info['search_query']
                        break
            
            # 最后的兜底方案，只有在实在无法推断的情况下使用
            if not location_type_override or not search_query_override:
                # 使用默认值但尝试从描述中提取更有意义的搜索词
                location_type_override = 'point_of_interest'
                
                # 从描述中提取名词作为搜索词
                words = description.split()
                if len(words) >= 2:
                    # 使用最长的词作为可能的搜索词，跳过常见的停用词
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'into', 'like', 'through', 'after', 'before', 'between', 'during', 'without', 'of', 'my', 'your', 'his', 'her', 'their', 'our', 'its', 'this', 'that', 'these', 'those'}
                    search_terms = [word for word in words if len(word) > 3 and word not in stop_words]
                    if search_terms:
                        search_terms.sort(key=len, reverse=True)
                        search_query_override = search_terms[0]
                    else:
                        search_query_override = 'place'
                else:
                    search_query_override = 'place'
            
            # 调用目标选择器获取目的地信息
            location, details = self.destination_selector.select_destination(
                persona,
                current_location,
                activity['activity_type'],
                activity['start_time'],
                day_of_week,
                self._calculate_available_time(activities, i),
                memory_patterns,
                location_type_override,
                search_query_override
            )

            # 更新活动信息
            activity['coordinates'] = location
            activity['distance'] = details.get('distance', 0)
            activity['travel_time'] = details.get('travel_time', 0)
            activity['transport_mode'] = details.get('transport_mode', 'walking')
            activity['location_name'] = details.get('name', '')
            
            # 更新当前位置
            current_location = location
            
            # 添加到增强的活动列表中
            enhanced_activities.append(activity)
            
        return enhanced_activities
    
        
    def _calculate_available_time(self, activities, current_index):
        """
        Calculate available time for an activity
        
        Args:
            activities: List of all activities
            current_index: Index of current activity
            
        Returns:
            int: Available time in minutes
        """
        if current_index >= len(activities):
            return 120  # Default 2 hours for invalid index
            
        activity = activities[current_index]
        
        # If this is the last activity
        if current_index == len(activities) - 1:
            return 120  # Default 2 hours for last activity
            
        # Calculate time between this activity and the next
        next_activity = activities[current_index + 1]
        start_time = datetime.strptime(activity['start_time'], '%H:%M')
        end_time = datetime.strptime(next_activity['start_time'], '%H:%M')
        
        # Handle day crossing
        if end_time < start_time:
            end_time += timedelta(days=1)
            
        available_minutes = (end_time - start_time).total_seconds() / 60
        return available_minutes
    
    def _generate_activities_with_llm(self, persona, date, day_of_week, memory_patterns=None, is_weekend=False):
        """Generate activities using LLM, with error handling and retry mechanism"""
                
        # 获取personas的工作状态
        has_job = getattr(persona, 'has_job', True)  # 默认为True，兼容旧数据
        
        # 构建提示，添加工作状态信息
        job_info = ""
        if not has_job:
            job_info = "\nIMPORTANT: This person DOES NOT HAVE A JOB. Do not generate any work-related activities. Focus on activities such as job searching, hobbies, errands, or leisure activities instead."
        
        prompt = ACTIVITY_GENERATION_PROMPT.format(
            gender=persona.gender,
            age=persona.age,
            race=persona.race,
            education=persona.education,
            household_income=persona.get_household_income(),
            household_vehicles=persona.get_household_vehicles(),
            occupation=persona.occupation,
            day_of_week=day_of_week,
            date=date,
            home_location=persona.home,
            work_location=persona.work,
            memory_patterns=memory_patterns
        ) + job_info

        try:
            # Generate activities using LLM
            response = client.chat.completions.create(
                model=self.act_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract activities from the response
            activities = self._extract_activities_from_text(response.choices[0].message.content)

            # Normalize activities to ensure they start at 00:00 and end at 23:59
            activities = self._normalize_daily_activities(activities)

            return activities
            
        except Exception as e:
            print(f"Error generating activities: {e}")
            return self._generate_default_activities(persona)
    
    def _normalize_daily_activities(self, activities):
        """
        Ensure that the day's activities start at 00:00 and end at 23:59.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            list: Normalized list of activities
        """
        if not activities:
            return activities
            
        # Sort activities by start time
        activities = sorted(activities, key=lambda x: x['start_time'])
        
        # Check if the first activity starts at 00:00
        if activities[0]['start_time'] != '00:00':
            # Adjust the first activity to start at 00:00
            activities[0]['start_time'] = '00:00'
        
        # Check if the last activity ends at 23:59
        last_activity = activities[-1]
        if last_activity['end_time'] != '23:59':
            # Add a new activity from the last activity's end time to 23:59
            new_activity = {
                'activity_type': 'leisure',
                'start_time': last_activity['end_time'],
                'end_time': '23:59',
                'description': 'Relaxing at home before bedtime.',
                'location_type': 'home'
            }
            activities.append(new_activity)
        
        return activities
    
    def _format_time(self, time_str):
        """
        Format time string to HH:MM.
        
        Args:
            time_str: Time string from LLM
        
        Returns:
            str: Formatted time string (HH:MM)
        """
        # 首先检查格式为(HH:MM)的时间
        parenthesis_match = re.search(r'\((\d{1,2}):(\d{2})\)', time_str)
        if parenthesis_match:
            hour = int(parenthesis_match.group(1))
            minute = parenthesis_match.group(2)
            return f"{hour:02d}:{minute}"
            
        # First, check if it's an ISO 8601 format datetime string
        iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', time_str)
        if iso_match:
            # Extract time part
            hours, minutes = iso_match.group(4), iso_match.group(5)
            return f"{hours}:{minutes}"
            
        # Try different formats
        formats = [
            '%H:%M',  # 14:30
            '%I:%M %p',  # 2:30 PM
            '%I%p',  # 2PM
            '%I %p',  # 2 PM
            '%Y-%m-%dT%H:%M:%S',  # ISO 8601 format: 2025-02-03T12:00:00
            '%Y-%m-%d %H:%M:%S'   # Date time format: 2025-02-03 12:00:00
        ]
        
        # First, clean up the string
        time_str = time_str.strip().upper().replace('.', '')
        
        # Try to parse with each format
        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.strftime('%H:%M')
            except ValueError:
                continue
            
        # If parsing fails, try to extract time part from string
        time_pattern = re.search(r'(\d{1,2})[:\.](\d{2})(?:\s*([APap][Mm])?)?', time_str)
        if time_pattern:
            hour = int(time_pattern.group(1))
            minute = time_pattern.group(2)
            ampm = time_pattern.group(3)
            
            # Handle AM/PM
            if ampm and ampm.upper() == 'PM' and hour < 12:
                hour += 12
            elif ampm and ampm.upper() == 'AM' and hour == 12:
                hour = 0
                
            return f"{hour:02d}:{minute}"
        
        # If all formats fail, raise ValueError
        raise ValueError(f"Could not parse time string: {time_str}")
    
    def _fix_json_array(self, json_str):
        """
        Fix common JSON formatting issues in the input string
        
        Args:
            json_str: JSON string that may contain formatting errors
            
        Returns:
            str: Fixed JSON string
        """
        try:
            # First try direct parsing
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        # Preprocessing: Handle characters that might cause problems
        # 1. Handle apostrophe issues
        json_str = re.sub(r'(\w)\'s\b', r'\1s', json_str)  # Replace all 's with s
        json_str = re.sub(r'(\w)\'(\w)', r'\1\2', json_str)  # Remove apostrophes in words
        
        # 2. Replace non-standard quotes
        json_str = json_str.replace("'", '"')
        
        # 3. Protect JSON string content
        protected_strings = {}
        string_pattern = r'"([^"]*)"'
        
        def protect_string(match):
            content = match.group(1)
            key = f"__STRING_{len(protected_strings)}__"
            # Clean string content
            content = content.replace('\\', '\\\\')  # Escape backslashes
            content = content.replace('"', '\\"')    # Escape quotes
            content = re.sub(r'[\n\r\t]', ' ', content)  # Replace newlines and tabs
            protected_strings[key] = content
            return f'"{key}"'
        
        # Protect string content
        json_str = re.sub(string_pattern, protect_string, json_str)
        
        # 4. Fix common JSON formatting issues
        # Ensure property names have double quotes
        json_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', json_str)
        
        # Fix commas between objects
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix boolean values and null
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Fix unquoted string values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', json_str)
        
        # Fix commas between properties
        json_str = re.sub(r'"\s*}\s*"', '","', json_str)
        json_str = re.sub(r'"\s*]\s*"', '","', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 5. Restore protected strings
        for key, value in protected_strings.items():
            json_str = json_str.replace(f'"{key}"', f'"{value}"')
        
        # 6. Handle array formatting
        # If it looks like multiple objects but not wrapped in an array
        if json_str.strip().startswith('{') and json_str.strip().endswith('}'):
            if re.search(r'}\s*,\s*{', json_str):
                json_str = f'[{json_str}]'
        
        # 7. Final repair attempts
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # Handle specific error cases
            if 'Expecting \',\' delimiter' in str(e):
                # Try adding comma at error position
                match = re.search(r'line (\d+) column (\d+)', str(e))
                if match:
                    lines = json_str.split('\n')
                    line_num = int(match.group(1)) - 1
                    col_num = int(match.group(2))
                    if 0 <= line_num < len(lines):
                        line = lines[line_num]
                        if col_num < len(line):
                            lines[line_num] = line[:col_num] + ',' + line[col_num:]
                            json_str = '\n'.join(lines)
            
            # If still failing, try extracting and fixing individual objects
            matches = list(re.finditer(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', json_str))
            if matches:
                fixed_objects = []
                for match in matches:
                    obj_str = match.group(0)
                    try:
                        # Validate each object can be parsed
                        json.loads(obj_str)
                        fixed_objects.append(obj_str)
                    except:
                        # If individual object fails parsing, try basic fixes
                        fixed_obj = re.sub(r'([{,])\s*(\w+)(?=\s*:)', r'\1"\2"', obj_str)
                        try:
                            json.loads(fixed_obj)
                            fixed_objects.append(fixed_obj)
                        except:
                            continue
                
                if fixed_objects:
                    return f'[{",".join(fixed_objects)}]'
            
            # If all repair attempts fail, return original string
            return json_str
    
    def _extract_activities_from_text(self, text):
        """
        Extract activities from plain text when JSON parsing fails
        
        Args:
            text: Text containing activity information
            
        Returns:
            list: Extracted activities
        """
        activities = []
        
        # Look for activity blocks
        activity_blocks = re.findall(r'{[^{}]*}', text)
        
        # Try to parse each block
        for block in activity_blocks:
            try:
                # Try to fix and parse each block
                fixed_block = self._fix_json_array(block)
                activity = json.loads(fixed_block)
                
                # Validate required fields
                if all(key in activity for key in ['activity_type', 'start_time', 'end_time', 'description']):
                    activities.append(activity)
            except:
                continue
        
        # If block parsing failed, try line-by-line parsing
        if not activities:
            current_activity = {}
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new activity
                if '"activity_type"' in line or "'activity_type'" in line or "activity_type" in line:
                    # Save the previous activity if it has all required fields
                    if current_activity and all(key in current_activity for key in ['activity_type', 'start_time', 'end_time', 'description']):
                        activities.append(current_activity)
                    # Start a new activity
                    current_activity = {}
                
                # Try to extract key-value pairs
                for key in ['activity_type', 'start_time', 'end_time', 'description', 'transport_mode']:
                    key_patterns = [f'"{key}"', f"'{key}'", key]
                    if any(pattern in line for pattern in key_patterns):
                        # Extract value if there's a colon
                        if ':' in line:
                            value_part = line.split(':', 1)[1].strip()
                            # Clean up the value
                            value = value_part.strip('",\'').strip()
                            if value:
                                current_activity[key] = value
            
            # Add the last activity if it's complete
            if current_activity and all(key in current_activity for key in ['activity_type', 'start_time', 'end_time', 'description']):
                activities.append(current_activity)
        
        return activities
    
    def _generate_default_activities(self, persona):
        """
        Generate default activities if LLM generation fails
        
        Args:
            persona: Persona object
            
        Returns:
            list: List of default activities
        """
        # Basic schedule
        default_activities = [
            {
                'activity_type': 'sleep',
                'start_time': '00:00',
                'end_time': '08:00',
                'description': 'Sleeping at home',
                'location_type': 'home'
            },
            {
                'activity_type': 'work',
                'start_time': '09:00',
                'end_time': '17:00',
                'description': 'Working at office',
                'location_type': 'work'
            },
            {
                'activity_type': 'leisure',
                'start_time': '18:00',
                'end_time': '22:00',
                'description': 'Relaxing at home',
                'location_type': 'home'
            },
            {
                'activity_type': 'sleep',
                'start_time': '22:00',
                'end_time': '23:59',
                'description': 'Sleeping at home',
                'location_type': 'home'
            }
        ]
        
        # Normalize the default activities to ensure they start at 00:00 and end at 23:59
        return self._normalize_daily_activities(default_activities)