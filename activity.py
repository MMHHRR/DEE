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
    LLM_MODEL,
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
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.activity_queue = []  # For batch processing of activities
        self.config = config  # Store config for destination selector
    
    @cached
    def analyze_memory_patterns(self, memory, persona=None):
        """
        Analyze the activity patterns in the historical memory.
        Try to use LLM summary first, fallback to basic statistics if LLM fails.
        
        Args:
            memory: Memory object, containing historical activity records
            persona: Optional Persona object for additional context
            
        Returns:
            dict: Dictionary containing activity pattern analysis results
        """
        patterns = {
            'summaries': [],  # LLM generated summaries
            'frequent_locations': {},  # Frequent locations (fallback)
            'time_preferences': {},    # Time preferences (fallback)
            'travel_times': {},        # Travel times for different activities
            'activity_durations': {},  # Duration of different activities
            'distances': {},           # Travel distances
            'transport_modes': {}      # Transport mode preferences
        }
        
        # Try LLM summary for each day
        for day in memory.days:
            # Filter out 3 AM data
            filtered_activities = [activity for activity in day['activities'] if activity.get('start_time') != '03:00']
            day['activities'] = filtered_activities
            try:
                summary = self._generate_activities_summary(filtered_activities)
                if summary and not summary.startswith("Unable to generate"):
                    patterns['summaries'].append(summary)
            except Exception as e:
                print(f"LLM summary generation failed: {e}")

            # 无论LLM是否成功,都收集统计信息
            for activity in filtered_activities:
                # 收集位置频率
                location = activity.get('location_type', 'unknown')
                if location in patterns['frequent_locations']:
                    patterns['frequent_locations'][location] += 1
                else:
                    patterns['frequent_locations'][location] = 1
                    
                activity_type = activity.get('activity_type')
                start_time = activity.get('start_time')
                
                # 收集时间偏好
                if activity_type not in patterns['time_preferences']:
                    patterns['time_preferences'][activity_type] = []
                patterns['time_preferences'][activity_type].append(start_time)
                
                # 收集行程时间
                travel_time = activity.get('travel_time', 0)
                if activity_type not in patterns['travel_times']:
                    patterns['travel_times'][activity_type] = []
                patterns['travel_times'][activity_type].append(travel_time)
                
                # 收集活动持续时间
                duration = activity.get('activity_duration', 0)
                if activity_type not in patterns['activity_durations']:
                    patterns['activity_durations'][activity_type] = []
                patterns['activity_durations'][activity_type].append(duration)
                
                # 收集距离信息
                distance = activity.get('distance', 0)
                if activity_type not in patterns['distances']:
                    patterns['distances'][activity_type] = []
                # 只添加非负的距离值
                if distance >= 0:
                    patterns['distances'][activity_type].append(distance)
                
                # 收集交通方式
                mode = activity.get('transport_mode', 'unknown')
                # 只记录非None的交通方式
                if mode is not None and mode != 'unknown':
                    if mode not in patterns['transport_modes']:
                        patterns['transport_modes'][mode] = 1
                    else:
                        patterns['transport_modes'][mode] += 1
        
        return patterns

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
                    {"role": "user", "content": f"Please summarize the following activities for the day in a concise, coherent paragraph: {activities_json}"}
                ],
                max_tokens=100,  # Reduced for faster response
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return "Unable to generate activity summary."
    
    @cached
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
                        memory_patterns
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
            
            # 3. 处理在工作场所的工作活动 - 直接使用工作场所坐标，不需要调用谷歌地图API
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
                    neighborhood_location = generate_random_location_near(persona.home, max_distance_km=0.8)
                    activity['coordinates'] = neighborhood_location
                    activity['distance'] = calculate_distance(current_location, neighborhood_location)
                    activity['travel_time'] = estimate_travel_time(current_location, neighborhood_location, 'walking')[0]
                    activity['transport_mode'] = 'walking'  # 这种活动通常是步行的
                    activity['location_name'] = 'Neighborhood Walking Path'
                    current_location = neighborhood_location
                elif current_location == persona.work:
                    # 使用改进的函数生成在工作地附近0.3-0.8公里范围内的道路上的随机位置
                    workplace_area_location = generate_random_location_near(persona.work, max_distance_km=0.6)
                    activity['coordinates'] = workplace_area_location
                    activity['distance'] = calculate_distance(current_location, workplace_area_location)
                    activity['travel_time'] = estimate_travel_time(current_location, workplace_area_location, 'walking')[0]
                    activity['transport_mode'] = 'walking'  # 这种活动通常是步行的
                    activity['location_name'] = 'Work Area Walking Path'
                    current_location = workplace_area_location
                else:
                    # 如果当前不在家也不在工作地，就在当前位置附近生成一个道路上的随机位置
                    nearby_location = generate_random_location_near(current_location, max_distance_km=0.5)
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
                'bank': {'place_type': 'bank', 'search_query': 'bank financial institution'},
                'grocery': {'place_type': 'grocery_or_supermarket', 'search_query': 'grocery store'},
                'supermarket': {'place_type': 'grocery_or_supermarket', 'search_query': 'supermarket'},
                'restaurant': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dining': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'dinner': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                'lunch': {'place_type': 'restaurant', 'search_query': 'restaurant cafe'},
                'breakfast': {'place_type': 'cafe', 'search_query': 'breakfast cafe'},
                'cafe': {'place_type': 'cafe', 'search_query': 'cafe'},
                'coffee': {'place_type': 'cafe', 'search_query': 'coffee shop'},
                'gym': {'place_type': 'gym', 'search_query': 'gym fitness'},
                'fitness': {'place_type': 'gym', 'search_query': 'fitness center'},
                'workout': {'place_type': 'gym', 'search_query': 'gym'},
                'exercise': {'place_type': 'gym', 'search_query': 'fitness center'},
                'park': {'place_type': 'park', 'search_query': 'park'},
                'cinema': {'place_type': 'movie_theater', 'search_query': 'cinema'},
                'movie': {'place_type': 'movie_theater', 'search_query': 'movie theater'},
                'library': {'place_type': 'library', 'search_query': 'library'},
                'hospital': {'place_type': 'hospital', 'search_query': 'hospital'},
                'doctor': {'place_type': 'doctor', 'search_query': 'doctor clinic'},
                'medical': {'place_type': 'doctor', 'search_query': 'medical center'},
                'clinic': {'place_type': 'doctor', 'search_query': 'medical clinic'},
                'shop': {'place_type': 'store', 'search_query': 'retail store'},
                'shopping': {'place_type': 'store', 'search_query': 'shopping retail'},
                'mall': {'place_type': 'shopping_mall', 'search_query': 'shopping mall'},
                'bar': {'place_type': 'bar', 'search_query': 'bar pub'},
                'pub': {'place_type': 'bar', 'search_query': 'pub'},
                'school': {'place_type': 'school', 'search_query': 'school'},
                'university': {'place_type': 'university', 'search_query': 'university'},
                'college': {'place_type': 'university', 'search_query': 'college'},
                'hair': {'place_type': 'beauty_salon', 'search_query': 'hair salon'},
                'salon': {'place_type': 'beauty_salon', 'search_query': 'beauty salon'},
                'barber': {'place_type': 'beauty_salon', 'search_query': 'barber shop'},
                'grocery store': {'place_type': 'grocery_or_supermarket', 'search_query': 'grocery store'},
                'convenience store': {'place_type': 'convenience_store', 'search_query': 'convenience store'},
                'gas station': {'place_type': 'gas_station', 'search_query': 'gas station'},
                'pharmacy': {'place_type': 'pharmacy', 'search_query': 'pharmacy drugstore'},
                'drugstore': {'place_type': 'pharmacy', 'search_query': 'drugstore'},
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
            
            # 对于特定类型的活动，增加特定地点的匹配可能性
            if not location_type_override:
                activity_specific_places = {
                    'dining': {'place_type': 'restaurant', 'search_query': 'restaurant cafe'},
                    'shopping': {'place_type': 'store', 'search_query': 'shopping retail'},
                    'recreation': {'place_type': 'park', 'search_query': 'park recreation area'},
                    'leisure': {'place_type': 'point_of_interest', 'search_query': 'entertainment venue'},
                    'healthcare': {'place_type': 'doctor', 'search_query': 'doctor medical center'},
                    'education': {'place_type': 'school', 'search_query': 'education school'},
                    'social': {'place_type': 'bar', 'search_query': 'social venue cafe'},
                    'errands': {'place_type': 'store', 'search_query': 'convenience services'}
                }
                
                if activity['activity_type'] in activity_specific_places:
                    location_type_override = activity_specific_places[activity['activity_type']]['place_type']
                    search_query_override = activity_specific_places[activity['activity_type']]['search_query']
            
            # 使用location_type字段作为备选
            if not location_type_override and activity['location_type'] and activity['location_type'] != 'other':
                # 地点类型映射
                location_type_map = {
                    'bank': {'place_type': 'bank', 'search_query': 'bank'},
                    'restaurant': {'place_type': 'restaurant', 'search_query': 'restaurant'},
                    'cafe': {'place_type': 'cafe', 'search_query': 'cafe'},
                    'gym': {'place_type': 'gym', 'search_query': 'gym'},
                    'park': {'place_type': 'park', 'search_query': 'park'},
                    'store': {'place_type': 'store', 'search_query': 'store'},
                    'shopping_mall': {'place_type': 'shopping_mall', 'search_query': 'shopping mall'},
                    'hospital': {'place_type': 'hospital', 'search_query': 'hospital'},
                    'doctor': {'place_type': 'doctor', 'search_query': 'doctor clinic'},
                    'school': {'place_type': 'school', 'search_query': 'school'},
                    'university': {'place_type': 'university', 'search_query': 'university'},
                    'library': {'place_type': 'library', 'search_query': 'library'},
                    'bar': {'place_type': 'bar', 'search_query': 'bar pub'},
                    'beauty_salon': {'place_type': 'beauty_salon', 'search_query': 'beauty salon'},
                    'pharmacy': {'place_type': 'pharmacy', 'search_query': 'pharmacy'}
                }
                
                if activity['location_type'] in location_type_map:
                    location_type_override = location_type_map[activity['location_type']]['place_type']
                    search_query_override = location_type_map[activity['location_type']]['search_query']
                else:
                    # 默认使用location_type作为搜索词
                    location_type_override = 'point_of_interest'
                    search_query_override = activity['location_type']
            
            # 调用目标选择器获取目的地信息
            destination_params = {}
            if location_type_override:
                destination_params = {
                    'place_type': location_type_override,
                    'search_query': search_query_override,
                    'distance_preference': 3,  # 默认中等距离偏好
                    'price_level': 2  # 默认中等价格水平
                }
                # print(f"为活动类型 {activity['activity_type']} 确定了地点类型: {location_type_override}, 搜索词: {search_query_override}")
                
            location, details = self.destination_selector.select_destination(
                persona,
                current_location,
                activity['activity_type'],
                activity['start_time'],
                day_of_week,
                self._calculate_available_time(activities, i),
                memory_patterns,
                location_type_override=destination_params if destination_params else None
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
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract activities from the response
            activities = self._extract_activities_from_text(response.choices[0].message.content)

            return activities
            
        except Exception as e:
            print(f"Error generating activities: {e}")
            return self._generate_default_activities(persona)
    
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
        return [
            {
                'activity_type': 'sleep',
                'start_time': '03:00',
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
                'end_time': '27:00',
                'description': 'Sleeping at home',
                'location_type': 'home'
            }
        ]