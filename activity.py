"""
Activity module for the LLM-based mobility simulation.
Uses LLM to generate and manage daily activity plans.
"""

import json
import re
import openai
import random
import datetime
from config import (
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    ACTIVITY_GENERATION_PROMPT,
    ACTIVITY_REFINEMENT_PROMPT,
    ACTIVITY_TYPES,
    DEEPBRICKS_API_KEY,
    DEEPBRICKS_BASE_URL,
    TRANSPORT_MODES,
    BATCH_PROCESSING,
    BATCH_SIZE
)
from utils import get_day_of_week, normalize_transport_mode, cached

# Create OpenAI client
client = openai.OpenAI(
    api_key = DEEPBRICKS_API_KEY,
    base_url = DEEPBRICKS_BASE_URL,
)

class Activity:
    """
    Manages the generation and processing of daily activity plans.
    """
    
    def __init__(self):
        """Initialize the Activity generator."""
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.activity_queue = []  # For batch processing of activities
    
    @cached
    def analyze_memory_patterns(self, memory):
        """
        Analyze the activity patterns in the historical memory.
        Try to use LLM summary first, fallback to basic statistics if LLM fails.
        
        Args:
            memory: Memory object, containing historical activity records
            
        Returns:
            dict: Dictionary containing activity pattern analysis results
        """
        patterns = {
            'summaries': [],  # LLM generated summaries
            'frequent_locations': {},  # Frequent locations (fallback)
            'time_preferences': {}     # Time preferences (fallback)
        }
        
        # Try LLM summary for each day
        for day in memory.days:
            try:
                summary = self.generate_activities_summary(day['activities'])
                if summary and not summary.startswith("Unable to generate"):
                    patterns['summaries'].append(summary)
                    continue
            except Exception as e:
                print(f"LLM summary generation failed: {e}")
            
            # If LLM failed, collect statistics
            for activity in day['activities']:
                location = activity.get('location_type', 'unknown')
                if location in patterns['frequent_locations']:
                    patterns['frequent_locations'][location] += 1
                else:
                    patterns['frequent_locations'][location] = 1
                    
                activity_type = activity.get('activity_type')
                start_time = activity.get('start_time')
                if activity_type not in patterns['time_preferences']:
                    patterns['time_preferences'][activity_type] = []
                patterns['time_preferences'][activity_type].append(start_time)
        
        return patterns
    
    @cached
    def generate_daily_schedule(self, persona, date):
        """
        Generate a daily schedule for a given persona.
        Added caching to avoid repeated generation under the same conditions
        
        Args:
            persona: Persona object
            date: Date string (YYYY-MM-DD)
        
        Returns:
            list: List of activities for the day
        """
        day_of_week = get_day_of_week(date)
        
        # Analyze historical memory patterns (if any)
        memory_patterns = None
        if hasattr(persona, 'memory') and persona.memory and persona.memory.days:
            memory_patterns = self.analyze_memory_patterns(persona.memory)
        
        # Build cache key
        persona_key = f"{persona.id}:{persona.age}:{persona.gender}:{persona.income}"
        cache_key = f"daily_schedule:{persona_key}:{date}:{day_of_week}"
        
        # Generate schedule
        activities = self._generate_activities_with_llm(persona, date, day_of_week, memory_patterns)

        # Refine and potentially decompose activities
        refined_activities = []
        previous_activity = None
        for activity in activities:
            result = self.refine_activity(persona, activity, date, day_of_week, previous_activity)
            
            # Check if result is a list (decomposed activities) or dict (single refined activity)
            if isinstance(result, list):
                refined_activities.extend(result)  # Add all sub-activities
                previous_activity = result[-1]  # Use the last sub-activity as previous
            else:
                refined_activities.append(result)  # Add single refined activity
                previous_activity = result

        # Validate and correct activities
        validated_activities = self._validate_activities(refined_activities)
        
        return validated_activities
        
    def _generate_activities_with_llm(self, persona, date, day_of_week, memory_patterns=None):
        """Generate activities using LLM, with error handling and retry mechanism"""
        # Prepare the prompt, adding memory pattern information
        prompt = ACTIVITY_GENERATION_PROMPT.format(
            gender=persona.gender,
            age=persona.age,
            income=persona.income,
            consumption=persona.consumption,
            education=persona.education,
            day_of_week=day_of_week,
            date=date,
            home_location=persona.home,
            work_location=persona.work
        )
        
        # Add historical pattern prompts
        if memory_patterns:
            prompt += "\nBased on historical patterns, this person tends to:"
            
            # Add frequently visited places
            if memory_patterns['frequent_locations']:
                top_locations = sorted(
                    memory_patterns['frequent_locations'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                prompt += "\n- Visit these locations frequently: " + ", ".join([f"{loc[0]}" for loc in top_locations])
            
            # Add time preferences
            if memory_patterns['time_preferences']:
                for activity_type, times in memory_patterns['time_preferences'].items():
                    if len(times) >= 2:  # Only add if we have enough data
                        average_time = sum([int(t.split(':')[0]) for t in times]) / len(times)
                        prompt += f"\n- Start {activity_type} activities around {int(average_time)}:00"
        
        max_retries = 2  # 增加重试次数
        for attempt in range(max_retries + 1):
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
                
                # 验证时间连续性
                if activities and self._validate_time_continuity(activities):
                    return activities
                    
                # 如果时间不连续，但还有重试机会
                if attempt < max_retries:
                    # 添加更明确的错误提示到提示中
                    error_details = self._get_time_continuity_errors(activities)
                    prompt += f"\n\nPrevious attempt had time continuity issues: {error_details}\n"
                    prompt += "Please ensure EXACT time continuity between activities.\n"
                    continue
                    
                # 最后一次尝试失败，返回默认活动
                print("Failed to generate time-continuous activities, using default activities")
                return self._generate_default_activities(persona)
                
            except Exception as e:
                print(f"Error generating activities: {e}")
                if attempt < max_retries:
                    print(f"Retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                
                # Last retry failed, return default activities
                print("Error with LLM API, using default activities")
                return self._generate_default_activities(persona)
    
    def _validate_time_continuity(self, activities):
        """
        验证活动列表的时间连续性
        
        Args:
            activities: 活动列表
            
        Returns:
            bool: 是否时间连续
        """
        if not activities:
            return False
            
        # 按开始时间排序
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # 检查第一个活动是否从00:00开始
        first_start = self._format_time(sorted_activities[0].get('start_time', ''))
        if first_start != "00:00":
            return False
            
        # 检查最后一个活动是否在23:59结束
        last_end = self._format_time(sorted_activities[-1].get('end_time', ''))
        if last_end != "23:59":
            return False
            
        # 检查每个活动的结束时间是否等于下一个活动的开始时间
        for i in range(len(sorted_activities) - 1):
            current_end = self._format_time(sorted_activities[i].get('end_time', ''))
            next_start = self._format_time(sorted_activities[i + 1].get('start_time', ''))
            
            if current_end != next_start:
                return False
                
        return True
    
    def _get_time_continuity_errors(self, activities):
        """
        获取时间连续性错误的详细信息
        
        Args:
            activities: 活动列表
            
        Returns:
            str: 错误描述
        """
        if not activities:
            return "No activities generated"
            
        errors = []
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # 检查第一个活动
        first_start = self._format_time(sorted_activities[0].get('start_time', ''))
        if first_start != "00:00":
            errors.append(f"First activity should start at 00:00, not {first_start}")
            
        # 检查最后一个活动
        last_end = self._format_time(sorted_activities[-1].get('end_time', ''))
        if last_end != "23:59":
            errors.append(f"Last activity should end at 23:59, not {last_end}")
            
        # 检查活动之间的连续性
        for i in range(len(sorted_activities) - 1):
            current = sorted_activities[i]
            next_act = sorted_activities[i + 1]
            current_end = self._format_time(current.get('end_time', ''))
            next_start = self._format_time(next_act.get('start_time', ''))
            
            if current_end != next_start:
                errors.append(
                    f"Time gap between activities: {current.get('activity_type')} ends at {current_end} " +
                    f"but {next_act.get('activity_type')} starts at {next_start}"
                )
                
        return "; ".join(errors)
    
    def refine_activity(self, persona, activity, date, day_of_week, previous_activity=None):
        """
        Refine activity, adding more details and potentially adding transportation mode
        
        Args:
            persona: Persona object
            activity: Activity dictionary
            date: Date string (YYYY-MM-DD)
            day_of_week: Day of week
            previous_activity: Previous activity dictionary (optional)
        
        Returns:
            dict: Refined activity with transport_mode if needed
        """

        # Skip refinement for sleep activities (they don't need much detail)
        if activity.get('activity_type') in ['sleep', 'commuting', 'travel', 'work']:
            return activity
            
        # Check if the activity requires transportation based on previous and current locations
        requires_transport = False
        if previous_activity:
            prev_location = previous_activity.get('location_type', '')
            current_location = activity.get('location_type', '')
            requires_transport = self._needs_transportation(prev_location, current_location, activity.get('activity_type', ''))
        
        # If transport isn't required, just return the original activity
        if not requires_transport:
            return activity
            
        # Build the prompt
        prompt = ACTIVITY_REFINEMENT_PROMPT.format(
            gender=persona.gender,
            age=persona.age,
            income=persona.income,
            consumption=persona.consumption,
            education=persona.education,
            date=date,
            day_of_week=day_of_week,
            activity_description=activity.get('description', ''),
            location_type=activity.get('location_type', ''),
            start_time=activity.get('start_time', ''),
            end_time=activity.get('end_time', ''),
            previous_activity_type=previous_activity.get('activity_type', '') if previous_activity else '',
            previous_location=previous_activity.get('location_type', '') if previous_activity else '',
            previous_end_time=previous_activity.get('end_time', '') if previous_activity else '',
            requires_transportation=str(requires_transport).lower()
        )
        
        try:
            # Use LLM to refine the activity
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Extract the refined activity
            content = response.choices[0].message.content
            
            # Try to parse the JSON response
            try:
                # Find JSON object in the text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx == -1 or end_idx <= start_idx:
                    return activity
                    
                json_str = content[start_idx:end_idx]
                # Fix common JSON formatting issues
                fixed_json = self._fix_json_array(json_str)
                
                try:
                    refined_data = json.loads(fixed_json)
                    
                    # Handle single refined activity
                    refined_activity = self._create_refined_activity(refined_data, activity)
                    return refined_activity
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse refined activity JSON: {e}")
                    return activity
                    
            except Exception as e:
                print(f"Error processing LLM refinement response: {e}")
                return activity
            
        except Exception as e:
            print(f"Error refining activity: {e}")
            return activity
    
    def _create_refined_activity(self, refined_data, original_activity):
        """
        Create a refined activity from LLM response data and original activity
        
        Args:
            refined_data: Dict containing refined activity data
            original_activity: Original activity dict
            
        Returns:
            dict: Refined activity
        """
        # Start with a copy of the original activity
        refined_activity = original_activity.copy()
        
        # Update activity description if provided
        if 'description' in refined_data and refined_data['description']:
            refined_activity['description'] = refined_data.get('description')
        
        # Add transport mode if specified
        if 'transport_mode' in refined_data and refined_data['transport_mode']:
            # Normalize transport mode to ensure consistency
            refined_activity['transport_mode'] = normalize_transport_mode(refined_data.get('transport_mode'))
        
        # Optionally update activity type if the refinement changes it
        if 'activity_type' in refined_data and refined_data['activity_type'] in ACTIVITY_TYPES:
            refined_activity['activity_type'] = refined_data.get('activity_type')
        
        return refined_activity
            
    def _format_time_after_minutes(self, start_time, minutes):
        """
        计算给定开始时间后指定分钟数的时间
        
        Args:
            start_time: 开始时间 (HH:MM)
            minutes: 分钟数
            
        Returns:
            str: 计算后的时间 (HH:MM)
        """
        try:
            start_hour, start_minute = map(int, start_time.split(':'))
            total_minutes = start_hour * 60 + start_minute + minutes
            
            # 确保不超过23:59
            if total_minutes >= 24 * 60:
                total_minutes = 23 * 60 + 59
                
            new_hour = total_minutes // 60
            new_minute = total_minutes % 60
            
            return f"{new_hour:02d}:{new_minute:02d}"
        except:
            return start_time
    
    def _validate_activities(self, activities):
        """
        Validate and clean activities data.
        Support handling decomposed sub-activities, ensuring time continuity and correct activity types.
        
        Args:
            activities: List of activity dictionaries (including normal activities and decomposed sub-activities)
            
        Returns:
            list: Validated activities
        """
        if not activities:
            return []
            
        # First sort by start time to ensure proper time sequence
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # 添加睡眠时间并确保时间连续性
        final_activities = []
        last_end_time = "00:00"
        
        # 如果第一个活动不是从00:00开始，添加睡眠时间
        if sorted_activities and self._format_time(sorted_activities[0].get('start_time')) != "00:00":
            final_activities.append({
                'activity_type': 'sleep',
                'start_time': '00:00',
                'end_time': sorted_activities[0].get('start_time'),
                'description': 'Sleeping at home',
                'location_type': 'home'
            })
        
        # 预处理：调整travel活动的时间
        processed_activities = []
        for i, activity in enumerate(sorted_activities):
            if activity['activity_type'].lower() == 'travel':
                # 查找下一个非travel活动
                next_activity = None
                for next_act in sorted_activities[i+1:]:
                    if next_act['activity_type'].lower() != 'travel':
                        next_activity = next_act
                        break
                
                if next_activity:
                    # 调整travel活动的结束时间为下一个活动的开始时间
                    activity = activity.copy()
                    activity['end_time'] = next_activity['start_time']
            
            processed_activities.append(activity)
        
        # 使用处理后的活动列表
        sorted_activities = processed_activities
        
        for i, activity in enumerate(sorted_activities):
            # Skip invalid activities
            if 'activity_type' not in activity or 'start_time' not in activity or 'end_time' not in activity:
                continue
                
            # Get activity type and ensure it's valid
            activity_type = activity['activity_type'].lower()
            
            # Format times consistently
            try:
                start_time = self._format_time(activity['start_time'])
                end_time = self._format_time(activity['end_time'])
                
                # 确保结束时间不超过23:59
                if end_time > "23:59":
                    end_time = "23:59"
                    
            except:
                # Skip activities with invalid times
                continue
            
            # 创建更新后的活动
            updated_activity = {
                **activity,
                'activity_type': activity_type,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # 检查是否需要填补时间空隙
            if start_time > last_end_time:
                # 添加睡眠时间（凌晨或晚上）
                if (int(start_time.split(':')[0]) < 6 or  # 凌晨时间
                    int(last_end_time.split(':')[0]) >= 22):  # 晚上时间
                    gap_activity = {
                        'activity_type': 'sleep',
                        'start_time': last_end_time,
                        'end_time': start_time,
                        'description': 'Sleeping at home',
                        'location_type': 'home'
                    }
                    final_activities.append(gap_activity)
            
            # 检查与最后一个活动的重叠
            if final_activities:
                last_activity = final_activities[-1]
                if start_time < last_activity['end_time']:
                    # 处理重叠情况
                    if activity_type == last_activity['activity_type']:
                        # 如果是相同类型的活动，合并它们
                        last_activity['end_time'] = max(last_activity['end_time'], end_time)
                        continue
                    elif activity_type == 'travel' or last_activity['activity_type'] == 'travel':
                        # 如果其中一个是travel活动，调整时间
                        if activity_type == 'travel':
                            # travel活动开始时间为上一个活动的结束时间
                            updated_activity['start_time'] = last_activity['end_time']
                        else:
                            # 非travel活动开始时间为travel活动的结束时间
                            last_activity['end_time'] = start_time
                    elif activity_type == 'sleep' or last_activity['activity_type'] == 'sleep':
                        # 如果其中一个是睡眠活动，保留睡眠活动
                        if activity_type == 'sleep':
                            # 更新最后一个活动的结束时间
                            last_activity['end_time'] = start_time
                            final_activities.append(updated_activity)
                        continue
                    else:
                        # 调整当前活动的开始时间
                        start_time = last_activity['end_time']
                        updated_activity['start_time'] = start_time
                        
                        # 如果调整后开始时间晚于或等于结束时间，跳过此活动
                        if start_time >= end_time:
                            continue
            
            final_activities.append(updated_activity)
            last_end_time = end_time
        
        # 如果最后一个活动不是在23:59结束，添加睡眠时间
        if final_activities and final_activities[-1]['end_time'] != "23:59":
            final_activities.append({
                'activity_type': 'sleep',
                'start_time': final_activities[-1]['end_time'],
                'end_time': '23:59',
                'description': 'Sleeping at home',
                'location_type': 'home'
            })
        
        # 合并连续的相同类型活动
        merged_activities = []
        current_activity = None
        
        for activity in final_activities:
            if not current_activity:
                current_activity = activity.copy()
                continue
                
            if (current_activity['activity_type'] == activity['activity_type'] and
                current_activity['end_time'] == activity['start_time']):
                # 合并连续的相同类型活动
                current_activity['end_time'] = activity['end_time']
            else:
                merged_activities.append(current_activity)
                current_activity = activity.copy()
        
        if current_activity:
            merged_activities.append(current_activity)
        
        # 最后按开始时间重新排序
        return sorted(merged_activities, key=lambda x: x['start_time'])
        
    def _calculate_duration_minutes(self, start_time, end_time):
        """
        计算两个时间点之间的分钟差
        
        Args:
            start_time: 开始时间 (HH:MM)
            end_time: 结束时间 (HH:MM)
            
        Returns:
            int: 分钟差
        """
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        
        # 处理跨天的情况
        if end_hour < start_hour or (end_hour == start_hour and end_minute < start_minute):
            end_hour += 24
            
        return (end_hour - start_hour) * 60 + (end_minute - start_minute)
    
    def _correct_activity_type_based_on_description(self, activity_type, description):
        """
        Correct activity type based on description
        
        Args:
            activity_type: Original activity type
            description: Activity description
            
        Returns:
            str: Corrected activity type
        """
        description = description.lower()
        
        # Keyword mapping to activity types
        keyword_mapping = {
            'sleep': ['sleep', 'nap', 'bed', 'rest'],
            'work': ['work', 'meeting', 'office', 'job', 'task', 'email', 'project', 'client', 'presentation'],
            'shopping': ['shop', 'store', 'mall', 'purchase', 'buy', 'grocery', 'supermarket'],
            'dining': ['eat', 'lunch', 'dinner', 'breakfast', 'restaurant', 'cafe', 'meal', 'food', 'brunch'],
            'recreation': ['exercise', 'gym', 'workout', 'sport', 'run', 'jog', 'swim', 'yoga', 'fitness'],
            'healthcare': ['doctor', 'dentist', 'medical', 'health', 'therapy', 'hospital', 'clinic'],
            'social': ['friend', 'party', 'gathering', 'meet up', 'social', 'visit', 'guest'],
            'education': ['class', 'study', 'learn', 'school', 'course', 'lecture', 'university', 'college'],
            'leisure': ['relax', 'tv', 'movie', 'read', 'book', 'game', 'hobby', 'leisure'],
            'errands': ['errand', 'bank', 'post', 'atm', 'dry clean', 'pick up'],
            'home': ['home', 'house', 'apartment', 'clean', 'cook', 'laundry', 'chore']
        }
        
        # Check if description contains keywords for specific activity types
        best_match = None
        max_matches = 0
        
        for type_name, keywords in keyword_mapping.items():
            matches = sum(1 for keyword in keywords if keyword in description)
            if matches > max_matches:
                max_matches = matches
                best_match = type_name
        
        # Only modify if there's a clear match and the original type is incorrect
        if max_matches > 0 and best_match != activity_type:
            return best_match
        
        return activity_type
    
    def _needs_transportation(self, previous_location, current_location, activity_type):
        """
        Determine if an activity requires transportation
        
        Args:
            previous_location: Previous activity location type
            current_location: Current activity location type
            activity_type: Activity type
            
        Returns:
            bool: Whether transportation is required
        """
        # 工作活动永远不需要显示交通信息
        if activity_type == 'work':
            return False
            
        # If the activity is sleep or home, it usually doesn't require transportation
        if activity_type in ['sleep', 'home']:
            return False
        
        # If the locations are the same, no transportation is needed
        if previous_location and current_location and previous_location == current_location:
            return False
        
        # By default, assume transportation is needed
        return True
    
    def _format_time(self, time_str):
        """
        Format time string to HH:MM.
        
        Args:
            time_str: Time string from LLM
        
        Returns:
            str: Formatted time string (HH:MM)
        """
        # 首先，检查是否为ISO 8601格式的日期时间字符串
        iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', time_str)
        if iso_match:
            # 提取时间部分
            hours, minutes = iso_match.group(4), iso_match.group(5)
            return f"{hours}:{minutes}"
            
        # Try different formats
        formats = [
            '%H:%M',  # 14:30
            '%I:%M %p',  # 2:30 PM
            '%I%p',  # 2PM
            '%I %p',  # 2 PM
            '%Y-%m-%dT%H:%M:%S',  # ISO 8601格式: 2025-02-03T12:00:00
            '%Y-%m-%d %H:%M:%S'   # 日期时间格式: 2025-02-03 12:00:00
        ]
        
        # First, clean up the string
        time_str = time_str.strip().upper().replace('.', '')
        
        # Try to parse with each format
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(time_str, fmt)
                return dt.strftime('%H:%M')
            except ValueError:
                continue
            
        # 如果无法解析，尝试从字符串中提取时间部分
        time_pattern = re.search(r'(\d{1,2})[:\.](\d{2})(?:\s*([APap][Mm])?)?', time_str)
        if time_pattern:
            hour = int(time_pattern.group(1))
            minute = time_pattern.group(2)
            ampm = time_pattern.group(3)
            
            # 处理AM/PM
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
            # 首先尝试直接解析
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        # 预处理：处理可能导致问题的字符
        # 1. 处理撇号问题
        json_str = re.sub(r'(\w)\'s\b', r'\1s', json_str)  # 替换所有的's为s
        json_str = re.sub(r'(\w)\'(\w)', r'\1\2', json_str)  # 移除词中的撇号
        
        # 2. 替换不规范的引号
        json_str = json_str.replace("'", '"')
        
        # 3. 保护JSON字符串中的内容
        protected_strings = {}
        string_pattern = r'"([^"]*)"'
        
        def protect_string(match):
            content = match.group(1)
            key = f"__STRING_{len(protected_strings)}__"
            # 清理字符串内容
            content = content.replace('\\', '\\\\')  # 转义反斜杠
            content = content.replace('"', '\\"')    # 转义双引号
            content = re.sub(r'[\n\r\t]', ' ', content)  # 替换换行和制表符
            protected_strings[key] = content
            return f'"{key}"'
        
        # 保护字符串内容
        json_str = re.sub(string_pattern, protect_string, json_str)
        
        # 4. 修复常见的JSON格式问题
        # 确保属性名有双引号
        json_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', json_str)
        
        # 修复对象之间的逗号
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # 修复布尔值和null
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # 修复未加引号的字符串值
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', json_str)
        
        # 修复属性之间的逗号
        json_str = re.sub(r'"\s*}\s*"', '","', json_str)
        json_str = re.sub(r'"\s*]\s*"', '","', json_str)
        
        # 修复尾部逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 5. 恢复被保护的字符串
        for key, value in protected_strings.items():
            json_str = json_str.replace(f'"{key}"', f'"{value}"')
        
        # 6. 处理数组格式
        # 如果看起来是多个对象但没有被数组包装
        if json_str.strip().startswith('{') and json_str.strip().endswith('}'):
            if re.search(r'}\s*,\s*{', json_str):
                json_str = f'[{json_str}]'
        
        # 7. 最后的修复尝试
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # 处理特定的错误情况
            if 'Expecting \',\' delimiter' in str(e):
                # 尝试在错误位置添加逗号
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
            
            # 如果仍然失败，尝试提取和修复单个对象
            matches = list(re.finditer(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', json_str))
            if matches:
                fixed_objects = []
                for match in matches:
                    obj_str = match.group(0)
                    try:
                        # 验证每个对象是否可以解析
                        json.loads(obj_str)
                        fixed_objects.append(obj_str)
                    except:
                        # 如果单个对象解析失败，尝试基本的修复
                        fixed_obj = re.sub(r'([{,])\s*(\w+)(?=\s*:)', r'\1"\2"', obj_str)
                        try:
                            json.loads(fixed_obj)
                            fixed_objects.append(fixed_obj)
                        except:
                            continue
                
                if fixed_objects:
                    return f'[{",".join(fixed_objects)}]'
            
            # 如果所有修复尝试都失败，返回原始字符串
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
    
    def _is_valid_activity(self, activity):
        """
        Check if an activity dictionary is valid
        
        Args:
            activity: Activity dictionary
            
        Returns:
            bool: Whether the activity is valid
        """
        # Check required fields
        required_fields = ['activity_type', 'start_time', 'end_time', 'description']
        if not all(field in activity for field in required_fields):
            return False
        
        # Check activity type
        activity_type = activity.get('activity_type', '').lower()
        if activity_type not in [t.lower() for t in ACTIVITY_TYPES]:
            return False
        
        return True
    
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
                'start_time': '00:00',
                'end_time': '07:00',
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
                'end_time': '21:00',
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
    
    @cached
    def generate_activities_summary(self, activities, persona=None):
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
                max_tokens=150,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return "Unable to generate activity summary."