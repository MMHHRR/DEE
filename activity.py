"""
Activity module for the LLM-based mobility simulation.
Uses LLM to generate and manage daily activity plans.
"""

import json
import re
import openai
import random
import datetime
import pandas as pd
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
from utils import get_day_of_week, normalize_transport_mode, cached, time_to_minutes

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
            # 过滤掉凌晨3点的数据
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
                max_tokens=150,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return "Unable to generate activity summary."
    
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
            memory_patterns = self.analyze_memory_patterns(persona.memory, persona)
        
        # Build cache key with more demographic details
        persona_key = f"{persona.id}:{persona.age}:{persona.gender}:{persona.education}:{persona.occupation}:{persona.race}:{persona.home}:{persona.work}"

        cache_key = f"daily_schedule:{persona_key}:{date}:{day_of_week}"
        
        # Check if it's a weekend, get weekend/weekday activity preferences from history
        is_weekend = day_of_week in ['Saturday', 'Sunday']
        
        # Generate schedule
        activities = self._generate_activities_with_llm(
            persona, 
            date, 
            day_of_week, 
            memory_patterns, 
            is_weekend=is_weekend,
        )

        print("---------------------------------------------------")
        print(activities)
        print("---------------------------------------------------")

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
        
    def _generate_activities_with_llm(self, persona, date, day_of_week, memory_patterns=None, is_weekend=False):
        """Generate activities using LLM, with error handling and retry mechanism"""
                
        prompt = ACTIVITY_GENERATION_PROMPT.format(
            gender=persona.gender,
            age=persona.age,
            race=persona.race,
            education=persona.education,
            occupation=persona.occupation,
            day_of_week=day_of_week,
            date=date,
            home_location=persona.home,
            work_location=persona.work,
            memory_patterns=memory_patterns
        )

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
        
    
    def _validate_time_continuity(self, activities):
        """
        Validate time continuity of activity list
        
        Args:
            activities: List of activities
            
        Returns:
            bool: Whether time is continuous
        """
        if not activities:
            return False
            
        # Sort by start time
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # Check if first activity starts at 03:00
        first_start = self._format_time(sorted_activities[0].get('start_time', ''))
        if first_start != "00:00":
            return False
            
        # Check if last activity ends at 27:00 (next day 03:00)
        last_end = self._format_time(sorted_activities[-1].get('end_time', ''))
        if last_end != "27:00":
            return False
            
        # Check if each activity's end time equals the next activity's start time
        for i in range(len(sorted_activities) - 1):
            current_end = self._format_time(sorted_activities[i].get('end_time', ''))
            next_start = self._format_time(sorted_activities[i + 1].get('start_time', ''))
            
            if current_end != next_start:
                return False
                
        return True
    
    def _get_time_continuity_errors(self, activities):
        """
        Get detailed information about time continuity errors
        
        Args:
            activities: List of activities
            
        Returns:
            str: Error description
        """
        if not activities:
            return "No activities generated"
            
        errors = []
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '00:00')))
        
        # Check first activity
        first_start = self._format_time(sorted_activities[0].get('start_time', ''))
        if first_start != "00:00":
            errors.append(f"First activity should start at 03:00, not {first_start}")
            
        # Check last activity
        last_end = self._format_time(sorted_activities[-1].get('end_time', ''))
        if last_end != "24:00":
            errors.append(f"Last activity should end at 27:00 (next day 03:00), not {last_end}")
            
        # Check continuity between activities
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
            
        # Safely get more demographic attributes, if not available use default values
        education = getattr(persona, 'education', 'unknown')
        occupation = getattr(persona, 'occupation', 'unknown')
        race = getattr(persona, 'race', 'unknown')
        # disability = getattr(persona, 'disability', False)
        # disability_type = getattr(persona, 'disability_type', None)
        
        # Build the prompt
        prompt = ACTIVITY_REFINEMENT_PROMPT.format(
            gender=persona.gender,
            age=persona.age,
            education=education,
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
        
        # Add more demographic information
        demographic_info = f"\n\nAdditional demographic information:"
        demographic_info += f"\n- Occupation: {occupation}"
        demographic_info += f"\n- Race/ethnicity: {race}"
        
        # if disability:
        #     demographic_info += f"\n- Has disability: Yes"
        #     if disability_type:
        #         demographic_info += f" (Type: {disability_type})"
        
        prompt += demographic_info
        
        try:
            # Use LLM to refine the activity
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=150
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
        Calculate time after adding specified minutes to a start time
        
        Args:
            start_time: Start time (HH:MM)
            minutes: Minutes to add
            
        Returns:
            str: Calculated time (HH:MM)
        """
        try:
            start_hour, start_minute = map(int, start_time.split(':'))
            total_minutes = start_hour * 60 + start_minute + minutes
            
            # Ensure time doesn't exceed 23:59
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
        sorted_activities = sorted(activities, key=lambda x: self._format_time(x.get('start_time', '03:00')))
        
        # Add sleep time and ensure time continuity
        final_activities = []
        last_end_time = "03:00"
        
        # If first activity doesn't start at 03:00, add sleep time
        if sorted_activities and self._format_time(sorted_activities[0].get('start_time')) != "03:00":
            final_activities.append({
                'activity_type': 'sleep',
                'start_time': '03:00',
                'end_time': sorted_activities[0].get('start_time'),
                'description': 'Sleeping at home',
                'location_type': 'home'
            })
        
        # Preprocessing: Adjust travel activity times
        processed_activities = []
        for i, activity in enumerate(sorted_activities):
            if activity['activity_type'].lower() == 'travel':
                # Find next non-travel activity
                next_activity = None
                for next_act in sorted_activities[i+1:]:
                    if next_act['activity_type'].lower() != 'travel':
                        next_activity = next_act
                        break
                
                if next_activity:
                    # Adjust travel activity end time to next activity's start time
                    activity = activity.copy()
                    activity['end_time'] = next_activity['start_time']
            
            processed_activities.append(activity)
        
        # Use processed activity list
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
                
                # Ensure end time doesn't exceed next day 03:00
                if end_time > "27:00":  # Convert to 24-hour format's 03:00
                    end_time = "27:00"  # Equivalent to next day 03:00
                    
            except:
                # Skip activities with invalid times
                continue
            
            # Create updated activity
            updated_activity = {
                **activity,
                'activity_type': activity_type,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # Check if time gap needs to be filled
            if start_time > last_end_time:
                # Add sleep time (early morning or night)
                if (int(start_time.split(':')[0]) < 8 or  # Morning before 8 AM
                    int(last_end_time.split(':')[0]) >= 22):  # Night after 10 PM
                    gap_activity = {
                        'activity_type': 'sleep',
                        'start_time': last_end_time,
                        'end_time': start_time,
                        'description': 'Sleeping at home',
                        'location_type': 'home'
                    }
                    final_activities.append(gap_activity)
            
            # Check overlap with last activity
            if final_activities:
                last_activity = final_activities[-1]
                if start_time < last_activity['end_time']:
                    # Handle overlap
                    if activity_type == last_activity['activity_type']:
                        # If same activity type, merge them
                        last_activity['end_time'] = max(last_activity['end_time'], end_time)
                        continue
                    elif activity_type == 'travel' or last_activity['activity_type'] == 'travel':
                        # If one is a travel activity, adjust times
                        if activity_type == 'travel':
                            # Travel activity start time becomes last activity's end time
                            updated_activity['start_time'] = last_activity['end_time']
                        else:
                            # Non-travel activity start time becomes travel activity's end time
                            last_activity['end_time'] = start_time
                    elif activity_type == 'sleep' or last_activity['activity_type'] == 'sleep':
                        # If one is a sleep activity, preserve sleep activity
                        if activity_type == 'sleep':
                            # Update last activity's end time
                            last_activity['end_time'] = start_time
                            final_activities.append(updated_activity)
                        continue
                    else:
                        # Adjust current activity's start time
                        start_time = last_activity['end_time']
                        updated_activity['start_time'] = start_time
                        
                        # Skip if adjusted start time is later than or equal to end time
                        if start_time >= end_time:
                            continue
            
            final_activities.append(updated_activity)
            last_end_time = end_time
        
        # If last activity doesn't end at 03:00 (next day), add sleep time
        if final_activities and final_activities[-1]['end_time'] != "27:00":  # 27:00 represents next day 03:00
            final_activities.append({
                'activity_type': 'sleep',
                'start_time': final_activities[-1]['end_time'],
                'end_time': '27:00',  # Next day 03:00
                'description': 'Sleeping at home',
                'location_type': 'home'
            })
        
        # Merge consecutive activities of the same type
        merged_activities = []
        current_activity = None
        
        for activity in final_activities:
            if not current_activity:
                current_activity = activity.copy()
                continue
                
            if (current_activity['activity_type'] == activity['activity_type'] and
                current_activity['end_time'] == activity['start_time']):
                # Merge consecutive activities of the same type
                current_activity['end_time'] = activity['end_time']
            else:
                merged_activities.append(current_activity)
                current_activity = activity.copy()
        
        if current_activity:
            merged_activities.append(current_activity)
        
        # Sort by start time again
        return sorted(merged_activities, key=lambda x: x['start_time'])
        
    def _calculate_duration_minutes(self, start_time, end_time):
        """
        Calculate minutes between two time points
        
        Args:
            start_time: Start time (HH:MM)
            end_time: End time (HH:MM)
            
        Returns:
            int: Minute difference
        """
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        
        # Handle overnight cases
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
        # If the activity is sleep or home, it usually doesn't require transportation
        if activity_type in ['sleep', 'home', 'work']:
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
                dt = datetime.datetime.strptime(time_str, fmt)
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
    
    # def _is_valid_activity(self, activity):
    #     """
    #     Check if an activity dictionary is valid
        
    #     Args:
    #         activity: Activity dictionary
            
    #     Returns:
    #         bool: Whether the activity is valid
    #     """
    #     # Check required fields
    #     required_fields = ['activity_type', 'start_time', 'end_time', 'description']
    #     if not all(field in activity for field in required_fields):
    #         return False
        
    #     # Check activity type
    #     activity_type = activity.get('activity_type', '').lower()
    #     if activity_type not in [t.lower() for t in ACTIVITY_TYPES]:
    #         return False
        
    #     return True
    
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