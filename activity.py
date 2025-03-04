"""
Activity module for the LLM-based mobility simulation.
Uses LLM to generate and manage daily activity plans.
"""

import json
import datetime
import re
import openai
import random
from config import (
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    ACTIVITY_GENERATION_PROMPT,
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
        Added caching to avoid repeated analysis
        
        Args:
            memory: Memory object, containing historical activity records
            
        Returns:
            dict: Dictionary containing activity pattern analysis results
        """
        patterns = {
            'frequent_locations': {},  # Frequent locations
            'time_preferences': {}     # Time preferences
        }
        
        # Analyze the data of historical days
        for day in memory.days:
            # Analyze location frequency
            for activity in day['activities']:
                location = tuple(activity.get('location', [0, 0]))
                if location in patterns['frequent_locations']:
                    patterns['frequent_locations'][location] += 1
                else:
                    patterns['frequent_locations'][location] = 1
                    
                # Analyze time preferences
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
        
        # Validate and correct activities
        validated_activities = self._validate_activities(activities)
        
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
            date=date
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
        
        max_retries = 2
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
                
                # If activities were extracted, return the results
                if activities:
                    return activities
                    
                # If no activities were extracted, but there are still retry attempts, continue trying
                if attempt < max_retries:
                    print(f"Failed to extract activities, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                    
                # Last retry failed, return default activities
                print("Failed to generate activities with LLM, using default activities")
                return self._generate_default_activities(persona)
                
            except Exception as e:
                print(f"Error generating activities: {e}")
                if attempt < max_retries:
                    print(f"Retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                
                # Last retry failed, return default activities
                print("Error with LLM API, using default activities")
                return self._generate_default_activities(persona)
    
    def refine_activity(self, persona, activity, date):
        """
        Refine an activity with more details.
        实现批量处理以减少API调用
        
        Args:
            persona: Persona object
            activity: Activity dictionary
            date: Date string (YYYY-MM-DD)
        
        Returns:
            dict: Refined activity with more details
        """
        # If batch processing is enabled, add activity to queue
        if BATCH_PROCESSING:
            # Add to queue
            self.activity_queue.append((persona, activity, date))
            
            # If queue is full or last activity, process entire batch
            if len(self.activity_queue) >= BATCH_SIZE:
                batch_results = self._process_activity_batch(self.activity_queue)
                self.activity_queue = []  # Clear queue
                
                # Find result for current activity
                for batch_persona, batch_activity, batch_date, refined_activity in batch_results:
                    if (batch_persona.id == persona.id and 
                        batch_activity['activity_type'] == activity['activity_type'] and
                        batch_activity['start_time'] == activity['start_time']):
                        return refined_activity
                
                # If no matching result found, process current activity separately
                return self._refine_single_activity(persona, activity, date)
            
            # Queue not full, return original activity (will be processed later)
            return activity
        else:
            # No batch processing, process single activity directly
            return self._refine_single_activity(persona, activity, date)

    def _process_activity_batch(self, activity_queue):
        """
        处理一批活动的细化
        
        Args:
            activity_queue: 活动队列，格式为[(persona, activity, date),...]
            
        Returns:
            list: [(persona, original_activity, date, refined_activity),...]
        """
        results = []
        
        try:
            if not activity_queue:
                return results
                
            # Build prompts for batch processing
            prompts = []
            for i, (persona, activity, date) in enumerate(activity_queue):
                day_of_week = get_day_of_week(date)
                prompt = f"Activity {i+1}:\n"
                prompt += f"Description: {activity.get('description', 'No description')}\n"
                prompt += f"Activity Type: {activity.get('activity_type', 'Unknown')}\n"
                prompt += f"Time: {activity.get('start_time', '00:00')} - {activity.get('end_time', '00:00')}\n"
                prompt += f"Person: {persona.age} year old {persona.gender}, income ${persona.income}, {persona.education} education\n"
                prompt += f"Day: {day_of_week}, {date}\n\n"
                prompts.append(prompt)
                
            # Build full prompt
            full_prompt = "You are helping to refine the details of a daily activity schedule. " \
                          "For each activity, provide more specific details about the location and environment.\n\n"
            full_prompt += "".join(prompts)
            full_prompt += "\nFor each activity, provide:\n" \
                          "1. A more refined description of what the person would do\n" \
                          "2. The type of location where this would occur\n" \
                          "3. Any equipment or items they would need\n" \
                          "4. Environmental characteristics of the location\n" \
                          "5. Appropriate transportation mode (walking, driving, public_transit, cycling, rideshare)\n\n" \
                          "Format your response as a JSON array:\n" \
                          "[\n" \
                          "  {\n" \
                          "    \"activity_index\": 1,\n" \
                          "    \"refined_description\": \"...\",\n" \
                          "    \"location_type\": \"...\",\n" \
                          "    \"equipment\": \"...\",\n" \
                          "    \"environment\": \"...\",\n" \
                          "    \"transport_mode\": \"...\"\n" \
                          "  },\n" \
                          "  ...\n" \
                          "]"
            
            # Call LLM for batch refinement
            refinements = []  # Default to empty list
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract returned JSON, including potential fixes
                text = response.choices[0].message.content
                try:
                    # Try to parse JSON
                    refinements = json.loads(text)
                    if not isinstance(refinements, list):
                        refinements = []
                except Exception:
                    # Try extracting JSON from text
                    try:
                        json_text = re.search(r'\[.*\]', text, re.DOTALL)
                        if json_text:
                            json_text = self._fix_json_array(json_text.group(0))
                            refinements = json.loads(json_text)
                        else:
                            refinements = []
                    except Exception as je:
                        print(f"Error extracting JSON from response: {je}")
                        refinements = []
            except Exception as e:
                print(f"Error calling LLM API: {e}")
                # API call failed, keep empty refinements list
            
            # Apply refined activities
            for refinement in refinements:
                if 'activity_index' in refinement:
                    index = int(refinement['activity_index']) - 1
                    if 0 <= index < len(activity_queue):
                        persona, activity, date = activity_queue[index]
                        
                        # Copy original activity and update
                        refined_activity = activity.copy()
                        
                        if 'refined_description' in refinement:
                            refined_activity['description'] = refinement['refined_description']
                        
                        if 'location_type' in refinement:
                            refined_activity['location_type'] = refinement['location_type']
                        
                        if 'equipment' in refinement:
                            refined_activity['equipment'] = refinement['equipment']
                        
                        if 'environment' in refinement:
                            refined_activity['environment'] = refinement['environment']
                            
                        if 'transport_mode' in refinement:
                            refined_activity['transport_mode'] = normalize_transport_mode(refinement['transport_mode'])
                        
                        results.append((persona, activity, date, refined_activity))
            
            # Process unrefined activities
            processed_indices = [int(r.get('activity_index', 0)) - 1 for r in refinements]
            for i, (persona, activity, date) in enumerate(activity_queue):
                if i not in processed_indices:
                    results.append((persona, activity, date, activity))  # Use original activity
        
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Batch processing failed, return original activity
            for persona, activity, date in activity_queue:
                results.append((persona, activity, date, activity))
        
        return results

    @cached
    def _refine_single_activity(self, persona, activity, date):
        """处理单个活动的细化，可被缓存"""
        # Simplify batch processing to single activity processing
        results = self._process_activity_batch([(persona, activity, date)])
        if results:
            _, _, _, refined_activity = results[0]
            return refined_activity
        return activity
    
    def _validate_activities(self, activities):
        """
        Validate and clean activities data.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            list: Validated activities
        """
        validated = []
        previous_location_type = None
        
        for activity in activities:
            # Skip invalid activities
            if 'activity_type' not in activity or 'start_time' not in activity or 'end_time' not in activity:
                continue
                
            # Get activity type and ensure it's valid
            activity_type = activity['activity_type'].lower()
            
            # Check description for more accurate activity type
            if 'description' in activity and activity['description']:
                activity_type = self._correct_activity_type_based_on_description(
                    activity_type, 
                    activity['description']
                )
            
            # Format times consistently
            try:
                start_time = self._format_time(activity['start_time'])
                end_time = self._format_time(activity['end_time'])
            except:
                # Skip activities with invalid times
                continue
            
            # Get current activity location type
            location_type = activity.get('location_type', '')
            
            # Get transport mode if available
            transport_mode = activity.get('transport_mode', '')
            
            # Normalize transport mode if specified
            if transport_mode:
                transport_mode = normalize_transport_mode(transport_mode)
            
            # If no transportation mode specified and not first activity, determine default transportation based on activity type and location
            if not transport_mode and len(validated) > 0:
                # Home activities usually don't require transportation
                if location_type == 'home':
                    transport_mode = 'walking'  # Default walking at home
                # Work commute
                elif location_type == 'work' or activity_type == 'work':
                    transport_mode = 'driving'  # Default driving to work
                # Short activities default to walking
                elif activity_type in ['shopping', 'dining', 'leisure', 'errands']:
                    transport_mode = 'walking'
                # Medium/long activities default to public transportation
                elif activity_type in ['recreation', 'healthcare', 'social', 'education']:
                    transport_mode = 'public_transit'
                else:
                    transport_mode = 'driving'  # Other cases default to driving
            
            # Create validated activity
            validated_activity = {
                'activity_type': activity_type,
                'start_time': start_time,
                'end_time': end_time,
                'description': activity['description'],
            }
            
            # Add location type if available
            if location_type:
                validated_activity['location_type'] = location_type
                previous_location_type = location_type
            
            # Add transport mode if available
            if transport_mode:
                validated_activity['transport_mode'] = transport_mode
            
            validated.append(validated_activity)
        
        # Sort by start time
        validated.sort(key=lambda x: x['start_time'])
        
        return validated
    
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
        # Try different formats
        formats = [
            '%H:%M',  # 14:30
            '%I:%M %p',  # 2:30 PM
            '%I%p',  # 2PM
            '%I %p'  # 2 PM
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
        # Replace single quotes with double quotes
        fixed = json_str.replace("'", '"')
        
        # Ensure property names have double quotes
        fixed = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', fixed)
        
        # Fix missing commas between objects
        fixed = re.sub(r'}\s*{', '},{', fixed)
        
        # Fix boolean values and null
        fixed = fixed.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        return fixed
    
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
        Generate a default list of activities for a day based on persona characteristics.
        
        Args:
            persona: Persona object with demographic information
            
        Returns:
            list: List of activity dictionaries
        """
        # Generate activities without sleep - sleep will be added in main.py
        activities = []
        
        # Work day activities
        if persona.works and get_day_of_week(persona.current_date) in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            activities.extend([
                {
                    "activity_type": "home",
                    "description": "Morning routine and breakfast",
                    "start_time": "06:00",
                    "end_time": "08:00",
                    "location_type": "home"
                },
                {
                    "activity_type": "work",
                    "description": "Morning office session",
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "location_type": "workplace"
                },
                {
                    "activity_type": "dining",
                    "description": "Lunch break",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "location_type": "restaurant"
                },
                {
                    "activity_type": "work",
                    "description": "Afternoon office session",
                    "start_time": "13:00",
                    "end_time": "18:00",
                    "location_type": "workplace"
                },
                {
                    "activity_type": "dining",
                    "description": "Dinner and evening relaxation",
                    "start_time": "19:00",
                    "end_time": "21:30",
                    "location_type": "home",
                    "transport_mode": "driving"
                }
            ])
        else:
            # Weekend activities
            activities.extend([
                {
                    "activity_type": "home",
                    "description": "Morning routine and breakfast",
                    "start_time": "07:00",
                    "end_time": "09:00",
                    "location_type": "home"
                },
                {
                    "activity_type": "leisure",
                    "description": "Morning relaxation time",
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "location_type": "home"
                },
                {
                    "activity_type": "dining",
                    "description": "Lunch time",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "location_type": "home"
                }
            ])
            
            # Add shopping activity for weekends
            if random.random() > 0.5:
                activities.append({
                    "activity_type": "shopping",
                    "description": "Buying groceries",
                    "start_time": "15:00",
                    "end_time": "17:00",
                    "location_type": "supermarket",
                    "transport_mode": "driving"
                })
            
            # Add evening activity
            activities.append({
                "activity_type": "dining",
                "description": "Dinner time",
                "start_time": "18:00",
                "end_time": "19:30",
                "location_type": "home",
                "transport_mode": "driving"
            })
            
            # Add social or leisure evening activity
            if random.random() > 0.6:
                activities.append({
                    "activity_type": "social",
                    "description": "Evening gathering with friends",
                    "start_time": "20:00",
                    "end_time": "21:30",
                    "location_type": "restaurant",
                    "transport_mode": "driving"
                })
            else:
                activities.append({
                    "activity_type": "leisure",
                    "description": "Evening relaxation",
                    "start_time": "20:00",
                    "end_time": "21:30",
                    "location_type": "home"
                })
        
        # Only return non-sleep activities, sleep activities will be handled separately in main.py
        return activities
        
    # Simplified sleep time methods that just return fixed times
    def _generate_sleep_end_time(self, age):
        return "06:00"
        
    def _generate_sleep_start_time(self, age):
        return "22:00" 