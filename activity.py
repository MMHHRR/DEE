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
    TRANSPORT_MODES
)
from utils import get_day_of_week, normalize_transport_mode

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
    
    def analyze_memory_patterns(self, memory):
        """
        Analyze the activity patterns in the historical memory.
        
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
    
    def generate_daily_schedule(self, persona, date):
        """
        Generate a daily activity schedule for a persona on a specific date.
        
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
        
        # If there are memory patterns, add them to the prompt
        if memory_patterns:
            prompt += "\n\nBased on historical patterns:"
            
            # Add frequent locations
            if memory_patterns['frequent_locations']:
                top_locations = sorted(memory_patterns['frequent_locations'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                prompt += f"\nMost visited locations: {[loc[0] for loc in top_locations]}"
            
            # Add time preferences
            if memory_patterns['time_preferences']:
                prompt += "\nTypical timing for activities:"
                for activity_type, times in memory_patterns['time_preferences'].items():
                    avg_time = sum([int(t.split(':')[0]) for t in times]) / len(times)
                    prompt += f"\n- {activity_type}: typically around {int(avg_time)}:00"
        
        # Call LLM to generate activity schedule
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a human behavior simulator that creates realistic daily schedules."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and parse the generated schedule
            generated_text = response.choices[0].message.content
            # print(f"Raw LLM response (first 100 chars): {generated_text[:100]}...")
            
            # Clean the text for better parsing
            cleaned_text = generated_text.strip()
            
            # Enhanced JSON parsing logic with better error handling
            activities = []
            
            # Step 1: Try direct JSON parsing
            try:
                activities = json.loads(cleaned_text)
                print("Successfully parsed complete response as JSON")
            except json.JSONDecodeError as e:
                # print(f"Direct JSON parsing failed: {e}")
                pass
                
                # Step 2: Try to extract a JSON array
                try:
                    # Look for array pattern with non-greedy matching
                    array_pattern = r'\[\s*\{.*?\}\s*\]'
                    array_match = re.search(array_pattern, cleaned_text, re.DOTALL)
                    
                    if array_match:
                        array_content = array_match.group(0)
                        # print(f"Found potential JSON array: {array_content[:50]}...")
                        
                        try:
                            # Try to parse the extracted array
                            activities = json.loads(array_content)
                            print("Successfully parsed extracted JSON array")
                        except json.JSONDecodeError as inner_e:
                            print(f"Parsing extracted array failed: {inner_e}")
                            
                            # Try to fix common JSON formatting issues
                            fixed_array = self._fix_json_array(array_content)
                            try:
                                activities = json.loads(fixed_array)
                                print("Successfully parsed fixed JSON array")
                            except json.JSONDecodeError:
                                print("Failed to fix and parse JSON array")
                    else:
                        print("No JSON array pattern found")
                        
                        # Try to extract individual activity objects
                        object_pattern = r'\{\s*"activity_type".*?\}'
                        objects = re.findall(object_pattern, cleaned_text, re.DOTALL)
                        
                        if objects:
                            print(f"Found {len(objects)} individual activity objects")
                            for obj in objects:
                                try:
                                    activity = json.loads(obj)
                                    activities.append(activity)
                                except json.JSONDecodeError:
                                    try:
                                        fixed_obj = self._fix_json_array(obj)
                                        activity = json.loads(fixed_obj)
                                        activities.append(activity)
                                    except:
                                        continue
                except Exception as ex:
                    print(f"Error extracting JSON: {ex}")
                
                # Step 3: If all above fails, try manual text parsing
                if not activities:
                    print("Attempting manual text parsing")
                    activities = self._extract_activities_from_text(cleaned_text)
            
            # Step 4: If everything fails, use default activities
            if not activities:
                print("No valid activities found, using defaults")
                activities = self._generate_default_activities(persona)
            
            # Validate and clean up activities
            validated_activities = self._validate_activities(activities)
            
            return validated_activities
            
        except Exception as e:
            print(f"Error generating activity schedule: {e}")
            # Return default activities as fallback
            default_activities = self._generate_default_activities(persona)
            return self._validate_activities(default_activities)
    
    def _validate_activities(self, activities):
        """
        Validate and clean up activities generated by the LLM.
        
        Args:
            activities: List of activity dictionaries
        
        Returns:
            list: Validated and cleaned activities
        """
        validated = []
        previous_location_type = "home"  # Assume starting location is home
        
        for activity in activities:
            # Ensure all required fields exist
            if not all(key in activity for key in ['activity_type', 'start_time', 'end_time', 'description']):
                continue
            
            # Clean activity type
            activity_type = activity['activity_type'].lower()
            
            # Correct activity type based on description
            corrected_type = self._correct_activity_type_based_on_description(activity_type, activity['description'])
            if corrected_type != activity_type:
                #print(f"Fix activity type: '{activity_type}' -> '{corrected_type}' based on description: '{activity['description']}'")
                activity_type = corrected_type
            
            # Ensure activity type is in the allowed list
            if activity_type not in [a.lower() for a in ACTIVITY_TYPES]:
                # Find the closest match
                activity_type = min(ACTIVITY_TYPES, key=lambda x: abs(len(x) - len(activity_type)))
            
            # Ensure time format is correct (HH:MM)
            try:
                start_time = self._format_time(activity['start_time'])
                end_time = self._format_time(activity['end_time'])
            except ValueError:
                continue
            
            # Get current activity location type
            location_type = activity.get('location_type', '')
            
            # Determine if transportation is needed
            needs_transport = self._needs_transportation(previous_location_type, location_type, activity_type)
            
            # Normalize transport mode
            if needs_transport:
                transport_mode = normalize_transport_mode(activity.get('transport_mode', 'walking'))
            else:
                transport_mode = None  # If no transportation is needed, set to None
            
            # Create validated activity
            validated_activity = {
                'activity_type': activity_type,
                'start_time': start_time,
                'end_time': end_time,
                'description': activity['description'],
            }
            
            # Only add transport mode if needed
            if transport_mode:
                validated_activity['transport_mode'] = transport_mode
            
            # Add location type if available
            if location_type:
                validated_activity['location_type'] = location_type
                previous_location_type = location_type
            
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
    
    def refine_activity(self, persona, activity, date):
        """
        Refine an activity with more detailed information.
        
        Args:
            persona: Persona object
            activity: Activity dictionary
            date: Date string (YYYY-MM-DD)
        
        Returns:
            dict: Refined activity with additional details
        """
        # Copy original activity
        refined = activity.copy()
        
        # For specific types of activities (sleep, home, work), do not generate additional details
        if activity['activity_type'].lower() in ['sleep', 'home', 'work']:
            # Set basic location types for these fixed position activities
            if activity['activity_type'].lower() == 'sleep' or activity['activity_type'].lower() == 'home':
                refined['location_type'] = 'home'
            elif activity['activity_type'].lower() == 'work':
                refined['location_type'] = 'workplace'
            
            # Ensure a reasonable transport_mode (if needed)
            if 'transport_mode' not in refined or not refined['transport_mode']:
                # Most fixed position activities don't require transportation, return None
                refined['transport_mode'] = None
                
            return refined
        
        # Check the activity description, if it mentions activities at home, set location_type to home directly
        description_lower = activity['description'].lower()
        if 'at home' in description_lower or 'home' in description_lower and ('cook' in description_lower or 'dinner' in description_lower or 'breakfast' in description_lower or 'lunch' in description_lower):
            refined['location_type'] = 'home'
            return refined
        
        # For other activity types, generate detailed descriptions
        try:
            # Prepare the prompt
            prompt = f"""
            Based on the following information, provide a more detailed description of this activity:
            
            Person:
            - Gender: {persona.gender}
            - Age: {persona.age}
            - Income: ${persona.income}
            - Consumption habits: {persona.consumption}
            - Education: {persona.education}
            
            Activity:
            - Type: {activity['activity_type']}
            - Description: {activity['description']}
            - Time: {activity['start_time']} to {activity['end_time']}
            - Day: {get_day_of_week(date)}
            
            Provide your response as a JSON object with the following fields:
            {{
                "detailed_description": "A paragraph describing what exactly the person will do during this activity, considering their demographics",
                "location_type": "A specific type of place where this activity would occur (e.g., 'upscale restaurant' rather than just 'restaurant')",
                "specific_preferences": "What preferences might this person have for this type of activity, considering their income and consumption habits"
            }}
            
            IMPORTANT: If the activity takes place at home (e.g., "dinner at home", "cooking at home"), always set "location_type" to EXACTLY "home" without any additional details.
            """
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an activity detail generator for human daily activities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the JSON response
            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            
            if json_match:
                try:
                    details = json.loads(json_match.group(1))
                    
                    # Add detailed information to the refined activity
                    for key, value in details.items():
                        refined[key] = value
                    
                    # Confirm again: If the activity is at home, force location_type to "home"
                    if 'location_type' in refined:
                        if 'home' in refined['location_type'].lower() or ('at home' in description_lower and not refined['location_type'] == 'home'):
                            refined['location_type'] = 'home'
                    
                except Exception as e:
                    print(f"Error parsing activity details: {e}")
                
        except Exception as e:
            print(f"Error generating activity details: {e}")
        
        # Final check: Ensure the location_type for home activities is "home"
        if 'location_type' in refined and ('home' in refined['location_type'].lower() and not refined['location_type'] == 'home'):
            refined['location_type'] = 'home'
        
        # If the activity is at home, force location_type to "home"
        if 'at home' in activity['description'].lower():
            refined['location_type'] = 'home'
        
        return refined
    
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
                    "location_type": "home",
                    "transport_mode": None
                },
                {
                    "activity_type": "work",
                    "description": "Morning office session",
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "location_type": "workplace",
                    "transport_mode": "driving"
                },
                {
                    "activity_type": "dining",
                    "description": "Lunch break",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "location_type": "restaurant",
                    "transport_mode": "walking"
                },
                {
                    "activity_type": "work",
                    "description": "Afternoon office session",
                    "start_time": "13:00",
                    "end_time": "18:00",
                    "location_type": "workplace",
                    "transport_mode": None
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
                    "location_type": "home",
                    "transport_mode": None
                },
                {
                    "activity_type": "leisure",
                    "description": "Morning relaxation time",
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "location_type": "home",
                    "transport_mode": None
                },
                {
                    "activity_type": "dining",
                    "description": "Lunch time",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "location_type": "home",
                    "transport_mode": None
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
                    "location_type": "home",
                    "transport_mode": None
                })
        
        # Only return non-sleep activities, sleep activities will be handled separately in main.py
        return activities
        
    # Simplified sleep time methods that just return fixed times
    def _generate_sleep_end_time(self, age):
        return "06:00"
        
    def _generate_sleep_start_time(self, age):
        return "22:00" 