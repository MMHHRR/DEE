"""
Memory module for the LLM-based mobility simulation.
Records daily mobility trajectories and activities.
"""

import os
import json
from config import RESULTS_DIR
from utils import get_day_of_week

class Memory:
    """
    Records and manages the memory of daily mobility trajectories.
    Only passes the most recent days of activity records to LLM to reduce context length,
    but maintains complete history for data storage and analysis.
    """
    
    def __init__(self, persona_id, memory_days=2):
        """
        Initialize a Memory instance.
        
        Args:
            persona_id: Identifier for the persona
            memory_days: Number of recent days to pass to LLM context (default: 2)
        """
        self.persona_id = persona_id
        self.persona_info = None
        self.days = []
        self.current_day = None
        self.memory_days = memory_days
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Create efficient indexes for location lookups
        self.location_index = {}  # Location index for efficient location queries
    
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
            'race': persona.race,
            'education': persona.education,
            'occupation': persona.occupation,
            'income': persona.get_household_income(),
            'household_vehicles': persona.get_household_vehicles(),
            'disability': persona.disability,
            'disability_type': persona.disability_type,
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
    
    def record_mobility_event(self, activity_type=None, 
                             start_time=None, end_time=None, 
                             description=None, location_name=None,
                             location_type=None, coordinates=None, location=None,
                             timestamp=None, transport_mode=None, 
                             distance=None, travel_time=None,
                             start_location=None, end_location=None,
                             **additional_fields):
        """
        Record any mobility event as a unified activity
        
        Args:
            activity_type: Type of activity (e.g., 'travel', 'dining', 'work', etc.)
            start_time: Time string (HH:MM) when activity starts
            end_time: Time string (HH:MM) when activity ends
            description: Description of the activity
            location_name: Name of the location
            location_type: Type of location (e.g., 'restaurant', 'park', 'home')
            coordinates: Coordinates in (lat, lon) format
            location: Location coordinates (alternative to coordinates)
            timestamp: Current timestamp string
            transport_mode: Mode of transportation (if relevant)
            distance: Travel distance (if relevant)
            travel_time: Time spent traveling in minutes (if relevant)
            start_location: Start location coordinates (if relevant)
            end_location: End location coordinates (if relevant)
            additional_fields: Any additional fields to include in the activity record
        """
        if self.current_day is None:
            print("No current day to record mobility event")
            return
            
        # Create unified activity record
        activity_record = {
            'activity_type': activity_type,
            'start_time': start_time,
            'end_time': end_time,
            'description': description,
            'location_name': location_name,
            'location_type': location_type,
            'timestamp': timestamp,
            'transport_mode': transport_mode,
            'distance': distance,
            'travel_time': travel_time
        }
        
        # Add location info
        if coordinates:
            activity_record['coordinates'] = coordinates
        elif location:
            activity_record['location'] = location
            
        # Add travel-specific information if provided
        if start_location:
            activity_record['from_location'] = start_location
        if end_location:
            activity_record['to_location'] = end_location
            
        # Add any additional fields
        if additional_fields:
            activity_record.update(additional_fields)
        
        # Clean up None values
        activity_record = {k: v for k, v in activity_record.items() if v is not None}
        
        # Add to activities list
        self.current_day['activities'].append(activity_record)
        
        # Update location index if we have a location name
        if location_name:
            if coordinates:
                self.location_index[location_name] = coordinates
            elif location:
                self.location_index[location_name] = location
            elif end_location and activity_type == 'travel':
                # For travel activities, use end_location for the destination
                self.location_index[location_name] = end_location
    
    def end_day(self):
        """
        End current day recording and add to days list.
        Maintains all historical data but only passes memory_days to LLM.
        """
        if self.current_day:
            # Add current day to history (keeps all historical data)
            self.days.append(self.current_day)
            self.current_day = None
    
    def get_recent_activities(self, activity_type=None, days=None):
        """
        Get recent activities, optionally filtered by type.
        Only returns the specified number of most recent days.
        
        Args:
            activity_type: Optional activity type filter
            days: Number of days to retrieve, defaults to memory_days
            
        Returns:
            list: List of activities matching the criteria
        """
        if days is None:
            days = self.memory_days
            
        days = min(days, len(self.days))
        
        # Only get the most recent n days
        recent_days = self.days[-days:]
        
        activities = []
        for day in recent_days:
            for activity in day['activities']:
                if activity_type is None or activity['activity_type'] == activity_type:
                    activities.append(activity)
                    
        return activities
    
    def get_mobility_patterns(self):
        """
        Extract basic mobility pattern info from stored days,
        needed for activity.py's analyze_memory_patterns function.
        Only uses memory_days for pattern analysis to reduce LLM context.
        
        Returns:
            dict: Dictionary with mobility pattern information for activity analysis
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
        
        # Get only the most recent memory_days days for pattern analysis
        recent_days = self.days[-self.memory_days:] if len(self.days) > self.memory_days else self.days
        
        # Collect all activities from recent days
        for day in recent_days:
            for activity in day['activities']:
                # Collect location frequencies
                location = activity.get('location_type', 'unknown')
                patterns['frequent_locations'][location] = patterns['frequent_locations'].get(location, 0) + 1
                
                activity_type = activity.get('activity_type')
                start_time = activity.get('start_time')
                
                # Collect time preferences
                if activity_type not in patterns['time_preferences']:
                    patterns['time_preferences'][activity_type] = []
                if start_time:
                    patterns['time_preferences'][activity_type].append(start_time)
                
                # Collect travel times
                travel_time = activity.get('travel_time', 0)
                if activity_type not in patterns['travel_times']:
                    patterns['travel_times'][activity_type] = []
                patterns['travel_times'][activity_type].append(travel_time)
                
                # Collect activity durations
                if 'start_time' in activity and 'end_time' in activity:
                    start = activity.get('start_time')
                    end = activity.get('end_time')
                    try:
                        from utils import time_difference_minutes
                        duration = time_difference_minutes(start, end)
                        if activity_type not in patterns['activity_durations']:
                            patterns['activity_durations'][activity_type] = []
                        patterns['activity_durations'][activity_type].append(duration)
                    except:
                        pass
                
                # Collect distance information
                distance = activity.get('distance', 0)
                if activity_type not in patterns['distances']:
                    patterns['distances'][activity_type] = []
                # Only add non-negative distance values
                if distance >= 0:
                    patterns['distances'][activity_type].append(distance)
                
                # Collect transport modes
                if 'transport_mode' in activity:
                    mode = activity['transport_mode']
                    if mode and mode != 'unknown':
                        patterns['transport_modes'][mode] = patterns['transport_modes'].get(mode, 0) + 1
                
        return patterns

    def save_to_file(self, file_path):
        """
        Save memory data to JSON files - one main file and separate files for each day.
        Saves ALL historical days, not just the memory_days limit.
        Only saves activity data, not trajectory data.
        
        Args:
            file_path: Path to save the main file
        """
        try:
            # Create directory if it doesn't exist
            base_dir = os.path.dirname(file_path)
            os.makedirs(base_dir, exist_ok=True)
            
            # Create a directory for daily activities
            household_id = self.persona_id
            daily_dir = os.path.join(base_dir, f"household_{household_id}_activities")
            os.makedirs(daily_dir, exist_ok=True)
            
            # Prepare main data - only persona info and summary of ALL days
            main_data = {
                'persona_id': self.persona_id,
                'persona_info': self.persona_info,
                'days_summary': [{'date': day['date'], 'day_of_week': day['day_of_week']} for day in self.days],
                'total_days_saved': len(self.days),
                'days_used_for_llm_context': self.memory_days
            }
            
            # Save main data to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(main_data, f, indent=2)
                
            # Save ALL days to separate files - only activities, not trajectory
            for day in self.days:
                date = day['date']
                date_filename = date.replace('-', '')  # Remove hyphens for filename
                day_file = os.path.join(daily_dir, f"{date_filename}.json")
                
                # Create a copy of the day data with only activities, without trajectory
                day_data = {
                    'date': day['date'],
                    'day_of_week': day['day_of_week'],
                    'activities': day['activities']
                }
                
                with open(day_file, 'w', encoding='utf-8') as f:
                    json.dump(day_data, f, indent=2)
                
            # print(f"Memory data saved to {file_path} with all {len(self.days)} daily activities in {daily_dir}")
            
        except Exception as e:
            print(f"Error saving memory data: {e}")
               
    def save_llm_days_to_csv(self, output_dir=None, persona_id=None, start_year=2025):
        """
        Save only LLM-generated mobility data to a CSV file (days from start_year and later).
        
        This method is similar to save_to_csv but filters only days from the specified start_year or later,
        which are typically the days generated by LLM rather than historical data.
        
        Args:
            output_dir: Directory to save the CSV file (defaults to RESULTS_DIR)
            persona_id: Person ID to use in the file (defaults to 1)
            start_year: Start year for filtering LLM-generated days (defaults to 2025)
            
        Returns:
            str: Path to the saved CSV file
        """
        import pandas as pd
        import os
        from utils import time_difference_minutes
        
        if output_dir is None:
            output_dir = RESULTS_DIR
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household ID (sampno) from persona_id
        if '_' in str(self.persona_id):
            # If persona_id contains an underscore, extract the household_id part
            household_id = str(self.persona_id).split('_')[0]
        else:
            household_id = self.persona_id
            
        # Use provided persona_id or default to 1
        if persona_id is None:
            persona_id = 1
            
        # Prepare data rows for CSV
        rows = []
        
        # Track the last known location for each day
        last_coords = {}  # day_index -> (lat, lon)
        
        # Filter to include only LLM generated days (start_year and later)
        llm_generated_days = []
        for day in self.days:
            # Check if the date starts with start_year or later
            date = day.get('date', '')
            if date and date.startswith(str(start_year)) or (len(date) >= 4 and int(date[:4]) > start_year):
                llm_generated_days.append(day)
        
        if not llm_generated_days:
            print(f"No days found from {start_year} or later")
            return None
        
        # Process each LLM-generated day in memory
        for day_index, day in enumerate(llm_generated_days, 1):
            
            # 记录每天的第一个活动
            is_first_activity_of_day = True
            
            # 获取home位置的坐标
            home_lat, home_lon = 0, 0
            if self.persona_info and 'home' in self.persona_info and self.persona_info['home']:
                home_info = self.persona_info['home']
                # 处理home是字典的情况
                if isinstance(home_info, dict) and 'coordinates' in home_info:
                    home_coords = home_info.get('coordinates')
                    if home_coords and isinstance(home_coords, (list, tuple)) and len(home_coords) >= 2:
                        home_lat, home_lon = home_coords[0], home_coords[1]
                # 处理home直接是坐标元组的情况
                elif isinstance(home_info, (list, tuple)) and len(home_info) >= 2:
                    home_lat, home_lon = home_info[0], home_info[1]
                    
            # Process each activity for this day
            for activity in day['activities']:
                # Extract location name
                actype = activity.get('activity_type', 'Unknown')
                locname = activity.get('location_name', 'Unknown')
                
                # Extract arrival and departure times
                arrtime = activity.get('start_time', '')
                deptime = activity.get('end_time', '')
                
                # Calculate travel time (in minutes) and activity duration
                travtime = activity.get('travel_time', 0)
                
                # Calculate activity duration if start and end times are available
                actdur = 0
                if arrtime and deptime:
                    try:
                        actdur = time_difference_minutes(arrtime, deptime)
                    except:
                        actdur = 0
                
                # Extract distance
                distance = activity.get('distance', 0)
                
                # Extract transport mode
                transportmode = activity.get('transport_mode', '')
                
                # Extract coordinates (lon, lat)
                lon, lat = 0, 0
                start_lon, start_lat = 0, 0
                end_lon, end_lat = 0, 0
                
                # Try different ways to get location coordinates
                if 'coordinates' in activity and activity['coordinates']:
                    # Check if coordinates are in (lat, lon) or (lon, lat) format
                    coords = activity['coordinates']
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        # Assume coordinates are in (lat, lon) format
                        lat, lon = coords[0], coords[1]
                        # For current activity's end location
                        end_lat, end_lon = lat, lon
                elif 'location' in activity and activity['location']:
                    # Try location field as alternative
                    loc = activity['location']
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        lat, lon = loc[0], loc[1]
                        # For current activity's end location
                        end_lat, end_lon = lat, lon
                elif 'to_location' in activity and activity['to_location']:
                    # If we have a to_location but no coordinates/location, use to_location
                    to_loc = activity['to_location']
                    if isinstance(to_loc, (list, tuple)) and len(to_loc) >= 2:
                        lat, lon = to_loc[0], to_loc[1]
                        end_lat, end_lon = lat, lon
                
                # For travel activities, try to get explicitly defined start and end coordinates
                if 'from_location' in activity and activity['from_location']:
                    # Use explicit from_location if available
                    from_loc = activity['from_location']
                    if isinstance(from_loc, (list, tuple)) and len(from_loc) >= 2:
                        start_lat, start_lon = from_loc[0], from_loc[1]
                elif day_index in last_coords:
                    # Use last known coordinates for this day as the starting point
                    prev_coords = last_coords[day_index]
                    if isinstance(prev_coords, (list, tuple)) and len(prev_coords) >= 2:
                        start_lat, start_lon = prev_coords[0], prev_coords[1]
                
                # 如果是每天的第一个活动且没有起始坐标，使用home的坐标
                if is_first_activity_of_day and (start_lat == 0 and start_lon == 0) and (home_lat != 0 and home_lon != 0):
                    start_lat, start_lon = home_lat, home_lon
                
                # For travel activities, try to get explicitly defined end coordinates
                if 'to_location' in activity and activity['to_location']:
                    # Use explicit to_location if available
                    to_loc = activity['to_location']
                    if isinstance(to_loc, (list, tuple)) and len(to_loc) >= 2:
                        end_lat, end_lon = to_loc[0], to_loc[1]
                
                # If we don't have start coordinates but have end coordinates, and this is not the first activity
                if (start_lat == 0 and start_lon == 0) and (end_lat != 0 and end_lon != 0):
                    if day_index in last_coords:
                        # 使用上一个活动的位置作为起点
                        start_lat, start_lon = last_coords[day_index]
                
                # Update last known coordinates for next activity
                if end_lat != 0 and end_lon != 0:
                    last_coords[day_index] = (end_lat, end_lon)
                
                # Create a row for this activity
                row = {
                    'sampno': household_id,
                    'perno': persona_id,
                    'dayno': day_index,
                    'actype': actype,
                    'locname': locname,
                    'arrtime': arrtime,
                    'deptime': deptime,
                    'travtime': travtime,
                    'actdur': actdur,
                    'distance': distance,
                    'transportmode': transportmode,
                    'lon': lon,
                    'lat': lat,
                    'start_lon': start_lon,
                    'start_lat': start_lat,
                    'end_lon': end_lon,
                    'end_lat': end_lat
                }
                
                rows.append(row)
                
                # 第一个活动处理完毕后，更新标志
                is_first_activity_of_day = False
        
        # Create DataFrame and save to CSV
        if rows:
            df = pd.DataFrame(rows)
            
            # Define column order to match the specified format
            columns = ['sampno', 'perno', 'dayno', 'actype', 'locname', 'arrtime', 'deptime', 
                      'travtime', 'actdur', 'distance', 'transportmode', 'lon', 'lat',
                      'start_lon', 'start_lat', 'end_lon', 'end_lat']
            
            # Reorder columns and save to CSV
            csv_path = os.path.join(output_dir, f"household_{household_id}_persona_{persona_id}_llm_activities.csv")
            df[columns].to_csv(csv_path, index=False)
            
            print(f"Saved LLM-generated activity records to {csv_path}")
            return csv_path
        else:
            print("No LLM-generated activities to save to CSV")
            return None 