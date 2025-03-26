"""
Memory module for the LLM-based mobility simulation.
Records daily mobility trajectories and activities.
"""

import os
import json
import datetime
from config import RESULTS_DIR
from utils import visualize_trajectory

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
        self.persona_info = {
            'id': persona.id,
            'name': persona.name,
            'gender': persona.gender,
            'age': persona.age,
            'income': persona.income,
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
            activity_record['location_name'] = self._get_location_name(activity_record)
            
        # Add to activities list
        self.current_day['activities'].append(activity_record)
        
        # Create trajectory point
        self._add_trajectory_point(activity_record, location, timestamp)
        
        # Update statistics
        self._update_statistics(activity_record, location, timestamp)
    
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
    
    def _find_location_name(self, location):
        """
        Find location name from previous activities at same location
        
        Args:
            location: Location coordinates
            
        Returns:
            str: Location name if found, otherwise "Unknown Location"
        """
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
        """
        if self.current_day:
            self.days.append(self.current_day)
            self.current_day = None
    
    def save_to_file(self, file_path):
        """
        Save memory data to a JSON file.
        
        Args:
            file_path: Path to save the file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data
            data = {
                'persona_id': self.persona_id,
                'persona_info': self.persona_info,
                'days': self.days,
                'statistics': {
                    'location_frequency': self.location_frequency,
                    'activity_type_frequency': self.activity_type_frequency,
                    'time_preference': self.time_preference
                }
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            print(f"Memory data saved to {file_path}")
            
            # Generate visualization for each day
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
                visualize_trajectory(day['trajectory'], map_file)
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