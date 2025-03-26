"""
Persona module for the LLM-based mobility simulation.
Defines the Persona class to represent individuals with their attributes and characteristics.
"""

import os
import pandas as pd 
import datetime
from config import RESULTS_DIR

class Memory:
    """
    Simplified memory object for storing historical trajectory data loaded from CSV files
    """
    def __init__(self):
        self.days = []  # Structure expected by activity.py's analyze_memory_patterns function

class Persona:
    """
    Represents an individual with specific attributes and characteristics.
    """
    
    def __init__(self, persona_data):
        """
        Initialize a Persona instance with data.
        
        Args:
            persona_data (dict): Dictionary containing persona attributes
        """
        self.id = persona_data.get('id')
        self.name = persona_data.get('name', f"Person-{self.id}")
        self.home = tuple(persona_data.get('home', [0, 0]))
        self.work = tuple(persona_data.get('work', [0, 0]))
        self.gender = persona_data.get('gender', 'unknown')
        self.race = persona_data.get('race', 'unknown')
        self.age = persona_data.get('age', 0)
        
        # Additional attributes
        self.occupation = persona_data.get('occupation', 'unknown')
        self.education = persona_data.get('education', 'unknown')
        self.disability = persona_data.get('disability', False)
        self.disability_type = persona_data.get('disability_type', None)
        
        # Current state
        self.current_location = self.home  # Start at home
        self.current_activity = None       # Complete activity information
        self.current_activity_type = None  # Current activity type (string)
        
        # Historical data memory
        self.memory = None
        
        # CSV data file paths
        self.person_csv = "data/person.csv"
        self.gps_place_csv = "data/gps_place.csv"
        self.location_csv = "data/location.csv"
    
    def load_historical_data(self, household_id=None, person_id=None):
        """
        Load historical travel data from CSV files
        
        Args:
            household_id (int): Household ID (sampno), if not specified uses self.id
            person_id (int): Person ID (perno), if not specified finds matching person
            
        Returns:
            bool: Whether data was successfully loaded
        """
        try:
            # If ID not specified, use current persona's ID
            if household_id is None:
                household_id = self.id
                
            # Ensure CSV files exist
            if not all(os.path.exists(csv_file) for csv_file in [self.person_csv, self.gps_place_csv, self.location_csv]):
                print(f"Cannot find CSV files: {self.person_csv}, {self.gps_place_csv}, or {self.location_csv}")
                return False
                
            # Read person.csv
            person_df = pd.read_csv(self.person_csv)
            
            # If person_id not specified, find person matching current persona features
            if person_id is None:
                matching_persons = person_df[person_df['sampno'] == household_id]
                if matching_persons.empty:
                    print(f"No records found for household ID {household_id}")
                    return False
                    
                # Try to match by age and gender
                if hasattr(self, 'age') and self.age > 0:
                    age_diff = abs(matching_persons['age'] - self.age)
                    min_diff_idx = age_diff.idxmin()
                    best_match = matching_persons.loc[min_diff_idx]
                    person_id = best_match['perno']
                else:
                    # If no age information, use the first person
                    person_id = matching_persons.iloc[0]['perno']
                
                # Update persona attributes
                selected_person = person_df[(person_df['sampno'] == household_id) & (person_df['perno'] == person_id)].iloc[0]
                self.age = selected_person['age']
                self.gender = 'male' if selected_person['sex'] == 1 else 'female'
                
                # Process occupation information
                if 'occup' in selected_person:
                    self.occupation = self._determine_occupation(selected_person['occup'])
                
                # Process race information
                if 'race' in selected_person:
                    self.race = self._determine_race(selected_person['race'])
                
                # Process disability status
                if 'disab' in selected_person:
                    self.disability = self._has_disability(selected_person['disab'])
                    
                # Process disability type
                if 'dtype' in selected_person:
                    self.disability_type = self._determine_disability_type(selected_person['dtype'])
                
                # Process education level
                if 'educ' in selected_person:
                    self.education = self._determine_education(selected_person['educ'])
                
            # Read location.csv to get location information
            location_df = pd.read_csv(self.location_csv)
            
            # Read gps_place.csv to get travel records
            places_df = pd.read_csv(self.gps_place_csv)
            
            # Filter current person's records
            person_places = places_df[(places_df['sampno'] == household_id) & (places_df['perno'] == person_id)]
            
            if person_places.empty:
                print(f"No travel records found for household ID={household_id}, person ID={person_id}")
                return False
                
            # Create Memory object
            self.memory = Memory()
            
            # Organize data by day - considering day starts at 3:00 AM
            for day_no, day_group in person_places.groupby('dayno'):
                year = 2017
                month = 10
                
                day_date = datetime.datetime(year, month, day_no+1)
                day_data = {
                    'date': day_date.strftime("%Y-%m-%d"),
                    'day_of_week': day_date.strftime("%A"),
                    'day_start_time': "03:00",
                    'day_end_time': "03:00",
                    'activities': []
                }
                
                # Sort by arrival time
                day_places = day_group.sort_values('arrtime')
                
                # Skip if only one record exists
                if len(day_places) <= 1:
                    continue

                # Process each location record
                for i, place in enumerate(day_places.iterrows()):
                    place = place[1]  # Get the Series from the tuple
                    
                    # Get location type
                    loc_no = place['locno']
                    location_record = location_df[
                        (location_df['sampno'] == household_id) & 
                        (location_df['locno'] == loc_no)
                    ]
                    
                    # Determine location type
                    location_type = "other"
                    if not location_record.empty:
                        loc_type = location_record.iloc[0]['loctype']
                        if loc_type == 1:
                            location_type = "home"
                        elif loc_type == 2:
                            location_type = "work"
                        elif loc_type == 3:
                            location_type = "school"
                        elif loc_type == 4:
                            location_type = "transit"
                        else:
                            location_type = "other"
                    
                    # 处理时间
                    try:
                        arr_time = self._parse_time(place['arrtime'])
                        dep_time = self._parse_time(place['deptime'])
                        
                        # 调整跨天的时间（3:00-27:00）
                        arr_h, arr_m = map(int, arr_time.split(':'))
                        dep_h, dep_m = map(int, dep_time.split(':'))
                        
                        if arr_h < 3:
                            arr_h += 24
                            arr_time = f"{arr_h:02d}:{arr_m:02d}"
                        
                        if dep_h < 3:
                            dep_h += 24
                            dep_time = f"{dep_h:02d}:{dep_m:02d}"
                    except:
                        continue
                    
                    # 获取坐标
                    coords = []
                    if not location_record.empty:
                        if 'wgslat' in location_record.columns and 'wgslon' in location_record.columns:
                            lat = location_record.iloc[0]['wgslat']
                            lon = location_record.iloc[0]['wgslon']
                            coords = [float(lat), float(lon)]
                    
                    # Get movement related data
                    travel_time = float(place['travtime']) if not pd.isna(place['travtime']) else 0
                    activity_duration = float(place['actdur']) if not pd.isna(place['actdur']) else 0
                    distance = float(place['distance']) if not pd.isna(place['distance']) else 0
                    
                    # Get transport mode (only when travel_time > 0)
                    transport_mode = None
                    if not pd.isna(place['mode']):
                        transport_mode = self._determine_transport_mode(place['mode'])
                    
                    # Create activity record
                    activity = {
                        'location_type': location_type,
                        'activity_type': self._determine_activity_type(place, location_type),
                        'start_time': arr_time,
                        'end_time': dep_time,
                        'transport_mode': transport_mode,
                        'location': coords if coords else None,
                        'travel_time': int(travel_time),
                        'activity_duration': int(activity_duration),
                        'distance': round(distance, 2)
                    }

                    day_data['activities'].append(activity)
                
                # Add to memory if activities exist
                if day_data['activities']:
                    self.memory.days.append(day_data)
            
            return len(self.memory.days) > 0
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return False
    
    def _parse_time(self, time_str):
        """Parse time string from CSV file"""
        if pd.isna(time_str):
            return "03:00"  # 默认返回一天的开始时间
            
        try:
            # 如果是纯时间格式 (HH:MM)
            if ':' in time_str and len(time_str.split(':')) == 2:
                time_parts = time_str.split(':')
                try:
                    hour, minute = map(int, time_parts)
                    if 0 <= minute <= 59:
                        if hour == 24:  # 特殊处理24:00
                            return "24:00"
                        elif 0 <= hour <= 23:
                            return f"{hour:02d}:{minute:02d}"
                except ValueError:
                    pass

            # 尝试解析完整的日期时间格式
            formats = [
                '%Y-%m-%d %H:%M:%S',  # 2017-10-02 12:40:20
                '%Y/%m/%d %H:%M:%S',  # 2017/10/02 12:40:20
                '%Y-%m-%d %H:%M',     # 2017-10-02 12:40
                '%Y/%m/%d %H:%M',     # 2017/10/02 12:40
                '%m/%d/%Y %H:%M:%S',  # 10/02/2017 12:40:20
                '%m/%d/%Y %H:%M',     # 10/02/2017 12:40
                '%Y-%m-%d %I:%M:%S %p',  # 2017-10-02 12:40:20 PM
                '%Y-%m-%d %I:%M %p'      # 2017-10-02 12:40 PM
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(time_str, fmt)
                    # 如果时间是凌晨3点前，将其视为前一天的延续
                    if dt.hour < 3:
                        hour = dt.hour + 24
                        return f"{hour:02d}:{dt.minute:02d}"
                    return f"{dt.hour:02d}:{dt.minute:02d}"
                except ValueError:
                    continue
                
            return "03:00"  # 如果所有格式都无法解析，返回一天的开始时间
            
        except Exception as e:
            print(f"Error parsing time: {time_str}, error: {e}")
            return "03:00"  # 返回一天的开始时间
    
    def _determine_activity_type(self, place, location_type):
        """ 
        Args:
            place: Place record from GPS data
            location_type: Type of location (home, work, school, etc.)
            
        Returns:
            str: Basic activity type based on location
        """
        # 直接根据位置类型返回对应的活动类型
        if location_type == "home":
            return "home"
        elif location_type == "work":
            return "work"
        elif location_type == "school":
            return "education"
        elif location_type == "transit":
            return "travel"
        else:
            return "other"
    
    def _determine_transport_mode(self, mode_code):
        """Determine transport mode from code in CSV"""
        if pd.isna(mode_code):
            return None
            
        # Convert mode_code to int if it's not already
        try:
            mode_code = int(mode_code)
        except (ValueError, TypeError):
            return None
            
        mode_mapping = {
            -8: "dont_know",           # I don't know
            -7: "prefer_not_answer",   # I prefer not to answer
            -1: "appropriate_skip",    # Appropriate skip
            101: "walk",              # Walk
            102: "own_bike",          # My own bike
            103: "divvy_bike",        # Divvy bike
            104: "zagster_bike",      # Zagster bike
            201: "motorcycle_moped",  # Motorcycle/moped
            202: "auto_driver",       # Auto / van / truck (as the driver)
            203: "auto_passenger",    # Auto / van / truck (as the passenger)
            301: "carpool_vanpool",   # Carpool/vanpool
            401: "school_bus",        # School bus
            500: "rail_and_bus",      # Rail and Bus
            501: "bus_cta_pace",      # Bus (CTA, PACE, Huskie Line, Indiana)
            502: "dial_a_ride",       # Dial-a-Ride
            503: "call_n_ride",       # Call-n-Ride
            504: "paratransit",       # Paratransit
            505: "train",             # Train (CTA, METRA, South Shore Line)
            506: "local_transit",     # Local transit (NIRPC region)
            509: "transit_unspecified", # Transit (specific mode not reported or imputed)
            601: "private_shuttle",    # Private shuttle bus
            701: "taxi",              # Taxi
            702: "private_limo",      # Private limo
            703: "private_car",       # Private car
            704: "uber_lyft",         # Uber/Lyft
            705: "shared_ride",       # Via/Uber Pool/Lyft Line (shared ride)
            801: "airplane"           # Airplane
        }
        
        # Return exact code mapping or None if not found
        return mode_mapping.get(mode_code, None)
    
    def _determine_occupation(self, occup_code):
        """Determine occupation category"""
        if pd.isna(occup_code):
            return "unknown"
            
        occup_mapping = {
            -8: "unknown",  # I don't know
            -7: "unknown",  # I prefer not to answer
            -1: "unknown",  # Appropriate skip
            11: "management",
            13: "business_financial",
            15: "computer_mathematical",
            17: "architecture_engineering",
            19: "life_physical_social_science",
            21: "community_social_service",
            23: "legal",
            25: "education",
            27: "arts_entertainment_media",
            29: "healthcare_practitioners",
            31: "healthcare_support",
            33: "protective_service",
            35: "food_preparation_serving",
            37: "building_maintenance",
            39: "personal_care_service",
            41: "sales",
            43: "office_administrative",
            45: "farming_fishing_forestry",
            47: "construction_extraction",
            49: "installation_maintenance_repair",
            51: "production",
            53: "transportation_material_moving",
            55: "military",
            97: "other"
        }
        
        return occup_mapping.get(int(occup_code), "unknown")
    
    def _determine_race(self, race_code):
        """Determine race category"""
        if pd.isna(race_code):
            return "unknown"
            
        race_mapping = {
            -8: "unknown",  # I don't know
            -7: "unknown",  # I prefer not to answer
            1: "white",
            2: "black",
            3: "asian",
            4: "american_indian_alaska_native",
            5: "pacific_islander",
            6: "multiracial",
            97: "other"
        }
        
        return race_mapping.get(int(race_code), "unknown")
    
    def _has_disability(self, disab_code):
        """Determine if person has a disability"""
        if pd.isna(disab_code):
            return False
            
        disab_mapping = {
            -8: False,  # I don't know
            -7: False,  # I prefer not to answer
            -1: False,  # Appropriate skip
            1: True,    # Yes
            2: False    # No
        }
        
        return disab_mapping.get(int(disab_code), False)
    
    def _determine_disability_type(self, dtype_code):
        """Determine disability type"""
        if pd.isna(dtype_code):
            return None
            
        dtype_mapping = {
            -8: None,  # I don't know
            -7: None,  # I prefer not to answer
            -1: None,  # Appropriate skip
            1: "visual_impairment",
            2: "hearing_impairment",
            3: "mobility_cane_walker",
            4: "wheelchair_non_transferable",
            5: "wheelchair_transferable",
            6: "mental_emotional",
            97: "other"
        }
        
        return dtype_mapping.get(int(dtype_code), None)
    
    def _determine_education(self, educ_code):
        """Determine education level"""
        if pd.isna(educ_code):
            return "unknown"
            
        educ_mapping = {
            -9: "unknown",  # Not ascertained
            -8: "unknown",  # I don't know
            -7: "unknown",  # I prefer not to answer
            1: "less_than_high_school",
            2: "high_school",
            3: "some_college",
            4: "associate_degree",
            5: "bachelor_degree",
            6: "graduate_degree",
            97: "other"
        }
        
        return educ_mapping.get(int(educ_code), "unknown")
    
    def get_location_info(self):
        """
        Get location information for the persona.
        
        Returns:
            dict: Dictionary with location information
        """
        return {
            'home': self.home,
            'work': self.work,
            'current_location': self.current_location
        }
    
    def update_current_location(self, location):
        """
        Update the current location of the persona.
        
        Args:
            location: Tuple of (latitude, longitude)
        """
        self.current_location = location
    
    def update_current_activity(self, activity):
        """
        Update the current activity of the persona.
        
        Args:
            activity: Activity information dictionary
        """
        self.current_activity = activity
        self.current_activity_type = activity.get('activity_type', None) if activity else None
    
