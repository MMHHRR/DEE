"""
Persona module for the LLM-based mobility simulation.
Defines the Persona class to represent individuals with their attributes and characteristics.
"""

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
        self.education = persona_data.get('education', 'unknown')
        self.income = persona_data.get('income', 0)
        self.consumption = persona_data.get('consumption', 'medium')
        self.age = persona_data.get('age', 0)
        
        # Other attributes that might be added later
        self.household_size = persona_data.get('household_size', 1)
        self.has_children = persona_data.get('has_children', False)
        self.has_car = persona_data.get('has_car', True)
        self.has_bike = persona_data.get('has_bike', False)
        
        # Transportation preference - supports two attribute names
        self.preferred_transport = persona_data.get('preferred_transport', 'driving')
        # For compatibility, providing transportation_preference as an alias
        self.transportation_preference = self.preferred_transport
        
        # Current state
        self.current_location = self.home  # Start at home
        self.current_activity = None       # Complete activity information
        self.current_activity_type = None  # Current activity type (string)
    
    def get_basic_info(self):
        """
        Get basic information about the persona.
        
        Returns:
            dict: Dictionary with basic persona information
        """
        return {
            'id': self.id,
            'gender': self.gender,
            'age': self.age,
            'income': self.income,
            'consumption': self.consumption,
            'education': self.education,
            'race': self.race
        }
    
    def get_location_info(self):
        """
        Get location information about the persona.
        
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
            location (tuple): (latitude, longitude)
        """
        self.current_location = location
    
    def update_current_activity(self, activity):
        """
        Update the current activity of the persona.
        
        Args:
            activity (dict): Activity dictionary
        """
        self.current_activity = activity
        if activity and 'activity_type' in activity:
            self.current_activity_type = activity['activity_type']
    
    def to_dict(self):
        """
        Convert persona to dictionary.
        
        Returns:
            dict: Dictionary representation of persona
        """
        data = {
            'id': self.id,
            'name': self.name,
            'gender': self.gender,
            'age': self.age,
            'race': self.race,
            'education': self.education,
            'income': self.income,
            'consumption': self.consumption
        }
        
        # Include location data if available
        if hasattr(self, 'home') and self.home:
            data['home'] = self.home
        
        if hasattr(self, 'work') and self.work:
            data['work'] = self.work
            
        if hasattr(self, 'current_location') and self.current_location:
            data['current_location'] = self.current_location
            
        # Add transportation preference (consistently use preferred_transport)
        if hasattr(self, 'preferred_transport') and self.preferred_transport:
            data['preferred_transport'] = self.preferred_transport
            
        # Include vehicle availability if defined
        for attr in ['has_car', 'has_bike']:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)
        
        # Include current activity if available
        if hasattr(self, 'current_activity_type') and self.current_activity_type:
            data['current_activity_type'] = self.current_activity_type
            
        return data
    
    def __str__(self):
        """
        Return string representation of persona.
        
        Returns:
            str: String representation
        """
        return (
            f"Persona {self.id}: {self.name}\n"
            f"Age: {self.age}, Gender: {self.gender}\n"
            f"Income: ${self.income}, Education: {self.education}\n"
            f"Consumption: {self.consumption}"
        ) 