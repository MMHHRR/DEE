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
        self.preferred_transport = persona_data.get('preferred_transport', 'driving')
        
        # Current state
        self.current_location = self.home  # Start at home
        self.current_activity = None
    
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
            activity (str): Current activity
        """
        self.current_activity = activity
    
    def to_dict(self):
        """
        Convert persona to dictionary representation.
        
        Returns:
            dict: Dictionary representation of the persona
        """
        return {
            'id': self.id,
            'home': self.home,
            'work': self.work,
            'gender': self.gender,
            'race': self.race,
            'education': self.education,
            'income': self.income,
            'consumption': self.consumption,
            'age': self.age,
            'household_size': self.household_size,
            'has_children': self.has_children,
            'has_car': self.has_car,
            'has_bike': self.has_bike,
            'preferred_transport': self.preferred_transport,
            'current_location': self.current_location,
            'current_activity': self.current_activity
        }
    
    def __str__(self):
        """String representation of the persona."""
        return f"Persona {self.id}: {self.gender}, {self.age} years old, {self.education} education, ${self.income} income" 