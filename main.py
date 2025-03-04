"""
Main module for the LLM-based mobility simulation.
Coordinates all components to simulate human daily mobility.
"""

import os
import json
import random
import datetime
import numpy as np
from tqdm import tqdm
import traceback

from config import (
    PERSONA_DATA_PATH,
    RESULTS_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    ENVIRONMENTAL_FACTORS,
    TRANSPORT_MODES
)
from persona import Persona
from activity import Activity
from destination import Destination
from memory import Memory
from utils import (
    load_json,
    calculate_distance,
    generate_date_range,
    get_day_of_week,
    time_difference_minutes,
    parse_time,
    normalize_transport_mode
)

# def simulate_environmental_exposure(location, activity_type):
#     """
#     Simulate environmental exposure for a given location and activity type.
#     This is a placeholder that would be replaced by actual environmental data.
    
#     Args:
#         location: (latitude, longitude)
#         activity_type: Type of activity
    
#     Returns:
#         dict: Dictionary of environmental exposures
#     """
#     # Safety check for location parameter
#     if not location or not isinstance(location, (list, tuple)) or len(location) < 2:
#         print(f"Invalid location format: {location}. Using default values.")
#         # Return default values if location is invalid
#         return {
#             'air_quality': 50.0,
#             'noise_level': 50.0,
#             'green_space': 50.0,
#             'urban_density': 50.0,
#             'traffic_density': 50.0
#         }
    
#     # In a real application, this would query environmental data APIs
#     # For now, we'll generate random values with some correlation to the activity type
    
#     exposures = {}
    
#     # Safety check for activity_type
#     if not isinstance(activity_type, str):
#         activity_type = str(activity_type)
    
#     activity_type = activity_type.lower()
    
#     # Air quality (lower is better, 0-100)
#     if activity_type in ['work', 'shopping', 'dining']:
#         # Urban activities tend to have worse air quality
#         exposures['air_quality'] = random.uniform(40, 80)
#     else:
#         exposures['air_quality'] = random.uniform(20, 60)
    
#     # Noise level (lower is better, 0-100)
#     if activity_type in ['work', 'shopping', 'social']:
#         exposures['noise_level'] = random.uniform(40, 80)
#     else:
#         exposures['noise_level'] = random.uniform(20, 50)
    
#     # Green space availability (higher is better, 0-100)
#     if activity_type in ['recreation', 'leisure']:
#         exposures['green_space'] = random.uniform(40, 90)
#     else:
#         exposures['green_space'] = random.uniform(10, 50)
    
#     # Urban density (lower for suburban/rural areas, 0-100)
#     if activity_type in ['work', 'shopping']:
#         exposures['urban_density'] = random.uniform(60, 90)
#     else:
#         exposures['urban_density'] = random.uniform(30, 70)
    
#     # Traffic density (lower is better, 0-100)
#     if activity_type in ['work', 'shopping']:
#         exposures['traffic_density'] = random.uniform(50, 90)
#     else:
#         exposures['traffic_density'] = random.uniform(20, 60)
    
#     return exposures

def estimate_travel_time(start_location, end_location, transport_mode, persona=None):
    """
    Estimate travel time between two locations based on distance and transport mode.
    
    Args:
        start_location: (latitude, longitude)
        end_location: (latitude, longitude)
        transport_mode: Mode of transportation
        persona: 人物对象，包含交通偏好信息
    
    Returns:
        tuple: (travel_time_minutes, selected_transport_mode)
    """
    distance_km = calculate_distance(start_location, end_location)
    
    # 如果没有指定交通方式，根据距离和人物特征自动选择合适的交通方式
    if not transport_mode:
        # 默认交通方式选择逻辑
        if persona:
            # 考虑人物特征和偏好
            if distance_km < 0.5:
                # 非常短距离，通常选择步行
                transport_mode = 'walking'
            elif distance_km < 3:
                # 短距离，根据是否有自行车和偏好决定
                if persona.has_bike and (persona.preferred_transport == 'cycling' or random.random() < 0.7):
                    transport_mode = 'cycling'
                else:
                    transport_mode = 'walking'
            elif distance_km < 10:
                # 中等距离
                if persona.has_bike and persona.preferred_transport == 'cycling':
                    transport_mode = 'cycling'
                elif persona.has_car and persona.preferred_transport == 'driving':
                    transport_mode = 'driving'
                elif persona.preferred_transport == 'public_transit':
                    transport_mode = 'public_transit'
                else:
                    # 根据可用性选择
                    if persona.has_car:
                        transport_mode = 'driving'
                    elif persona.has_bike:
                        transport_mode = 'cycling'
                    else:
                        transport_mode = 'public_transit'
            else:
                # 长距离
                if persona.has_car:
                    transport_mode = 'driving'
                elif persona.preferred_transport == 'public_transit':
                    transport_mode = 'public_transit'
                else:
                    transport_mode = 'rideshare'
        else:
            # 如果没有人物信息，使用简单的基于距离的逻辑
            if distance_km < 1:
                transport_mode = 'walking'
            elif distance_km < 5:
                transport_mode = 'cycling'
            elif distance_km < 15:
                transport_mode = 'public_transit'
            else:
                transport_mode = 'driving'
    
    # Normalize transport mode
    transport_mode = normalize_transport_mode(transport_mode)
    
    # Average speeds in km/h
    speeds = {
        'walking': 5,
        'cycling': 15,
        'driving': 30,
        'public_transit': 20,
        'rideshare': 25
    }
    
    # Get speed for the transport mode, default to walking if not found
    speed_kmh = speeds.get(transport_mode, speeds['walking'])
    
    # Calculate time in minutes
    time_hours = distance_km / speed_kmh
    time_minutes = int(time_hours * 60)
    
    # Add some randomness to account for traffic, waiting times, etc.
    time_minutes = int(time_minutes * random.uniform(0.8, 1.2))
    
    # Minimum travel time
    return max(5, time_minutes), transport_mode

def format_time_after_minutes(start_time, minutes):
    """
    Calculate a new time after adding minutes to a start time.
    
    Args:
        start_time: Time string (HH:MM)
        minutes: Minutes to add
    
    Returns:
        str: New time string (HH:MM)
    """
    start_h, start_m = parse_time(start_time)
    
    # Add minutes
    total_minutes = start_h * 60 + start_m + minutes
    
    # Convert back to hours and minutes
    new_h = (total_minutes // 60) % 24
    new_m = total_minutes % 60
    
    return f"{new_h:02d}:{new_m:02d}"

def simulate_persona(persona_data, num_days=7, start_date=None):
    """
    Simulate daily activities for a persona over a period of time.
    
    Args:
        persona_data: Dictionary with persona data
        num_days: Number of days to simulate
        start_date: Starting date (YYYY-MM-DD format)
    
    Returns:
        Memory: Object containing the simulation results
    """
    try:
        # Create persona object
        persona = Persona(persona_data)
        
        # Initialize date
        if start_date is None:
            start_date = SIMULATION_START_DATE
            
        date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        
        # Initialize components
        activity_generator = Activity()
        destination_selector = Destination()
        
        # Initialize memory
        memory = Memory(persona.id)
        memory.initialize_persona(persona)
        
        # Add memory object to persona for use in generating activities
        persona.memory = memory
        
        # Generate dates
        dates = [date_obj + datetime.timedelta(days=i) for i in range(num_days)]
        dates_str = [d.strftime("%Y-%m-%d") for d in dates]
        
        # Create progress bar
        progress_bar = tqdm(dates_str, desc=f"Simulating persona {persona.id} activities")

        # Simulate each day
        for date in progress_bar:
            try:
                # Update persona's current date
                persona.current_date = date
                day_of_week = get_day_of_week(date)
                
                # Start a new day in memory
                memory.start_new_day(date)
                
                # Generate daily activities
                daily_activities = activity_generator.generate_daily_schedule(persona, date)
                
                # Filter out possible sleep activities generated by LLM
                daily_activities = [a for a in daily_activities if a['activity_type'].lower() != 'sleep']
                
                # Ensure activities are sorted correctly by time
                daily_activities.sort(key=lambda x: x['start_time'])
                
                # Process activity list
                processed_activities = []
                
                # Add morning sleep activity (00:00 to first activity start)
                first_activity = daily_activities[0] if daily_activities else None
                if not first_activity or parse_time(first_activity['start_time']) > parse_time("05:00"):
                    first_activity_start = first_activity['start_time'] if first_activity else "07:00"
                    morning_sleep = {
                        "activity_type": "sleep",
                        "description": "Nighttime rest",
                        "start_time": "00:00",
                        "end_time": first_activity_start,
                        "location_type": "home",
                        "transport_mode": None
                    }
                    processed_activities.append(morning_sleep)
                
                # Add all non-sleep activities
                processed_activities.extend(daily_activities)
                
                # Process each activity
                current_time = "00:00"
                persona.current_location = persona.home  # Start from home
                
                final_activities = []
                for i, activity in enumerate(processed_activities):
                    try:
                        # Get current activity details
                        activity_type = activity.get('activity_type', 'unknown')
                        start_time = activity.get('start_time', '00:00')
                        end_time = activity.get('end_time', '23:59')
                        
                        # Ensure activity start time is not earlier than current time
                        if parse_time(start_time) < parse_time(current_time):
                            start_time = current_time
                            activity['start_time'] = start_time
                        
                        # Refine activity details
                        if activity_type.lower() == 'sleep':
                            refined_activity = activity.copy()
                            refined_activity['location_type'] = 'home'
                        else:
                            refined_activity = activity_generator.refine_activity(persona, activity, date)
                        
                        # For fixed location activities (sleep, home, work), use the predefined locations
                        if activity_type.lower() == 'sleep' or refined_activity.get('location_type', '').lower() == 'home':
                            location = persona.home
                            refined_activity['location_name'] = "Home"
                        elif refined_activity.get('location_type', '').lower() in ['work', 'workplace']:
                            location = persona.work
                            refined_activity['location_name'] = "Workplace"
                        else:
                            try:
                                location = destination_selector.select_destination(
                                    persona, 
                                    refined_activity,
                                    date,
                                    start_time,
                                    day_of_week
                                )
                            except Exception as dest_e:
                                print(f"Error selecting destination: {dest_e}. Using home location.")
                                location = persona.home
                                refined_activity['location_name'] = "Home (fallback)"
                        
                        # Safety check for location format
                        if not location or not isinstance(location, (list, tuple)) or len(location) < 2:
                            print(f"Invalid location format after destination selection: {location}. Using home.")
                            location = persona.home
                            refined_activity['location_name'] = "Home (invalid location)"
                        
                        # Estimate travel time if location is different from current location
                        transport_mode = refined_activity.get('transport_mode')
                        
                        if activity_type.lower() != 'sleep' and i > 0 and persona.current_location != location:
                            try:
                                # 不使用预定义的交通方式，而是由系统根据距离和人物特征自动选择
                                travel_time, selected_transport_mode = estimate_travel_time(
                                    persona.current_location, 
                                    location, 
                                    None,  # 不预先指定交通方式
                                    persona  # 传入人物对象
                                )
                                
                                # 使用选择的交通方式
                                transport_mode = selected_transport_mode
                                
                                # Record travel information
                                travel_start_time = current_time
                                travel_end_time = format_time_after_minutes(travel_start_time, travel_time)
                                
                                # Update activity start time to account for travel time
                                if parse_time(start_time) < parse_time(travel_end_time):
                                    start_time = travel_end_time
                                    refined_activity['start_time'] = start_time
                                
                                # Record travel activity
                                memory.record_travel(
                                    persona.current_location,
                                    location,
                                    travel_start_time,
                                    travel_end_time,
                                    transport_mode
                                )
                                
                                # Update current time
                                current_time = travel_end_time
                                
                            except Exception as travel_e:
                                print(f"Error recording travel: {travel_e}")
                        
                        # Update current location
                        persona.current_location = location
                        
                        # Record activity in memory
                        memory.record_activity(
                            refined_activity,
                            location,
                            refined_activity.get('start_time', '00:00')
                        )
                        
                        # Update current time to activity end time
                        current_time = end_time
                        
                        final_activities.append(refined_activity)
                        
                    except Exception as activity_e:
                        print(f"Error processing activity: {activity_e}")
                
                # End the day
                memory.end_day()
                
                # Analyze the day's activity patterns and update memory
                if hasattr(activity_generator, 'analyze_memory_patterns'):
                    patterns = activity_generator.analyze_memory_patterns(memory)
                    memory.update_patterns(patterns)
                
            except Exception as day_e:
                print(f"Error simulating day {date}: {day_e}")
        
        # Save and return memory
        memory.save_memory()
        memory.create_visualizations()
        
        return memory
    except Exception as e:
        print(f"Error simulating persona {persona_data.get('id', 'unknown')}: {e}")
        raise

def main():
    """
    Main function to run the simulation.
    """
    try:
        print("Starting LLM-based mobility simulation...")
        
        # Load personas from JSON file
        with open(PERSONA_DATA_PATH, 'r') as f:
            personas_data = json.load(f)
        
        print(f"Loaded {len(personas_data)} personas.")
        
        # Simulate activities for each persona
        for persona_data in tqdm(personas_data, desc="Simulating persona activities"):
            try:
                simulate_persona(
                    persona_data, 
                    num_days=NUM_DAYS_TO_SIMULATE, 
                    start_date=SIMULATION_START_DATE
                )
            except Exception as e:
                print(f"Error simulating persona {persona_data.get('id', 'unknown')}: {e}")
        
        print("Simulation completed.")
    except Exception as e:
        print(f"Error in main simulation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 