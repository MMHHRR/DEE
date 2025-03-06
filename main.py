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
import multiprocessing
import concurrent.futures

from config import (
    PERSONA_DATA_PATH,
    RESULTS_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    ENVIRONMENTAL_FACTORS,
    TRANSPORT_MODES,
    BATCH_PROCESSING,
    BATCH_SIZE
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
    normalize_transport_mode,
    estimate_travel_time,
    cached,
    cache,
    time_to_minutes
)


@cached
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

def simulate_single_day(persona, date, activity_generator, destination_selector, memory):
    """
    Simulate daily activities, ensuring the use of specified transportation modes
    
    Args:
        persona: Persona object
        date: Date string (YYYY-MM-DD)
        activity_generator: Activity generator instance
        destination_selector: Destination selector instance
        memory: Memory instance for recording activities
        
    Returns:
        bool: Whether the simulation was successful
    """
    try:
        # Update memory
        print(f"Simulating activities for {persona.name} on {date}...")
        memory.start_new_day(date)
        
        # 1. Generate schedule (including transportation modes)
        daily_activities = activity_generator.generate_daily_schedule(persona, date)
        
        # Ensure daily activities data is valid
        if not daily_activities:
            print(f"Warning: No activities generated for {persona.name} on {date}")
            return False
            
        # Ensure activities are sorted by time
        daily_activities.sort(key=lambda x: x['start_time'])
        
        # Remove duplicate activities (based on start time and activity type)
        activity_keys = set()
        unique_activities = []
        for activity in daily_activities:
            activity_key = f"{activity['start_time']}_{activity['activity_type']}"
            if activity_key not in activity_keys:
                activity_keys.add(activity_key)
                unique_activities.append(activity)
        
        # Update schedule with deduplicated activities
        daily_activities = unique_activities
        
        # 2. Select specific destinations for each activity and handle movement
        for i, activity in enumerate(daily_activities):
            # Get activity information
            activity_type = activity.get('activity_type')
            start_time = activity.get('start_time')
            end_time = activity.get('end_time')
            description = activity.get('description', '')
            location_type = activity.get('location_type', '')
            
            # Calculate available time (minutes)
            available_minutes = time_difference_minutes(start_time, end_time)
            
            # Get activity specified transportation mode
            transport_mode = activity.get('transport_mode')
            if transport_mode:
                transport_mode = normalize_transport_mode(transport_mode)
            
            try:
                print(f"  - {start_time}: {activity_type} - {description}")
                
                # Check if it's a fixed location (home or work place)
                if location_type.lower() == 'home' and hasattr(persona, 'home'):
                    destination = persona.home
                    details = {'name': 'Home', 'address': 'Home location'}
                elif location_type.lower() == 'work' and hasattr(persona, 'work'):
                    destination = persona.work
                    details = {'name': 'Work', 'address': 'Work location'}
                else:
                    # Select destination location - pass transportation mode and time window
                    destination, details = destination_selector.select_destination(
                        persona, 
                        persona.current_location,  # Current location
                        activity_type,  # Activity type
                        start_time,  # Start time
                        get_day_of_week(date),  # Day of week
                        available_minutes,  # Available time
                        transport_mode  # Activity specified transportation mode
                    )
                
                # Store destination details in activity
                activity['location_name'] = details.get('name', 'Unknown location')
                
                # 3. Handle movement and record to memory
                if i > 0 and not are_locations_same(persona.current_location, destination):
                    prev_activity = daily_activities[i-1]
                    prev_end_time = prev_activity['end_time']
                    
                    # Use activity specified transportation mode to calculate travel time
                    travel_time, actual_transport_mode = estimate_travel_time(
                        persona.current_location, 
                        destination, 
                        transport_mode,
                        persona
                    )
                    
                    # Calculate travel arrival time
                    arrival_time = format_time_after_minutes(prev_end_time, travel_time)
                    
                    # Print transportation information only once, use activity location name, not Home
                    location_name = activity['location_name']
                    if location_type.lower() in ['work', 'home']:
                        location_name = location_type.capitalize()
                    
                    print(f"(Traveling to {location_name} by {actual_transport_mode}, " +
                          f"estimated travel time: {travel_time} min)")
                    
                    # Record travel to memory
                    memory.record_travel(
                        persona.current_location,
                        destination,
                        prev_end_time,
                        arrival_time,
                        actual_transport_mode
                    )
                    
                    # Check for time conflicts
                    if time_to_minutes(arrival_time) > time_to_minutes(start_time):
                        # print(f"Schedule conflict: Arrival time {arrival_time}, Original activity start time {start_time}")
                        # Automatically adjust activity start time
                        start_time = arrival_time
                        # print(f"Auto-adjusted: New activity start time is {start_time}")
                
                # Record activity to memory
                memory.record_activity(activity, destination, start_time)
                
                # Update current location and activity
                persona.current_location = destination
                persona.update_current_activity(activity)
                
            except Exception as activity_error:
                print(f"Error processing activity: {activity_error}")
                continue
        
        # Complete day simulation
        memory.end_day()
        return True
        
    except Exception as e:
        print(f"Error simulating day: {e}")
        if memory.current_day:
            memory.end_day()  # Ensure end date recorded
        return False

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
        
        # Check if using parallel processing, usually start parallel processing when simulating more than 3 days
        if num_days > 3 and not BATCH_PROCESSING:
            # Parallel processing multiple dates
            max_workers = min(multiprocessing.cpu_count(), num_days)  # 使用最小的合理进程数
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create date processing tasks
                future_to_date = {
                    executor.submit(
                        simulate_single_day, 
                        persona, 
                        date, 
                        activity_generator, 
                        destination_selector, 
                        memory
                    ): date for date in dates_str
                }
                
                # Process results
                for future in concurrent.futures.as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        day_result = future.result()
                        if day_result:
                            # Merge results into memory
                            memory.days.append(day_result)
                    except Exception as e:
                        print(f"Error processing {date}: {e}")
        else:
            # Sequential processing for small number of dates or when using batch processing
            for date in tqdm(dates_str, desc=f"Simulating days for {persona.id}"):
                simulate_single_day(persona, date, activity_generator, destination_selector, memory)
        
        # Save memory to file
        memory.save_memory()
        
        # Generate visualizations
        memory.create_visualizations()
        
        # Print cache statistics
        if hasattr(cache, 'stats'):
            cache_stats = cache.stats()
            print(f"\nCache statistics: {cache_stats}")
            print(f"Cache hit ratio: {cache_stats['hit_ratio']:.2f}")
        
        return memory
    except Exception as e:
        print(f"Error simulating persona {persona_data.get('id')}: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function to run the simulation."""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load personas data
    personas_data = load_json(PERSONA_DATA_PATH)
    
    # Parallel processing multiple people based on CPU core count
    max_workers = max(1, multiprocessing.cpu_count() - 1)  # Reserve one core for system
    
    if len(personas_data) > 1 and not BATCH_PROCESSING:
        # Parallel processing multiple people
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create person processing tasks
            future_to_persona = {
                executor.submit(
                    simulate_persona, 
                    persona, 
                    NUM_DAYS_TO_SIMULATE, 
                    SIMULATION_START_DATE
                ): persona for persona in personas_data
            }
            
            # Process results
            for future in concurrent.futures.as_completed(future_to_persona):
                persona = future_to_persona[future]
                try:
                    memory = future.result()
                except Exception as e:
                    print(f"Error processing persona {persona['id']}: {e}")
    else:
        # Sequential processing people
        for i, persona_data in enumerate(personas_data):
            print(f"\n\n--- Simulating Persona {i+1}/{len(personas_data)}: {persona_data['id']} ---\n")
            simulate_persona(persona_data, NUM_DAYS_TO_SIMULATE, SIMULATION_START_DATE)
    
    print("\nSimulation completed.")

def are_locations_same(loc1, loc2, threshold=0.001):
    """
    Check if two location coordinates are close enough to be considered the same place
    
    Args:
        loc1: First location coordinates (latitude, longitude)
        loc2: Second location coordinates (latitude, longitude)
        threshold: Maximum distance difference (in degrees) to consider two locations the same
        
    Returns:
        bool: True if the two locations are close enough
    """
    if not loc1 or not loc2:
        return False
        
    # Check if the latitude and longitude of the two coordinates are close enough
    return (abs(loc1[0] - loc2[0]) < threshold and 
            abs(loc1[1] - loc2[1]) < threshold)

if __name__ == "__main__":
    main() 