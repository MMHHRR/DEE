"""
Main module for the LLM-based mobility simulation.
Coordinates all components to simulate human daily mobility.
"""

import os
import json
import datetime
import pandas as pd
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc

from config import (
    RESULTS_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    MEMORY_DAYS,
    PERSON_CSV_PATH,
    LOCATION_CSV_PATH,
    GPS_PLACE_CSV_PATH,
    HOUSEHOLD_CSV_PATH
)
from persona import Persona
from activity import Activity
from destination import Destination
from memory import Memory
from utils import (
    load_json,
    save_json,
    calculate_distance,
    generate_date_range,
    get_day_of_week,
    time_difference_minutes,
    parse_time,
    normalize_transport_mode,
    estimate_travel_time,
    cached,
    format_time_after_minutes,
    time_to_minutes,
    compress_trajectory_data,
    generate_summary_report
)


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
        
        # 2. Store daily activities in memory
        for idx, activity in enumerate(daily_activities):
            try:
                # Extract activity details directly from activity dictionary
                activity_type = activity.get('activity_type')
                start_time = activity.get('start_time')
                end_time = activity.get('end_time')
                description = activity.get('description')
                location_name = activity.get('location_name')
                location_type = activity.get('location_type')
                coordinates = activity.get('coordinates')
                transport_mode = activity.get('transport_mode')
                distance = activity.get('distance', 0)
                travel_time = activity.get('travel_time', 0)
                from_location = activity.get('from_location')
                to_location = activity.get('to_location')
                
                # Print travel information if applicable
                if transport_mode and transport_mode != 'No need to travel' and travel_time > 0:
                    print(f"(Traveling to {location_name} by {transport_mode}, "
                          f"estimated travel time: {travel_time} min)")
                
                # Record activity in memory with all available details
                memory.record_mobility_event(
                    activity_type=activity_type,
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    location_name=location_name,
                    location_type=location_type,
                    coordinates=coordinates,
                    transport_mode=transport_mode,
                    distance=distance,
                    travel_time=travel_time,
                    start_location=from_location,
                    end_location=to_location
                )
                
                # Update current location and activity if coordinates are provided
                if coordinates:
                    persona.current_location = coordinates
                persona.update_current_activity(activity)
                
            except Exception as activity_error:
                print(f"Error processing activity: {activity_error}")
                traceback.print_exc()
                continue
        
        # Complete day simulation
        memory.end_day()
        return True
        
    except Exception as e:
        print(f"Error simulating day: {e}")
        traceback.print_exc()
        if memory.current_day:
            memory.end_day()  # Ensure end date recorded
        return False


def simulate_persona(persona_data, num_days=7, start_date=None, memory_days=2, household_id=None, person_id=None):
    """
    Simulate daily activities for a persona over a period of time.
    
    Args:
        persona_data: Persona data
        num_days: Number of days to simulate
        start_date: Starting date (YYYY-MM-DD format)
        memory_days: Number of days to keep in memory
        household_id: Specified household ID to use for loading historical data
        person_id: Specified person ID to use for loading historical data
    
    Returns:
        Memory: Object containing the simulation results
    """
    try:
        persona = Persona(persona_data)
        
        # Initialize date
        if start_date is None:
            start_date = SIMULATION_START_DATE
            
        date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        
        # Initialize components
        activity_generator = Activity()
        destination_selector = Destination()
        memory = Memory(persona.id, memory_days=memory_days)
        memory.initialize_persona(persona)
        
        # Try to load historical data if CSV files exist
        try:
            if os.path.exists(PERSON_CSV_PATH) and os.path.exists(LOCATION_CSV_PATH) and os.path.exists(GPS_PLACE_CSV_PATH) and os.path.exists(HOUSEHOLD_CSV_PATH):
                print(f"Attempting to load historical data for {persona.name}...")
                
                if persona.load_historical_data(household_id=household_id, person_id=person_id):
                    print(f"Successfully loaded historical data for household ID {household_id}, person ID {person_id}")
                else:
                    print(f"Failed to load historical data for household ID {household_id}, person ID {person_id}")

        except Exception as load_error:
            print(f"Error loading historical data: {load_error}")
        
        # Generate date range for simulation
        date_range = generate_date_range(start_date, num_days)
        
        # Simulate each day
        for date in date_range:
            simulate_single_day(persona, date, activity_generator, destination_selector, memory)
            
            # Clean memory after each simulation
            gc.collect()
        
        return memory
    except Exception as e:
        print(f"Error in persona simulation: {e}")
        traceback.print_exc()
        return None    


def load_household_ids():
    """
    Load list of household and person IDs from the GPS place CSV file
    
    Returns:
        list: List of tuples (household_id, person_id)
    """
    try:
        # Read household IDs and person IDs from the GPS place CSV file using pandas
        if os.path.exists(GPS_PLACE_CSV_PATH):
            # Read 'sampno' and 'perno' columns
            place_df = pd.read_csv(GPS_PLACE_CSV_PATH, usecols=['sampno', 'perno'])
            # Get unique combinations of sampno and perno
            household_person_pairs = place_df[['sampno', 'perno']].drop_duplicates().values.tolist()
            
            print(f"Successfully loaded {len(household_person_pairs)} household-person pairs from {GPS_PLACE_CSV_PATH}")
            return household_person_pairs
        else:
            # If the file doesn't exist, use default values
            print(f"Warning: Could not find {GPS_PLACE_CSV_PATH}, using default household-person pairs for testing")
            return [(1, 1)]  # Default test pair
        
    except Exception as e:
        print(f"Error loading household IDs: {e}")
        return [(1, 1)]  # Default test pair


def main(args=None):
    """Main entry point for the simulation."""
    try:
        # Parse command line arguments
        if args is None:
            parser = argparse.ArgumentParser(description='Run mobility simulation')
            parser.add_argument('--days', type=int, default=NUM_DAYS_TO_SIMULATE,
                              help='Number of days to simulate')
            parser.add_argument('--start_date', type=str, default=SIMULATION_START_DATE,
                              help='Start date (YYYY-MM-DD)')
            parser.add_argument('--output', type=str, default=RESULTS_DIR,
                              help='Output directory')
            parser.add_argument('--workers', type=int, default=1,
                              help='Number of parallel workers (currently not used)')
            parser.add_argument('--memory_days', type=int, default=MEMORY_DAYS,
                              help='Number of days to keep in memory')
            parser.add_argument('--summary', action='store_true', default=True,
                              help='Generate summary report')
            parser.add_argument('--compress', action='store_true', default=False,
                              help='Compress output files')
            parser.add_argument('--household_ids', type=str, default='',
                              help='Comma-separated list of household IDs to simulate (optional)')
            
            args = parser.parse_args()
        
        # Create results directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Load household-person pairs
        household_person_pairs = []
        if args.household_ids:
            # If household IDs are provided in command line, use them with default person ID 1
            household_ids = [int(hid.strip()) for hid in args.household_ids.split(',') if hid.strip()]
            household_person_pairs = [(hid, 1) for hid in household_ids]
            print(f"Using command line provided {len(household_person_pairs)} household IDs for simulation")
        else:
            # Otherwise load from file
            household_person_pairs = load_household_ids()
            print(f"Will simulate {len(household_person_pairs)} household-person pairs")
            
        if not household_person_pairs:
            print("No valid household-person pairs found, cannot proceed with simulation")
            return {}
            
        # Simulate each household-person pair
        results = {}
        
        # Sequential processing for each household-person pair
        for household_id, person_id in tqdm(household_person_pairs, desc="Simulating household-person pairs"):
            # Create a basic persona for each household-person pair
            persona_data = {
                'id': f"{household_id}_{person_id}",
                'name': f"Person-{household_id}-{person_id}"
            }
            
            # Simulate persona with specified household and person IDs
            memory = simulate_persona(
                persona_data, 
                num_days=args.days, 
                start_date=args.start_date,
                memory_days=args.memory_days,
                household_id=household_id,
                person_id=person_id
            )
            
            if memory:
                pair_id = f"{household_id}_{person_id}"
                results[pair_id] = memory
                
                # Save activity data
                output_file = os.path.join(args.output, f"{pair_id}.json")
                memory.save_to_file(output_file)
                print(f"Saved activity data for household {household_id}, person {person_id} to {output_file}")
                
                # Save activity data to CSV format
                memory.save_to_csv(output_dir=args.output, persona_id=person_id)
                
                # Compress data if needed
                if args.compress:
                    try:
                        compress_trajectory_data(output_file, method='gzip')
                    except Exception as e:
                        print(f"Error compressing data: {e}")
        
        # Generate summary report
        if args.summary:
            summary_file = generate_summary_report(results, args.output)
            print(f"Generated summary report: {summary_file}")
            
        
        print(f"Completed simulation for {len(household_person_pairs)} household-person pairs")
        return results
        
    except Exception as e:
        print(f"Main function error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 