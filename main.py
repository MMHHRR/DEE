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
import concurrent.futures
from functools import partial
import time
import threading
import queue
from threading import Semaphore, Lock

from config import (
    RESULTS_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    MEMORY_DAYS,
    PERSON_CSV_PATH,
    LOCATION_CSV_PATH,
    GPS_PLACE_CSV_PATH,
    HOUSEHOLD_CSV_PATH,
    BATCH_SIZE
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
        
        # Assign the memory object to persona, ensuring the same memory object is always used
        persona.memory = memory
        
        # Try to load historical data if CSV files exist
        try:
            if os.path.exists(PERSON_CSV_PATH) and os.path.exists(LOCATION_CSV_PATH) and os.path.exists(GPS_PLACE_CSV_PATH) and os.path.exists(HOUSEHOLD_CSV_PATH):
                
                if persona.load_historical_data(household_id=household_id, person_id=person_id):
                    print(f"Successfully loaded historical data for household ID {household_id}, person ID {person_id}")
                    
                    # Merge the days data from load_historical_data into our memory
                    if hasattr(persona, 'memory') and persona.memory and persona.memory.days:
                        # Save the original CSV loaded days data
                        csv_loaded_days = persona.memory.days
                        
                        # If the CSV loaded memory is not the same object as our memory
                        if id(persona.memory) != id(memory):                            
                            # Merge the CSV loaded days data into our memory
                            memory.days.extend(csv_loaded_days)
                            memory.initialize_persona(persona)
                            persona.memory = memory
                            
                        else:
                            print("CSV data already in the same memory object")
                    else:
                        print("No historical days loaded from CSV")
                        
                    # Update the persona information in memory, ensuring the latest household coordinates are used
                    memory.initialize_persona(persona)
                else:
                    print(f"Failed to load historical data for household ID {household_id}, person ID {person_id}")

        except Exception as load_error:
            print(f"Error loading historical data: {load_error}")
            traceback.print_exc()
        
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
            parser.add_argument('--workers', type=int, default=4,
                              help='Number of parallel workers')
            parser.add_argument('--memory_days', type=int, default=MEMORY_DAYS,
                              help='Number of days to keep in memory')
            parser.add_argument('--summary', action='store_true', default=True,
                              help='Generate summary report')
            parser.add_argument('--compress', action='store_true', default=False,
                              help='Compress output files')
            parser.add_argument('--household_ids', type=str, default='',
                              help='Comma-separated list of household IDs to simulate (optional)')
            parser.add_argument('--max_concurrent_llm', type=int, default=5, 
                              help='Maximum number of concurrent LLM requests')
            parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                              help='Number of simulations to batch process')
            parser.add_argument('--max_batches', type=int, default=10,
                              help='Maximum number of batches to process (None for all)')
            parser.add_argument('--llm_rate_limit', type=float, default=0.5,
                              help='Minimum seconds between LLM requests to avoid rate limiting')
            parser.add_argument('--no_threading', action='store_true', default=True,
                              help='Disable multi-threading for debugging (runs in single thread)') ##True是单线程，False是多线程
            
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
        
        # Create a rate limiting semaphore for LLM API calls
        llm_semaphore = Semaphore(args.max_concurrent_llm)
        rate_limit_lock = Lock()
        last_request_time = [time.time() - args.llm_rate_limit]  # Use list for mutable reference
        
        # Patch Activity.generate_daily_schedule to implement rate limiting
        original_generate_daily_schedule = Activity.generate_daily_schedule
        
        def rate_limited_generate_daily_schedule(self, *method_args, **kwargs):
            with llm_semaphore:
                # Apply rate limiting
                with rate_limit_lock:
                    time_since_last = time.time() - last_request_time[0]
                    if time_since_last < args.llm_rate_limit:  # Use the command line args object
                        sleep_time = args.llm_rate_limit - time_since_last
                        time.sleep(sleep_time)
                    last_request_time[0] = time.time()
                
                # Call original method
                return original_generate_daily_schedule(self, *method_args, **kwargs)
        
        # Apply the patch
        Activity.generate_daily_schedule = rate_limited_generate_daily_schedule
        
        # Simulate each household-person pair
        results = {}
        results_lock = Lock()  # Lock for thread-safe results dictionary updates
        
        # Define worker function for parallel processing
        def process_household_person_pair(pair, args_dict):
            household_id, person_id = pair
            try:
                # Create a basic persona for each household-person pair
                persona_data = {
                    'id': f"{household_id}_{person_id}",
                    'name': f"Person-{household_id}-{person_id}"
                }
                
                # Simulate persona with specified household and person IDs
                memory = simulate_persona(
                    persona_data, 
                    num_days=args_dict['days'], 
                    start_date=args_dict['start_date'],
                    memory_days=args_dict['memory_days'],
                    household_id=household_id,
                    person_id=person_id
                )
                
                if memory:
                    pair_id = f"{household_id}_{person_id}"
                    
                    # Save activity data
                    output_file = os.path.join(args_dict['output'], f"{pair_id}.json")
                    memory.save_to_file(output_file)
                    
                    # Save activity data to CSV format
                    memory.save_to_csv(output_dir=args_dict['output'], persona_id=person_id)
                    
                    # Compress data if needed
                    if args_dict['compress']:
                        try:
                            compress_trajectory_data(output_file, method='gzip')
                        except Exception as e:
                            print(f"Error compressing data: {e}")
                    
                    # Thread-safe results update
                    with results_lock:
                        # Update progress information
                        print(f"✓ Completed household {household_id}, person {person_id}")
                        return pair_id, memory
                return None
            except Exception as e:
                print(f"Error processing household {household_id}, person {person_id}: {e}")
                traceback.print_exc()
                return None
        
        # Prepare arguments for worker function
        args_dict = {
            'days': args.days,
            'start_date': args.start_date,
            'memory_days': args.memory_days,
            'output': args.output,
            'compress': args.compress
        }
        
        # Determine number of workers
        workers = min(args.workers, len(household_person_pairs))
        workers = max(1, workers)  # Ensure at least 1 worker
        
        # Process in batches to avoid memory issues and improve throughput
        def process_batches():
            total_pairs = len(household_person_pairs)
            batch_size = min(args.batch_size, total_pairs)
            total_batches = (total_pairs + batch_size - 1) // batch_size
            
            # Limit number of batches if requested
            if args.max_batches is not None:
                max_batches = min(args.max_batches, total_batches)
                print(f"Limiting processing to {max_batches} batches out of {total_batches} total")
            else:
                max_batches = total_batches
            
            batch_counter = 0
            for i in range(0, total_pairs, batch_size):
                if batch_counter >= max_batches:
                    print(f"Reached maximum number of batches ({max_batches}), stopping.")
                    break
                    
                batch = household_person_pairs[i:i + batch_size]
                print(f"\nProcessing batch {batch_counter + 1}/{max_batches} " 
                      f"({len(batch)} household-person pairs)")
                
                if args.no_threading:
                    # Single-threaded mode for debugging
                    print("Running in single-thread mode for debugging")
                    for pair in tqdm(batch, desc="Processing pairs"):
                        result = process_household_person_pair(pair, args_dict)
                        if result:
                            pair_id, memory = result
                            with results_lock:
                                results[pair_id] = memory
                else:
                    # Use ThreadPoolExecutor for IO-bound operations (better for API calls)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        # Submit all tasks in this batch
                        futures = {executor.submit(process_household_person_pair, pair, args_dict): pair 
                                  for pair in batch}
                        
                        # Process results as they complete
                        for future in tqdm(concurrent.futures.as_completed(futures), 
                                          total=len(futures), 
                                          desc="Processing pairs"):
                            pair = futures[future]
                            result = future.result()
                            if result:
                                pair_id, memory = result
                                with results_lock:
                                    results[pair_id] = memory
                
                batch_counter += 1
                # Explicit garbage collection between batches
                gc.collect()
        
        # Start batch processing
        print(f"Starting simulation with {workers} parallel workers, " 
              f"max {args.max_concurrent_llm} concurrent LLM requests, "
              f"and {args.batch_size} simulations per batch")
        if args.no_threading:
            print("Warning: Threading disabled. Running in single-thread mode for debugging.")
        process_batches()
        
        # Generate summary report
        if args.summary:
            summary_file = generate_summary_report(results, args.output)
            print(f"Generated summary report: {summary_file}")
            
        # Restore original method
        Activity.generate_daily_schedule = original_generate_daily_schedule
        
        print(f"Completed simulation for {len(results)} household-person pairs")
        return results
        
    except Exception as e:
        print(f"Main function error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 