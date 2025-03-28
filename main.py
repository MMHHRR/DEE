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
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc

from config import (
    RESULTS_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    BATCH_PROCESSING,
    BATCH_SIZE,
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
    batch_save_trajectories,
    compress_trajectory_data,
    generate_summary_report
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
                    
                    # Print transportation information once
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
                        # Automatically adjust activity start time
                        start_time = arrival_time
                
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


def simulate_parallel(persona_list, num_days=7, start_date=None, max_workers=4, memory_days=2, batch_size=10, household_person_pairs=None):
    """
    Simulate multiple personas in parallel using multiprocessing.
    
    Args:
        persona_list: List of persona data
        num_days: Number of days to simulate
        start_date: Starting date
        max_workers: Maximum number of parallel workers
        memory_days: Number of days to keep in memory
        batch_size: Number of personas to process in each batch
        household_person_pairs: List of (household_id, person_id) tuples to simulate
        
    Returns:
        dict: Dictionary mapping household_persona IDs to Memory objects
    """
    results = {}
    
    # If no household-person pairs are provided, use default values
    if household_person_pairs is None or len(household_person_pairs) == 0:
        household_person_pairs = [(20000228, 1), (20001882, 1)]
        print(f"Parallel simulation using {len(household_person_pairs)} default household-person pairs")
    
    # Process each household-person pair with each persona template
    for household_id, person_id in tqdm(household_person_pairs, desc="Simulating household-person pairs"):
        # Create a basic persona data for this household-person pair
        for persona in persona_list:
            # Create a unique ID for this combination
            persona['id'] = f"{household_id}_{person_id}_{persona.get('id', 'unknown')}"
            persona['name'] = f"Person-{household_id}-{person_id}"
    
    # Process persona list in batches to reduce memory usage
    for batch_start in range(0, len(persona_list), batch_size):
        batch_end = min(batch_start + batch_size, len(persona_list))
        current_batch = persona_list[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end}/{len(persona_list)})")
        
        batch_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a mapping of futures to persona IDs
            future_to_persona = {}
            
            for persona in current_batch:
                # Extract household_id and person_id from persona ID
                parts = persona['id'].split('_')
                if len(parts) >= 2:
                    household_id = int(parts[0])
                    person_id = int(parts[1])
                    
                    future = executor.submit(
                        simulate_persona, 
                        persona, 
                        num_days, 
                        start_date,
                        memory_days,
                        household_id,
                        person_id
                    )
                    future_to_persona[future] = persona['id']
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_persona):
                persona_id = future_to_persona[future]
                try:
                    memory = future.result()
                    if memory:
                        batch_results[persona_id] = memory
                        print(f"Completed simulation for {persona_id}")
                    else:
                        print(f"Failed to simulate {persona_id}")
                except Exception as e:
                    print(f"Exception occurred during simulation for {persona_id}: {e}")
        
        # Save current batch results
        for persona_id, memory in batch_results.items():
            results[persona_id] = memory
            
            # Immediately save results to file to free memory
            output_file = os.path.join(RESULTS_DIR, f"{persona_id}.json")
            memory.save_to_file(output_file)
            print(f"Saved results for {persona_id} to {output_file}")
            
            # Save activity data to CSV format
            parts = persona_id.split('_')
            if len(parts) >= 2:
                person_id_str = parts[1]
                csv_path = memory.save_to_csv(output_dir=RESULTS_DIR, persona_id=person_id_str)
            
            # Compress data to save disk space
            try:
                compress_trajectory_data(output_file, method='gzip')
                print(f"Compressed trajectory data for {persona_id}")
            except Exception as e:
                print(f"Error compressing data: {e}")
        
        # Clean up current batch results to free memory
        batch_results.clear()
        gc.collect()
    
    return results


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
            return [(20000228, 1), (20001882, 1)]
        
    except Exception as e:
        print(f"Error loading household IDs: {e}")
        print(f"Using default household-person pairs for testing")
        return [(20000228, 1), (20001882, 1)]


def create_batch_visualizations(results_dir, max_personas_per_vis=10):
    """
    为模拟结果创建批量可视化
    
    Args:
        results_dir: 结果目录
        max_personas_per_vis: 每个可视化包含的最大persona数量
    """
    import os
    import json
    import folium
    from folium.plugins import MarkerCluster
    
    # 查找所有persona结果文件
    persona_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json") and not filename.endswith(".gz"):
            persona_files.append(os.path.join(results_dir, filename))
    
    if not persona_files:
        print("No persona result files found")
        return
    
    # 按批次创建可视化
    for batch_idx in range(0, len(persona_files), max_personas_per_vis):
        batch_files = persona_files[batch_idx:batch_idx + max_personas_per_vis]
        
        # 创建地图
        m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)  # 以芝加哥为中心
        
        # 为每个persona创建一个特征组
        for persona_file in batch_files:
            try:
                # 读取persona数据
                with open(persona_file, 'r') as f:
                    data = json.load(f)
                
                persona_id = data.get('persona_id', 'unknown')
                
                # 创建此persona的特征组
                fg = folium.FeatureGroup(name=f"Persona {persona_id}")
                
                # 添加轨迹点
                for day in data.get('days', []):
                    date = day.get('date', '')
                    
                    # 创建轨迹线
                    locations = []
                    for point in day.get('trajectory', []):
                        if 'location' in point and point['location']:
                            locations.append(point['location'])
                    
                    if len(locations) > 1:
                        # 添加轨迹线
                        folium.PolyLine(
                            locations,
                            color=f'#{hash(str(persona_id)) % 0xFFFFFF:06x}',  # 基于persona_id生成唯一颜色
                            weight=2,
                            opacity=0.7,
                            popup=f"Persona {persona_id} on {date}"
                        ).add_to(fg)
                
                # 将特征组添加到地图
                fg.add_to(m)
                
            except Exception as e:
                print(f"Error processing {persona_file}: {e}")
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 保存地图
        output_file = os.path.join(results_dir, f"batch_visualization_{batch_idx // max_personas_per_vis + 1}.html")
        m.save(output_file)
        print(f"Created batch visualization: {output_file}")


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
            parser.add_argument('--batch', action='store_true', default=BATCH_PROCESSING,
                              help='Use batch processing')
            parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                              help='Batch size for processing')
            parser.add_argument('--workers', type=int, default=4,
                              help='Number of parallel workers')
            parser.add_argument('--memory_days', type=int, default=MEMORY_DAYS,
                              help='Number of days to keep in memory')
            parser.add_argument('--compress', action='store_true', default=False,
                              help='Compress output files')
            parser.add_argument('--summary', action='store_true', default=True,
                              help='Generate summary report')
            parser.add_argument('--household_ids', type=str, default='',
                              help='Comma-separated list of household IDs to simulate (optional)')
            parser.add_argument('--visualize', action='store_true', default=False,
                              help='Generate trajectory visualizations')
            
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
        
        if args.batch:
            # Use batch processing for multiple household-person pairs
            print(f"Using batch processing mode to simulate {len(household_person_pairs)} household-person pairs")
            
            # Process in batches
            batch_size = args.batch_size
            for i in range(0, len(household_person_pairs), batch_size):
                batch = household_person_pairs[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(household_person_pairs))}/{len(household_person_pairs)})")
                
                batch_results = {}
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                    # Create a mapping of futures to pairs
                    future_to_pair = {}
                    for household_id, person_id in batch:
                        # Create a basic persona for each household-person pair
                        persona_data = {
                            'id': f"{household_id}_{person_id}",
                            'name': f"Person-{household_id}-{person_id}"
                        }
                        
                        future = executor.submit(
                            simulate_persona, 
                            persona_data, 
                            args.days, 
                            args.start_date,
                            args.memory_days,
                            household_id,
                            person_id
                        )
                        future_to_pair[future] = (household_id, person_id)
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_pair):
                        household_id, person_id = future_to_pair[future]
                        pair_id = f"{household_id}_{person_id}"
                        try:
                            memory = future.result()
                            if memory:
                                batch_results[pair_id] = memory
                                print(f"Completed simulation for household {household_id}, person {person_id}")
                                
                                # Save results
                                output_file = os.path.join(args.output, f"{pair_id}.json")
                                memory.save_to_file(output_file)
                                
                                # Save to CSV
                                csv_path = memory.save_to_csv(output_dir=args.output, persona_id=person_id)
                                
                                # Compress if needed
                                if args.compress:
                                    try:
                                        compress_trajectory_data(output_file, method='gzip')
                                    except Exception as e:
                                        print(f"Error compressing data: {e}")
                            else:
                                print(f"Failed to simulate household {household_id}, person {person_id}")
                        except Exception as e:
                            print(f"Exception during simulation for household {household_id}, person {person_id}: {e}")
                
                # Add batch results to overall results
                results.update(batch_results)
                
                # Clean up
                batch_results.clear()
                gc.collect()
        else:
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
                    csv_path = memory.save_to_csv(output_dir=args.output, persona_id=person_id)
                    
                    # Compress data if needed
                    if args.compress:
                        try:
                            compress_trajectory_data(output_file, method='gzip')
                        except Exception as e:
                            print(f"Error compressing data: {e}")
        
        # Save all activity data
        activity_path = os.path.join(args.output, 'activities')
        batch_save_trajectories(results, activity_path, format='merged')
        
        # Generate summary report
        if args.summary:
            summary_file = generate_summary_report(results, args.output)
            print(f"Generated summary report: {summary_file}")
            
            # Only create batch visualizations if specifically requested
            if args.visualize:
                create_batch_visualizations(args.output)
        
        print(f"Completed simulation for {len(household_person_pairs)} household-person pairs")
        return results
        
    except Exception as e:
        print(f"Main function error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 