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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import time
import threading
import queue
from threading import Semaphore, Lock
import random
import psutil

from config import (
    RESULTS_DIR,
    SAMPLE_DIR,
    NUM_DAYS_TO_SIMULATE,
    SIMULATION_START_DATE,
    MEMORY_DAYS,
    PERSON_CSV_PATH,
    LOCATION_CSV_PATH,
    GPS_PLACE_CSV_PATH,
    HOUSEHOLD_CSV_PATH,
    BATCH_SIZE,
    BATCH_PROCESSING
)
from persona import Persona
from activity import Activity
from destination import Destination
from memory import Memory
from utils import (
    generate_date_range,
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


def load_household_ids(stratified_sample=False, sample_size=None, seed=42, use_saved_sample=False, save_sample=False):
    """
    Load list of household and person IDs from the GPS place CSV file
    
    Args:
        stratified_sample (bool): Whether to use stratified sampling based on gender, age and income
        sample_size (int): Size of sample to take, None means all records
        seed (int): Random seed for reproducibility
        use_saved_sample (bool): Whether to use previously saved sample
        save_sample (bool): Whether to save the current sample
        
    Returns:
        list: List of tuples (household_id, person_id)
    """
    try:
        # 设置固定的随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        # 检查是否存在已保存的抽样结果
        sample_file = os.path.join(SAMPLE_DIR, f'stratified_sample_{sample_size}.json')
        
        if use_saved_sample and os.path.exists(sample_file):
            print(f"Loading saved stratified sample from {sample_file}")
            with open(sample_file, 'r') as f:
                saved_data = json.load(f)
                household_person_pairs = saved_data['household_person_pairs']
                print("\nSaved Sample Statistics:")
                print(f"Gender distribution: {saved_data['statistics']['gender']}")
                print(f"Age group distribution: {saved_data['statistics']['age']}")
                print(f"Income group distribution: {saved_data['statistics']['income']}")
                return household_person_pairs
        
        # Read household IDs and person IDs from the GPS place CSV file using pandas
        if os.path.exists(GPS_PLACE_CSV_PATH):
            # Read 'sampno' and 'perno' columns
            place_df = pd.read_csv(GPS_PLACE_CSV_PATH, usecols=['sampno', 'perno'])
            # Get unique combinations of sampno and perno
            household_person_pairs = place_df[['sampno', 'perno']].drop_duplicates()
            
            if stratified_sample and sample_size:
                # We need to get demographic data for stratification
                person_df = pd.read_csv(PERSON_CSV_PATH, low_memory=False)
                household_df = pd.read_csv(HOUSEHOLD_CSV_PATH, low_memory=False)
                
                # 将household_person_pairs与人口和家庭数据合并
                merged_df = household_person_pairs.merge(
                    person_df[['sampno', 'perno', 'sex', 'age']], 
                    on=['sampno', 'perno'], 
                    how='inner'
                )
                
                merged_df = merged_df.merge(
                    household_df[['sampno', 'hhinc']], 
                    on='sampno', 
                    how='inner'
                )
                
                # 创建分层变量
                # 性别分类
                merged_df['gender_group'] = merged_df['sex'].apply(lambda x: 'male' if x == 1 else 'female')
                
                # 年龄分组 - 使用默认分箱
                age_bins = [0, 18, 35, 50, 65, 200]
                age_labels = ['child', 'young_adult', 'adult', 'middle_age', 'senior']
                merged_df['age_group'] = pd.cut(
                    merged_df['age'], 
                    bins=age_bins, 
                    labels=age_labels
                )
                
                # 计算收入的分位数（0%, 25%, 50%, 75%, 100%）
                quantiles = [0, 0.25, 0.5, 0.75, 1.0]
                income_bins = [0] + list(merged_df['hhinc'].quantile(quantiles[1:]))
                # 确保最小值为0，处理可能的舍入误差
                income_bins[0] = 0
                # 确保所有值唯一且递增
                income_bins = sorted(set(income_bins))
                
                # 收入标签
                income_labels = ['low', 'medium_low', 'medium', 'high']
                
                # 应用收入分组
                merged_df['income_group'] = pd.cut(
                    merged_df['hhinc'], 
                    bins=income_bins, 
                    labels=income_labels
                )
                
                # 进行分层抽样
                try:
                    stratified_df = merged_df.groupby(['gender_group', 'age_group', 'income_group'], dropna=False).apply(
                        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(merged_df)))), random_state=seed)
                    ).reset_index(drop=True)
                    
                    # 如果分层抽样后的样本量少于总体需要的样本量，从剩余样本中随机抽取补充
                    if len(stratified_df) < sample_size and len(stratified_df) < len(merged_df):
                        # 找出已被抽样的索引
                        sampled_indices = set(zip(stratified_df['sampno'], stratified_df['perno']))
                        
                        # 找出未被抽样的数据
                        unsampled_df = merged_df[~merged_df.apply(lambda row: (row['sampno'], row['perno']) in sampled_indices, axis=1)]
                        
                        # 确定需要额外抽样的数量
                        extra_samples_needed = min(sample_size - len(stratified_df), len(unsampled_df))
                        
                        if extra_samples_needed > 0:
                            # 随机抽取额外样本
                            extra_samples = unsampled_df.sample(extra_samples_needed, random_state=seed)
                            
                            # 将额外样本添加到分层样本中
                            stratified_df = pd.concat([stratified_df, extra_samples])
                
                except Exception as e:
                    print(f"Error during stratified sampling, falling back to random sampling: {e}")
                    traceback.print_exc()
                    # 如果分层抽样失败，则进行简单随机抽样
                    if sample_size and sample_size < len(household_person_pairs):
                        household_person_pairs = household_person_pairs.sample(sample_size, random_state=seed).values.tolist()
                        return household_person_pairs
                
                # 提取最终的 household_id 和 person_id 对
                household_person_pairs = stratified_df[['sampno', 'perno']].values.tolist()
                
                print(f"Successfully created stratified sample of {len(household_person_pairs)} household-person pairs")
                
                # 打印样本的分层统计信息
                print("\nStratified Sample Statistics:")
                gender_stats = stratified_df['gender_group'].value_counts().to_dict()
                age_stats = stratified_df['age_group'].value_counts().to_dict()
                income_stats = stratified_df['income_group'].value_counts().to_dict()
                print(f"Gender distribution: {gender_stats}")
                print(f"Age group distribution: {age_stats}")
                print(f"Income group distribution: {income_stats}")
                
                # 如果请求保存样本，则保存到文件
                # if save_sample:
                os.makedirs(SAMPLE_DIR, exist_ok=True)
                sample_data = {
                    'household_person_pairs': household_person_pairs,
                    'statistics': {
                        'gender': gender_stats,
                        'age': age_stats,
                        'income': income_stats
                    }
                }
                with open(sample_file, 'w') as f:
                    json.dump(sample_data, f, indent=2)
                print(f"\nSaved stratified sample to {sample_file}")
                
            else:
                # 如果不使用分层抽样但指定了样本大小，则进行简单随机抽样
                if sample_size and sample_size < len(household_person_pairs):
                    household_person_pairs = household_person_pairs.sample(sample_size, random_state=seed)
                
                # 转换为列表
                household_person_pairs = household_person_pairs.values.tolist()
            
            print(f"Successfully loaded {len(household_person_pairs)} household-person pairs from {GPS_PLACE_CSV_PATH}")
            return household_person_pairs
        else:
            # If the file doesn't exist, use default values
            print(f"Warning: Could not find {GPS_PLACE_CSV_PATH}, using default household-person pairs for testing")
            return [(1, 1)]  # Default test pair
        
    except Exception as e:
        print(f"Error loading household IDs: {e}")
        traceback.print_exc()
        return [(1, 1)]  # Default test pair


def process_household_person_pair(pair, args_dict):
    household_id, person_id = pair
    try:
        # Create a basic persona for each household-person pair
        persona_data = {
            'id': f"{household_id}_{person_id}",
            'name': f"Person-{household_id}-{person_id}",
            'gender': 'unknown',  # 将在load_historical_data中被适当更新
            'age': 0,             # 将在load_historical_data中被适当更新
            'race': 'unknown',    # 将在load_historical_data中被适当更新
            'education': 'unknown', # 将在load_historical_data中被适当更新
            'occupation': 'unknown', # 将在load_historical_data中被适当更新
            'household_income': 50000, # 默认值，将在load_historical_data中被适当更新
            'household_vehicles': 0   # 默认值，将在load_historical_data中被适当更新
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
            
            # Save activity data to CSV format (only LLM generated days)
            memory.save_llm_days_to_csv(output_dir=args_dict['output'], persona_id=person_id)
            
            # Compress data if needed
            if args_dict['compress']:
                try:
                    compress_trajectory_data(output_file, method='gzip')
                except Exception as e:
                    print(f"Error compressing data: {e}")
            
            # Return results for collection
            print(f"✓ Completed household {household_id}, person {person_id}")
            return pair_id, memory
        return None
    except Exception as e:
        print(f"Error processing household {household_id}, person {person_id}: {e}")
        traceback.print_exc()
        return None


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
            parser.add_argument('--workers', type=int, default=0,
                              help='Number of parallel workers (0 for auto-detection)')
            parser.add_argument('--memory_days', type=int, default=MEMORY_DAYS,
                              help='Number of days to keep in memory')
            parser.add_argument('--summary', action='store_true', default=False,
                              help='Generate summary report')
            parser.add_argument('--compress', action='store_true', default=False,
                              help='Compress output files')
            parser.add_argument('--household_ids', type=str, default='',
                              help='Comma-separated list of household IDs to simulate (optional)')
            parser.add_argument('--max_concurrent_llm', type=int, default=10, 
                              help='Maximum number of concurrent LLM requests')
            parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                              help='Number of simulations to batch process')
            parser.add_argument('--max_batches', type=int, default=None,
                              help='Maximum number of batches to process (None for all)')
            parser.add_argument('--llm_rate_limit', type=float, default=0.1,
                              help='Minimum seconds between LLM requests to avoid rate limiting')
            parser.add_argument('--random_seed', type=int, default=42,
                              help='Random seed for reproducibility')
            parser.add_argument('--no_threading', action='store_true', default=True,
                              help='Disable multi-threading for debugging (runs in single thread)')
            parser.add_argument('--use_processes', action='store_true', default=BATCH_PROCESSING,
                              help='Use processes instead of threads for better performance on CPU-bound tasks')
            
            # 分层抽样相关参数
            parser.add_argument('--stratified_sample', action='store_true', default=True,
                              help='Use stratified sampling based on gender, age and income')
            parser.add_argument('--sample_size', type=int, default=300,
                              help='Number of household-person pairs to sample (None for all)')
            parser.add_argument('--use_saved_sample', action='store_true', default=True,
                              help='Use previously saved stratified sample')
            
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
            # Otherwise load from file with stratified sampling if requested
            household_person_pairs = load_household_ids(
                stratified_sample=args.stratified_sample,
                sample_size=args.sample_size,
                seed=args.random_seed,
                use_saved_sample=args.use_saved_sample
            )
            print(f"Will simulate {len(household_person_pairs)} household-person pairs")
            
        if not household_person_pairs:
            print("No valid household-person pairs found, cannot proceed with simulation")
            return {}
            
        # 如果使用分层抽样，确保样本量不超过max_batches * batch_size
        if args.stratified_sample and args.max_batches is not None:
            max_total_samples = args.max_batches * args.batch_size
            if len(household_person_pairs) > max_total_samples:
                print(f"Warning: Stratified sample size ({len(household_person_pairs)}) exceeds max_batches * batch_size ({max_total_samples})")
                print("Adjusting sample size to match max_batches * batch_size")
                household_person_pairs = household_person_pairs[:max_total_samples]
        
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
        # 注意：process_household_person_pair 已移至外部，不再在此处定义
        
        # Prepare arguments for worker function
        args_dict = {
            'days': args.days,
            'start_date': args.start_date,
            'memory_days': args.memory_days,
            'output': args.output,
            'compress': args.compress
        }
        
        # 自动检测系统并行度
        if args.workers <= 0:
            # 使用可用CPU核心数，但最多使用物理核心数的80%，至少1个
            cpu_count = psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
            available_workers = max(1, int(cpu_count * 0.8))
            workers = available_workers
            print(f"Auto-detected {cpu_count} CPU cores, using {workers} worker{'s' if workers > 1 else ''}")
        else:
            workers = args.workers
        
        # 确保至少使用1个worker，但不超过数据量
        workers = min(workers, len(household_person_pairs))
        workers = max(1, workers)  
        
        # 根据系统内存动态调整批处理大小
        def get_optimal_batch_size(requested_size):
            try:
                # 获取系统可用内存 (GB)
                available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                # 每个模拟估计使用的内存 (GB)，可根据实际观察调整
                estimated_memory_per_simulation = 0.1
                # 根据可用内存计算最佳批处理大小，确保不使用超过70%的可用内存
                memory_based_size = int(available_memory_gb * 0.7 / estimated_memory_per_simulation)
                # 使用请求的批处理大小和基于内存的大小中较小的一个
                optimal_size = min(requested_size, memory_based_size)
                # 确保至少处理1个
                return max(1, optimal_size)
            except Exception as e:
                print(f"Warning: Error calculating optimal batch size: {e}")
                # 默认返回原始请求大小
                return requested_size
        
        # 优化批处理大小
        optimized_batch_size = get_optimal_batch_size(args.batch_size)
        if optimized_batch_size != args.batch_size:
            print(f"Adjusted batch size from {args.batch_size} to {optimized_batch_size} based on available system memory")
        
        # Process in batches to avoid memory issues and improve throughput
        def process_batches():
            total_pairs = len(household_person_pairs)
            batch_size = min(optimized_batch_size, total_pairs)
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
                    # 使用ProcessPoolExecutor处理CPU密集型任务
                    if args.use_processes:
                        print(f"Using ProcessPoolExecutor with {workers} workers")
                        with ProcessPoolExecutor(max_workers=workers) as executor:
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
                    else:
                        # 使用ThreadPoolExecutor处理IO密集型任务
                        print(f"Using ThreadPoolExecutor with {workers} workers")
                        with ThreadPoolExecutor(max_workers=workers) as executor:
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
              f"and {optimized_batch_size} simulations per batch")
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