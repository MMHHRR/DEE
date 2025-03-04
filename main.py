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
    """模拟单日活动，确保使用活动中指定的交通方式"""
    try:
        # 更新到内存
        print(f"Simulating activities for {persona.name} on {date}...")
        memory.start_new_day(date)
        
        # 1. 生成日程表（包括交通方式）
        daily_activities = activity_generator.generate_daily_schedule(persona, date)
        
        # 确保一天的活动数据有效
        if not daily_activities:
            print(f"Warning: No activities generated for {persona.name} on {date}")
            return False
            
        # 确保活动按时间排序
        daily_activities.sort(key=lambda x: x['start_time'])
        
        # 去除重复的活动（基于开始时间和活动类型）
        unique_activities = []
        activity_keys = set()
        for activity in daily_activities:
            activity_key = f"{activity['start_time']}_{activity['activity_type']}"
            if activity_key not in activity_keys:
                activity_keys.add(activity_key)
                unique_activities.append(activity)
        
        # 更新日程表为去重后的活动
        daily_activities = unique_activities
        
        # 2. 细化每项活动
        refined_activities = []
        for activity in daily_activities:
            refined_activity = activity_generator.refine_activity(persona, activity, date)
            refined_activities.append(refined_activity)
        
        # 3. 为每项活动选择具体目的地并处理移动
        for i, activity in enumerate(refined_activities):
            # 获取活动信息
            activity_type = activity.get('activity_type')
            start_time = activity.get('start_time')
            end_time = activity.get('end_time')
            description = activity.get('description', '')
            location_type = activity.get('location_type', '')
            
            # 计算可用时间（分钟）
            available_minutes = time_difference_minutes(start_time, end_time)
            
            # 获取活动指定的交通方式
            transport_mode = activity.get('transport_mode', 'walking')
            transport_mode = normalize_transport_mode(transport_mode)
            
            try:
                print(f"  - {start_time}: {activity_type} - {description}")
                
                # 检查是否是固定位置（家或工作地点）
                if location_type.lower() == 'home' and hasattr(persona, 'home'):
                    destination = persona.home
                    details = {'name': 'Home', 'address': 'Home location'}
                elif location_type.lower() == 'work' and hasattr(persona, 'work'):
                    destination = persona.work
                    details = {'name': 'Work', 'address': 'Work location'}
                else:
                    # 选择目的地地点 - 传递交通方式和时间窗口
                    destination, details = destination_selector.select_destination(
                        persona, 
                        persona.current_location,  # 当前位置
                        activity_type,  # 活动类型
                        start_time,  # 开始时间
                        get_day_of_week(date),  # 星期几
                        available_minutes,  # 可用时间
                        transport_mode  # 活动指定的交通方式
                    )
                
                # 存储目的地详情到活动中
                activity['location_name'] = details.get('name', 'Unknown location')
                activity['price_level'] = details.get('price_level', 0)
                activity['location_rating'] = details.get('rating', 0)
                
                # 4. 处理移动并记录到内存
                if i > 0 and not are_locations_same(persona.current_location, destination):
                    prev_activity = refined_activities[i-1]
                    prev_end_time = prev_activity['end_time']
                    
                    # 使用活动中指定的交通方式计算旅行时间
                    travel_time, actual_transport_mode = estimate_travel_time(
                        persona.current_location, 
                        destination, 
                        transport_mode,
                        persona
                    )
                    
                    # 计算旅行后的到达时间
                    arrival_time = format_time_after_minutes(prev_end_time, travel_time)
                    
                    # 只打印一次交通信息，目的地使用活动地点名称，不使用Home
                    location_name = activity['location_name']
                    if location_type.lower() in ['work', 'home']:
                        location_name = location_type.capitalize()
                    
                    print(f"    (Traveling to {location_name} by {actual_transport_mode}, " +
                          f"estimated travel time: {travel_time} min)")
                    
                    # 记录旅行到内存
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
                
                # 更新当前位置和活动
                persona.current_location = destination
                persona.update_current_activity(activity)
                
            except Exception as activity_error:
                print(f"Error processing activity: {activity_error}")
                continue
        
        # 完成日模拟
        memory.end_day()
        return True
        
    except Exception as e:
        print(f"Error simulating day: {e}")
        if memory.current_day:
            memory.end_day()  # 确保结束日期记录
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
        
        # 判断是否使用并行处理，通常超过3天模拟时开启并行
        if num_days > 3 and not BATCH_PROCESSING:
            # 并行处理多个日期
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), num_days)) as executor:
                # 创建日期处理任务
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
                
                # 处理结果
                for future in concurrent.futures.as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        day_result = future.result()
                        if day_result:
                            # 合并结果到内存中
                            memory.days.append(day_result)
                    except Exception as e:
                        print(f"Error processing {date}: {e}")
        else:
            # 对于少量日期或启用批处理时，顺序处理
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
    
    # 根据CPU核心数并行处理多个人
    max_workers = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统
    
    if len(personas_data) > 1 and not BATCH_PROCESSING:
        # 并行处理多个人物
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建人物处理任务
            future_to_persona = {
                executor.submit(
                    simulate_persona, 
                    persona, 
                    NUM_DAYS_TO_SIMULATE, 
                    SIMULATION_START_DATE
                ): persona for persona in personas_data
            }
            
            # 处理结果
            for future in concurrent.futures.as_completed(future_to_persona):
                persona = future_to_persona[future]
                try:
                    memory = future.result()
                except Exception as e:
                    print(f"Error processing persona {persona['id']}: {e}")
    else:
        # 顺序处理人物
        for i, persona_data in enumerate(personas_data):
            print(f"\n\n--- Simulating Persona {i+1}/{len(personas_data)}: {persona_data['id']} ---\n")
            simulate_persona(persona_data, NUM_DAYS_TO_SIMULATE, SIMULATION_START_DATE)
    
    print("\nSimulation completed.")

def are_locations_same(loc1, loc2, threshold=0.001):
    """
    检查两个位置坐标是否足够接近可以被视为同一地点
    
    Args:
        loc1: 第一个位置坐标 (latitude, longitude)
        loc2: 第二个位置坐标 (latitude, longitude)
        threshold: 判断两个位置相同的最大距离差异（度）
        
    Returns:
        bool: 如果两个位置足够接近，返回True
    """
    if not loc1 or not loc2:
        return False
        
    # 检查两个坐标的纬度和经度是否足够接近
    return (abs(loc1[0] - loc2[0]) < threshold and 
            abs(loc1[1] - loc2[1]) < threshold)

if __name__ == "__main__":
    main() 