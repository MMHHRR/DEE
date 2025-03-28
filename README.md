# LLM-based Mobility Simulation

This project simulates human daily mobility patterns using LLM (Large Language Model) for activity generation and trajectory planning.

## Overview

The simulation creates realistic daily activity schedules and travel patterns for virtual personas. It uses LLMs to generate activities based on demographic information, and then plans their movements through space and time.

<img src=".\data\framework.png">

## Key Features

- Person-centric simulation with demographic attributes
- LLM-based activity scheduling and destination selection
- Realistic travel patterns with various transportation modes
- Visualization of daily trajectories
- Historical data loading from CSV files (optional)

## Project Structure

- `main.py`: Main entry point for the simulation
- `config.py`: Configuration settings and prompt templates
- `persona.py`: Persona class representing individuals
- `activity.py`: Activity generation using LLMs
- `destination.py`: Destination selection for activities
- `memory.py`: Recording and tracking mobility patterns
- `utils.py`: Helper functions for calculations and visualization
- `data/`: Contains persona definitions and simulation results

## Usage

1. Set up your `.env` file with API keys:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
DEEPBRICKS_API_KEY=your_openai_api_key
DEEPBRICKS_BASE_URL=https://api.openai.com/v1
```

2. Customize persona data in `data/personas.json`

3. Run the simulation:
```
python main.py
```

4. View results in the `data/results/` directory

## Requirements

See `requirements.txt` for dependencies. Main requirements:
- Python 3.8+
- OpenAI API (or compatible)
- GeoPy
- Folium (for visualization)
- Matplotlib
- Pandas

## Configuration

You can modify simulation parameters in `config.py`:
- Number of days to simulate
- Starting date
- Activity types and transportation modes
- LLM model settings
- Prompt templates

## TODO List
📌实现大规模LLM-Based Agent日程模拟（约4S每天日程）  
📌与芝加哥出行数据进行对齐（对个体的历史移动进行分析）     
🔴需要检查一下memory对历史pattern是否正确输入（***）    
🔴实现轨迹计算并保存+道路感知？（会增加运算时间）（***）    
🔴整合暴露计算？（需要考虑shap文件结合+暴露计算（涉及时间加权等内容））（***）     
🔴模拟加速，加入缓存和并行（**）


## License

[Your license information]