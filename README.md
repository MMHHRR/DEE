# LLM-based Mobility Simulation

This project simulates human daily mobility patterns using LLM (Large Language Model) for activity generation.

## Overview

The simulation creates realistic daily activity schedules and travel patterns for virtual personas. It uses LLMs to generate activities based on demographic information, and then plans their movements through space and time.

<img src=".\data\framework.png">

## Key Features

- Person-centric simulation with demographic attributes and historical mobility patterns
- LLM-based activity scheduling and destination selection
- Realistic travel patterns with various transportation modes

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

2. Chicago travel data in `data/....csv`
- `data/gps_place.csv` mobility history
- `data/household.csv` houshold income
- `data/location_new.csv` location types (updated)
- `data/person.csv` demographic attribute

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
- Matplotlib
- Pandas

## Configuration

You can modify simulation parameters in `config.py`:
- Number of days to simulate (def=7days)
- Number of days to memory (def=2days)
- LLM model settings (model, maxtoken, temprature)
- Prompt templates (IMPORTANT)

## TODO List
- Construct the evaluation metric (IMPORTANT)
- Compare to different model (Statistic Distribution)
- Add green exposure calculation (shp file need)
- Writing paper (ASAP)

## Paper Coming Soon🤗