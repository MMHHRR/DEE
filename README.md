# LLM-based Mobility Simulation

This project simulates human daily mobility patterns using LLM (Large Language Model) for activity generation and trajectory planning.

## Overview

The simulation creates realistic daily activity schedules and travel patterns for virtual personas. It uses LLMs to generate activities based on demographic information, and then plans their movements through space and time.

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

## License

[Your license information]