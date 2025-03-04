# LLM-Based Human Mobility Simulation

This project uses large language models (LLMs) to simulate human daily mobility behavior for environmental exposure research.

## Project Structure

- `persona.py`: Defines individual attributes and characteristics
- `activity.py`: Generates activity plans using LLM
- `destination.py`: Retrieves locations using Google Maps API or OSM
- `memory.py`: Records daily mobility trajectories
- `config.py`: Configuration settings
- `utils.py`: Utility functions
- `main.py`: Main program to coordinate components
- `data/`: Directory for storing persona data and simulation results

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   ```

## Usage

1. Place your persona data in `data/personas.json`
2. Run the simulation: `python main.py`
3. Results will be stored in `data/results/`

## Configuration

Modify `config.py` to adjust simulation parameters:
- Number of days to simulate: `NUM_DAYS_TO_SIMULATE = 3`
- LLM model to use: `LLM_MODEL = "gpt-4o-mini"`
- Prompt templates: `ACTIVITY_GENERATION_PROMPT`

## TODO List
- Add validation section (compare to real world): matrix framework
- Add peception modual (VLM perception environment)