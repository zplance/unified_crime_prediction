import prompt_inventory
from pathlib import Path
import json

# Folder that holds THIS source file (…/workspace/src)
SRC_DIR = Path(__file__).resolve().parent
# One level up (…/workspace)
ROOT_DIR = SRC_DIR.parent
# clean_data directory (…/workspace/clean_data)
DATA_DIR = ROOT_DIR / "clean_data"

def prompt_loader(city_name):
    """
    Load the prompt template for the specified city.
    
    Args:
        city_name (str): The name of the city for which to load the prompt.
        
    Returns:
        str: The prompt template with placeholders for city-specific information.
    """

    if city_name == 'Dallas':
        return prompt_inventory.dallas_all_in_one_prompt

def default_data_loader(city_name: str):
    """Return JSON text for the requested city."""
    city_name = city_name.lower()

    if city_name == "dallas":
        json_path = DATA_DIR / "dallas_crime_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            
            return json.loads(f.read())

    raise ValueError(f"Unsupported city: {city_name}")

