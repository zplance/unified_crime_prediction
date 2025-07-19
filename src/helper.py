import prompt_inventory
from pathlib import Path

import json
import tiktoken
import pandas as pd

# Folder that holds THIS source file (…/workspace/src)
SRC_DIR = Path(__file__).resolve().parent
# One level up (…/workspace)
ROOT_DIR = SRC_DIR.parent
# clean_data directory (…/workspace/clean_data)
# DATA_DIR = ROOT_DIR / "clean_data"

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
        json_path = ROOT_DIR / "clean_data/dallas_crime_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            
            return json.loads(f.read())

    raise ValueError(f"Unsupported city: {city_name}")


def offline_record_loader(city_name, date, lon_bin, lat_bin, data_path=None):
    """
    Load offline records for a given city and date.
    
    Args:
        city_name (str): Name of the city.
        date (str): Date in 'YYYY-MM-DD' format.
        lon_bin (float): Longitude bin.
        lat_bin (float): Latitude bin.
        data_path (str, optional): Path to the data file. Defaults to None.
        
    Returns:
        dict: Offline records for the specified city and date.
    """
    
    if city_name not in ['Dallas'] and not data_path:
        raise ValueError("data_path must be provided.")
    

    if city_name == "Dallas":
        df = pd.read_csv(ROOT_DIR / "raw_data_download/dallas_crime_recrods.csv", low_memory=False)
        # df[(df.date1 == date) & (df.lat_bin == lat_bin) & (df.lon_bin == lon_bin)].to_dict(orient='records')
    
    else:
        df = pd.read_csv(data_path, low_memory=False)

    return df[(df.date1 == date) & (df.lat_bin == lat_bin) & (df.lon_bin == lon_bin)]

def get_encoder(model_name: str):
    """
    Return a tiktoken encoding instance for the given model.
    - deepseek-r1, qwen3, gemma3 → cl100k_base
    - llama3.*              → p50k_base
    - otherwise             → try encoding_for_model(), fallback to cl100k_base
    """
    m = model_name.lower()

    # All these use the same CL100K base tokenizer:
    if m.startswith(("deepseek-r1", "qwen3", "gemma3")):
        return tiktoken.get_encoding("cl100k_base")

    # Llama3-family uses the older P50K BPE
    if m.startswith("llama3"):
        return tiktoken.get_encoding("p50k_base")

    # fallback to tiktoken’s built‑in mapping, then to CL100K
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")