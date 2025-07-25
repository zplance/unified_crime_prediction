import asyncio
import json
import logging
import ollama
import pickle
import tiktoken
import time
import datetime

import numpy as np
from pathlib import Path
from pydantic import BaseModel

import helper
import tool_kit_help


class CrimeAmount(BaseModel):
  crime_amount: int

class OllamaPredictor:
    def __init__(self, model_name, seed, city_name, offline = True, data_path=None):
        self.model_name = model_name
        self.seed = seed
        self.results = {}
        self.city_name = city_name
        self.offline = offline
        if self.city_name in ['Dallas', 'Chicago', 'New York', 'San Francisco']:
            self.data = helper.default_data_loader(self.city_name)
        else:
            if data_path is None:
                raise ValueError("data_path must be provided for cities other than Dallas, Chicago, New York, or San Francisco.")
            
            else:
                with open(data_path, 'rb') as f:
                    self.data = pickle.load(f)
        self.client = ollama.Client(host="http://localhost:11434") ## If need the update the local host address, please do.
        
        # Create logger for this instance
        self.logger = logging.getLogger(__name__)
        
    async def async_predict_batch(self, date, offline, longitude, latitude):
        crime_task = asyncio.to_thread(
            tool_kit_help.fetch_crime_record,
            self.city_name,
            date,
            offline,
            longitude,
            latitude
        )
        weather_task = asyncio.to_thread(
            tool_kit_help.fetch_weather,
            longitude=longitude,
            latitude=latitude,
            start_date=date
        )
        crime_data, weather_data = await asyncio.gather(crime_task, weather_task)
        return crime_data, weather_data
    
    def test_ollama_connection(self):
        """Test if Ollama is responsive"""
        try:
            self.client.list()
            return True
        except Exception as e:
            self.logger.warning(f"Ollama connection test failed: {e}")
            return False
    
    def predict(self):

        predictions = []

        input_token_counts = []
        output_token_counts= []
        total_token_counts = []
        true_pred_diff = []

        encoder = helper.get_encoder(self.model_name)
        # self.logger.info("Starting prediction loop — %d dates to process", len(self.data))
        
        # start_test = datetime.date(2025, 1, 1)
        # end_test   = datetime.date(2025, 1, 2)
        # test_dates = [
        #     d for d in self.data.keys()
        #     if start_test <= datetime.date.fromisoformat(d) <= end_test
        # ]
        
        for date in self.data:
            for l in self.data[date]:

                max_retries = 10  # Increased from 3 to 10
                retry_count = 0
                success = False
                pred = 0  # Initialize pred with default value
                
                crime_data, weather_data = asyncio.run(
                    self.async_predict_batch(date, self.offline, l['coordinates'][0], l['coordinates'][1])
                )
                
                prompt_string = helper.prompt_loader(self.city_name).format(
                            longitude=l['coordinates'][0],
                            latitude=l['coordinates'][1],
                            last_year_sum=l['last_year_sum'],
                            prediction_date=date,
                            past_14_days_crime_time_series=l['past_14_days'],
                            cycle_data_points=l['cycle_data'],
                            event_level_crime_summary=crime_data,
                            weather_information=weather_data
                        )
                
                while retry_count < max_retries and not success:
                    try:
                        # Test Ollama connection before attempting generation
                        if not self.test_ollama_connection():
                            self.logger.error(f"Ollama connection test failed on attempt {retry_count + 1}")
                            raise ConnectionError("Ollama service is not responsive")
                        
                        # Single-turn completion
                        # Format prompt using fetched data
                        resp = self.client.generate(
                            model=self.model_name,
                            prompt=prompt_string,
                            options={"seed": self.seed},
                            format=CrimeAmount.model_json_schema(),)
                        
                        # count tokens
                        input_count  = len(encoder.encode(prompt_string))
                        output_count = len(encoder.encode(resp.response))+len(encoder.encode(resp.thinking or "")) # add model name to output count
                        total_count  = input_count + output_count
                              
                        input_token_counts.append(input_count)
                        output_token_counts.append(output_count)
                        total_token_counts.append(total_count)
                        
                        pred = int(json.loads(resp.response)["crime_amount"])
                        if pred > 0:
                            print(f'{date}, {(l['coordinates'][0],l['coordinates'][1])}: {pred}')
                        predictions.append(pred)
                        success = True

                    except (ValueError, KeyError) as e:
                        retry_count += 1
                        self.logger.warning(f"JSON parsing error on attempt {retry_count}/{max_retries}: {e}")
                        if retry_count == max_retries:
                            pred = 0  # Set default value for failed prediction
                            predictions.append(pred)
                            self.logger.error(f"Failed after {max_retries} attempts due to JSON parsing error, using default value 0")
                            print(f"Failed after {max_retries} attempts (JSON parsing), using default value 0")
                        else:
                            print(f"JSON parsing attempt {retry_count} failed, retrying in 3 seconds...")
                            time.sleep(3)  # Wait 3 seconds between retries
                    
                    except (ConnectionError, ollama._types.ResponseError) as e:
                        retry_count += 1
                        error_msg = str(e)
                        self.logger.warning(f"Ollama error on attempt {retry_count}/{max_retries}: {error_msg}")
                        
                        if retry_count == max_retries:
                            pred = 0  # Set default value for failed prediction
                            predictions.append(pred)
                            self.logger.error(f"Failed after {max_retries} attempts due to Ollama error, using default value 0")
                            print(f"Failed after {max_retries} attempts (Ollama error: {error_msg}), using default value 0")
                        else:
                            print(f"Ollama error attempt {retry_count}/{max_retries}: {error_msg}")
                            print(f"Retrying in 3 seconds...")
                            time.sleep(3)  # Wait 3 seconds between retries
                    
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        self.logger.warning(f"Unexpected error on attempt {retry_count}/{max_retries}: {error_msg}")
                        
                        if retry_count == max_retries:
                            pred = 0  # Set default value for failed prediction
                            predictions.append(pred)
                            self.logger.error(f"Failed after {max_retries} attempts due to unexpected error, using default value 0")
                            print(f"Failed after {max_retries} attempts (Unexpected error: {error_msg}), using default value 0")
                        else:
                            print(f"Unexpected error attempt {retry_count}/{max_retries}: {error_msg}")
                            print(f"Retrying in 3 seconds...")
                            time.sleep(3)  # Wait 3 seconds between retries
                
                # Calculate difference for this location (moved inside the location loop)
                true_pred_diff.append(int(round(l['today_value'] - pred, 0)))

        self.results = {
            'model_name': self.model_name,
            'seed': self.seed,
            'rmse': np.sqrt(np.mean(np.square(true_pred_diff))),  # Calculate RMSE from differences
            'mae': np.mean(np.abs(true_pred_diff)),  # Calculate MAE from differences
            'avg_input_tokens': np.mean(input_token_counts),
            'avg_output_tokens': np.mean(output_token_counts),
            'avg_total_tokens': np.mean(total_token_counts),
            
            'details': {
                'predictions': predictions,
                'input_tokens': input_token_counts,
                'output_tokens': output_token_counts,
                'total_tokens': total_token_counts,
                'true_pred_differences': true_pred_diff
            }
        }


def save_checkpoint(results, run_number, city, model, results_dir):
    """Save individual run results as checkpoint"""
    checkpoint_file = results_dir / f"{city}_{model}_run_{run_number}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoints(city, model, results_dir, num_runs=3):
    """Load existing checkpoint files if they exist"""
    checkpoints = []
    for i in range(1, num_runs + 1):
        checkpoint_file = results_dir / f"{city}_{model}_run_{i}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoints.append(json.load(f))
            print(f"Loaded checkpoint: {checkpoint_file}")
        else:
            checkpoints.append(None)
    return checkpoints


def calculate_final_results(checkpoints, city, model, results_dir):
    """Calculate final results from checkpoint files"""
    # Filter out None values (missing checkpoints)
    valid_results = [r for r in checkpoints if r is not None]
    
    if not valid_results:
        print("No valid checkpoint files found!")
        return None
    
    print(f"\nCalculating final results from {len(valid_results)} runs:")
    
    # Display individual run results
    for i, result in enumerate(valid_results):
        run_num = i + 1
        print(f"Run {run_num} - RMSE: {result['rmse']:.2f}, MAE: {result['mae']:.2f}")
    
    # Calculate mean values across runs
    mean_rmse = np.mean([r['rmse'] for r in valid_results])
    mean_mae = np.mean([r['mae'] for r in valid_results])
    mean_input_tokens = np.mean([r['avg_input_tokens'] for r in valid_results])
    mean_output_tokens = np.mean([r['avg_output_tokens'] for r in valid_results])
    mean_total_tokens = np.mean([r['avg_total_tokens'] for r in valid_results])
    
    print(f"\nMean values across {len(valid_results)} runs:")
    print(f"Mean RMSE: {mean_rmse:.2f}")
    print(f"Mean MAE: {mean_mae:.2f}")
    print(f"Mean Input Tokens: {mean_input_tokens:.2f}")
    print(f"Mean Output Tokens: {mean_output_tokens:.2f}")
    print(f"Mean Total Tokens: {mean_total_tokens:.2f}")
    
    # Create combined results
    final_results = {
        'model_name': model,
        'city_name': city,
        'num_runs': len(valid_results),
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae,
        'mean_input_tokens': mean_input_tokens,
        'mean_output_tokens': mean_output_tokens,
        'mean_total_tokens': mean_total_tokens,
        'individual_runs': valid_results
    }
    
    # Save final combined results
    final_output_path = results_dir / f"{city}_{model}_final_results.json"
    with open(final_output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Final results saved: {final_output_path}")
    
    return final_results


if __name__ == "__main__":
    print('Starting crime prediction using Ollama...')
    import argparse
    import logging
    import random
    import time
    from pathlib import Path

    # Folder that holds THIS source file (…/workspace/src)
    SRC_DIR = Path(__file__).resolve().parent
    # One level up (…/workspace)
    ROOT_DIR = SRC_DIR.parent

    RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")          # e.g. 20250720_143022
    LOG_DIR   = ROOT_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    LOG_FILE  = LOG_DIR / f"crime_predict_{RUN_STAMP}.txt"   # one file per script run

    logging.basicConfig(
        level=logging.INFO,                               # or DEBUG if you like
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=LOG_FILE,
        filemode="w"                                      # always start fresh
    )
    logger = logging.getLogger(__name__)
    logger.info("Log file created at %s", LOG_FILE)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run crime predictions using Ollama')
    parser.add_argument('--model', type=str, default='llama2', help='Model name to use')
    parser.add_argument('--city', type=str, default='Dallas', help='City name')
    parser.add_argument('--data_path', type=str, default='clean_data/dallads_crime_data.json', help='Path to data file')
    parser.add_argument('--offline', default = True, help='Use offline mode for data fetching')
    parser.add_argument('--resume', action='store_true', help='Resume from existing checkpoints')
    parser.add_argument('--calculate_only', action='store_true', help='Only calculate results from existing checkpoints')

    args = parser.parse_args()
    print('successfully parsed arguments:', args)

    # Create results directory if it doesn't exist
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # If only calculating results from existing checkpoints
    if args.calculate_only:
        print("Loading existing checkpoints and calculating final results...")
        checkpoints = load_checkpoints(args.city, args.model, results_dir)
        final_results = calculate_final_results(checkpoints, args.city, args.model, results_dir)
        if final_results:
            print("Final results calculation completed.")
        else:
            print("No valid checkpoints found for calculation.")
        exit()

    rand_seed = random.randint(1, 10000)
    num_runs = 3
    
    # Load existing checkpoints if resuming
    if args.resume:
        print("Checking for existing checkpoints...")
        existing_checkpoints = load_checkpoints(args.city, args.model, results_dir, num_runs)
    else:
        existing_checkpoints = [None] * num_runs
    
    results = []
    
    rand_seed_set = [6177,3186, 6889]  # Fixed seeds for reproducibility, can be randomized if needed
    
    for i in range(num_runs):
        rand_seed = rand_seed_set[i]  # Use the fixed seed for each runs
        run_number = i + 1
        
        # Skip if checkpoint already exists and we're resuming
        if args.resume and existing_checkpoints[i] is not None:
            print(f"Run {run_number} already completed, loading from checkpoint...")
            results.append(existing_checkpoints[i])
            continue
        
        print(f"Starting Run {run_number} with seed {rand_seed} for city {args.city} using model {args.model}")
        
        try:
            predictor = OllamaPredictor(
                model_name=args.model,
                seed=rand_seed,
                city_name=args.city,
                offline = args.offline,
                data_path=args.data_path
            )
            
            predictor.predict()
            
            # Save checkpoint immediately after each run
            checkpoint_file = save_checkpoint(
                predictor.results, 
                run_number, 
                args.city, 
                args.model, 
                results_dir
            )
            
            results.append(predictor.results)
            print(f"Run {run_number} completed successfully!")
            
        except Exception as e:
            print(f"Run {run_number} failed with error: {e}")
            logger.error(f"Run {run_number} failed: {e}")
            # Continue with next run even if one fails
            continue
    
    # Calculate and save final results
    if results:
        print('\nAll predictions completed, calculating final results...')
        final_results = calculate_final_results(results, args.city, args.model, results_dir)
        print("Prediction completed and all results saved.")
    else:
        print("No successful runs completed.")