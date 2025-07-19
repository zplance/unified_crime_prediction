import pickle

import ollama
import time
import numpy as np
import json
import helper
import tool_kit_help
import tiktoken
import asyncio
from pydantic import BaseModel

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
    
    def predict(self):

        predictions = []

        input_token_counts = []
        output_token_counts= []
        total_token_counts = []
        true_pred_diff = []

        encoder = helper.get_encoder(self.model_name)

        for date in self.data:
            for l in self.data[date]:

                max_retries = 3
                retry_count = 0
                success = False
                
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
                        if retry_count == max_retries:
                            predictions.append(0.0)
                            print(f"Failed after {max_retries} attempts, using default value 0.0")
                        else:
                            print(f"Attempt {retry_count} failed, retrying...")
                            time.sleep(1)  # Add small delay between retries
            
            true_pred_diff.append(int(round(l['today_value'] - pred, 0))) ##

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
        
    # def save_results(self, output_path):
    #     with open(output_path, 'w') as f:
    #         json.dump(self.results, f)

if __name__ == "__main__":
    print('Starting crime prediction using Ollama...')
    import random
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run crime predictions using Ollama')
    parser.add_argument('--model', type=str, default='llama2', help='Model name to use')
    parser.add_argument('--city', type=str, default='Dallas', help='City name')
    parser.add_argument('--data_path', type=str, default='clean_data/dallads_crime_data.json', help='Path to data file')
    parser.add_argument('--offline', default = True, help='Use offline mode for data fetching')

    args = parser.parse_args()
    print('successfully parsed arguments:', args)


    rand_seed = random.randint(1, 1000)
    results = []
    
    for i in range(3):  # Using fixed 2 runs
        print(f"Run {i+1} with seed {rand_seed} for city {args.city} using model {args.model}")
        predictor = OllamaPredictor(
            model_name=args.model,
            seed=rand_seed,
            city_name=args.city,
            offline = args.offline,
            data_path=args.data_path
        )
        
        predictor.predict()
        results.append(predictor.results)
    
    print('Finish all predictions, results are as follows:')

    for i in range(len(results)):
        print(f"Run {i+1} - RMSE: {results[i]['rmse']:.2f}, MAE: {results[i]['mae']:.2f}")
        
    # Calculate mean values across runs
    mean_rmse = np.mean([r['rmse'] for r in results])
    mean_mae = np.mean([r['mae'] for r in results])
    print(f"\nMean values across {len(results)} runs:")
    print(f"Mean RMSE: {mean_rmse:.2f}")
    print(f"Mean MAE: {mean_mae:.2f}")

    # Save results to JSON file
    output_path = f'results/{args.city}_{args.model}_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Prediction completed and results saved.")