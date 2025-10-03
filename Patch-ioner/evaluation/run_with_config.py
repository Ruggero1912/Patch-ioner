#!/usr/bin/env python3
"""
Example launcher script showing how to use custom configurations.
"""

import json
import sys
import os

# Add the current directory to path to import run-tasks
sys.path.append(os.path.dirname(__file__))

from run_tasks import ExperimentRunner

def main(config_file = "experiment_configs.json", log_to_file: bool = False):
    """Run experiments with custom configuration file."""
    
    # Load configuration from JSON file
    
    
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        print("Please create it or use the default configurations in run-tasks.py")
        return
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Handle both old format (list) and new format (dict with experiments)
    if isinstance(config_data, list):
        # Old format - direct list of experiments
        experiment_configs = config_data
        print(f"Loaded {len(experiment_configs)} experiments from {config_file} (legacy format)")
    else:
        # New format - structured with global defaults
        experiment_configs = config_data.get('experiments', [])
        
        # Apply global settings to each experiment
        global_defaults = config_data.get('global_defaults', {})
        output_directory = config_data.get('output_directory')
        
        for config in experiment_configs:
            if global_defaults:
                config['global_defaults'] = {**global_defaults, **config.get('global_defaults', {})}
            if output_directory:
                config['output_directory'] = output_directory
        
        print(f"Loaded {len(experiment_configs)} experiments from {config_file}")
        if global_defaults:
            print(f"Global defaults: {global_defaults}")
        if output_directory:
            print(f"Output directory: {output_directory}")
    
    # Initialize runner and execute
    runner = ExperimentRunner(results_dir=output_directory)
    results = runner.run_experiments(experiment_configs, log_to_file=log_to_file)
    
    print(f"\nCompleted {len(results)} experiments")
    print(f"Results saved in: {runner.results_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiments with custom configuration file.")
    parser.add_argument('--config', type=str, default="experiment_configs.json",
                        help="Path to the configuration file (default: experiment_configs.json)")
    parser.add_argument('--log-to-file', action='store_true', default=False,
                        help='Log output of each task to a file in the results directory')

    args = parser.parse_args()
    main(args.config, log_to_file=args.log_to_file)