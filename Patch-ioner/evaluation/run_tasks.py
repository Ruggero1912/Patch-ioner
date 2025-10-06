#!/usr/bin/env python3
"""
Multi-task evaluation script for decap-dino models.

This script runs multiple evaluation experiments across different tasks and models,
similar to the scripts in /evaluation_script but with a more flexible configuration system.

The script iterates over a list of experiment configurations, where each configuration
contains:
- Model configuration (model_name, etc.)
- Task-specific settings for each evaluation task

Task settings can be specified in two ways:
1. Single configuration: tasks: { task_name: {...settings...} }
2. Multiple configurations: tasks: { task_name: [{...settings1...}, {...settings2...}] }

When multiple configurations are provided for a task, each will be run separately
with unique screen sessions and result tracking.

Results are stored in CSV files in the results directory.

IMPORTANT: This script now includes automatic result checking to avoid re-running
experiments that have already been completed. For each task and model configuration,
it checks if results already exist in the corresponding CSV file. If results exist,
the task is skipped.

Usage examples:
  python run_tasks.py                    # Run with default configs, skip existing results
  python run_tasks.py --force-rerun      # Run all experiments, even if results exist
  python run_tasks.py --show-existing    # Show existing results without running experiments
  python run_tasks.py --config custom.json  # Use custom configuration file

Key parameters used for result checking:
- model_name: The model being evaluated
- evaluation_dataset: The dataset being used
- caption_from: Where captions are generated from (e.g., 'patches')
- batch_size: Batch size used for evaluation
- gaussian_variance: Gaussian weighting variance (if applicable)
- use_gaussian_weighting: Whether Gaussian weighting is enabled
"""

import sys
import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pandas as pd

# Add decap module to path
if '../decap' not in sys.path:
    sys.path.append('../decap')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_tasks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import utilities
try:
    sys.path.append('../evaluation_script')
    from utils import get_gpu_with_most_memory
except ImportError:
    logger.warning("Could not import utils.py. Using basic GPU selection.")
    import torch
    def get_gpu_with_most_memory():
        if torch.cuda.is_available():
            return f"cuda:0"
        return "cpu"


class ExperimentRunner:
    """
    Handles running evaluation experiments across multiple tasks and models.
    
    Supports both single and multiple configurations per task:
    - Single config: tasks: { task_name: {...settings...} }
    - Multiple configs: tasks: { task_name: [{...settings1...}, {...settings2...}] }
    
    When multiple configurations are specified for a task, each configuration
    will be run separately with unique screen sessions and result tracking.
    """
    
    def __init__(self, base_dir: str = os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))), results_dir = Path(__file__).parent / "results", force_rerun: bool = False):
        """
        Initialize the ExperimentRunner with the base directory and subdirectories for decap and results.
        
        Args:
            base_dir: Base directory path
            results_dir: Directory for storing results
            force_rerun: If True, run experiments even if results already exist
        """
        self.base_dir = Path(base_dir)
        self.decap_dir = self.base_dir / "decap"
        results_dir = Path(results_dir) if isinstance(results_dir, str) else results_dir if results_dir else Path(self.base_dir / "results")
        results_dir = results_dir.resolve()
        self.results_dir = results_dir
        self.force_rerun = force_rerun
        
        # Initialize ResultsCollector instance for better result checking
        self._collect_results_instance = None

        print(f"Using base directory: {self.base_dir}")
        print(f"Using results directory: {self.results_dir}")
        if force_rerun:
            print("Force rerun enabled - will run experiments even if results exist")

        # Global default parameters
        self.global_defaults = {
            'batch_size': 16,
            'gaussian_variance': 1.0,
            'conda_env': 'decapdino',
            'use_gaussian_weighting': False, # TODO: check if this default is appropriate
            'representation_cleaning_clean_after_projection': False, # TODO: check if this default is appropriate
        }
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Task definitions with their script paths and default output directories
        self.tasks = {
            'image_captioning': {
                'script': 'eval-image-captioning/eval_image_captioning.py',
                'default_output_dir': self.results_dir / 'image_captioning',
                'base_args': [], # No base args for image captioning
                'default_datasets': ['coco-test.json'] # flickr30k_test_coco.json
            },
            'dense_captioning': {
                'script': 'eval-dense-captioning/eval_densecap.py', 
                'default_output_dir': self.results_dir / 'dense_captioning',
                'base_args': ['--compute_scores', 'True'],
                'default_datasets': ['vg12'] # vgcoco
            },
            'narratives': {
                'script': 'eval-trace-captioning/eval_trace_captioning.py',
                'default_output_dir': self.results_dir / 'trace_captioning', 
                'base_args': [], # No base args for trace captioning
                'default_datasets': ['trace_capt_coco_test.json'] # trace_capt_flickr30k_test.json
            },
            'controllable_captioning': {
                'script': 'eval-region-set-captioning/eval_set_captioning.py',
                'default_output_dir': self.results_dir / 'eval_region_set_captioning',
                'base_args': [], 
                'default_datasets': ['coco_entities_test.json'] # flickr30k_entities_test.json
            }
        }
        
        # Create default output directories
        for task_info in self.tasks.values():
            task_info['default_output_dir'].mkdir(exist_ok=True)

    def _get_collect_results_instance(self):
        """Get or create a ResultsCollector instance for result checking."""
        if self._collect_results_instance is None:
            try:
                # Import here to avoid circular imports
                from collect_results import ResultsCollector
                self._collect_results_instance = ResultsCollector(
                    #base_dir=str(self.base_dir),
                    results_dir=str(self.results_dir)
                )
            except ImportError as e:
                logger.warning(f"Could not import ResultsCollector. Using legacy result checking. Error: {e}")
                self._collect_results_instance = None
            except Exception as e:
                logger.warning(f"Failed to create ResultsCollector instance: {e}")
                self._collect_results_instance = None
        return self._collect_results_instance
    
    def set_config_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply global defaults and config-level defaults to the experiment configuration."""
        # Create a copy to avoid modifying the original
        config_with_defaults = json.loads(json.dumps(config))
        
        # Apply global defaults from config file if present
        global_defaults = config_with_defaults.get('global_defaults', {})
        merged_defaults = {**self.global_defaults, **global_defaults}
        
        # Set output directory from config or use default
        if 'output_directory' in config_with_defaults:
            self.results_dir = Path(config_with_defaults['output_directory']).resolve()
            self.results_dir.mkdir(exist_ok=True)
            # Update task output directories
            for task_name, task_info in self.tasks.items():
                task_info['default_output_dir'] = self.results_dir / task_name
                task_info['default_output_dir'].mkdir(exist_ok=True)
        
        # Apply defaults to each task
        for task_name, task_settings in config_with_defaults.get('tasks', {}).items():
            if task_name in self.tasks:
                # Handle both single dict and list of dicts for task settings
                if isinstance(task_settings, list):
                    # Multiple task settings - apply defaults to each
                    for settings_dict in task_settings:
                        # Use default dataset if none specified
                        if 'datasets' not in settings_dict and 'evaluation_dataset' not in settings_dict:
                            settings_dict['datasets'] = self.tasks[task_name]['default_datasets']
                        
                        # Apply global defaults
                        for key, default_value in merged_defaults.items():
                            if key not in settings_dict:
                                settings_dict[key] = default_value
                else:
                    # Single task settings dict - apply defaults as before
                    # Use default dataset if none specified
                    if 'datasets' not in task_settings and 'evaluation_dataset' not in task_settings:
                        task_settings['datasets'] = self.tasks[task_name]['default_datasets']
                    
                    # Apply global defaults
                    for key, default_value in merged_defaults.items():
                        if key not in task_settings:
                            task_settings[key] = default_value
        
        return config_with_defaults

    def run_experiment(self, config: Dict[str, Any], log_to_file: bool = False) -> Dict[str, Any]:
        """
        Run a single experiment configuration across all specified tasks.
        
        Args:
            config: Experiment configuration containing model settings and task-specific parameters
            
        Returns:
            Dictionary with experiment results and status
        """
        # Apply defaults to the configuration
        config = self.set_config_defaults(config)
        
        model_name = config['model_name']
        logger.info(f"Starting experiment for model: {model_name}")
        
        results = {
            'model_name': model_name,
            'config': config,
            'task_results': {},
            'start_time': time.time()
        }

        conda_env = config['conda_env'] if 'conda_env' in config else None
        
        
        # Run each task specified in the config
        for task_name, task_settings in config.get('tasks', {}).items():
            if task_name not in self.tasks:
                logger.warning(f"Unknown task: {task_name}. Skipping.")
                continue

            # Handle both single dict and list of dicts for task settings
            if isinstance(task_settings, list):
                # Multiple task settings - run each configuration
                task_results = []
                for i, settings_dict in enumerate(task_settings):
                    if conda_env is not None:
                        settings_dict['conda_env'] = conda_env
                    
                    logger.info(f"Running task: {task_name} (configuration {i+1}/{len(task_settings)})")
                    task_result = self._run_single_task(
                        task_name, model_name, settings_dict, config, config_index=i, log_to_file=log_to_file
                    )
                    task_result['config_index'] = i
                    task_results.append(task_result)
                
                results['task_results'][task_name] = task_results
            else:
                # Single task settings dict - run as before
                if conda_env is not None:
                    task_settings['conda_env'] = conda_env
                    
                logger.info(f"Running task: {task_name}")
                task_result = self._run_single_task(
                    task_name, model_name, task_settings, config, log_to_file=log_to_file
                )
                results['task_results'][task_name] = task_result
        
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        logger.info(f"Completed experiment for {model_name} in {results['duration']:.2f} seconds")
        return results

    def _run_single_task(self, task_name: str, model_name: str, 
                        task_settings: Dict[str, Any], config: Dict[str, Any] = None, 
                        config_index: Optional[int] = None, log_to_file: bool = False) -> Dict[str, Any]:
        """Run a single task evaluation."""
        task_info = self.tasks[task_name]
        script_path = self.decap_dir / task_info['script']
        
        result = {
            'task': task_name,
            'status': 'started',
            'start_time': time.time()
        }

        # Get available GPU
        try:
            device = get_gpu_with_most_memory()
            logger.info(f"Using device: {device}")
        except Exception as e:
            logger.warning(f"Could not get optimal GPU: {e}. Using cuda:0")
            device = "cuda:0"
        
        
        try:
            # Build command arguments
            cmd_args = self._build_command_args(
                task_name, model_name, task_settings, device, task_info
            )
            
            # Generate screen session name
            screen_prefix = config.get('screen_prefix', '') if config else ''
            screen_name = self._generate_screen_name(task_name, model_name, task_settings, config_index, screen_prefix)
            
            # Check if screen session already exists
            if self._screen_exists(screen_name):
                logger.warning(f"Screen session {screen_name} already exists. Skipping.")
                result['status'] = 'skipped'
                result['reason'] = 'screen_session_exists'
                return result
            
            # Check if all results already exist before creating screen session
            datasets = task_settings.get('datasets', [task_settings.get('evaluation_dataset')])
            if not isinstance(datasets, list):
                datasets = [datasets]
            
            datasets_to_run = []
            if task_name == "controllable_captioning":
                    logger.info(f"Debugging controllable_captioning task. datasets to run: {datasets}")
            for dataset in datasets:
                
                if dataset is None:
                    logger.warning(f"Dataset is None for task {task_name}. Skipping.")
                    continue  
                if not self._check_results_exist(task_name, model_name, task_settings, dataset):
                    logger.info(f"Results do not exist for {task_name} with {model_name} on dataset {dataset}. Will run.")
                    datasets_to_run.append(dataset)
            
            if not datasets_to_run:
                logger.info(f"All results already exist for {task_name} with {model_name}. Skipping.")
                result['status'] = 'skipped'
                result['reason'] = 'results_already_exist'
                return result
            
            # Run the evaluation
            sleep_time = 10
            print(f"Sleeping for {sleep_time} seconds before starting task")
            time.sleep(sleep_time)# Get available GPU

            success = self._execute_task(script_path, cmd_args, screen_name, task_settings, task_info, log_to_file=log_to_file)

            if success:
                result['status'] = 'completed'
                logger.info(f"Successfully started {task_name} for {model_name}")
            else:
                result['status'] = 'failed'
                logger.error(f"Failed to start {task_name} for {model_name}")
                
        except Exception as e:
            logger.error(f"Error running {task_name} for {model_name}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        
        return result

    def _build_command_args(self, task_name: str, model_name: str, 
                           task_settings: Dict[str, Any], device: str,
                           task_info: Dict[str, Any]) -> List[str]:
        """Build command line arguments for the task."""
        args = [
            '--model_name', model_name,
            '--device', device
        ]
        
        # Add base arguments for this task type
        args.extend(task_info.get('base_args', []))
        
        # Add task-specific arguments (excluding special handling parameters)
        exclude_keys = {'datasets', 'conda_env', 'compute_scores'}  # Exclude compute_scores to avoid duplication
        for key, value in task_settings.items():
            if key in exclude_keys:
                continue
            
            arg_name = f'--{key}'
            if value == "": # case in which it is a flag
                # If value is empty string, just add the flag
                # This allows for boolean flags like --keep_img_ratio
                # to be passed without a value
                args.append(arg_name)
            elif key in {'use_gaussian_weighting', 'compute_scores', 'representation_cleaning_clean_after_projection'} and isinstance(value, bool):
                # For boolean flags, only add the flag if True
                if value:
                    args.append(arg_name)
            else:
                args.extend([arg_name, str(value)])
        
        return args

    def _generate_screen_name(self, task_name: str, model_name: str, 
                            task_settings: Dict[str, Any], config_index: Optional[int] = None,
                            screen_prefix: str = "") -> str:
        """Generate a unique screen session name."""
        # Create a short identifier from task settings
        settings_id = []
        
        if 'evaluation_dataset' in task_settings:
            dataset = task_settings['evaluation_dataset']
            if isinstance(dataset, str):
                settings_id.append(dataset.split('.')[0][:10])
        
        if 'caption_from' in task_settings:
            settings_id.append(task_settings['caption_from'])
            
        if 'use_gaussian_weighting' in task_settings:
            if task_settings['use_gaussian_weighting']:
                settings_id.append('gw')
        
        if 'representation_cleaning_type' in task_settings:
            if task_settings['representation_cleaning_type']:
                settings_id.append(f"clean{task_settings['representation_cleaning_type'][:8]}")
        
        if 'representation_cleaning_clean_after_projection' in task_settings:
            if task_settings['representation_cleaning_clean_after_projection']:
                settings_id.append('cleanpostproj')
        
        if 'caption_bboxes_type' in task_settings:
            settings_id.append('bboxcapt')
            
        # Add configuration index if multiple configurations exist
        if config_index is not None:
            settings_id.append(f'cfg{config_index}')
            
        settings_str = '_'.join(settings_id) if settings_id else 'default'
        
        # Truncate model name if too long
        model_short = model_name.replace('.k', '')[-15:] if len(model_name) > 15 else model_name
        
        # Build screen name with optional prefix
        screen_name = f"{task_name}_{model_short}_{settings_str}"
        if screen_prefix:
            screen_name = f"{screen_prefix}_{screen_name}"
        
        return screen_name

    def _screen_exists(self, screen_name: str) -> bool:
        """Check if a screen session with the given name exists."""
        try:
            result = subprocess.run(
                ['screen', '-list'], 
                capture_output=True, 
                text=True
            )
            return screen_name in result.stdout
        except Exception:
            return False
    
    def _check_results_exist(self, task_name: str, model_name: str, 
                           task_settings: Dict[str, Any], dataset: str) -> bool:
        """
        Check if results already exist in the CSV file for the given configuration.
        
        Args:
            task_name: Name of the task (e.g., 'image_captioning', 'dense_captioning')
            model_name: Name of the model being evaluated
            task_settings: Task-specific settings dictionary
            dataset: Dataset name being evaluated
            
        Returns:
            True if results already exist, False otherwise
        """
        # If force_rerun is enabled, always return False (don't skip)
        if self.force_rerun:
            logger.info(f"Force rerun enabled, skipping results check for {task_name}/{model_name}/{dataset}")
            return False

        # Try to use the more robust ResultsCollector logic if available
        collect_results = self._get_collect_results_instance()
        if collect_results is not None:
            try:
                # Create a config object that matches what ResultsCollector expects
                config = {
                    'model_name': model_name,
                    'tasks': {
                        task_name: {**task_settings, 'evaluation_dataset': dataset}
                    },
                    'global_defaults': self.global_defaults
                }
                
                # Use get_results_for_config to check if results exist
                result = collect_results.get_results_for_config(config, task_name, config_index=0)
                if result is not None:
                    logger.info(f"Results already exist for {model_name} on {dataset} for task {task_name} (via ResultsCollector)")
                    return True
                else:
                    logger.info(f"No results found for {model_name} on {dataset} for task {task_name} (via ResultsCollector)")
                    return False
                    
            except Exception as e:
                logger.warning(f"Error using ResultsCollector for result checking: {e}. Falling back to legacy method.")
                # Fall through to legacy method
        
        # Legacy result checking method (fallback)
        try:
            # Get the CSV output path for this task
            task_info = self.tasks[task_name]
            csv_output_path = task_info['default_output_dir'] / "results.csv"
            
            # Check if CSV file exists
            if not csv_output_path.exists():
                logger.info(f"CSV file does not exist: {csv_output_path}")
                return False
            
            # Read the CSV file
            df = pd.read_csv(csv_output_path)
            
            # If CSV is empty, no results exist
            if df.empty:
                logger.info(f"CSV file is empty: {csv_output_path}")
                return False
            
            # Build the configuration identifier based on the key parameters
            # These are the parameters that uniquely identify a configuration
            config_params = {
                'model_name': model_name,
                'evaluation_dataset': dataset
            }
            
            # Add task-specific parameters that affect results
            key_params = ['caption_from', 'use_gaussian_weighting']

            if task_name != "image_captioning":
                key_params += ['representation_cleaning_clean_after_projection', 'representation_cleaning_type']
            
            if task_name != "narratives":
                key_params += ['gaussian_variance']
            
            if task_name == "dense_captioning":
                key_params += ['caption_bboxes_type']

            for param in key_params:
                if param in task_settings:
                    config_params[param] = task_settings[param]
                elif param == 'use_gaussian_weighting':
                    # If use_gaussian_weighting is missing from task_settings, it means False (uniform weighting)
                    config_params[param] = False
                elif param == 'representation_cleaning_clean_after_projection':
                    # If representation_cleaning_clean_after_projection is missing from task_settings, it means False
                    config_params[param] = False
            
            # Check if a row with matching configuration exists
            mask = pd.Series([True] * len(df))
            debug_conditions = {}
            for param, value in config_params.items():
                if param in df.columns:
                    if value == "":
                        value = True  # Handle empty string as a flag
                    if param == "evaluation_dataset":
                        # For evaluation_dataset, we can match the dataset name directly
                        mask &= (df[param].str.contains(value, na=False))
                    else:
                        mask &= (df[param] == value)
                    debug_conditions[param] = value
                else:
                    # If the parameter column doesn't exist in the CSV,
                    # we can't guarantee this configuration was tested
                    logger.info(f"Parameter '{param}' not found in CSV columns: {list(df.columns)}")
                    return False
            
            if task_name in ["image_captioning", "narratives"]:
                logger.info(f"Debug conditions for {task_name}: {debug_conditions}")
            
            matching_rows = df[mask]
            
            if not matching_rows.empty:
                logger.info(f"Results already exist for {model_name} on {dataset} for task {task_name}")
                logger.info(f"Found {len(matching_rows)} matching rows in {csv_output_path}")
                return True
            else:
                logger.info(f"No matching results found for {model_name} on {dataset} for task {task_name}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking existing results for {task_name}/{model_name}/{dataset}: {e}")
            # If we can't check, assume results don't exist and proceed with the task
            return False

    def get_existing_results_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a summary of existing results for each task.
        
        Returns:
            Dictionary mapping task names to lists of existing result configurations
        """
        summary = {}
        
        for task_name, task_info in self.tasks.items():
            csv_path = task_info['default_output_dir'] / "results.csv"
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        # Get unique configurations
                        config_columns = ['model_name', 'evaluation_dataset']
                        optional_columns = ['caption_from', 'batch_size', 'gaussian_variance', 'use_gaussian_weighting']
                        
                        available_columns = [col for col in config_columns + optional_columns if col in df.columns]
                        
                        if available_columns:
                            unique_configs = df[available_columns].drop_duplicates().to_dict('records')
                            summary[task_name] = unique_configs
                        else:
                            summary[task_name] = []
                    else:
                        summary[task_name] = []
                except Exception as e:
                    logger.warning(f"Error reading results for {task_name}: {e}")
                    summary[task_name] = []
            else:
                summary[task_name] = []
        
        return summary

    def _execute_task(self, script_path: Path, args: List[str], 
                     screen_name: str, task_settings: Dict[str, Any], 
                     task_info: Dict[str, Any], log_to_file : bool = False) -> bool:
        """Execute the task evaluation script."""
        # Get conda environment name from task settings
        conda_env = task_settings.get('conda_env', 'decapdino')
        
        # Handle multiple datasets if specified
        datasets = task_settings.get('datasets', [task_settings.get('evaluation_dataset')])
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        success = True
        executed_any = False  # Track if we actually executed any tasks
        
        for dataset in datasets:
            if dataset is None:
                continue
            
            # Extract model name from args (it should be after '--model_name')
            model_name = None
            for i, arg in enumerate(args):
                if arg == '--model_name' and i + 1 < len(args):
                    model_name = args[i + 1]
                    break
            
            if model_name is None:
                logger.error("Could not extract model_name from arguments")
                success = False
                continue
            
            # Get task name from the script path or task_info
            task_name = None
            for name, info in self.tasks.items():
                if info['script'] == str(script_path.relative_to(self.decap_dir)):
                    task_name = name
                    break
            
            if task_name is None:
                logger.error(f"Could not determine task name for script: {script_path}")
                success = False
                continue
            
            # Check if results already exist for this configuration
            if self._check_results_exist(task_name, model_name, task_settings, dataset):
                logger.info(f"Skipping {task_name} for {model_name} on {dataset} - results already exist")
                continue
                
            # Add dataset to args
            dataset_args = args + ['--evaluation_dataset', dataset]
            
            # Add CSV scores output path
            csv_output_path = task_info['default_output_dir'] / "results.csv"
            dataset_args.extend(['--csv_scores_output', str(csv_output_path)])

            if log_to_file:
                log_file = task_info['default_output_dir'] / f"{screen_name}.log"
                dataset_args.extend([f'> {log_file} 2>&1'])
            
            # Build the full command
            cmd = [
                'screen', '-dmS', f"{screen_name}_{(dataset.split('.')[0] if '.' in dataset else dataset)[:5]}",
                'bash', '-c',
                f"cd {script_path.parent} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && python {script_path.name} {' '.join(dataset_args)}"
            ]
            
            try:
                logger.info(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                logger.info(f"Successfully started screen session for {dataset}")
                executed_any = True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to execute command for {dataset}: {e}")
                success = False
            except Exception as e:
                logger.error(f"Unexpected error executing command for {dataset}: {e}")
                success = False
        
        # If no tasks were executed because all results already exist, still consider it successful
        if not executed_any:
            logger.info("All datasets for this task already have results - no execution needed")
        
        return success

    def run_experiments(self, experiment_configs: List[Dict[str, Any]], log_to_file : bool = False) -> List[Dict[str, Any]]:
        """Run multiple experiment configurations."""
        logger.info(f"Starting {len(experiment_configs)} experiments")
        
        all_results = []
        for i, config in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}")
            result = self.run_experiment(config, log_to_file=log_to_file)
            all_results.append(result)
            
            # Save intermediate results
            #self._save_results(all_results, f"intermediate_results_{int(time.time())}.json")
        
        # Save final results
        #self._save_results(all_results, "final_results.json")
        
        logger.info("All experiments completed")
        return all_results

    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save experiment results to JSON file."""
        results_file = self.results_dir / filename
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def get_default_experiment_configs() -> List[Dict[str, Any]]:
    """
    Define default experiment configurations with global defaults.
    
    Each configuration contains:
    - model_name: The model to evaluate
    - tasks: Dictionary of task-specific settings
    - global_defaults: Default parameters applied to all tasks
    - output_directory: Optional custom output directory
    """
    
    # Common model configurations
    models = [
        'viecap_b16_37patches.k',
        'meacap_invlm_b16_37patches.k',
        'meacap_invlm_b16_14patches.k',
        'viecap_b16_14patches.k'
    ]
    
    configs = []
    
    for model in models:
        config = {
            'model_name': model,
            'global_defaults': {
                'batch_size': 16,
                'gaussian_variance': 1.0,
                'conda_env': 'decapdino'
            },
            'tasks': {
                'image_captioning': {
                    'caption_from': 'patches',
                    #'overwrite_inference' : '' # uncomment if you want to overwrite existing results
                    # datasets will use default: ['coco-test.json']
                },
                'dense_captioning': {
                    'caption_from': 'patches',
                    'overwrite_inference': 'false',
                    'representation_cleaning_type': ''
                    # datasets will use default: ['vg12'] 
                },
                'controllable_captioning': {
                    'caption_from': 'patches',
                    'representation_cleaning_type': ''
                    #'overwrite_inference': '', # uncomment if you want to overwrite existing results
                    # datasets will use default: ['flickr30k_entities_test.json']
                },
                'narratives': {
                    'caption_from': 'patches',
                    'representation_cleaning_type': ''
                    #'overwrite_inference': '', # uncomment if you want to overwrite existing results
                    # datasets will use default: ['test.json']
                }
            }
        }
        configs.append(config)
    
    return configs

def main():
    """Main function to run the experiment suite."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multi-task evaluation experiments')
    parser.add_argument('--force-rerun', action='store_true', 
                       help='Force re-running experiments even if results already exist')
    parser.add_argument('--config', type=str, 
                       help='Path to JSON configuration file (optional)')
    parser.add_argument('--log-to-file', action='store_true', default=False,
                       help='Log output of each task to a file in the results directory')
    parser.add_argument('--show-existing', action='store_true',
                       help='Show existing results and exit without running experiments')
    args = parser.parse_args()
    
    # Initialize the experiment runner
    runner = ExperimentRunner(force_rerun=args.force_rerun)
    
    # If user wants to see existing results, show them and exit
    if args.show_existing:
        existing_results = runner.get_existing_results_summary()
        print("\n" + "="*60)
        print("EXISTING RESULTS SUMMARY")
        print("="*60)
        
        for task_name, configs in existing_results.items():
            print(f"\n{task_name.upper()}:")
            if configs:
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
            else:
                print("  No results found")
        
        print(f"\nResults are stored in: {runner.results_dir}")
        return
    
    # Get experiment configurations
    if args.config:
        with open(args.config, 'r') as f:
            experiment_configs = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Use default configurations
        experiment_configs = get_default_experiment_configs()
        logger.info("Using default experiment configurations")
    
    logger.info(f"Loaded {len(experiment_configs)} experiment configurations")
    
    # Run all experiments
    results = runner.run_experiments(experiment_configs, log_to_file=bool(args.log_to_file))
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for result in results:
        model_name = result['model_name']
        duration = result.get('duration', 0)
        task_count = len(result.get('task_results', {}))
        
        print(f"Model: {model_name}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Tasks: {task_count}")
        
        for task_name, task_result in result.get('task_results', {}).items():
            if isinstance(task_result, list):
                # Multiple task configurations
                print(f"    {task_name}:")
                for i, single_result in enumerate(task_result):
                    status = single_result.get('status', 'unknown')
                    reason = single_result.get('reason', '')
                    status_display = f"{status}" + (f" ({reason})" if reason else "")
                    print(f"      Config {i+1}: {status_display}")
            else:
                # Single task configuration
                status = task_result.get('status', 'unknown')
                reason = task_result.get('reason', '')
                status_display = f"{status}" + (f" ({reason})" if reason else "")
                print(f"    {task_name}: {status_display}")
        print()
    
    print(f"Results saved in: {runner.results_dir}")
    print("Check run_tasks.log for detailed execution logs")


if __name__ == "__main__":
    main()
