#!/usr/bin/env python3
"""
Results collection script for decap-dino experiments.

This script loads configurations from a JSON file, searches for experiment results 
in CSV files, and builds dataframes for each task containing evaluation results 
of all methods in the configuration.

Features:
- Loads experiment configurations from JSON files
- Searches CSV result files for matching configurations
- Builds DataFrames for each task (narratives, image_captioning, dense_captioning, controllable_captioning)
- Handles missing results gracefully (creates rows with empty scores)
- Applies task-specific score transformations (multiplies by 100 where appropriate)
- Checks if screen sessions are running for each configuration
- Supports both single and multiple configurations per task (handles task_settings as dict or list)
- Can be used as a script or imported as a module

Task Settings Support:
- Single configuration: tasks: { task_name: {...settings...} }
- Multiple configurations: tasks: { task_name: [{...settings1...}, {...settings2...}] }

When multiple configurations are provided for a task, each will create a separate
row in the resulting DataFrame.

Tasks and their score columns:
- narratives: METEOR, CIDEr, SPICE, RefPAC-S, Bleu_4, ROUGE_L, CLIP-S, PAC-S (×100)
- image_captioning: METEOR, CIDEr, SPICE, RefPAC-S, Bleu_4, ROUGE_L, CLIP-S, PAC-S (×100)
- controllable_captioning: METEOR, CIDEr, SPICE, RefPAC-S, Bleu_4, ROUGE_L, CLIP-S, PAC-S (no scaling)
- dense_captioning: METEOR, CIDEr, SPICE, RefPAC-S, Bleu_4, ROUGE_L, map_score (×100)

Usage as script:
    python collect_results.py --config Backbone_ablations.json
    python collect_results.py --config my_experiments.json --results-dir ./custom_results
    python collect_results.py --config config.json --output results.pkl --format pickle

Usage as module:
    from collect_results import collect_results_from_config
    results = collect_results_from_config('config.json')
    df_narratives = results['narratives']
    df_image = results['image_captioning']
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

# Add current directory to path for imports
if '.' not in sys.path:
    sys.path.append('.')

# Import the ExperimentRunner class to reuse its functionality
from run_tasks import ExperimentRunner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) 

# Import utils if available
try:
    from utils import get_model_infos
except ImportError:
    logger.warning("Could not import get_model_infos from utils.py. Using dummy function.")
    def get_model_infos(model_name):
        """Dummy function if utils.py is not available."""
        return model_name, "unknown", "unknown"


class ResultsCollector:
    """Collects and formats experiment results from CSV files."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the ResultsCollector.
        
        Args:
            results_dir: Directory containing the results. If None, uses the default from ExperimentRunner
        """
        # Use ExperimentRunner to get the default paths and task definitions
        self.runner = ExperimentRunner()
        
        if results_dir:
            self.results_dir = Path(results_dir).resolve()
            # Update task output directories to use custom results dir
            for task_name, task_info in self.runner.tasks.items():
                task_info['default_output_dir'] = self.results_dir / task_name
        else:
            self.results_dir = self.runner.results_dir
        
        logger.info(f"Using results directory: {self.results_dir}")
        
        # Define score columns for each task
        self.task_score_columns = {
            'narratives': {
                'main_scores': ['METEOR', 'CIDEr', 'SPICE', 'RefPAC-S', 'Bleu_4', 'ROUGE_L', 'CLIP-S', 'PAC-S'],
                'std_scores': ['METEOR_std', 'CIDEr_std', 'SPICE_std', 'RefPAC-S_std', 'Bleu_4_std', 'ROUGE_L_std', 'CLIP-S_std', 'PAC-S_std'],
                'time_scores': ['avg_inference_time_per_image', 'std_inference_time_per_image'],
                'multiply_by_100': False,  # All scores except time should be multiplied by 100
                'config_columns': ['use_gaussian_weighting', 'caption_from', 'evaluation_dataset', 'use_attention_weighting', 'representation_cleaning_type', 'representation_cleaning_clean_after_projection']
            },
            'image_captioning': {
                'main_scores': ['METEOR', 'CIDEr', 'SPICE', 'RefPAC-S', 'Bleu_4', 'ROUGE_L', 'CLIP-S', 'PAC-S'],
                'std_scores': ['METEOR_std', 'CIDEr_std', 'SPICE_std', 'RefPAC-S_std', 'Bleu_4_std', 'ROUGE_L_std', 'CLIP-S_std', 'PAC-S_std'],
                'time_scores': ['avg_inference_time_per_image', 'std_inference_time_per_image'],
                'multiply_by_100': True,
                'config_columns': ['use_gaussian_weighting', 'caption_from', 'evaluation_dataset', 'gaussian_variance']
            },
            'controllable_captioning': {
                'main_scores': ['METEOR', 'CIDEr', 'SPICE', 'RefPAC-S', 'Bleu_4', 'ROUGE_L', 'CLIP-S', 'PAC-S'],
                'std_scores': ['METEOR_std', 'CIDEr_std', 'SPICE_std', 'RefPAC-S_std', 'Bleu_4_std', 'ROUGE_L_std', 'CLIP-S_std', 'PAC-S_std'],
                'time_scores': ['avg_inference_time_per_image', 'std_inference_time_per_image'],
                'multiply_by_100': False,  # Based on the notebook, these are not multiplied by 100
                'config_columns': ['use_gaussian_weighting', 'use_attn_map_for_bboxes', 'caption_from', 'evaluation_dataset', 'representation_cleaning_type', 'representation_cleaning_clean_after_projection']
            },
            'dense_captioning': {
                'main_scores': ['METEOR', 'CIDEr', 'SPICE', 'RefPAC-S', 'Bleu_4', 'ROUGE_L', 'map_score', 'CLIP-S', 'PAC-S', 'CLIP-S_cropped', 'PAC-S_cropped'],
                'std_scores': ['METEOR_std', 'CIDEr_std', 'SPICE_std', 'RefPAC-S_std', 'Bleu_4_std', 'ROUGE_L_std', 'map_score_std', 'CLIP-S_std', 'PAC-S_std', 'CLIP-S_cropped_std', 'PAC-S_cropped_std'],
                'time_scores': ['avg_inference_time_per_image', 'std_inference_time_per_image'],
                'multiply_by_100': True,
                'config_columns': ['use_gaussian_weighting', 'use_attn_map_for_bboxes', 'caption_from', 'evaluation_dataset', 'caption_bboxes_type', 'representation_cleaning_type', 'representation_cleaning_clean_after_projection']
            }
        }
    
    def load_config(self, config_path: str) -> List[Dict[str, Any]]:
        """
        Load experiment configurations from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            List of experiment configurations
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Handle different configuration formats
        if isinstance(config_data, list):
            # Format 1: Direct array of configurations
            configs = config_data
        elif isinstance(config_data, dict) and 'experiments' in config_data:
            # Format 2: Object with experiments array and global settings
            configs = config_data['experiments']
            
            # Apply global defaults and output directory to each config
            global_defaults = config_data.get('global_defaults', {})
            output_directory = config_data.get('output_directory')
            
            for config in configs:
                # Add global defaults if not already present
                if 'global_defaults' not in config:
                    config['global_defaults'] = global_defaults
                else:
                    # Merge global defaults
                    merged_defaults = {**global_defaults, **config['global_defaults']}
                    config['global_defaults'] = merged_defaults
                
                # Add output directory if specified
                if output_directory and 'output_directory' not in config:
                    config['output_directory'] = output_directory
                    
            # Update results directory if specified in config
            if output_directory:
                self.results_dir = Path(output_directory).resolve()
                # Update task output directories
                for task_name, task_info in self.runner.tasks.items():
                    task_info['default_output_dir'] = self.results_dir / task_name
                logger.info(f"Updated results directory from config: {self.results_dir}")
                    
        else:
            # Format 3: Simple dictionary format (convert to list)
            configs = [config_data] if isinstance(config_data, dict) else config_data
        
        logger.info(f"Loaded {len(configs)} configurations from {config_path}")
        return configs
    
    def get_results_for_config(self, config: Dict[str, Any], task_name: str, config_index: int = 0) -> Optional[pd.Series]:
        """
        Get results for a specific configuration and task from the CSV file.
        
        Args:
            config: Experiment configuration
            task_name: Name of the task
            config_index: Index of the configuration if task_settings is a list
            
        Returns:
            Pandas Series with results, or None if not found
        """
        if task_name not in self.runner.tasks:
            logger.warning(f"Unknown task: {task_name}")
            return None
        
        # Get the CSV file path for this task
        task_info = self.runner.tasks[task_name]
        csv_path = task_info['default_output_dir'] / "results.csv"
        
        if not csv_path.exists():
            logger.debug(f"CSV file does not exist: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            # treat empty values for 'representation_cleaning_type' as "" 
            for k in ['representation_cleaning_type']:
                if k in df.columns:
                    df[k] = df[k].fillna("")
            if df.empty:
                logger.debug(f"CSV file is empty: {csv_path}")
                return None
            
            # Extract task settings for this task from the configuration
            task_settings_raw = config.get('tasks', {}).get(task_name, {})
            
            # Handle both single dict and list of dicts for task settings
            if isinstance(task_settings_raw, list):
                if config_index >= len(task_settings_raw):
                    logger.warning(f"Config index {config_index} out of range for task {task_name}")
                    return None
                task_settings = task_settings_raw[config_index]
            else:
                task_settings = task_settings_raw
            
            # Apply global defaults
            global_defaults = config.get('global_defaults', {})
            for key, value in global_defaults.items():
                if key not in task_settings:
                    task_settings[key] = value
            
            # Build search criteria
            search_criteria = {
                'model_name': config['model_name']
            }
            
            # Add task-specific search criteria
            datasets = task_settings.get('datasets', [task_settings.get('evaluation_dataset')])
            if datasets and datasets[0]:
                search_criteria['evaluation_dataset'] = datasets[0]
            
            # Add other important parameters
            key_params = ['caption_from', 'batch_size', 'gaussian_variance', 'use_gaussian_weighting', 
                         'use_attention_weighting', 'use_attn_map_for_bboxes', 'caption_bboxes_type',
                         'representation_cleaning_type', 'representation_cleaning_clean_after_projection']
            
            for param in key_params:
                if param in task_settings:
                    search_criteria[param] = task_settings[param]
            
            # Search for matching row
            mask = pd.Series([True] * len(df))
            debug_info = {}
            for param, value in search_criteria.items():
                if param in ['use_gaussian_weighting', 'use_attention_weighting', 'use_attn_map_for_bboxes', 'representation_cleaning_clean_after_projection'] and value == "":
                    value = True
                    logger.debug(f"Using default value True for parameter '{param}' in task '{task_name}' since it is empty string")
                if param in df.columns:
                    # Special handling for evaluation_dataset to handle ./ prefix differences
                    if param == 'evaluation_dataset':
                        # Try to match with and without ./ prefix
                        value_no_prefix = value.lstrip('./')
                        mask_condition = (df[param] == value) | (df[param] == f'./{value_no_prefix}') # | (df[param].str.lstrip('./') == value_no_prefix)
                        mask &= mask_condition
                    elif param == 'representation_cleaning_clean_after_projection':
                        # If representation_cleaning_type is none or not specified, ignore this parameter
                        #print(f"Checking representation_cleaning_clean_after_projection with representation_cleaning_type={search_criteria.get('representation_cleaning_type')}")
                        if  ('representation_cleaning_type' not in search_criteria or search_criteria['representation_cleaning_type'] in ['none', None, '']):
                            #print(f"Ignoring parameter '{param}' in search criteria for task '{task_name}' since representation_cleaning_type is none or not specified")
                            continue
                    else:
                        # Handle NaN values in CSV - if CSV has NaN and we're looking for a specific value,
                        # we might want to be more lenient for certain parameters
                        if param in ['batch_size'] and pd.isna(df[param]).any():
                            # For batch_size, if CSV has NaN, skip this criterion
                            mask_with_nan = (df[param] == value) | pd.isna(df[param])
                            mask &= mask_with_nan
                        else:
                            mask &= (df[param] == value)
                    debug_info[param] = value
                else:
                    logger.debug(f"Parameter '{param}' not found in CSV columns for {task_name}")
            

            matching_rows = df[mask]


            # Debug info for proxyclip models
            #if 'proxyclip' in config['model_name'].lower():
            #    logger.info(f"DEBUG: Searching for {config['model_name']} in {task_name}")
            #    logger.info(f"DEBUG: Search criteria: {debug_info}")
            #    logger.info(f"DEBUG: Found {len(matching_rows)} matching rows after applying mask")
            #    if len(df[df['model_name'] == config['model_name']]) > 0:
            #        logger.info(f"DEBUG: Model exists in CSV but may not match criteria")
            #        proxyclip_row = df[df['model_name'] == config['model_name']].iloc[0]
            #        logger.info(f"DEBUG: Actual CSV values: {proxyclip_row[list(debug_info.keys())].to_dict()}")
            
            # debug why dense captioning might not find results
            if task_name == 'dense_captioning' and matching_rows.empty:
                logger.debug(f"DEBUG: No matching rows for dense_captioning with criteria: {debug_info}")
                if 'model_name' in df.columns and (df['model_name'] == config['model_name']).any():
                    logger.debug(f"DEBUG: Model {config['model_name']} exists in CSV but may not match other criteria")
                    model_rows = df[df['model_name'] == config['model_name']]
                    logger.debug(f"DEBUG: Found {len(model_rows)} rows with model_name={config['model_name']}")
                    for idx, row in model_rows.iterrows():
                        logger.debug(f"DEBUG: Row {idx}: {row[list(debug_info.keys())].to_dict()}")
                else:
                    logger.debug(f"DEBUG: Model {config['model_name']} does not exist in CSV at all")
            
            
            if not matching_rows.empty:
                # check if multiple rows found
                if len(matching_rows) > 1:
                    logger.warning(f"Multiple matching rows found for {config['model_name']} in {task_name} with criteria {debug_info}.")
                    # check if the rows are identical in score columns
                    score_columns = (self.task_score_columns[task_name]['main_scores'] + 
                                     self.task_score_columns[task_name]['std_scores'] + 
                                     self.task_score_columns[task_name]['time_scores'])
                    # drop the rows with some NaN values in score columns
                    # Considera solo le colonne che esistono nel CSV
                    available_score_columns = [c for c in score_columns if c in matching_rows.columns]

                    #matching_rows = matching_rows.dropna(subset=score_columns, how='all')
                    

                    if available_score_columns:
                        # drop rows that have all NaN in available score columns
                        matching_rows = matching_rows.dropna(subset=available_score_columns, how='all')
                    
                    if matching_rows[available_score_columns].nunique().sum() > len(score_columns): # it means not all rows are identical
                        logger.warning(f"{len(matching_rows)} distinct matching rows found for {config['model_name']} in {task_name}. Using the best one.")
                        # select the row with the best main score (first main score)
                        if 'CIDEr' in matching_rows.columns:
                            main_score = 'CIDEr'
                        else:
                            main_score = self.task_score_columns[task_name]['main_scores'][0]
                        if main_score in matching_rows.columns:
                            matching_rows = matching_rows.sort_values(by=main_score, ascending=False)
                    else:
                        logger.info(f"All matching rows have identical scores for {config['model_name']} in {task_name}. Using the first one.")
                result = matching_rows.iloc[0].copy()
                logger.debug(f"Found results for {config['model_name']} in {task_name}")
                return result
            else:
                logger.info(f"No results found for {config['model_name']} in {task_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading results for {task_name}/{config['model_name']}: {e}")
            #raise e
            return None
    
    def _check_screen_running(self, config: Dict[str, Any], task_name: str, config_index: Optional[int] = None) -> bool:
        """Check if there's a screen session running for the given task configuration."""
        # Get task settings
        task_settings_raw = config.get('tasks', {}).get(task_name, {})
        global_defaults = config.get('global_defaults', {})
        
        # Handle both single dict and list of dicts for task settings
        if isinstance(task_settings_raw, list):
            #if config_index is None or config_index >= len(task_settings_raw):
            #    # If no specific index provided or out of range, check first config
            #    config_index = 0
            #if config_index >= len(task_settings_raw):
            #    return False
            task_settings = task_settings_raw[config_index]
        else:
            task_settings = task_settings_raw
        
        # Merge settings
        merged_settings = {**global_defaults, **task_settings}
        
        # Generate screen name using the ExperimentRunner's method
        screen_name = self.runner._generate_screen_name(task_name, config['model_name'], merged_settings) # , config_index
        
        # Check if screen exists using the ExperimentRunner's method
        return self.runner._screen_exists(screen_name)
    
    def create_task_dataframe(self, configs: List[Dict[str, Any]], task_name: str) -> pd.DataFrame:
        """
        Create a dataframe for a specific task with results from all configurations.
        
        Args:
            configs: List of experiment configurations
            task_name: Name of the task
            
        Returns:
            DataFrame with results for all configurations, including a 'screen_running' 
            boolean column indicating if a screen session is active for each config
        """
        logger.info(f"Creating dataframe for task: {task_name}")
        
        if task_name not in self.task_score_columns:
            logger.error(f"Task {task_name} not supported")
            return pd.DataFrame()
        
        task_config = self.task_score_columns[task_name]
        
        # Initialize list to store rows
        rows = []
        
        for config_index, config in enumerate(configs):
            # Check if this configuration includes this task
            if task_name not in config.get('tasks', {}):
                logger.debug(f"Task {task_name} not found in config for {config['model_name']}")
                continue
            
            # Get task settings - handle both single dict and list of dicts
            task_settings_raw = config.get('tasks', {}).get(task_name, {})
            
            # Handle both single dict and list of dicts for task settings
            if isinstance(task_settings_raw, list):
                # Multiple task settings - create a row for each configuration
                for settings_index, task_settings in enumerate(task_settings_raw):
                    row_data = self._create_single_row(config, task_name, task_config, settings_index)# , settings_index)
                    if row_data:
                        rows.append(row_data)
            else:
                # Single task settings dict - create one row
                row_data = self._create_single_row(config, task_name, task_config, 0) #, 0
                if row_data:
                    rows.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        if df.empty:
            logger.warning(f"No data found for task {task_name}")
            return df
        
        # Fill NaN values with empty strings for display
        df.fillna('', inplace=True)
        
        # Apply score multiplication if needed
        if task_config['multiply_by_100']:
            score_columns_to_multiply = task_config['main_scores'] + task_config['std_scores']
            for col in score_columns_to_multiply:
                if col in df.columns:
                    # Only multiply non-empty, numeric values
                    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                    df.loc[numeric_mask, col] = pd.to_numeric(df.loc[numeric_mask, col]) * 100
        
        # Add model info columns if get_model_infos is available
        try:
            df_infos = df['model_name'].apply(get_model_infos).apply(pd.Series)
            df_infos.columns = ['model', 'n_patches', 'backbone']
            df = pd.concat([df, df_infos], axis=1)
        except Exception as e:
            logger.warning(f"Could not add model info columns: {e}")
        
        # Select and order columns
        base_columns = ['model_name']
        if 'model' in df.columns:
            base_columns.extend(['model', 'n_patches', 'backbone'])
        
        # Add screen running status column
        status_columns = ['screen_running'] if 'screen_running' in df.columns else []
        
        score_columns = []
        for col_list in [task_config['main_scores'], task_config['std_scores'], task_config['time_scores']]:
            score_columns.extend([col for col in col_list if col in df.columns])
        
        config_columns = [col for col in task_config['config_columns'] if col in df.columns]
        
        final_columns = base_columns + status_columns + score_columns + config_columns
        
        # Only keep columns that exist in the dataframe
        final_columns = [col for col in final_columns if col in df.columns]
        df = df[final_columns]
        
        logger.info(f"Created dataframe for {task_name} with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _create_single_row(self, config: Dict[str, Any], task_name: str, task_config: Dict[str, Any], config_index: int) -> Dict[str, Any]:
        """
        Create a single row of data for a task configuration.
        
        Args:
            config: Experiment configuration
            task_name: Name of the task
            task_config: Task configuration with score columns info
            config_index: Index of the configuration within the task settings
            
        Returns:
            Dictionary representing a single row of data, or None if no valid configuration
        """
        # Get results for this configuration and task
        result_row = self.get_results_for_config(config, task_name, config_index) # , config_index
        
        if result_row is not None:
            # We found results, use them
            row_data = result_row.to_dict()
            # Add screen running status even if we have results
            row_data['screen_running'] = self._check_screen_running(config, task_name, config_index) # , config_index
        else:
            # No results found, create empty row with config info
            row_data = {
                'model_name': config['model_name']
            }
            
            # Get task settings - handle both single dict and list of dicts
            task_settings_raw = config.get('tasks', {}).get(task_name, {})
            global_defaults = config.get('global_defaults', {})
            
            # Handle both single dict and list of dicts for task settings
            if isinstance(task_settings_raw, list):
                if config_index >= len(task_settings_raw):
                    return None  # Invalid configuration index
                task_settings = task_settings_raw[config_index]
            else:
                task_settings = task_settings_raw
            
            # Merge settings
            merged_settings = {**global_defaults, **task_settings}
            
            for param in task_config['config_columns']:
                if param in merged_settings:
                    row_data[param] = merged_settings[param]
            
            # Add empty scores
            all_score_columns = (task_config['main_scores'] + 
                               task_config['std_scores'] + 
                               task_config['time_scores'])
            
            for col in all_score_columns:
                row_data[col] = None
        
        # Add screen running status
        row_data['screen_running'] = self._check_screen_running(config, task_name, config_index) # , config_index
        
        return row_data
    
    def print_summary_report(self, results: Dict[str, pd.DataFrame]):
        """
        Print a summary report of the collected results.
        
        Args:
            results: Dictionary of task results
        """
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY")
        print("="*80)
        
        for task_name, df in results.items():
            print(f"\n{task_name.upper()} RESULTS:")
            print("-" * 50)
            
            if df.empty:
                print("  No results found")
                continue
            
            print(f"  Total configurations: {len(df)}")
            
            # Show models
            if 'model_name' in df.columns:
                models = df['model_name'].tolist()
                print(f"  Models: {models}")
            
            # Show datasets
            if 'evaluation_dataset' in df.columns:
                datasets = df['evaluation_dataset'].unique().tolist()
                print(f"  Datasets: {datasets}")
            
            # Show score columns with data
            task_config = self.task_score_columns.get(task_name, {})
            score_cols = task_config.get('main_scores', [])
            
            if score_cols:
                print(f"  Score columns:")
                for col in score_cols:
                    if col in df.columns:
                        # Count non-empty values
                        non_empty = df[col].replace('', pd.NA).notna().sum()
                        print(f"    {col}: {non_empty}/{len(df)} values")
            
            # Show screen running status
            if 'screen_running' in df.columns:
                running_count = df['screen_running'].sum()
                print(f"  Running screen sessions: {running_count}/{len(df)}")
            
            # Show sample row
            if not df.empty:
                print(f"  Sample configuration:")
                sample_row = df.iloc[0]
                for col in ['model_name', 'caption_from', 'evaluation_dataset']:
                    if col in df.columns:
                        print(f"    {col}: {sample_row[col]}")
                if 'screen_running' in df.columns:
                    print(f"    screen_running: {sample_row['screen_running']}")

    def collect_all_results(self, config_path: str) -> Dict[str, pd.DataFrame]:
        """
        Collect results for all tasks from the given configuration file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Dictionary mapping task names to their result DataFrames
        """
        # Load configurations
        configs = self.load_config(config_path)
        
        # Apply defaults using ExperimentRunner logic
        processed_configs = []
        for config in configs:
            processed_config = self.runner.set_config_defaults(config)
            processed_configs.append(processed_config)
        
        # Collect results for each task
        results = {}
        
        for task_name in self.task_score_columns.keys():
            df = self.create_task_dataframe(processed_configs, task_name)
            if not df.empty:
                results[task_name] = df
            else:
                logger.warning(f"No results collected for task: {task_name}")
        
        return results


def collect_results_from_config(config_path: str, results_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to collect results from a configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        results_dir: Directory containing result CSV files (optional)
        
    Returns:
        Dictionary mapping task names to their result DataFrames
        
    Example:
        >>> results = collect_results_from_config('sample_config.json')
        >>> df_narratives = results['narratives']
        >>> df_image = results['image_captioning']
    """
    collector = ResultsCollector(results_dir=results_dir)
    return collector.collect_all_results(config_path)


def main():
    """Main function to collect and display results."""
    parser = argparse.ArgumentParser(description='Collect experiment results from CSV files')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--results-dir', type=str,
                       help='Directory containing result CSV files (optional)')
    parser.add_argument('--output', type=str,
                       help='Output file to save results (optional)')
    parser.add_argument('--format', choices=['json', 'pickle'], default='pickle',
                       help='Output format for saving results')
    args = parser.parse_args()
    
    # Initialize collector
    collector = ResultsCollector(results_dir=args.results_dir)
    
    # Collect results
    try:
        results = collector.collect_all_results(args.config)
        
        # Display summary
        collector.print_summary_report(results)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'json':
                # Convert DataFrames to JSON
                results_json = {}
                for task_name, df in results.items():
                    results_json[task_name] = df.to_dict('records')
                
                with open(output_path, 'w') as f:
                    json.dump(results_json, f, indent=2, default=str)
                    
            elif args.format == 'pickle':
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(results, f)
            
            print(f"\nResults saved to: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error collecting results: {e}")
        raise


if __name__ == "__main__":
    main()
