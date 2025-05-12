import sys
import os
import pandas as pd
import numpy as np
# Add project root to Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

"""
Optimized data processing functions for DSSAT output
"""
# OPTIMIZED: Import only specific pandas functions/modules
from pandas import DataFrame, Series, isna, to_datetime, api
from pandas.core.dtypes.common import is_datetime64_any_dtype
# Import to_numeric as a function to maintain code compatibility
from pandas import to_numeric

# OPTIMIZED: Import only specific numpy functions
from numpy import arange, min, max, full, isclose, mean

import logging
from typing import  List, Tuple
from functools import lru_cache
import config

logger = logging.getLogger(__name__)

# Global cache for frequently used lookups
_variable_info_cache = {}
_date_conversion_cache = {}

def standardize_dtypes(df: DataFrame) -> DataFrame:
    """Optimized data type standardization with vectorized operations."""
    if df is None or df.empty:
        return df
    
    # Drop all-NaN columns efficiently using boolean indexing
    non_empty_cols = df.notna().any()
    df = df.loc[:, non_empty_cols]
    
    # Drop columns that contain only empty strings or spaces
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        non_empty_string_cols = df[string_cols].apply(lambda x: x.str.strip().astype(bool).any())
        df = df.loc[:, ~df.columns.isin(string_cols) | non_empty_string_cols]
    
    # Define column groups for efficient bulk processing
    timestamp_cols = df.columns.intersection({"YEAR", "DOY", "DATE"})
    treatment_cols = df.columns.intersection({"TRT", "TRNO", "TR"})
    # Explicitly exclude the CR column from numeric conversion
    non_numeric_cols = timestamp_cols.union(treatment_cols).union({"CR"})
    potential_numeric_cols = df.columns.difference(non_numeric_cols)
    
    # Process timestamp columns
    for col in timestamp_cols:
        if col in df.columns:
            # Use categorical for low-cardinality timestamp columns
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(str)
    
    # Process treatment columns - always as categorical for efficiency
    for col in treatment_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Handle CR column specifically if present
    if 'CR' in df.columns:
        df['CR'] = df['CR'].astype('category')
    
    # Process numeric columns in bulk
    if len(potential_numeric_cols) > 0:
        # Convert to numeric in one operation
        numeric_df = df[potential_numeric_cols].apply(to_numeric, errors="coerce")
        
        # Find columns with < 10% NaN values after conversion
        valid_numeric_mask = numeric_df.isna().mean() < 0.1
        valid_numeric_cols = potential_numeric_cols[valid_numeric_mask]
        
        # Process each valid numeric column
        for col in valid_numeric_cols:
            df[col] = numeric_df[col]
    
    return df

@lru_cache(maxsize=1024)
def unified_date_convert(year=None, doy=None, date_str=None):
    """
    Convert various date formats to datetime with comprehensive error handling.
    
    Parameters:
        year (int, optional): Year component if providing separate year/doy
        doy (int, optional): Day of year component if providing separate year/doy
        date_str (str, optional): Date string in various formats
        
    Returns:
        pandas.Timestamp or pandas.NaT: Converted date or NaT if conversion fails
    """
    from pandas import to_datetime, NaT
    
    try:
        # Case 1: Handle year/doy format (most efficient path)
        if year is not None and doy is not None:
            try:
                year = int(float(year))
                doy = int(float(doy))
                if 1 <= doy <= 366 and 1900 <= year <= 2100:
                    return to_datetime(f"{year}{doy:03d}", format="%Y%j")
                else:
                    logger.debug(f"Invalid year or DOY range: year={year}, doy={doy}")
                    return NaT
            except Exception as e:
                logger.debug(f"Error converting year/doy: {e}")
                return NaT
        
        # Case 2: Handle date_str format
        if date_str is not None:
            # Clean input
            date_str = str(date_str).strip()
            
            # Check for empty or invalid strings
            if not date_str or date_str in ['-99', '-99.0', 'NA', 'NaN']:
                return NaT
                
            # Handle 7-digit format (YYYYDDD) - common in DSSAT files
            if len(date_str) == 7 and date_str.isdigit():
                try:
                    year_part = int(date_str[:4])
                    doy_part = int(date_str[4:])
                    if 1900 <= year_part <= 2100 and 1 <= doy_part <= 366:
                        return to_datetime(f"{year_part}{doy_part:03d}", format="%Y%j")
                except Exception:
                    pass
            
            # Handle 5-digit format (YYDDD) - also common in DSSAT
            elif len(date_str) == 5 and date_str.isdigit():
                try:
                    year_part = int(date_str[:2])
                    doy_part = int(date_str[2:])
                    century_prefix = "20" if year_part <= 30 else "19"
                    full_year = century_prefix + f"{year_part:02d}"
                    if 1 <= doy_part <= 366:
                        return to_datetime(f"{full_year}{doy_part:03d}", format="%Y%j")
                except Exception:
                    pass
            
            # Handle ISO format (YYYY-MM-DD) or similar
            try:
                return to_datetime(date_str)
            except Exception:
                pass
            
            # Handle additional common formats (with explicit format specification)
            formats_to_try = [
                "%Y%m%d",    # YYYYMMDD
                "%d/%m/%Y",  # DD/MM/YYYY
                "%m/%d/%Y",  # MM/DD/YYYY
                "%Y/%m/%d",  # YYYY/MM/DD
                "%d-%m-%Y",  # DD-MM-YYYY
                "%m-%d-%Y",  # MM-DD-YYYY
                "%b %d %Y",  # Month DD YYYY
                "%d %b %Y"   # DD Month YYYY
            ]
            
            for fmt in formats_to_try:
                try:
                    return to_datetime(date_str, format=fmt)
                except Exception:
                    continue
                    
            # If all attempts fail, log and return NaT
            logger.debug(f"Could not convert date string: {date_str}")
            return NaT
        
        # If neither year/doy nor date_str is provided
        logger.debug("No valid date input provided")
        return NaT
        
    except Exception as e:
        logger.debug(f"Unhandled exception in date conversion: {e}")
        return NaT

def vectorized_date_conversion(df: DataFrame, date_col: str = "DATE") -> DataFrame:
    """Apply efficient vectorized date conversion to DataFrame."""
    # Skip if already datetime
    if date_col in df.columns and is_datetime64_any_dtype(df[date_col]):
        return df
        
    if date_col in df.columns:
        # Direct vectorized conversion
        df[date_col] = to_datetime(df[date_col], errors="coerce")
    elif "YEAR" in df.columns and "DOY" in df.columns:
        # Combine year and day of year for efficient conversion
        year_col = df["YEAR"].astype(str)
        doy_col = df["DOY"].astype(str).str.zfill(3)
        df[date_col] = to_datetime(year_col + doy_col, format="%Y%j", errors="coerce")
    
    return df

def handle_missing_xvar(obs_data: DataFrame, x_var: str, sim_data: DataFrame = None) -> DataFrame:
    """Handle missing X variables in observed data - optimized version."""
    if obs_data is None or obs_data.empty:
        return obs_data
    
    # Create a copy only if we'll be modifying the DataFrame
    if f"{x_var}" not in obs_data.columns and x_var not in obs_data.columns:
        obs_data = obs_data.copy()
    else:
        # X variable already exists, just ensure it's properly named
        if x_var in obs_data.columns and f"{x_var}" not in obs_data.columns:
            obs_data = obs_data.copy()
            obs_data[f"{x_var}"] = obs_data[x_var]
        return obs_data
    
    # Efficiently convert DATE columns once
    if "DATE" in obs_data.columns and not is_datetime64_any_dtype(obs_data["DATE"]):
        obs_data = vectorized_date_conversion(obs_data)
        
    if sim_data is not None and "DATE" in sim_data.columns and not is_datetime64_any_dtype(sim_data["DATE"]):
        sim_data = vectorized_date_conversion(sim_data)
    
    # Handle special time variables using vectorized operations
    if x_var.upper() in ["DAP", "DOY", "DAS"]:
        if "DATE" in obs_data.columns:
            if x_var.upper() == "DOY":
                # Vectorized extraction of day of year
                obs_data[f"{x_var}"] = obs_data["DATE"].dt.dayofyear
            elif x_var.upper() in ["DAP", "DAS"]:
                # Vectorized date difference calculation
                start_date = sim_data["DATE"].min() if sim_data is not None and "DATE" in sim_data.columns else obs_data["DATE"].min()
                obs_data[f"{x_var}"] = (obs_data["DATE"] - start_date).dt.days
        else:
            logger.warning(f"Cannot create {x_var} without 'DATE' column in observed data")
            return obs_data
    
    # Try to infer from simulation data efficiently
    elif sim_data is not None and not sim_data.empty and "DATE" in sim_data.columns and f"{x_var}" in sim_data.columns:
        # Create mapping dict in one operation
        date_to_xvar = dict(zip(
            to_datetime(sim_data["DATE"].dropna()).unique(),
            sim_data[f"{x_var}"].dropna().unique()
        ))
        
        # Apply mapping to all values at once
        obs_dates = to_datetime(obs_data["DATE"].dropna())
        obs_data[f"{x_var}"] = obs_dates.map(date_to_xvar)
        
        if obs_data[f"{x_var}"].isna().any():
            logger.warning(f"Some values for {x_var} could not be inferred from simulation data")
    
    # Create sequence as last resort
    if f"{x_var}" not in obs_data.columns:
        logger.warning(f"Creating sequence for missing {x_var} in observed data")
        obs_data[f"{x_var}"] = arange(len(obs_data))  # Faster than range()
    
    # Efficient in-place forward fill
    obs_data[f"{x_var}"].fillna(method="ffill", inplace=True)
    return obs_data

@lru_cache(maxsize=8)
def parse_data_cde(data_cde_path: str = None) -> dict:
    """Parse DATA.CDE file and return a dictionary of variable information.
    Results are cached for better performance."""
    if data_cde_path is None:
        from config import DSSAT_BASE
        data_cde_path = f"{DSSAT_BASE}/DATA.CDE"
    
    # Return from cache if previously parsed
    if data_cde_path in _variable_info_cache:
        return _variable_info_cache[data_cde_path]
    
    variable_info = {}
    try:
        # Use with context for file handling
        with open(data_cde_path, "r", errors='replace') as f:
            lines = f.readlines()
        
        # One-pass processing with indexed access
        non_comment_lines = [i for i, line in enumerate(lines) 
                            if not line.startswith(("!", "*"))]
        
        header_idx = next((i for i in non_comment_lines if lines[i].startswith("@")), None)
        if header_idx is None:
            logger.error(f"No header found in DATA.CDE file: {data_cde_path}")
            return variable_info
        
        # Process data lines after header
        data_line_indices = [i for i in non_comment_lines if i > header_idx]
        
        for i in data_line_indices:
            line = lines[i]
            if len(line.strip()) == 0:
                continue
            
            # Fixed-width parsing is faster than splitting
            cde = line[0:6].strip()
            label = line[7:20].strip() if len(line) > 7 else ""
            description = line[21:70].strip() if len(line) > 21 else ""
            
            if cde:
                variable_info[cde] = {"label": label, "description": description}
        
        # Cache the result
        _variable_info_cache[data_cde_path] = variable_info
        
    except Exception as e:
        logger.error(f"Error parsing DATA.CDE: {e}")
    
    return variable_info

@lru_cache(maxsize=256)
def get_variable_info(variable_name: str, data_cde_path: str = None) -> tuple:
    """Get label and description for a variable, with caching."""
    try:
        variable_info = parse_data_cde(data_cde_path)
        if variable_name in variable_info:
            return (
                variable_info[variable_name]["label"],
                variable_info[variable_name]["description"],
            )
        return None, None
    
    except Exception as e:
        logger.error(f"Error getting variable info: {e}")
        return None, None

def get_evaluate_variable_pairs(data: DataFrame) -> List[Tuple[str, str, str]]:
    """
    Get pairs of simulated and measured variables from EVALUATE.OUT data.
    Optimized version with bulk processing.
    """
    pairs = []
    
    # Get all column names once
    columns = set(data.columns)
    metadata_cols = {'RUN', 'EXCODE', 'TRNO','TN','TRT','TR', 'RN', 'CR'}
    
    # Find all simulated variables in one pass
    sim_vars = [col for col in columns if col.endswith('S') and col not in metadata_cols]
    
    # Process each simulated variable
    for sim_var in sim_vars:
        base_name = sim_var[:-1]
        measured_var = base_name + 'M'
        
        # Skip if no matching measured variable
        if measured_var not in columns:
            continue
        
        # Check for missing values in bulk
        sim_all_missing = data[sim_var].isna().all()
        meas_all_missing = data[measured_var].isna().all()
        
        if sim_all_missing or meas_all_missing:
            continue
        
        # Get valid data points
        valid_mask = data[[sim_var, measured_var]].notna().all(axis=1)
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            continue
        
        # Efficient check for identical values (only for non-NaN values)
        valid_data = data.loc[valid_mask, [sim_var, measured_var]]
        
        if (valid_data[sim_var] == valid_data[measured_var]).all():
            logger.info(f"Skipping {base_name} - all simulated and measured values are identical")
            continue
        
        # Get variable info
        var_label, _ = get_variable_info(base_name)
        display_name = var_label if var_label else base_name
        pairs.append((display_name, sim_var, measured_var))
    
    return sorted(pairs)

def get_all_evaluate_variables(data: DataFrame) -> List[Tuple[str, str]]:
    """Get all variables from EVALUATE.OUT data with their descriptions.
    Optimized to process columns in bulk."""
    variables = []
    
    # Identify metadata columns to exclude
    metadata_cols = {'RUN', 'EXCODE', 'TRNO','TN','TRT','TR', 'RN', 'CR'}
    
    # Get columns with at least one non-missing value
    valid_cols = [col for col in data.columns 
                 if col not in metadata_cols and not data[col].isna().all()]
    
    # Get variable info for all columns at once
    for col in valid_cols:
        var_label, _ = get_variable_info(col)
        display_name = var_label if var_label else col
        variables.append((display_name, col))
    
    return sorted(variables)

def improved_smart_scale(
    data, variables, target_min=None, target_max=None, scaling_factors=None
):
    """Scale data columns to a target range for visualization.
    Optimized with vectorized calculations. Skips scaling if only one variable is provided."""
    scaled_data = {}
    
    # Filter to variables that exist in data
    available_vars = [var for var in variables if var in data.columns]
    
    # If only one variable is being processed, return original data
    if len(available_vars) == 1:
        var = available_vars[0]
        scaled_data[var] = data[var].copy()
        return scaled_data
    
    # Pre-convert to numeric where possible
    numeric_data = {var: pd.to_numeric(data[var], errors="coerce") 
                   for var in available_vars}
    
    for var in available_vars:
        # Skip completely empty columns
        values = numeric_data[var].dropna().values
        if len(values) == 0:
            continue
        
        # Use provided scaling factors if available
        if scaling_factors and var in scaling_factors:
            scale_factor, offset = scaling_factors[var]
        else:
            # Calculate scaling factors
            var_min, var_max = np.min(values), np.max(values)
            
            # Handle constant values
            if np.isclose(var_min, var_max):
                midpoint = (target_max + target_min) / 2
                scaled_data[var] = pd.Series(
                    np.full(len(data[var]), midpoint), 
                    index=data[var].index
                )
                continue
                
            # Calculate scaling parameters in one operation
            scale_factor = ((target_max + target_min)/2) / ((var_max +var_min)/20)
            offset = target_min - var_min * scale_factor
        
        # Apply scaling in one vectorized operation
        scaled_data[var] = numeric_data[var] * scale_factor + offset
    
    return scaled_data

# Add a background caching utility
class DataCacheManager:
    """Utility class for managing data caching with memory management."""
    
    def __init__(self):
        self.data_cache = {}
        self.variable_info = {}
        self.data_cde_cache = {}
        self.path_cache = {}
        self.cache_size_limit = 1024 * 1024 * 256  # Reduced to 256MB for better memory usage
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cache_data(self, key: str, data: pd.DataFrame, metadata: dict = None):
        """Cache DataFrame with memory management."""
        try:
            # Convert object dtypes to categories where beneficial
            for col in data.select_dtypes(include=['object']):
                if data[col].nunique() / len(data[col]) < 0.5:  # If less than 50% unique values
                    data[col] = data[col].astype('category')
            
            data_size = data.memory_usage(deep=True).sum()
            
            # Clear old entries if cache would exceed limit
            while (self.get_cache_size() + data_size > self.cache_size_limit 
                   and self.data_cache):
                oldest_key = min(self.path_cache.keys(), 
                               key=lambda k: self.path_cache[k].get('last_access', 0))
                self.clear_cache(oldest_key)
            
            self.data_cache[key] = data
            if metadata:
                metadata['last_access'] = pd.Timestamp.now().timestamp()
                self.path_cache[key] = metadata
                
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
    
    def get_cached_data(self, key: str) -> pd.DataFrame:
        """Retrieve cached data if available."""
        if key in self.data_cache:
            self.cache_hits += 1
            if key in self.path_cache:
                self.path_cache[key]['last_access'] = pd.Timestamp.now().timestamp()
            return self.data_cache[key]
        self.cache_misses += 1
        return None
        
    def optimize_memory(self, threshold_mb: int = 200):  # Reduced threshold
        """Optimize memory usage if it exceeds threshold."""
        current_size = self.get_cache_size()
        if current_size > threshold_mb * 1024 * 1024:
            # Remove least recently used items until under threshold
            while self.get_cache_size() > threshold_mb * 1024 * 1024 and self.path_cache:
                oldest_key = min(self.path_cache.keys(), 
                               key=lambda k: self.path_cache[k].get('last_access', 0))
                self.clear_cache(oldest_key)
            
            # Force garbage collection
            import gc
            gc.collect()
            
        # Log cache efficiency metrics
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            hit_rate = (self.cache_hits / total_requests) * 100
            logger.info(f"Cache hit rate: {hit_rate:.1f}%")

    def clear_cache(self, key: str = None):
        """Clear specific or all cached data."""
        if key:
            self.data_cache.pop(key, None)
            self.path_cache.pop(key, None)
        else:
            self.data_cache.clear()
            self.variable_info.clear()
            self.data_cde_cache.clear()
            self.path_cache.clear()
            get_variable_info.cache_clear()
            parse_data_cde.cache_clear()
            unified_date_convert.cache_clear()
    
    def get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        return sum(df.memory_usage(deep=True).sum() for df in self.data_cache.values())
    
    def optimize_memory(self, threshold_mb: int = 400):
        """Optimize memory usage if it exceeds threshold."""
        current_size = self.get_cache_size()
        if current_size > threshold_mb * 1024 * 1024:
            # Remove oldest items until under threshold
            while (self.get_cache_size() > threshold_mb * 1024 * 1024 
                   and self.data_cache):
                self.data_cache.pop(next(iter(self.data_cache)))
            
            # Force garbage collection
            import gc
            gc.collect()

# Create a singleton instance
cache_manager = DataCacheManager()