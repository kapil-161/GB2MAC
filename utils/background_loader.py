"""
Optimized background data loading utility for DSSAT Viewer
"""
import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BackgroundLoader:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, concurrent.futures.Future] = {}
        self.cache: Dict[str, Any] = {}
        self.cache_size_limit = 1000
        
    def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> None:
        """Queue data loading task in background"""
        if key in self.futures:
            # Already loading
            return
            
        if key in self.cache:
            # Data already cached
            return
            
        future = self.executor.submit(self._load_and_process, loader_func, *args, **kwargs)
        self.futures[key] = future
        
    def _load_and_process(self, loader_func: Callable, *args, **kwargs) -> Any:
        """Load and preprocess data with optimized memory usage"""
        try:
            data = loader_func(*args, **kwargs)
            
            if isinstance(data, pd.DataFrame):
                # Optimize memory usage for DataFrames
                data = self._optimize_dataframe(data)
                
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return None
            
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Convert string columns to categorical if beneficial
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
                    
            elif col_type == 'float64':
                # Downcast floats where possible
                if df[col].notnull().all():
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        
            elif col_type == 'int64':
                # Downcast integers where possible
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif min_val >= 0:
                    if max_val <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    
        return df
        
    def get_data(self, key: str, timeout: Optional[float] = None) -> Any:
        """Get loaded data with timeout"""
        if key in self.cache:
            return self.cache[key]
            
        if key not in self.futures:
            return None
            
        try:
            future = self.futures[key]
            result = future.result(timeout=timeout)
            
            # Manage cache size
            if len(self.cache) >= self.cache_size_limit:
                # Remove oldest entries
                oldest_keys = sorted(self.cache.keys())[:len(self.cache)//2]
                for old_key in oldest_keys:
                    del self.cache[old_key]
                    
            self.cache[key] = result
            del self.futures[key]
            return result
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"Timeout waiting for data: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting data for {key}: {e}", exc_info=True)
            del self.futures[key]
            return None
            
    def cancel_all(self) -> None:
        """Cancel all pending loads"""
        for future in self.futures.values():
            future.cancel()
        self.futures.clear()
        
    def clear_cache(self) -> None:
        """Clear the data cache"""
        self.cache.clear()