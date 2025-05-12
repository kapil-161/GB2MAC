"""
Model performance metrics calculation with minimal dependencies
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate model performance metrics using only NumPy."""
    
    @staticmethod
    def d_stat(measured, simulated):
        """Calculate Willmott's index of agreement (d-stat)."""
        try:
            # Convert inputs to numpy arrays and ensure they are 1D
            M = np.asarray(measured, dtype=float).flatten() if hasattr(measured, 'shape') else np.asarray([measured], dtype=float)
            S = np.asarray(simulated, dtype=float).flatten() if hasattr(simulated, 'shape') else np.asarray([simulated], dtype=float)
            
            # Skip calculation if arrays have different lengths or contain NaN values
            if len(M) != len(S) or np.isnan(M).any() or np.isnan(S).any():
                logger.warning("Invalid inputs for d-stat calculation")
                return 0.0
                
            M_mean = np.mean(M)
            
            numerator = np.sum((M - S) ** 2)
            denominator = np.sum((np.abs(M - M_mean) + np.abs(S - M_mean)) ** 2)
            
            return 1 - (numerator / denominator) if denominator != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating d-stat: {e}", exc_info=True)
            return 0.0
            
    @staticmethod
    def rmse(obs_values, sim_values):
        """Calculate Root Mean Square Error."""
        try:
            # Convert inputs to numpy arrays and ensure they are 1D
            obs = np.asarray(obs_values, dtype=float).flatten() if hasattr(obs_values, 'shape') else np.asarray([obs_values], dtype=float)
            sim = np.asarray(sim_values, dtype=float).flatten() if hasattr(sim_values, 'shape') else np.asarray([sim_values], dtype=float)
            
            # Skip calculation if arrays have different lengths or contain NaN values
            if len(obs) != len(sim) or np.isnan(obs).any() or np.isnan(sim).any():
                logger.warning("Invalid inputs for RMSE calculation")
                return 0.0
                
            return np.sqrt(np.mean((obs - sim) ** 2))
            
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}", exc_info=True)
            return 0.0
    
    @staticmethod
    def r_squared(x, y):
        """Calculate R-squared coefficient."""
        try:
            # Convert inputs to numpy arrays and ensure they are 1D
            x_arr = np.asarray(x, dtype=float).flatten() if hasattr(x, 'shape') else np.asarray([x], dtype=float)
            y_arr = np.asarray(y, dtype=float).flatten() if hasattr(y, 'shape') else np.asarray([y], dtype=float)
            
            # Skip calculation if arrays have different lengths or contain NaN values
            if len(x_arr) != len(y_arr) or len(x_arr) < 2 or np.isnan(x_arr).any() or np.isnan(y_arr).any():
                logger.warning("Invalid inputs for R-squared calculation")
                return 0.0
                
            correlation_matrix = np.corrcoef(x_arr, y_arr)
            return correlation_matrix[0, 1] ** 2
            
        except Exception as e:
            logger.error(f"Error calculating R-squared: {e}", exc_info=True)
            return 0.0
            
    @staticmethod
    def calculate_metrics(sim_values, obs_values, treatment_number):
        """Calculate multiple performance metrics using only NumPy."""
        try:
            # Convert inputs to numpy arrays and ensure they are 1D
            sim = np.asarray(sim_values, dtype=float).flatten() if hasattr(sim_values, 'shape') else np.asarray([sim_values], dtype=float)
            obs = np.asarray(obs_values, dtype=float).flatten() if hasattr(obs_values, 'shape') else np.asarray([obs_values], dtype=float)
            
            if len(sim) == 0 or len(obs) == 0:
                logger.warning("Empty input arrays")
                return None
                
            # Prepare data
            min_length = min(len(sim), len(obs))
            mask = ~np.isnan(sim[:min_length]) & ~np.isnan(obs[:min_length])
            sim = sim[:min_length][mask]
            obs = obs[:min_length][mask]
            
            if len(sim) == 0:
                logger.warning("No valid pairs after filtering")
                return None
                
            # Calculate metrics
            mean_obs = np.mean(obs)
            n = len(sim)
            
            rmse_value = MetricsCalculator.rmse(obs, sim)
            nrmse = (rmse_value / mean_obs) * 100 if mean_obs != 0 else None
            d_stat = MetricsCalculator.d_stat(obs, sim)
            
            return {
                "TRT": treatment_number,
                "n": n,
                "RMSE": rmse_value,
                "NRMSE": nrmse,
                "Willmott's d-stat": d_stat,
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return None