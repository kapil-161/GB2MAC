"""
Scatter Plot Widget for DSSAT Viewer
Replaces Dash/Plotly with PyQtGraph for scatter plotting
"""
import os
import sys
import logging
from typing import List, Dict,  Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, 
     QFrame, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

import config
from utils.dssat_paths import get_crop_details
from data.dssat_io import read_evaluate_file
from data.data_processing import (
    get_evaluate_variable_pairs, get_all_evaluate_variables,
    get_variable_info
)
from models.metrics import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)

class ScatterPlotWidget(QWidget):
    """
    Custom widget for scatter plot visualization using PyQtGraph
    
    Replaces Dash/Plotly scatter plots with PyQtGraph for better integration
    and performance in a desktop application.
    """
    
    # Signal when metrics are calculated
    metrics_calculated = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Store data and metrics
        self.evaluate_data = None
        self.metrics_data = []
        
        # Create color and symbol cycles
        self.colors = config.PLOT_COLORS
        self.symbols = config.MARKER_SYMBOLS
        
        # Treatment color mapping
        self.treatment_colors = {}
        
        # Performance optimizations
        self.batch_size = 5000  # Increased batch size for better performance
        self.enable_antialiasing = False
        
        # Configure plot defaults for maximum performance
        pg.setConfigOptions(
            antialias=False,
            useOpenGL=True,
            enableExperimental=True
        )
        
        # Add caching
        self._data_cache = {}
        self._plot_cache = {}
        self._metrics_cache = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main layout as grid to support multiple plots
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
        
        # Initially create a single plot
        self.plot_widgets = []
        self.add_plot_widget(0, 0)
    
    def add_plot_widget(self, row: int, col: int) -> pg.PlotWidget:
        """Add a plot widget to the grid and return it"""
        plot = pg.PlotWidget()
        plot.setBackground('w')  # White background
        plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Add to layout
        self.main_layout.addWidget(plot, row, col)
        self.plot_widgets.append(plot)
        return plot
    
    def clear_plots(self):
        """Clear all plots and reset the layout"""
        # Remove all widgets from layout
        for i in reversed(range(self.main_layout.count())): 
            widget = self.main_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Clear list of plot widgets
        self.plot_widgets = []
        
        # Reset treatment colors
        self.treatment_colors = {}
    
    def batch_render_points(self, x_vals, y_vals, brush, symbol, name=None):
        """Render points in batches for better performance"""
        if len(x_vals) <= self.batch_size:
            scatter = pg.ScatterPlotItem(
                x=x_vals, y=y_vals,
                pen=None, brush=brush,
                symbol=symbol, size=10,
                name=name
            )
            return scatter
            
        # Split into batches
        batches = []
        for i in range(0, len(x_vals), self.batch_size):
            batch_x = x_vals[i:i + self.batch_size]
            batch_y = y_vals[i:i + self.batch_size]
            
            scatter = pg.ScatterPlotItem(
                x=batch_x, y=batch_y,
                pen=None, brush=brush,
                symbol=symbol, size=10,
                name=None  # Only set name on first batch
            )
            batches.append(scatter)
            
        return batches

    def plot_sim_vs_meas(self, 
                    selected_folder: str,
                    selected_treatments: List[str],
                    selected_vars: List[Any],
                    treatment_names: Dict[str, str] = None):
        """
        Create simulated vs measured scatter plots
        
        Args:
            selected_folder: Selected crop folder
            selected_treatments: List of selected treatments
            selected_vars: List of variables to plot (from auto-pairing)
        """
        # Clear previous metrics
        self.metrics_data = []
        self.metrics_calculated.emit([])  # Always emit empty list when starting new plot
        
        # Read evaluate file
        if self.evaluate_data is None or self.evaluate_data.empty:
            self.evaluate_data = read_evaluate_file(selected_folder)
            if self.evaluate_data is None:
                logger.warning("No evaluation data available")
                return

        logger.info(f"Evaluate data loaded with columns: {self.evaluate_data.columns.tolist()}")
        
        # Get variable pairs for auto mode
        var_pairs = get_evaluate_variable_pairs(self.evaluate_data)
        logger.info(f"Found {len(var_pairs)} variable pairs")
        
        # Filter to selected variables
        selected_pairs = []
        for var_item in selected_vars:
            logger.info(f"Processing selected var: {var_item} of type {type(var_item)}")
            
            # Handle different formats of variables
            if isinstance(var_item, tuple) and len(var_item) == 3:
                # Direct tuple format (display_name, sim_var, meas_var)
                selected_pairs.append(var_item)
                logger.info(f"Added tuple directly: {var_item}")
            elif hasattr(var_item, 'data') and callable(var_item.data):
                # QListWidgetItem with UserRole data
                try:
                    var_data = var_item.data(Qt.ItemDataRole.UserRole)
                    if var_data and isinstance(var_data, tuple) and len(var_data) == 3:
                        selected_pairs.append(var_data)
                        logger.info(f"Added from QListWidgetItem data: {var_data}")
                    else:
                        # Fallback to text
                        var_text = var_item.text()
                        logger.info(f"Using text from QListWidgetItem: {var_text}")
                        # Find in var_pairs by display name
                        found = False
                        for pair in var_pairs:
                            if pair[0] == var_text:
                                selected_pairs.append(pair)
                                logger.info(f"Found matching pair: {pair}")
                                found = True
                                break
                        if not found:
                            logger.warning(f"Could not find matching pair for {var_text}")
                except Exception as e:
                    logger.error(f"Error getting data from QListWidgetItem: {e}", exc_info=True)
            elif isinstance(var_item, str):
                if '(' in var_item and ')' in var_item:
                    try:
                        # Parse string representation of tuple
                        parts = var_item.strip("()").split(", ")
                        display_name = parts[0].strip("'\"")
                        sim_var = parts[1].strip("'\"")
                        meas_var = parts[2].strip("'\"")
                        selected_pairs.append((display_name, sim_var, meas_var))
                        logger.info(f"Parsed tuple from string: {display_name}, {sim_var}, {meas_var}")
                    except Exception as e:
                        logger.error(f"Error parsing tuple string {var_item}: {e}", exc_info=True)
                else:
                    # Try to find in var_pairs by display name
                    found = False
                    for pair in var_pairs:
                        if pair[0] == var_item:
                            selected_pairs.append(pair)
                            logger.info(f"Found matching pair for string: {pair}")
                            found = True
                            break
                    if not found:
                        logger.warning(f"Could not find matching pair for string {var_item}")
        
        logger.info(f"Final selected pairs: {selected_pairs}")
        
        # Clear existing plots
        self.clear_plots()
        
        # Determine grid layout based on number of pairs
        n_pairs = len(selected_pairs)
        if n_pairs == 0:
            logger.warning("No variable pairs selected")
            return
                
        if n_pairs == 1:
            n_rows, n_cols = 1, 1
        elif n_pairs == 2:
            n_rows, n_cols = 1, 2
        elif n_pairs <= 4:
            n_rows, n_cols = 2, 2
        elif n_pairs <= 6:
            n_rows, n_cols = 2, 3
        elif n_pairs <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 4, 4  # Cap at 16 plots
        
        logger.info(f"Using grid layout: {n_rows}x{n_cols}")
        
        # Map treatments to colors for consistency
        self.treatment_colors = {}
        
        # First, find all treatments across all variable pairs
        all_treatments = set()
        for display_name, sim_var, meas_var in selected_pairs:
            if sim_var in self.evaluate_data.columns and meas_var in self.evaluate_data.columns:
                mask = self.evaluate_data[[sim_var, meas_var, 'TRNO']].notna().all(axis=1)
                valid_data = self.evaluate_data[mask]
                for trno in valid_data['TRNO'].unique():
                    all_treatments.add(str(trno))
                        
        # If there are selected treatments, filter to just those
        if selected_treatments:
            all_treatments = [t for t in all_treatments if t in selected_treatments]
                
        # Assign colors to treatments
        for j, trno in enumerate(sorted(all_treatments)):
            color_idx = j % len(self.colors)
            self.treatment_colors[trno] = color_idx
        
        # Create a main layout for all content
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(main_layout)
        
        # Create grid container for the plots
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(10)
        grid_widget.setLayout(grid_layout)
        
        # Create legend container
        legend_widget = QWidget()
        legend_widget.setMinimumWidth(150)
        legend_widget.setMaximumWidth(200)
        legend_layout = QVBoxLayout()
        legend_layout.setContentsMargins(5, 5, 5, 5)
        legend_widget.setLayout(legend_layout)
        
        # Add title to legend
        legend_title = QLabel("Legend")
        legend_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        legend_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend_layout.addWidget(legend_title)
        
        # Add 1:1 line to legend
        line_entry = QWidget()
        line_layout = QHBoxLayout()
        line_layout.setContentsMargins(5, 2, 5, 2)
        line_entry.setLayout(line_layout)
        
        line_sample = QFrame()
        line_sample.setFrameShape(QFrame.Shape.HLine)
        line_sample.setFrameShadow(QFrame.Shadow.Plain)
        line_sample.setStyleSheet("border: 1px dashed red;")
        line_sample.setFixedWidth(30)
        line_sample.setFixedHeight(2)
        
        line_label = QLabel("1:1 Line")
        
        line_layout.addWidget(line_sample)
        line_layout.addWidget(line_label)
        line_layout.addStretch(1)
        
        legend_layout.addWidget(line_entry)
        
        # Add treatments to legend
        for trno in sorted(all_treatments):
            j = sorted(all_treatments).index(trno)
            color_idx = self.treatment_colors.get(trno, j % len(self.colors))
            symbol_idx = j % len(self.symbols)
            
            color = self.colors[color_idx]
            symbol = self.symbols[symbol_idx]
            
            entry = QWidget()
            entry_layout = QHBoxLayout()
            entry_layout.setContentsMargins(5, 2, 5, 2)
            entry.setLayout(entry_layout)
            
            # Create a color sample
            color_sample = QFrame()
            color_sample.setFrameShape(QFrame.Shape.Box)
            color_sample.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            color_sample.setFixedWidth(15)
            color_sample.setFixedHeight(15)
            
            # Get treatment name if available
            if treatment_names and trno in treatment_names:
                trt_display = treatment_names[trno]
            else:
                trt_display = f"Treatment {trno}"
                
            treatment_label = QLabel(trt_display)
            
            entry_layout.addWidget(color_sample)
            entry_layout.addWidget(treatment_label)
            entry_layout.addStretch(1)
            
            legend_layout.addWidget(entry)
        
        # Add stretch to push items to the top
        legend_layout.addStretch(1)
        
        # Add grid and legend to main layout
        main_layout.addWidget(grid_widget, 85)  # 85% of width
        main_layout.addWidget(legend_widget, 15)  # 15% of width
        
        # Add main widget to the layout
        self.main_layout.addWidget(main_widget, 0, 0, 1, 1)
        
        # Create plots and add to grid
        for idx, (display_name, sim_var, meas_var) in enumerate(selected_pairs):
            if idx >= n_rows * n_cols:
                # Skip if too many pairs
                logger.warning(f"Skipping plot {idx+1} - grid full")
                continue
                    
            # Calculate row and column
            row = idx // n_cols
            col = idx % n_cols
                
            # Verify variables exist
            if sim_var not in self.evaluate_data.columns:
                logger.error(f"Simulated variable {sim_var} not in evaluate data")
                continue
                    
            if meas_var not in self.evaluate_data.columns:
                logger.error(f"Measured variable {meas_var} not in evaluate data")
                continue
                
            logger.info(f"Creating plot at ({row},{col}) for {display_name}: {sim_var} vs {meas_var}")
                
            # Create plot widget
            plot = pg.PlotWidget()
            plot.setBackground('w')  # White background
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Set title and labels
            plot.setTitle(display_name)
            plot.setLabel('bottom', 'Simulated')
            plot.setLabel('left', 'Measured')
            
            # Add to grid
            grid_layout.addWidget(plot, row, col)
            
            # Store in plot widgets list for reference
            self.plot_widgets.append(plot)
            
            # Get data for the pair
            try:
                # Filter out NaN values
                valid_mask = self.evaluate_data[[sim_var, meas_var, 'TRNO']].notna().all(axis=1)
                valid_data = self.evaluate_data[valid_mask].copy()
                    
                # Convert to numeric to ensure proper plotting
                valid_data[sim_var] = pd.to_numeric(valid_data[sim_var], errors='coerce')
                valid_data[meas_var] = pd.to_numeric(valid_data[meas_var], errors='coerce')
                    
                logger.info(f"Valid data points: {len(valid_data)}")
                    
                if valid_data.empty:
                    logger.warning(f"No valid data for {sim_var} vs {meas_var}")
                    continue
                    
                # Calculate range for 1:1 line
                all_vals = pd.concat([valid_data[sim_var], valid_data[meas_var]])
                min_val = all_vals.min()
                max_val = all_vals.max()
                    
                # Add padding
                range_span = max_val - min_val
                pad = range_span * 0.1 if range_span > 0 else 1.0  # Avoid division by zero
                range_min = min_val - pad
                range_max = max_val + pad
                    
                logger.info(f"1:1 line range: {range_min} to {range_max}")
                    
                # Add 1:1 line
                line_pen = pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine)
                plot.plot([range_min, range_max], [range_min, range_max], 
                        pen=line_pen)
                    
                # Calculate statistics for all data
                try:
                    sim_values = valid_data[sim_var].to_numpy()
                    meas_values = valid_data[meas_var].to_numpy()
                        
                    if len(sim_values) >= 2 and len(meas_values) >= 2:
                        # Calculate metrics
                        try:
                            # Calculate R-squared manually to avoid issues
                            sim_mean = np.mean(sim_values)
                            meas_mean = np.mean(meas_values)
                                
                            numerator = np.sum((sim_values - sim_mean) * (meas_values - meas_mean))
                            denom1 = np.sum((sim_values - sim_mean)**2)
                            denom2 = np.sum((meas_values - meas_mean)**2)
                                
                            if denom1 > 0 and denom2 > 0:
                                denominator = np.sqrt(denom1 * denom2)
                                r = numerator / denominator
                                r2 = r**2
                            else:
                                r2 = 0.0
                                    
                            # Calculate RMSE
                            rmse = np.sqrt(np.mean((meas_values - sim_values)**2))
                                
                            # Calculate d-stat
                            try:
                                d_stat = MetricsCalculator.d_stat(meas_values, sim_values)
                                if d_stat is None:
                                    d_stat = 0.0
                            except Exception as e:
                                logger.error(f"Error calculating d-stat: {e}")
                                d_stat = 0.0
                                
                            logger.info(f"Metrics: R²={r2:.3f}, RMSE={rmse:.3f}, d-stat={d_stat:.3f}")
                                
                            # Add to metrics data
                            self.metrics_data.append({
                                "Variable": display_name,
                                "n": len(meas_values),
                                "R²": round(r2, 3),
                                "RMSE": round(rmse, 3),
                                "d-stat": round(d_stat, 3),
                            })
                        except Exception as e:
                            logger.error(f"Error calculating metrics: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing values: {e}", exc_info=True)
                    
                # Add scatter plots for each treatment
                unique_treatments = sorted(valid_data['TRNO'].unique())
                logger.info(f"Found treatments: {unique_treatments}")
                    
                for j, trno in enumerate(unique_treatments):
                    # Check if treatment is in selected treatments
                    str_trno = str(trno)
                    if selected_treatments and str_trno not in selected_treatments:
                        logger.info(f"Skipping treatment {trno} - not in selected treatments")
                        continue
                            
                    trno_data = valid_data[valid_data['TRNO'] == trno]
                    logger.info(f"Treatment {trno} data: {len(trno_data)} points")
                        
                    # Get consistent color for this treatment
                    color_idx = self.treatment_colors.get(str_trno, j % len(self.colors))
                    symbol_idx = j % len(self.symbols)
                        
                    brush = pg.mkBrush(self.colors[color_idx])
                        
                    # Make sure we have points to plot
                    x_vals = trno_data[sim_var].values
                    y_vals = trno_data[meas_var].values
                        
                    if len(x_vals) == 0 or len(y_vals) == 0:
                        logger.warning(f"No points to plot for treatment {trno}")
                        continue
                        
                    # Convert symbols to PyQtGraph format
                    symbol = self.symbols[symbol_idx]
                    if symbol == 'circle':
                        pg_symbol = 'o'
                    elif symbol == 'square':
                        pg_symbol = 's'
                    elif symbol == 'diamond':
                        pg_symbol = 'd'
                    elif symbol == 'triangle-up':
                        pg_symbol = 't'
                    elif symbol == 'star':
                        pg_symbol = 'star'
                    else:
                        pg_symbol = 'o'  # Default
                        
                    try:
                        # Batch render points
                        scatter_items = self.batch_render_points(x_vals, y_vals, brush, pg_symbol)
                        if isinstance(scatter_items, list):
                            for item in scatter_items:
                                plot.addItem(item)
                        else:
                            plot.addItem(scatter_items)
                        logger.info(f"Added scatter plot for treatment {trno}")
                    except Exception as e:
                        logger.error(f"Error adding scatter plot: {e}", exc_info=True)
                    
                # Set equal aspect ratio
                plot.setAspectLocked(True)
                    
            except Exception as e:
                logger.error(f"Error plotting {display_name}: {e}", exc_info=True)
        
        # Always emit metrics at the end, even if empty
        logger.info(f"Emitting metrics for {len(self.metrics_data)} variables")
        self.metrics_calculated.emit(self.metrics_data)
    
    def plot_custom_scatter(self, selected_folder, selected_treatments, x_var, y_vars):
        """Create custom X-Y scatter plots with optimized performance"""
        # Read data just once
        if self.evaluate_data is None or self.evaluate_data.empty:
            self.evaluate_data = read_evaluate_file(selected_folder)
            if self.evaluate_data is None:
                return
                
        # Clear and create optimized plot
        self.clear_plots()
        plot = self.add_plot_widget(0, 0)
        plot.setAntialiasing(self.enable_antialiasing)
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode='peak')
        plot.useOpenGL(True)
        
        # Get variable names
        x_label, _ = get_variable_info(x_var)
        x_display = x_label or x_var
        
        plot.setTitle(f"Variables vs {x_display}")
        plot.setLabel('bottom', x_display)
        plot.setLabel('left', 'Values')
        
        # Vectorized data preparation
        base_mask = self.evaluate_data['TRNO'].astype(str).isin(selected_treatments)
        
        for y_var in y_vars:
            if y_var not in self.evaluate_data.columns:
                continue
                
            y_label, _ = get_variable_info(y_var)
            y_display = y_label or y_var
            
            # Efficient filtering
            var_mask = base_mask & self.evaluate_data[[x_var, y_var]].notna().all(axis=1)
            valid_data = self.evaluate_data[var_mask].copy()
            
            if valid_data.empty:
                continue
                
            # Calculate metrics using numpy operations
            x_values = valid_data[x_var].to_numpy()
            y_values = valid_data[y_var].to_numpy()
            
            if len(x_values) >= 2:
                x_mean = np.mean(x_values)
                y_mean = np.mean(y_values)
                
                # Vectorized calculations
                numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
                denom1 = np.sum((x_values - x_mean)**2)
                denom2 = np.sum((y_values - y_mean)**2)
                
                r2 = (numerator / np.sqrt(denom1 * denom2))**2 if denom1 > 0 and denom2 > 0 else 0.0
                rmse = np.sqrt(np.mean((y_values - x_values)**2))
                
                self.metrics_data.append({
                    "Variable": f"{y_display} vs {x_display}",
                    "n": len(x_values),
                    "R²": round(r2, 3),
                    "RMSE": round(rmse, 3),
                    "d-stat": round(MetricsCalculator.d_stat(y_values, x_values) or 0.0, 3)
                })
            
            # Plot for each treatment in batches
            for trno in valid_data['TRNO'].unique():
                if str(trno) not in selected_treatments:
                    continue
                    
                trno_data = valid_data[valid_data['TRNO'] == trno]
                
                # Get consistent plotting parameters
                color_idx = self.treatment_colors.get(str(trno), hash(str(trno)) % len(self.colors))
                symbol_idx = hash(str(trno)) % len(self.symbols)
                brush = pg.mkBrush(self.colors[color_idx])
                
                # Convert symbol
                symbol = self.symbols[symbol_idx]
                pg_symbol = {'circle': 'o', 'square': 's', 'diamond': 'd', 
                           'triangle-up': 't', 'star': 'star'}.get(symbol, 'o')
                
                x_vals = trno_data[x_var].values
                y_vals = trno_data[y_var].values
                
                if len(x_vals) > 0:
                    name = f'{y_display} - Treatment {trno}'
                    scatter_items = self.batch_render_points(x_vals, y_vals, brush, pg_symbol, name)
                    
                    if isinstance(scatter_items, list):
                        for item in scatter_items:
                            plot.addItem(item)
                    else:
                        plot.addItem(scatter_items)
        
        # Enable hardware acceleration and emit metrics
        plot.useOpenGL(True)
        if self.metrics_data:
            self.metrics_calculated.emit(self.metrics_data)
    
    def _get_cached_data(self, key):
        """Get data from cache with timestamp validation"""
        if key in self._data_cache:
            data, timestamp = self._data_cache[key]
            if (pd.Timestamp.now() - timestamp).total_seconds() < 300:  # 5 minute cache
                return data
        return None
        
    def _cache_data(self, key, data):
        """Cache data with current timestamp"""
        self._data_cache[key] = (data, pd.Timestamp.now())
        
    def process_data_efficiently(self, data, x_var, y_var):
        """Process data with vectorized operations"""
        cache_key = f"{x_var}_{y_var}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Vectorized operations
        valid_mask = data[[x_var, y_var]].notna().all(axis=1)
        valid_data = data[valid_mask].copy()
        
        if not valid_data.empty:
            # Convert to numeric using vectorized operations
            valid_data[x_var] = pd.to_numeric(valid_data[x_var], errors='coerce')
            valid_data[y_var] = pd.to_numeric(valid_data[y_var], errors='coerce')
            
            # Update cache
            self._cache_data(cache_key, valid_data)
            return valid_data
            
        return None
        
    def calculate_metrics_efficiently(self, x_values, y_values):
        """Calculate metrics using optimized numpy operations"""
        cache_key = f"{hash(str(x_values))}_{hash(str(y_values))}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        metrics = {}
        if len(x_values) >= 2:
            # Vectorized metric calculations
            x_mean = np.mean(x_values)
            y_mean = np.mean(y_values)
            
            # Compute differences once
            x_diff = x_values - x_mean
            y_diff = y_values - y_mean
            
            # R-squared calculation
            numerator = np.sum(x_diff * y_diff)
            denom1 = np.sum(x_diff ** 2)
            denom2 = np.sum(y_diff ** 2)
            
            if denom1 > 0 and denom2 > 0:
                r = numerator / np.sqrt(denom1 * denom2)
                metrics['r2'] = r ** 2
            else:
                metrics['r2'] = 0.0
                
            # RMSE calculation
            metrics['rmse'] = np.sqrt(np.mean((y_values - x_values) ** 2))
            
            # D-stat calculation
            try:
                d_stat = MetricsCalculator.d_stat(y_values, x_values)
                metrics['d_stat'] = d_stat if d_stat is not None else 0.0
            except:
                metrics['d_stat'] = 0.0
                
            self._cache_data(cache_key, metrics)
            
        return metrics