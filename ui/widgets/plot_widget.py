import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import  Qt, pyqtSignal, QCoreApplication
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import Qt
from ui.widgets.export import export_plot
# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

import config
from utils.dssat_paths import get_crop_details
from data.dssat_io import read_file, read_observed_data
from data.data_processing import (
    handle_missing_xvar, get_variable_info, improved_smart_scale,
    standardize_dtypes, unified_date_convert
)
from models.metrics import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)

# Enable OpenGL for better performance
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)

class PlotWidget(QWidget):
    """A widget for visualizing time series data with simulated and observed values."""
    
    metrics_calculated = pyqtSignal(list)  # Signal to emit calculated metrics

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the PlotWidget with default settings and UI."""
        super().__init__(parent)
        
        # Configuration settings
        self._colors = config.PLOT_COLORS
        self._marker_symbols = config.MARKER_SYMBOLS
        self._batch_size = 5000
        self._enable_antialiasing = True
        self._downsampling_enabled = True
        self._max_points_before_downsampling = 500
        
        # Data storage
        self._sim_data: Optional[pd.DataFrame] = None
        self._obs_data: Optional[pd.DataFrame] = None
        self._plot_items_metadata: List[Tuple[Any, Dict[str, Any]]] = []
        self._scale_factors: Dict[str, Tuple[float, float]] = {}
        self._scaled_vars: Dict[str, str] = {}
        
        # Placeholder for performance monitoring
        self._perf_monitor = type('PerfMonitor', (), {
            'start_timer': lambda *args, **kwargs: 0,
            'stop_timer': lambda *args, **kwargs: None
        })()
        
        # Set widget size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Initialize UI and plot settings
        self._setup_ui()
        self._configure_plot_settings()


    def _setup_ui(self) -> None:
        """Set up the user interface with plot and legend areas."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        # Left container for plot, scaling label, and export button
        left_container = QWidget()
        left_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_container.setLayout(left_layout)
        
        # Initialize plot widget
        self._plot_view = pg.PlotWidget()
        self._plot_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self._plot_view, 1)
        
        # Bottom container for scaling label and export button
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 5, 0)
        bottom_container.setLayout(bottom_layout)
        
        # Scaling label
        self._scaling_label = QLabel()
        self._scaling_label.setStyleSheet("padding: 5px; font-size: 10pt;")
        self._scaling_label.setWordWrap(True)
        self._scaling_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scaling_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bottom_layout.addWidget(self._scaling_label, 1)
        
        # Export button
        self._export_button = QPushButton("Export Plot")
        self._export_button.setStyleSheet("padding: 5px;")
        self._export_button.clicked.connect(self._export_plot)
        bottom_layout.addWidget(self._export_button)
        
        left_layout.addWidget(bottom_container)
        
        main_layout.addWidget(left_container, 80)
        
        # Legend scroll area (rest of the original code)
        legend_scroll_area = QScrollArea()
        legend_scroll_area.setWidgetResizable(True)
        legend_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        legend_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        legend_scroll_area.setFixedWidth(200)
        
        # Legend container
        self._legend_container = QWidget()
        self._legend_layout = QVBoxLayout()
        self._legend_layout.setSpacing(2)
        self._legend_layout.setContentsMargins(5, 0, 5, 0)
        self._legend_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._legend_container.setLayout(self._legend_layout)
        
        # Outer legend widget
        legend_outer_widget = QWidget()
        legend_outer_layout = QVBoxLayout()
        legend_outer_layout.setContentsMargins(0, 0, 0, 0)
        legend_outer_layout.addWidget(self._legend_container, 0, Qt.AlignmentFlag.AlignTop)
        legend_outer_layout.addStretch(1)
        legend_outer_widget.setLayout(legend_outer_layout)
        
        legend_scroll_area.setWidget(legend_outer_widget)
        main_layout.addWidget(legend_scroll_area, 20)

    def _configure_plot_settings(self) -> None:
        """Configure plot settings for appearance and performance."""
        pg.setConfigOptions(
            antialias=True,
            useOpenGL=False,
            enableExperimental=True,
            foreground='k',
            background='w'
        )
        
        self._plot_view.setDownsampling(mode='peak', auto=True)
        self._plot_view.setClipToView(True)
        self._plot_view.setAntialiasing(True)
        self._plot_view.setBackground('w')
        
        # First create a full box around the plot
        self._plot_view.showAxis('top')
        self._plot_view.showAxis('right')
        
        # Configure all four axes
        for pos in ['left', 'bottom', 'right', 'top']:
            axis = self._plot_view.getAxis(pos)
            axis.setPen(pg.mkPen(color='k', width=1.5))
            
            # For top and right axes, hide the text and ticks
            if pos in ['right', 'top']:
                axis.setStyle(showValues=False, tickLength=0)
        

        
        # Configure the grid
        self._plot_view.showGrid(x=True, y=True, alpha=0.1)
        
        # Style the axis labels
        label_style = {'color': '#000000', 'font-size': '12pt'}
        self._plot_view.getAxis('bottom').setLabel(text='Date', **label_style)
        self._plot_view.getAxis('left').setLabel(text='Value', **label_style)
    
    def on_resize(self, event: Any) -> None:
        """Handle widget resize events."""
        try:
            if self._sim_data is not None:
                self._plot_view.autoRange()
        except Exception as e:
            logger.warning(f"Error during plot resize: {str(e)}")
    
    def batch_date_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert YEAR and DOY columns to DATE in the dataframe."""
        try:
            if "YEAR" in df.columns and "DOY" in df.columns:
                df["DATE"] = df.apply(
                    lambda row: unified_date_convert(row["YEAR"], row["DOY"]),
                    axis=1
                )
                df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.warning(f"Error in batch date conversion: {e}")
            return df
    
    def _create_styled_line(self, x_values: np.ndarray, y_values: np.ndarray, 
                           color: str, var_idx: int) -> Tuple[pg.PlotDataItem, QPen]:
        """Create a styled line plot with distinct dash patterns."""
        qt_color = pg.mkColor(color)
        
        # Define dash patterns for visual distinction
        dash_patterns = [
            [],  # Solid line
            [12, 6],  # Long dashes
            [3, 5],  # Dots
            [12, 3, 3, 3],  # Dash-dot pattern
        ]
        
        pattern = dash_patterns[var_idx % len(dash_patterns)]
        pen = pg.mkPen(color=qt_color, width=2)
        
        if pattern:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(pattern)
        
        curve = pg.PlotDataItem(
            x=x_values,
            y=y_values,
            pen=pen,
            skipFiniteCheck=True,
            antialias=True,
            connect='finite'
        )
        
        return curve, pen
    
    def plot_time_series(
        self,
        selected_folder: str,
        selected_out_files: List[str],
        selected_experiment: str,
        selected_treatments: List[str],
        x_var: str,
        y_vars: List[str],
        treatment_names: Optional[Dict[str, str]] = None
    ) -> None:
        """Plot time series data for simulated and observed values."""
        try:
            # Clear existing plot and legend
            self._plot_view.clear()
            self._plot_items_metadata.clear()
            self._clear_legend()
            self.metrics_calculated.emit([])
            
            # Load and process simulation data
            sim_data = self._load_simulation_data(selected_folder, selected_out_files)
            if sim_data is None or sim_data.empty:
                logger.warning("No simulation data available")
                return
            
            # Load observed data if available
            obs_data = self._load_observed_data(
                selected_folder, selected_experiment, x_var, y_vars, sim_data
            )
            
            # Calculate scaling factors
            self._scale_factors = self._calculate_scaling_factors(sim_data, obs_data, y_vars)
            
            # Apply scaling to data
            sim_data = self._apply_scaling(sim_data, y_vars)
            if obs_data is not None and not obs_data.empty:
                obs_data = self._apply_scaling(obs_data, y_vars)
            
            self._sim_data = sim_data
            self._obs_data = obs_data
            
            # Update scaling label
            self._update_scaling_label(y_vars)
            
            # Set axis labels
            self._set_axis_labels(x_var, y_vars, sim_data)
            
            # Initialize legend
            legend_entries = {"Simulated": {}, "Observed": {}}
            self._legend_layout.addWidget(QLabel("<b>Legend</b>").setAlignment(Qt.AlignmentFlag.AlignCenter))
            
            # Plot data
            self._plot_datasets(sim_data, obs_data, x_var, y_vars, selected_treatments, 
                              treatment_names, legend_entries)
            
            # Update legend
            self._update_legend(legend_entries)
            
            # Configure date axis if needed
            if x_var == "DATE":
                self._configure_date_axis(sim_data, obs_data)
            else:
                # For numeric variables like DAS, DAP, etc.
                # Get all x values
                all_x_values = []
                if x_var in sim_data.columns:
                    sim_x_values = pd.to_numeric(sim_data[x_var].dropna(), errors='coerce')
                    all_x_values.extend(sim_x_values)
                
                if obs_data is not None and x_var in obs_data.columns:
                    obs_x_values = pd.to_numeric(obs_data[x_var].dropna(), errors='coerce')
                    all_x_values.extend(obs_x_values)
                
                if all_x_values:
                    # Get min and max values
                    min_x = min(all_x_values)
                    max_x = max(all_x_values)
                    
                    # Set appropriate tick spacing based on range
                    x_range = max_x - min_x
                    if x_range <= 20:
                        step = 1
                    elif x_range <= 50:
                        step = 5
                    elif x_range <= 100:
                        step = 10
                    else:
                        step = max(int(x_range / 10), 1)
                    
                    # Create ticks
                    major_ticks = []
                    for i in range(int(min_x), int(max_x) + 1, step):
                        major_ticks.append((i, str(i)))
                        
                    # Set ticks on x-axis
                    self._plot_view.getAxis('bottom').setTicks([major_ticks, []])
            
            # Set y-axis range
            self._set_y_axis_range()
            
            # Calculate metrics if observed data is available
            if obs_data is not None and not obs_data.empty:
                self.calculate_metrics(sim_data, obs_data, y_vars, selected_treatments, treatment_names)
                
        except Exception as e:
            logger.error(f"Error in plot_time_series: {str(e)}")
            raise
    
    def _load_simulation_data(self, selected_folder: str, selected_out_files: List[str]) -> Optional[pd.DataFrame]:
        """Load and process simulation data from specified files."""
        all_data = []
        crop_details = get_crop_details()
        crop_info = next(
            (crop for crop in crop_details if crop['name'].upper() == selected_folder.upper()),
            None
        )
        
        if not crop_info:
            logger.error(f"Could not find crop info for: {selected_folder}")
            return None
        
        folder_path = crop_info['directory'].strip()
        
        for out_file in selected_out_files:
            file_path = os.path.join(folder_path, out_file)
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                continue
                
            data = read_file(file_path)
            if data is None or data.empty:
                logger.warning(f"No data loaded from {file_path}")
                continue
                
            # Standardize column names and types
            data.columns = data.columns.str.strip().str.upper()
            data = self._standardize_treatment_column(data)
            data = self._process_date_column(data)
            data["source"] = "sim"
            data["FILE"] = out_file
            all_data.append(data)
        
        return pd.concat(all_data, ignore_index=True) if all_data else None
    
    def _standardize_treatment_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize the treatment column in the dataframe."""
        if "TRNO" in data.columns and "TRT" not in data.columns:
            data["TRT"] = data["TRNO"]
        elif "TRT" not in data.columns:
            data["TRT"] = "1"
        data["TRT"] = data["TRT"].astype(str)
        return data
    
    def _process_date_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process YEAR and DOY columns to create DATE column."""
        for col in ["YEAR", "DOY"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
                
        if "YEAR" in data.columns and "DOY" in data.columns:
            data["DATE"] = data.apply(
                lambda row: unified_date_convert(row["YEAR"], row["DOY"]),
                axis=1
            )
            data["DATE"] = data["DATE"].dt.strftime("%Y-%m-%d")
        return data
    
    def _load_observed_data(
        self, selected_folder: str, selected_experiment: str, x_var: str, 
        y_vars: List[str], sim_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Load and process observed data if available."""
        if not selected_experiment:
            return None
            
        obs_data = read_observed_data(selected_folder, selected_experiment, x_var, y_vars)
        if obs_data is None or obs_data.empty:
            return None
            
        obs_data["source"] = "obs"
        obs_data = handle_missing_xvar(obs_data, x_var, sim_data)
        
        if obs_data is not None:
            if "TRNO" in obs_data.columns:
                obs_data["TRT"] = obs_data["TRNO"].astype(str)
                obs_data = obs_data.drop(columns=["TRNO"])
                
            for var in y_vars:
                if var in obs_data.columns:
                    obs_data[var] = pd.to_numeric(obs_data[var], errors="coerce")
                    for missing_val in config.MISSING_VALUES:
                        obs_data.loc[obs_data[var] == missing_val, var] = np.nan
        
        return obs_data
    
    def _calculate_scaling_factors(
        self, sim_data: pd.DataFrame, obs_data: Optional[pd.DataFrame], y_vars: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate scaling factors for multiple y-variables."""
        if len(y_vars) <= 1:
            return {var: (1.0, 0) for var in y_vars if var in sim_data.columns}
            
        magnitudes = {}
        max_values = {}
        
        # Collect statistics for simulation data
        for var in y_vars:
            if var in sim_data.columns:
                sim_values = pd.to_numeric(sim_data[var], errors="coerce").dropna().values
                if len(sim_values) > 0 and not np.isclose(np.min(sim_values), np.max(sim_values)):
                    max_val = np.max(sim_values)
                    abs_values = np.abs(sim_values[sim_values != 0])
                    if len(abs_values) > 0:
                        magnitudes[var] = np.floor(np.log10(np.mean(abs_values)))
                        max_values[var] = max_val
        
        # Find target maximum from all data
        all_maxes = []
        for var in y_vars:
            if var in sim_data.columns:
                sim_values = pd.to_numeric(sim_data[var], errors="coerce").dropna()
                if len(sim_values) > 0:
                    all_maxes.append(sim_values.max())
            if obs_data is not None and var in obs_data.columns:
                obs_values = pd.to_numeric(obs_data[var], errors="coerce").dropna()
                if len(obs_values) > 0:
                    all_maxes.append(obs_values.max())
        
        target_max = max(all_maxes) if all_maxes else float('inf')
        target_threshold = target_max * 1.1
        
        scaling_factors = {}
        if len(magnitudes) >= 2:
            reference_magnitude = max(magnitudes.values())
            for var, magnitude in magnitudes.items():
                scale_factor = 10 ** (reference_magnitude - magnitude)
                if var in max_values:
                    scaled_max = max_values[var] * scale_factor
                    while scaled_max > target_threshold and scale_factor > 0.2:
                        scale_factor /= 10.0
                        scaled_max = max_values[var] * scale_factor
                scaling_factors[var] = (scale_factor, 0)
        
        # Add default scaling for remaining variables
        for var in y_vars:
            if var in sim_data.columns and var not in scaling_factors:
                scaling_factors[var] = (1.0, 0)
                
        return scaling_factors
    
    def _apply_scaling(self, data: pd.DataFrame, y_vars: List[str]) -> pd.DataFrame:
        """Apply scaling factors to the data."""
        scaled_data = data.copy()
        for var in y_vars:
            if var in scaled_data.columns and var in self._scale_factors:
                scale_factor, offset = self._scale_factors[var]
                scaled_data[f"{var}_original"] = scaled_data[var].copy()
                scaled_data[var] = scaled_data[var] * scale_factor + offset
        return scaled_data
    
    def _update_scaling_label(self, y_vars: List[str]) -> None:
        """Update the scaling label with scaling factors, but only for variables scaled by more than 1."""
        scaling_parts = [
            f"{get_variable_info(var)[0] or var} = {scale_factor:.2f} * {get_variable_info(var)[0] or var}"
            for var, (scale_factor, _) in self._scale_factors.items()
            if scale_factor > 1.0  # Only include variables scaled by more than 1
        ]
        
        if scaling_parts:
            self._scaling_label.setText("\n".join(scaling_parts))
        else:
            self._scaling_label.setText("")  # Clear the label if no variables are scaled by more than 1
        
    def _set_axis_labels(self, x_var: str, y_vars: List[str], sim_data: pd.DataFrame) -> None:
        """Set axis labels for the plot."""
        x_label, _ = get_variable_info(x_var)
        self._plot_view.setLabel('bottom', text=x_label or x_var, color='k')
        
        y_axis_label = ", ".join(
            get_variable_info(var)[0] or var
            for var in y_vars
            if var in sim_data.columns
        )
        self._plot_view.setLabel('left', text=y_axis_label, color='#000000', size='12pt')
        
        self._plot_view.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        self._plot_view.getAxis('left').setPen(pg.mkPen(color='k', width=1))
        self._plot_view.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
        self._plot_view.getAxis('left').setTextPen(pg.mkPen(color='k'))
    
    def _clear_legend(self) -> None:
        """Clear all items from the legend layout."""
        for i in reversed(range(self._legend_layout.count())):
            item = self._legend_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
    
    def _plot_datasets(
        self, sim_data: pd.DataFrame, obs_data: Optional[pd.DataFrame], x_var: str,
        y_vars: List[str], selected_treatments: List[str], treatment_names: Optional[Dict[str, str]],
        legend_entries: Dict[str, Dict[str, List]]
    ) -> None:
        """Plot simulated and observed datasets."""
        line_styles = [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, 
                      Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine]
        var_style_map = {var: line_styles[i % len(line_styles)] for i, var in enumerate(y_vars)}
        
        for dataset in [sim_data, obs_data]:
            if dataset is None or dataset.empty:
                continue
                
            source_type = dataset["source"].iloc[0]
            category = "Simulated" if source_type == "sim" else "Observed"
            
            for var_idx, var in enumerate(y_vars):
                var_label, _ = get_variable_info(var)
                display_name = var_label or var
                if display_name not in legend_entries[category]:
                    legend_entries[category][display_name] = []
                    
                for trt_idx, (trt_value, group) in enumerate(dataset.groupby("TRT")):
                    if trt_value not in selected_treatments or var not in group.columns or not group[var].notna().any():
                        continue
                        
                    trt_display = treatment_names.get(trt_value, f"Treatment {trt_value}") if treatment_names else f"Treatment {trt_value}"
                    color = self._colors[trt_idx % len(self._colors)]
                    
                    if source_type == "sim":
                        self._plot_simulated_data(
                            group, x_var, var, trt_value, trt_display, color, var_idx,
                            legend_entries[category][display_name]
                        )
                    else:
                        self._plot_observed_data(
                            group, x_var, var, trt_value, trt_display, color, var_idx, trt_idx,
                            selected_treatments, legend_entries[category][display_name]
                        )
    
    def _plot_simulated_data(
        self, group: pd.DataFrame, x_var: str, var: str, trt_value: str, 
        trt_display: str, color: str, var_idx: int, legend_entries: List
    ) -> None:
        """Plot simulated data as lines."""
        valid_mask = group[var].notna()
        x_values = group[valid_mask][x_var].values
        y_values = group[valid_mask][var].values
        
        if x_var == "DATE":
            try:
                x_dates = pd.to_datetime(x_values, errors='coerce')
                valid_date_mask = ~x_dates.isna()
                x_values = x_values[valid_date_mask]
                y_values = y_values[valid_date_mask]
                x_dates = x_dates[valid_date_mask]
                if len(x_dates) == 0:
                    return
                x_values = [d.timestamp() for d in x_dates]
            except Exception as e:
                logger.warning(f"Error converting dates: {e}")
                return
        
        if len(x_values) < 2 or len(y_values) < 2:
            return
            
        curve, pen = self._create_styled_line(x_values, y_values, color, var_idx)
        self._plot_view.addItem(curve)
        
        self._plot_items_metadata.append((curve, {
            'variable': var,
            'treatment': trt_value,
            'treatment_name': trt_display,
            'source': 'sim',
            'x_var': x_var
        }))
        
        legend_entries.append({
            "item": curve,
            "name": trt_display,
            "trt": trt_value,
            "pen": pen,
            "symbol": None
        })
    
    def _plot_observed_data(
        self, group: pd.DataFrame, x_var: str, var: str, trt_value: str, 
        trt_display: str, color: str, var_idx: int, trt_idx: int,
        selected_treatments: List[str], legend_entries: List
    ) -> None:
        """Plot observed data as scatter points."""
        valid_mask = group[var].notna()
        x_values = group[valid_mask][x_var].values
        y_values = group[valid_mask][var].values
        
        if x_var == "DATE":
            try:
                x_dates = pd.to_datetime(x_values, errors='coerce')
                valid_date_mask = ~x_dates.isna()
                x_values = x_values[valid_date_mask]
                y_values = y_values[valid_date_mask]
                x_dates = x_dates[valid_date_mask]
                if len(x_dates) == 0:
                    return
                x_values = [d.timestamp() for d in x_dates]
            except Exception as e:
                logger.warning(f"Error converting dates: {e}")
                return
        
        if len(x_values) < 1 or len(y_values) < 1:
            return
            
        symbol_idx = (trt_idx + var_idx * len(selected_treatments)) % len(self._marker_symbols)
        symbol = self._marker_symbols[symbol_idx]
        qt_color = pg.mkColor(color)
        symbol_pen = pg.mkPen(qt_color, width=2) if (var_idx + trt_idx) % 2 == 0 else None
        
        scatter = pg.ScatterPlotItem(
            x=x_values,
            y=y_values,
            symbol=symbol,
            size=7,
            pen=symbol_pen,
            brush=qt_color,
            name=None
        )
        
        self._plot_view.addItem(scatter)
        
        self._plot_items_metadata.append((scatter, {
            'variable': var,
            'treatment': trt_value,
            'treatment_name': trt_display,
            'source': 'obs',
            'x_var': x_var
        }))
        
        legend_entries.append({
            "item": scatter,
            "name": trt_display,
            "trt": trt_value,
            "brush": qt_color,
            "pen": symbol_pen,
            "symbol": symbol
        })
    
    def _update_legend(self, legend_entries: Dict[str, Dict[str, List]]) -> None:
        """Update the legend using a three-column layout (Observed | Simulated | Treatment)."""
        self._clear_legend()
        
        # Add legend title
        legend_title = QLabel("<b></b>")
        legend_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._legend_layout.addWidget(legend_title)
        
        # Create header row with three columns
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 2, 0, 2)
        header_layout.setSpacing(5)  # Reduced spacing between columns
        header_widget.setLayout(header_layout)
        
        # Create fixed-width column headers with precise alignment
        obs_header = QLabel("<b>Obs.</b>")
        obs_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        obs_header.setFixedWidth(30)  # Fixed width for consistency
        
        sim_header = QLabel("<b>Sim.</b>")
        sim_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sim_header.setFixedWidth(30)  # Fixed width for consistency
        
        # FIX: Left-align the Treatment header
        trt_header = QLabel("<b>Treatment</b>")
        trt_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Create a layout with fixed spacing
        header_layout.addWidget(obs_header)
        header_layout.addWidget(sim_header)
        header_layout.addWidget(trt_header, 1)  # Treatment takes remaining space, no stretch before it
        
        # Add header to layout
        self._legend_layout.addWidget(header_widget)
        
        # Horizontal separator
        try:
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Plain)  # Use Plain instead of Sunken
            self._legend_layout.addWidget(separator)
        except Exception as e:
            logger.warning(f"Error creating separator: {e}")
            # Create a simple spacer as fallback
            spacer = QWidget()
            spacer.setFixedHeight(1)
            spacer.setStyleSheet("background-color: #CCCCCC;")
            self._legend_layout.addWidget(spacer)
        
        # Organize data by variable first
        variables = set()
        for category in ["Simulated", "Observed"]:
            for var_name in legend_entries[category].keys():
                variables.add(var_name)
        
        # Create a mapping of treatments by variable
        var_treatments = {}
        for var_name in variables:
            var_treatments[var_name] = {}
            
            # Process simulated data
            if var_name in legend_entries["Simulated"]:
                for treatment in legend_entries["Simulated"][var_name]:
                    trt_id = treatment["trt"]
                    if trt_id not in var_treatments[var_name]:
                        var_treatments[var_name][trt_id] = {
                            "name": treatment["name"],
                            "sim": treatment,
                            "obs": None
                        }
                    else:
                        var_treatments[var_name][trt_id]["sim"] = treatment
                        
            # Process observed data
            if var_name in legend_entries["Observed"]:
                for treatment in legend_entries["Observed"][var_name]:
                    trt_id = treatment["trt"]
                    if trt_id not in var_treatments[var_name]:
                        var_treatments[var_name][trt_id] = {
                            "name": treatment["name"],
                            "sim": None,
                            "obs": treatment
                        }
                    else:
                        var_treatments[var_name][trt_id]["obs"] = treatment
        
        # Now create the legend entries by variable and treatment
        for var_name in sorted(variables):
            # FIX: Left-align the variable name
            var_widget = QWidget()
            var_layout = QHBoxLayout()
            var_layout.setContentsMargins(0, 5, 0, 2)  # Add padding at top
            var_layout.setSpacing(5)
            var_widget.setLayout(var_layout)
            
            # Create fixed-width spacers for obs and sim columns to maintain alignment
            var_layout.addSpacing(30)  # Space for Obs column
            var_layout.addSpacing(30)  # Space for Sim column
            
            # Left-align the variable name
            var_label, _ = get_variable_info(var_name)
            display_name = var_label or var_name
            var_label_widget = QLabel(f"<b>{display_name}</b>")
            var_label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft)
            var_layout.addWidget(var_label_widget, 1)  # Variable takes rest of space, left-aligned
            
            self._legend_layout.addWidget(var_widget)
            
            # Create a row for each treatment under this variable
            for trt_id, treatment_data in sorted(var_treatments[var_name].items()):
                # Row widget with fixed layout
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(5)  # Match header spacing exactly
                row_widget.setLayout(row_layout)
                
                # Make the entire row clickable for toggling
                row_widget.setCursor(Qt.CursorShape.PointingHandCursor)
                
                # Observed column with fixed width
                obs_widget = QWidget()
                obs_widget.setFixedWidth(30)  # Match the header width
                obs_layout = QHBoxLayout()
                obs_layout.setContentsMargins(0, 0, 0, 0)
                obs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                obs_widget.setLayout(obs_layout)
                
                if treatment_data["obs"] is not None:
                    try:
                        # Add tooltip for observed data
                        obs_tooltip = f"Observed\nVariable: {display_name}\nTreatment: {treatment_data['name']}"
                        obs_sample = self._create_sample_widget(
                            has_symbol=True,
                            symbol=treatment_data["obs"]["symbol"],
                            pen=treatment_data["obs"].get("pen"),
                            brush=treatment_data["obs"].get("brush"),
                            tooltip=obs_tooltip
                        )
                        obs_layout.addWidget(obs_sample)
                    except Exception as e:
                        logger.warning(f"Error creating obs sample: {e}")
                        placeholder = QLabel("O")
                        placeholder.setStyleSheet("color: #000000;")
                        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        obs_layout.addWidget(placeholder)
                else:
                    # Empty placeholder for alignment
                    placeholder = QLabel("-")
                    placeholder.setStyleSheet("color: #CCCCCC;")
                    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    obs_layout.addWidget(placeholder)
                
                # Simulated column with fixed width
                sim_widget = QWidget()
                sim_widget.setFixedWidth(30)  # Match the header width
                sim_layout = QHBoxLayout()
                sim_layout.setContentsMargins(0, 0, 0, 0)
                sim_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sim_widget.setLayout(sim_layout)
                
                if treatment_data["sim"] is not None:
                    try:
                        # Add tooltip for simulated data
                        sim_tooltip = f"Simulated\nVariable: {display_name}\nTreatment: {treatment_data['name']}"
                        sim_sample = self._create_sample_widget(
                            has_symbol=False,
                            pen=treatment_data["sim"]["pen"],
                            tooltip=sim_tooltip
                        )
                        sim_layout.addWidget(sim_sample)
                    except Exception as e:
                        logger.warning(f"Error creating sim sample: {e}")
                        placeholder = QLabel("—")
                        placeholder.setStyleSheet("color: #000000;")
                        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        sim_layout.addWidget(placeholder)
                else:
                    # Empty placeholder for alignment
                    placeholder = QLabel("-")
                    placeholder.setStyleSheet("color: #CCCCCC;")
                    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sim_layout.addWidget(placeholder)
                
                # Treatment name column - LEFT ALIGNED
                trt_label = QLabel(treatment_data["name"])
                trt_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                
                # Add widgets to match header exactly
                row_layout.addWidget(obs_widget)
                row_layout.addWidget(sim_widget)
                # FIX: No stretch before treatment name to keep it left-aligned
                row_layout.addWidget(trt_label, 1)  # Takes remaining space, left-aligned
                
                # Store the plot items for this row
                toggle_items = []
                if treatment_data["obs"] is not None and "item" in treatment_data["obs"]:
                    toggle_items.append(treatment_data["obs"]["item"])
                if treatment_data["sim"] is not None and "item" in treatment_data["sim"]:
                    toggle_items.append(treatment_data["sim"]["item"])
                
                # Create a click handler for this specific row
                row_widget.mousePressEvent = self._create_toggle_handler(
                    row_widget, toggle_items, var_name, trt_id
                )
                
                # Add the row to the legend
                self._legend_layout.addWidget(row_widget)
            
            # Add a separator after each variable except the last one
            if var_name != sorted(variables)[-1]:
                try:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.Shape.HLine)
                    separator.setFrameShadow(QFrame.Shadow.Plain)  # Use Plain instead of Sunken
                    separator.setStyleSheet("color: #EEEEEE;")
                    self._legend_layout.addWidget(separator)
                except Exception as e:
                    logger.warning(f"Error creating variable separator: {e}")
                    # Create a simple spacer as fallback
                    spacer = QWidget()
                    spacer.setFixedHeight(1)
                    spacer.setStyleSheet("color: #EEEEEE;")
                    self._legend_layout.addWidget(spacer)

    def _create_toggle_handler(self, row_widget, plot_items, var_name, trt_id):
        """Create a click handler function for a legend row that highlights the selected item."""
        # Track highlight state for this row
        row_widget.highlighted = False
        
        # Make sure original style is preserved
        original_style = row_widget.styleSheet()
        
        def toggle_highlight(event):
            # Toggle highlight state for this row
            current_state = row_widget.highlighted
            
            # Always reset all items first to ensure clean state
            self._reset_all_highlighted_items()
            
            # If it was already highlighted, we're done (toggle off)
            if current_state:
                row_widget.highlighted = False
                return
                
            # Otherwise highlight this row (toggle on)
            row_widget.highlighted = True
            row_widget.setStyleSheet(original_style + "background-color: #e6f2ff; border: 1px solid #99ccff;")
            
            # Find all items that need to be dimmed (not in this row)
            for item_info, metadata in self._plot_items_metadata:
                if item_info in plot_items:
                    # Highlight the selected items by making them brighter and thicker
                    if isinstance(item_info, pg.PlotDataItem) or isinstance(item_info, pg.PlotCurveItem):
                        # For lines: make them thicker
                        current_pen = item_info.opts['pen']
                        if not hasattr(item_info, '_original_pen'):
                            item_info._original_pen = current_pen
                        
                        # Create a thicker pen with the same color and style
                        highlight_pen = pg.mkPen(
                            color=current_pen.color(),
                            width=current_pen.width() * 2,  # Double the width
                            style=current_pen.style()
                        )
                        item_info.setPen(highlight_pen)
                        
                        # Bring to front
                        item_info.setZValue(100)
                    
                    elif isinstance(item_info, pg.ScatterPlotItem):
                        # For scatter points: make them larger
                        if not hasattr(item_info, '_original_size'):
                            item_info._original_size = item_info.opts['size']
                        
                        # Also store original brush if not already stored
                        if not hasattr(item_info, '_original_brush'):
                            item_info._original_brush = item_info.opts.get('brush')
                        
                        item_info.setSize(item_info._original_size * 1.5)  # 50% larger
                        item_info.setZValue(100)  # Bring to front
                else:
                    # Dim all other items
                    if isinstance(item_info, pg.PlotDataItem) or isinstance(item_info, pg.PlotCurveItem):
                        if not hasattr(item_info, '_original_pen'):
                            item_info._original_pen = item_info.opts['pen']
                        
                        # Create a dimmed version of the pen
                        dim_color = item_info._original_pen.color()
                        dim_color.setAlpha(50)  # Reduce opacity to 50%
                        
                        dim_pen = pg.mkPen(
                            color=dim_color,
                            width=item_info._original_pen.width(),
                            style=item_info._original_pen.style()
                        )
                        item_info.setPen(dim_pen)
                        
                        # Send to back
                        item_info.setZValue(0)
                    
                    elif isinstance(item_info, pg.ScatterPlotItem):
                        if not hasattr(item_info, '_original_size'):
                            item_info._original_size = item_info.opts['size']
                            
                        # Store original brush if not already stored
                        if not hasattr(item_info, '_original_brush'):
                            item_info._original_brush = item_info.opts.get('brush')
                            
                        # Create dimmed brush - properly handle QBrush objects
                        if item_info._original_brush:
                            # Extract the color from the brush
                            brush_color = item_info._original_brush.color()
                            # Create a new color with reduced alpha
                            dim_color = pg.QtGui.QColor(brush_color)
                            dim_color.setAlpha(50)
                            # Create a new brush with the dimmed color
                            dim_brush = pg.mkBrush(dim_color)
                            
                            # Apply dimmed brush
                            item_info.setBrush(dim_brush)
                        
                        item_info.setZValue(0)  # Send to back
        
        return toggle_highlight
    
    def _reset_all_highlighted_items(self):
        """Reset all plot items to their original appearance."""
        # Reset all legend rows
        for i in range(self._legend_layout.count()):
            item = self._legend_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'highlighted'):
                    widget.highlighted = False
                    widget.setStyleSheet("")  # Reset style
        
        # Reset all plot items
        for item_info, metadata in self._plot_items_metadata:
            if isinstance(item_info, pg.PlotDataItem) or isinstance(item_info, pg.PlotCurveItem):
                if hasattr(item_info, '_original_pen'):
                    item_info.setPen(item_info._original_pen)
            
            elif isinstance(item_info, pg.ScatterPlotItem):
                if hasattr(item_info, '_original_size'):
                    item_info.setSize(item_info._original_size)
                if hasattr(item_info, '_original_brush'):
                    item_info.setBrush(item_info._original_brush)
            
            # Reset z-value to default
            item_info.setZValue(1)


    def _create_sample_widget(self, has_symbol=False, pen=None, symbol=None, brush=None, tooltip=None):
        """Create a small widget that shows a line or point sample."""
        sample_widget = pg.PlotWidget(background=None)
        sample_widget.setFixedSize(20, 15)
        sample_widget.hideAxis('left')
        sample_widget.hideAxis('bottom')
        sample_widget.setMouseEnabled(False, False)
        
        if has_symbol and symbol is not None:
            # Create a scatter plot for observed data
            sample = pg.ScatterPlotItem(
                x=[0.5], y=[0.5],
                symbol=symbol,
                size=7,
                pen=pen,
                brush=brush
            )
        else:
            # Create a line for simulated data
            if pen is not None:
                sample_pen = pg.mkPen(
                    color=pen.color(),
                    width=pen.width(),
                    style=pen.style()
                )
                sample = pg.PlotDataItem(
                    x=[0, 1],
                    y=[0.5, 0.5],
                    pen=sample_pen,
                    skipFiniteCheck=True,
                    antialias=True
                )
            else:
                # Create an empty sample if no pen is provided
                sample = pg.PlotDataItem(x=[0, 0], y=[0, 0])
        
        sample_widget.addItem(sample)
        
        # Set tooltip if provided
        if tooltip:
            sample_widget.setToolTip(tooltip)
        
        return sample_widget


    def _configure_date_axis(self, sim_data: pd.DataFrame, obs_data: Optional[pd.DataFrame]) -> None:
        """Configure the date axis for the plot."""
        date_axis = pg.DateAxisItem(orientation='bottom')
        date_axis.setLabel(text="Date", color='k')
        
        all_dates = []
        if 'DATE' in sim_data.columns:
            sim_dates = pd.to_datetime(sim_data['DATE'].dropna(), errors='coerce')
            all_dates.extend(sim_dates)
        if obs_data is not None and 'DATE' in obs_data.columns:
            obs_dates = pd.to_datetime(obs_data['DATE'].dropna(), errors='coerce')
            all_dates.extend(obs_dates)
        
        all_dates = sorted(set(all_dates))
        if not all_dates:
            return
            
        min_date = min(all_dates)
        max_date = max(all_dates)
        time_span_days = (max_date - min_date).days
        
        if time_span_days <= 90:
            freq = 'D'
            date_format = '%Y-%m-%d'
            max_labels = 12
        elif time_span_days <= 365:
            freq = 'MS'
            date_format = '%Y-%m-%d'
            max_labels = 12
        else:
            freq = 'MS'
            date_format = '%Y-%m'
            max_labels = 12
        
        major_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
        if len(major_dates) > max_labels:
            step = len(major_dates) // max_labels
            major_dates = major_dates[::step]
        
        major_ticks = [(d.timestamp(), d.strftime(date_format)) for d in major_dates]
        minor_ticks = []
        for i in range(len(major_dates) - 1):
            mid_date = major_dates[i] + (major_dates[i + 1] - major_dates[i]) / 2
            if min_date <= mid_date <= max_date:
                minor_ticks.append((mid_date.timestamp(), ''))
        
        date_axis.setTicks([major_ticks, minor_ticks])
        self._plot_view.setAxisItems({'bottom': date_axis})
    
    def _set_y_axis_range(self) -> None:
        """Set the y-axis range based on data."""
        y_max = float('-inf')
        valid_data_found = False
        
        for item, metadata in self._plot_items_metadata:
            try:
                y_values = None
                if isinstance(item, (pg.PlotCurveItem, pg.PlotDataItem)):
                    if hasattr(item, 'yData') and item.yData is not None:
                        y_values = np.array(item.yData, dtype=np.float64)
                elif isinstance(item, pg.ScatterPlotItem):
                    if hasattr(item, 'data') and 'y' in item.data and item.data['y'] is not None:
                        y_values = np.array(item.data['y'], dtype=np.float64)
                        
                if y_values is not None and len(y_values) > 0:
                    valid_y_values = y_values[~np.isnan(y_values)]
                    if len(valid_y_values) > 0:
                        y_max = max(y_max, np.max(valid_y_values))
                        valid_data_found = True
            except Exception as e:
                logger.warning(f"Error calculating max for item {metadata.get('variable', 'unknown')}: {str(e)}")
                continue
        
        if not valid_data_found:
            logger.warning("No valid y_max from plot items, checking raw data")
            for dataset in [self._sim_data, self._obs_data]:
                if dataset is not None and not dataset.empty:
                    for var in dataset.columns:
                        valid_y_values = pd.to_numeric(dataset[var], errors='coerce').dropna()
                        if len(valid_y_values) > 0:
                            y_max = max(y_max, valid_y_values.max())
                            valid_data_found = True
        
        if not valid_data_found:
            y_max = 10.0
            logger.warning(f"No valid data found, setting default y_max to {y_max}")
        
        y_max_with_padding = y_max + y_max * 0.1
        self._plot_view.getViewBox().setLimits(yMin=0)
        self._plot_view.enableAutoRange()
    
    def update_plot_for_resize(self) -> None:
        """Update the plot geometry on widget resize."""
        if hasattr(self, '_plot_view'):
            self._plot_view.updateGeometry()
            if self._sim_data is not None:
                self._plot_view.autoRange()
    
    def calculate_metrics(
        self, sim_data: pd.DataFrame, obs_data: pd.DataFrame, y_vars: List[str],
        selected_treatments: List[str], treatment_names: Optional[Dict[str, str]] = None
    ) -> None:
        """Calculate metrics comparing simulated and observed data."""
        if obs_data is None or obs_data.empty or not y_vars or not selected_treatments:
            logger.warning("No valid data for metrics calculation")
            self.metrics_calculated.emit([])
            return
            
        metrics_data = []
        has_valid_data = False
        
        for var in y_vars:
            if var not in sim_data.columns or var not in obs_data.columns:
                logger.warning(f"Variable {var} not found in both datasets")
                continue
                
            for trt in selected_treatments:
                try:
                    sim_trt_data = sim_data[sim_data['TRT'] == trt]
                    obs_trt_data = obs_data[obs_data['TRT'] == trt]
                    
                    if sim_trt_data.empty or obs_trt_data.empty:
                        logger.info(f"No data for treatment {trt}, variable {var}")
                        continue
                        
                    common_dates = set(sim_trt_data['DATE']) & set(obs_trt_data['DATE'])
                    if not common_dates:
                        logger.warning(f"No common dates for treatment {trt}, variable {var}")
                        continue
                        
                    sim_values = []
                    obs_values = []
                    for date in common_dates:
                        try:
                            sim_val = sim_trt_data[sim_trt_data['DATE'] == date][var].values
                            obs_val = obs_trt_data[obs_trt_data['DATE'] == date][var].values
                            if len(sim_val) > 0 and len(obs_val) > 0 and not pd.isna(sim_val[0]) and not pd.isna(obs_val[0]):
                                sim_values.append(float(sim_val[0]))
                                obs_values.append(float(obs_val[0]))
                        except Exception as e:
                            logger.warning(f"Error processing date {date}: {e}")
                            continue
                    
                    trt_name = treatment_names.get(trt, trt) if treatment_names else trt
                    var_label, _ = get_variable_info(var)
                    display_name = var_label or var
                    
                    if len(sim_values) < 2 or len(obs_values) < 2:
                        logger.warning(f"Insufficient data for treatment {trt}, variable {var}")
                        metrics_data.append({
                            "Variable": f"{display_name} - {trt_name}",
                            "n": len(sim_values),
                            "RMSE": 0.0,
                            "d-stat": 0.0,
                        })
                        continue
                        
                    try:
                        sim_vals = np.array(sim_values, dtype=float)
                        obs_vals = np.array(obs_values, dtype=float)
                        rmse = MetricsCalculator.rmse(obs_vals, sim_vals)
                        d_stat_val = MetricsCalculator.d_stat(obs_vals, sim_vals)
                        
                        metrics_data.append({
                            "Variable": f"{display_name} - {trt_name}",
                            "n": len(sim_values),
                            "RMSE": round(rmse, 3),
                            "d-stat": round(d_stat_val, 3),
                        })
                        has_valid_data = True
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {e}")
                        metrics_data.append({
                            "Variable": f"{display_name} - {trt_name}",
                            "n": len(sim_values),
                            "RMSE": 0.0,
                            "d-stat": 0.0,
                        })
                except Exception as e:
                    logger.error(f"Error processing treatment {trt} for variable {var}: {e}")
                    continue
        
        self.metrics_calculated.emit(metrics_data if has_valid_data else [])
    
    def calculate_scale_factors(self, data: pd.DataFrame, y_vars: List[str]) -> Dict[str, float]:
        """Calculate scale factors for variables based on their magnitudes."""
        scale_factors = {}
        
        for var in y_vars:
            if var not in data.columns:
                scale_factors[var] = 1.0
                continue
                
            valid_values = data[var].dropna().values
            if len(valid_values) == 0 or np.max(np.abs(valid_values)) == 0:
                scale_factors[var] = 1.0
                continue
                
            magnitude = np.floor(np.log10(np.max(np.abs(valid_values))))
            scale_factors[var] = 10 ** (-magnitude + 1) if magnitude > 2 or magnitude < 0 else 1.0
        
        return scale_factors
    
    def apply_scaling(self, data: pd.DataFrame, y_vars: List[str]) -> pd.DataFrame:
        """Apply scaling to data and store scaled variable names."""
        scaled_data = data.copy()
        self._scale_factors = self.calculate_scale_factors(data, y_vars)
        self._scaled_vars = {}
        
        for var in y_vars:
            if var in self._scale_factors and self._scale_factors[var] != 1.0:
                scaled_name = f"{var}_scaled"
                scaled_data[scaled_name] = data[var] * self._scale_factors[var]
                self._scaled_vars[var] = scaled_name
            else:
                self._scaled_vars[var] = var
                
        return scaled_data
    
    def _export_plot(self) -> None:
        """Export the plot with title, legend (with visible symbols), and scaling labels."""
        # Pass the necessary components to the export function
        export_plot(
            plot_widget=self._plot_view,
            plot_items_metadata=self._plot_items_metadata,
            scaling_label=self._scaling_label,
            get_variable_info_func=get_variable_info
        )
    