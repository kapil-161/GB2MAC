"""
Data Table Widget for DSSAT Viewer
Provides tabular view of DSSAT output data
"""
import os
import sys
import logging
from typing import Any
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QHeaderView,
    QPushButton, QHBoxLayout, QFileDialog, QGroupBox, QLabel, QComboBox,
    QTabWidget, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QAbstractTableModel, pyqtSlot
from utils.performance_monitor import PerformanceMonitor, function_timer

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

# Configure logging
logger = logging.getLogger(__name__)

class PandasTableModel(QAbstractTableModel):
    """Table model for pandas DataFrame to display in QTableView"""
    
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.index)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            # Format based on data type
            if pd.isna(value):
                return "NA"
            elif isinstance(value, (float, np.float64)):
                return f"{value:.4f}"
            elif isinstance(value, (int, np.int64)):
                return str(value)
            else:
                return str(value)
                
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        return None
    
    def sort(self, column, order):
        """Sort table by given column number"""
        self.layoutAboutToBeChanged.emit()
        col_name = self._data.columns[column]
        self._data = self._data.sort_values(
            col_name, 
            ascending=(order == Qt.SortOrder.AscendingOrder)
        )
        self.layoutChanged.emit()


class DataTableWidget(QTableWidget):
    """
    Widget for displaying tabular data from DSSAT outputs
    
    Features include:
    - Display data in a sortable, filterable table
    - Export data to CSV/Excel
    - Filter by treatment, variable, etc.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.perf_monitor = PerformanceMonitor()
        self.setup_ui()
        
        # Store data for both simulated and observed
        self._sim_data = None
        self._filtered_sim_data = None
        self._obs_data = None
        self._filtered_obs_data = None
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Export button
        self.export_button = QPushButton("Export Data")
        self.export_button.setToolTip("Export table to CSV/Excel")
        controls_layout.addWidget(self.export_button)
        
        # Filter controls
        filter_group = QGroupBox("Filter")
        filter_layout = QHBoxLayout()
        filter_group.setLayout(filter_layout)
        
        self.filter_column = QComboBox()
        filter_layout.addWidget(QLabel("Column:"))
        filter_layout.addWidget(self.filter_column)
        
        self.filter_value = QComboBox()
        self.filter_value.setEditable(True)
        filter_layout.addWidget(QLabel("Value:"))
        filter_layout.addWidget(self.filter_value)
        
        self.apply_filter_button = QPushButton("Apply")
        filter_layout.addWidget(self.apply_filter_button)
        
        self.clear_filter_button = QPushButton("Clear")
        filter_layout.addWidget(self.clear_filter_button)
        
        controls_layout.addWidget(filter_group)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create separate table views for simulated and observed data
        self.sim_table_view = QTableView()
        self.obs_table_view = QTableView()
        
        for table in [self.sim_table_view, self.obs_table_view]:
            table.setSortingEnabled(True)
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(True)
        
        # Add tables to tabs
        self.tab_widget.addTab(self.sim_table_view, "Simulated Data")
        self.tab_widget.addTab(self.obs_table_view, "Observed Data")
        
        layout.addWidget(self.tab_widget)
        self.tab_widget.setCurrentIndex(1)  # Start with observed data tab
        # Connect tab changed signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Connect other signals
        self.export_button.clicked.connect(self.export_data)
        self.apply_filter_button.clicked.connect(self.apply_filter)
        self.clear_filter_button.clicked.connect(self.clear_filter)
        self.filter_column.currentIndexChanged.connect(self.update_filter_values)
    
        

    def set_data(self, sim_data: pd.DataFrame = None, obs_data: pd.DataFrame = None):
        """
        Set the data to display in the tables
        
        Args:
            sim_data: DataFrame containing simulated DSSAT output data
            obs_data: DataFrame containing observed data
        """
        # Add debug logging for original data
        if obs_data is not None:
            print("Original obs data shape:", obs_data.shape)
            print("First few rows of obs data:")
            print(obs_data.head())
        
        # Clean data by removing empty columns
        if sim_data is not None and not sim_data.empty:
            sim_data = self._remove_empty_columns(sim_data)
            
        if obs_data is not None and not obs_data.empty:
            obs_data = self._remove_empty_columns(obs_data)
            print("After removing empty columns:", obs_data.shape)
        
        logger.info(f"Setting data - Sim data shape: {sim_data.shape if sim_data is not None else None}")
        logger.info(f"Setting data - Obs data shape: {obs_data.shape if obs_data is not None else None}")
    
    # Rest of the method remains the same...
        
        # Handle simulated data
        if sim_data is not None and not sim_data.empty:
            self._sim_data = sim_data.copy()
            self._filtered_sim_data = self._sim_data.copy()
            sim_model = PandasTableModel(self._filtered_sim_data)
            self.sim_table_view.setModel(sim_model)
            self.sim_table_view.resizeColumnsToContents()
            logger.debug(f"Simulated data loaded successfully. Row count: {sim_model.rowCount()}")
        else:
            self._sim_data = None
            self._filtered_sim_data = None
            self.sim_table_view.setModel(None)
            logger.warning("No simulation data provided or data is empty")
        
        # Handle observed data
        if obs_data is not None and not obs_data.empty:
            logger.debug(f"Observed data columns: {obs_data.columns.tolist()}")
            logger.debug(f"Observed data first rows: {obs_data.head().to_dict()}")
            
            # Store a copy to prevent modifications
            self._obs_data = obs_data.copy()
            self._filtered_obs_data = self._obs_data.copy()
            
            # Create and set model for observed data
            obs_model = PandasTableModel(self._filtered_obs_data)
            self.obs_table_view.setModel(obs_model)
            self.obs_table_view.resizeColumnsToContents()
            logger.debug(f"Observed data loaded successfully. Row count: {obs_model.rowCount()}")
        else:
            logger.warning("No observed data provided or data is empty")
            self._obs_data = None
            self._filtered_obs_data = None
            self.obs_table_view.setModel(None)
        
        # Update filter controls based on active tab
        self.update_filter_columns()
        
        # Force refresh of the current tab's data
        self.on_tab_changed(self.tab_widget.currentIndex())

    def _remove_empty_columns(self, df):
        """
        Remove empty columns (all NaN, 0, or empty string) from DataFrame
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with empty columns removed
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying original data
        df_copy = df.copy()
        
        # For each column, check if it's empty
        cols_to_drop = []
        for col in df_copy.columns:
            # Check if column is categorical
            if pd.api.types.is_categorical_dtype(df_copy[col]):
                # For categorical columns, check if they're all NaN
                if df_copy[col].isna().all():
                    cols_to_drop.append(col)
                # Otherwise, don't try to modify categorical columns
                continue
                
            # Check if all values are NaN
            if df_copy[col].isna().all():
                cols_to_drop.append(col)
                continue
                
            # For numeric columns, check if all values are 0 or NaN
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                # Don't modify the column directly, just check its values
                if ((df_copy[col] == 0) | df_copy[col].isna()).all():
                    cols_to_drop.append(col)
                    continue
                    
            # For string columns, check if all values are empty string or NaN
            elif pd.api.types.is_string_dtype(df_copy[col]):
                # Don't modify the column directly, just check its values
                if ((df_copy[col] == '') | df_copy[col].isna()).all():
                    cols_to_drop.append(col)
                    continue
        
        # Drop the identified empty columns
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} empty columns: {cols_to_drop}")
            df_copy = df_copy.drop(columns=cols_to_drop)
        
        return df_copy
    def clear(self):
        """Clear the table"""
        self._sim_data = None
        self._filtered_sim_data = None
        self._obs_data = None
        self._filtered_obs_data = None
        self.sim_table_view.setModel(None)
        self.obs_table_view.setModel(None)
        self.filter_column.clear()
        self.filter_value.clear()
    
    def update_filter_columns(self):
        """Update the filter column dropdown with available columns"""
        self.filter_column.clear()
        
        # Get data based on current tab
        current_data = self._sim_data if self.tab_widget.currentIndex() == 0 else self._obs_data
        
        if current_data is not None:
            # Add columns to filter dropdown
            for col in current_data.columns:
                self.filter_column.addItem(str(col))
    
    def update_filter_values(self):
        """Update filter values based on selected column"""
        current_data = self._sim_data if self.tab_widget.currentIndex() == 0 else self._obs_data
        
        if current_data is None:
            return
            
        self.filter_value.clear()
        
        column = self.filter_column.currentText()
        if column and column in current_data.columns:
            # Get unique values for the column
            unique_values = current_data[column].dropna().unique()
            
            # Add values to filter dropdown
            for value in sorted(unique_values):
                self.filter_value.addItem(str(value))
    
    def apply_filter(self):
        """Apply filter to the current tab's data"""
        # Get current tab's data and view
        is_sim = self.tab_widget.currentIndex() == 0
        data = self._sim_data if is_sim else self._obs_data
        view = self.sim_table_view if is_sim else self.obs_table_view
        
        if data is None:
            return
            
        column = self.filter_column.currentText()
        filter_text = self.filter_value.currentText()
        
        if column and filter_text and column in data.columns:
            try:
                # Filter logic remains the same
                if pd.api.types.is_numeric_dtype(data[column]):
                    try:
                        filter_value = float(filter_text)
                        filtered_data = data[data[column] == filter_value]
                    except ValueError:
                        filtered_data = data[data[column].astype(str).str.contains(filter_text)]
                else:
                    filtered_data = data[data[column].astype(str).str.contains(filter_text)]
                
                # Update the appropriate filtered data and model
                if is_sim:
                    self._filtered_sim_data = filtered_data
                else:
                    self._filtered_obs_data = filtered_data
                    
                model = PandasTableModel(filtered_data)
                view.setModel(model)
                view.resizeColumnsToContents()
                
                logger.debug(f"Filter applied. Filtered data shape: {filtered_data.shape}")
                
            except Exception as e:
                logger.error(f"Error applying filter: {e}")
    
    def clear_filter(self):
        """Clear filter and show all data"""
        is_sim = self.tab_widget.currentIndex() == 0
        
        if is_sim and self._sim_data is not None:
            self._filtered_sim_data = self._sim_data.copy()
            model = PandasTableModel(self._filtered_sim_data)
            self.sim_table_view.setModel(model)
            self.sim_table_view.resizeColumnsToContents()
            logger.debug(f"Filter cleared for simulated data. Row count: {model.rowCount()}")
        elif not is_sim and self._obs_data is not None:
            self._filtered_obs_data = self._obs_data.copy()
            model = PandasTableModel(self._filtered_obs_data)
            self.obs_table_view.setModel(model)
            self.obs_table_view.resizeColumnsToContents()
            logger.debug(f"Filter cleared for observed data. Row count: {model.rowCount()}")
    
    def export_data(self):
        """Export data to CSV or Excel"""
        is_sim = self.tab_widget.currentIndex() == 0
        data_to_export = self._filtered_sim_data if is_sim else self._filtered_obs_data
        
        if data_to_export is None or data_to_export.empty:
            logger.warning("No data to export")
            return
            
        # Ask for file name and type
        file_path, filter_type = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.csv'):
                data_to_export.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                data_to_export.to_excel(file_path, index=False)
            else:
                # Add extension based on filter
                if 'CSV' in filter_type:
                    file_path += '.csv'
                    data_to_export.to_csv(file_path, index=False)
                else:
                    file_path += '.xlsx'
                    data_to_export.to_excel(file_path, index=False)
            
            logger.info(f"Data exported successfully to {file_path}")
                    
        except Exception as e:
            logger.error(f"Error exporting data: {e}")

    def on_tab_changed(self, index):
        """Handle tab changes"""
        logger.debug(f"Tab changed to index {index}")
        
        # Update filter controls
        self.update_filter_columns()
        self.update_filter_values()

        # Ensure models are up to date after tab change
        if index == 0 and self._filtered_sim_data is not None:
            sim_model = PandasTableModel(self._filtered_sim_data)
            self.sim_table_view.setModel(sim_model)
            self.sim_table_view.resizeColumnsToContents()
            logger.debug(f"Refreshed simulated data model. Row count: {sim_model.rowCount()}")
        elif index == 1 and self._filtered_obs_data is not None:
            obs_model = PandasTableModel(self._filtered_obs_data)
            self.obs_table_view.setModel(obs_model)
            self.obs_table_view.resizeColumnsToContents()
            logger.debug(f"Refreshed observed data model. Row count: {obs_model.rowCount()}")

        # Debug info
        if index == 0:
            logger.debug(f"Sim data shape: {self._sim_data.shape if self._sim_data is not None else None}")
        else:
            logger.debug(f"Obs data shape: {self._obs_data.shape if self._obs_data is not None else None}")

    @function_timer("data_operations")
    def load_data(self, data):
        """Load data into the table with performance tracking"""
        timer_id = self.perf_monitor.start_timer("ui", "table_population")
        
        # Clear existing items
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(0)
        
        if data is None or data.empty:
            self.perf_monitor.stop_timer(timer_id, "No data to load")
            return
            
        # Set dimensions
        self.setRowCount(len(data))
        self.setColumnCount(len(data.columns))
        
        # Set headers
        self.setHorizontalHeaderLabels(data.columns)
        
        # Populate data
        for row in range(len(data)):
            for col in range(len(data.columns)):
                value = str(data.iloc[row, col])
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, col, item)
        
        duration = self.perf_monitor.stop_timer(timer_id, f"Loaded {len(data)} rows")
        return duration