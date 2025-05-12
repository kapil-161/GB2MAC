"""
Metrics Table Widget for DSSAT Viewer
Displays model performance metrics
"""
import os
import sys
import logging
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QHeaderView,
    QPushButton,  QFileDialog, QLabel,
    QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QAbstractTableModel,  pyqtSlot
from PyQt6.QtGui import QFont, QColor

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

# Configure logging
logger = logging.getLogger(__name__)

class MetricsTableModel(QAbstractTableModel):
    """Table model for metrics data to display in QTableView"""
    
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data
        # Define column headers based on typical metrics
        self._headers = ["Variable", "n", "R²", "RMSE", "d-stat"]
        # Map data keys to headers (for flexibility in data format)
        self._key_map = {
            "Variable": ["Variable", "variable", "var"],
            "n": ["n", "N", "samples", "count"],
            "R²": ["R²", "R2", "r_squared", "rsquared", "r-squared"],
            "RMSE": ["RMSE", "rmse", "root_mean_square_error"],
            "d-stat": ["d-stat", "Willmott's d-stat", "d_stat", "dstat", "willmott_d"]
        }

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None
            
        col_name = self._headers[index.column()]
        
        # Find the value using the key map
        value = None
        for key in self._key_map[col_name]:
            if key in self._data[index.row()]:
                value = self._data[index.row()][key]
                break
                
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "NA"
            elif isinstance(value, float):
                # Format floating point numbers
                return f"{value:.4f}"
            else:
                return str(value)
                
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col_name == "Variable":
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            else:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
        elif role == Qt.ItemDataRole.FontRole:
            if col_name == "Variable":
                font = QFont()
                font.setBold(True)
                return font
                
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Add coloring for good/bad metric values
            if col_name == "R²" and isinstance(value, float):
                if value > 0.8:
                    return QColor(200, 255, 200)  # Light green
                elif value < 0.5:
                    return QColor(255, 200, 200)  # Light red
            elif col_name == "d-stat" and isinstance(value, float):
                if value > 0.8:
                    return QColor(200, 255, 200)  # Light green
                elif value < 0.5:
                    return QColor(255, 200, 200)  # Light red
                    
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None
    
    def sort(self, column, order):
        """Sort table by given column number"""
        self.layoutAboutToBeChanged.emit()
        col_name = self._headers[column]
        
        # Find the key in the data that corresponds to this header
        key_to_sort = None
        for key in self._key_map[col_name]:
            if any(key in item for item in self._data):
                key_to_sort = key
                break
                
        if key_to_sort:
            # Sort by the identified key
            self._data = sorted(
                self._data,
                key=lambda x: (key_to_sort not in x, x.get(key_to_sort, 0)),
                reverse=(order != Qt.SortOrder.AscendingOrder)
            )
            
        self.layoutChanged.emit()


class MetricsTableWidget(QWidget):
    """
    Widget for displaying model performance metrics
    
    Shows calculated metrics for model evaluation with export capability
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Store metrics data
        self._metrics_data = []
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title and description
        self.title_label = QLabel("Model Performance Metrics")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.title_label)
        
        self.description_label = QLabel(
            "This table shows performance metrics for simulated versus measured data."
        )
        layout.addWidget(self.description_label)
        
        # Table view
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        # Change from Stretch to Interactive - will be set properly in set_metrics
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.verticalHeader().setVisible(False)
        layout.addWidget(self.table_view)
        
        # Export button 
        self.export_button = QPushButton("Export Metrics")
        self.export_button.setToolTip("Export metrics to CSV")
        layout.addWidget(self.export_button)
        
        # Connect signals
        self.export_button.clicked.connect(self.export_metrics)
    
    def set_metrics(self, metrics_data: List[Dict[str, Any]]):
        """
        Set the metrics data to display
        
        Args:
            metrics_data: List of dictionaries containing metrics data
        """
        if not metrics_data:
            self.clear()
            return
            
        self._metrics_data = metrics_data
        
        # Update model
        model = MetricsTableModel(self._metrics_data)
        self.table_view.setModel(model)
        
        # First, set to ResizeToContents to get width based on content
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        # Let the view calculate sizes
        self.table_view.horizontalHeader().setStretchLastSection(False)
        
        # Get the calculated widths and set them explicitly
        widths = [self.table_view.columnWidth(i) for i in range(model.columnCount())]
        
        # Give extra space to the Variable column (usually first column)
        if widths and len(widths) > 0:
            widths[0] = max(widths[0], 250)  # Minimum width for Variable column
        
        # Set fixed width mode so we can set the widths manually
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        
        # Apply the widths
        for i, width in enumerate(widths):
            self.table_view.setColumnWidth(i, width + 20)  # Add some padding
    
    def clear(self):
        """Clear the metrics table"""
        self._metrics_data = []
        self.table_view.setModel(None)
    
    def export_metrics(self):
        """Export metrics data to CSV"""
        if not self._metrics_data:
            return
            
        # Ask for file name
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics",
            "",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
            
        try:
            # Convert metrics to DataFrame
            df = pd.DataFrame(self._metrics_data)
            
            # Export to CSV
            if not file_path.endswith('.csv'):
                file_path += '.csv'
                
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")


class MetricsDialog(QDialog):
    """Dialog for displaying metrics in a dedicated window"""
    
    def __init__(self, metrics_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Performance Metrics")
        self.resize(800, 500)
        
        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Metrics table widget
        self.metrics_widget = MetricsTableWidget()
        layout.addWidget(self.metrics_widget)
        
        # Set metrics if provided
        if metrics_data:
            self.metrics_widget.set_metrics(metrics_data)
            
            # Adjust dialog width based on table width
            table_width = sum(self.metrics_widget.table_view.columnWidth(i) 
                            for i in range(self.metrics_widget.table_view.model().columnCount()))
            table_width += 40  # Add some padding
            self.resize(max(table_width, 600), self.height())
        
        # Add dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def set_metrics(self, metrics_data):
        """Set metrics data"""
        self.metrics_widget.set_metrics(metrics_data)