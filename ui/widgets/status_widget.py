"""
Status Widget for DSSAT Viewer
Provides status messages and notifications
"""
import os
import sys
import logging

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar
)
from PyQt6.QtCore import  QTimer,  pyqtSlot

# Configure logging
logger = logging.getLogger(__name__)

class StatusWidget(QWidget):
    """
    Widget for displaying status messages and notifications
    
    Shows success, error, and warning messages as well as progress indicators
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Message auto-clear timer
        self.clear_timer = QTimer(self)
        self.clear_timer.timeout.connect(self.clear)
        self.clear_timer.setSingleShot(True)
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        self.setLayout(layout)
        
        # Status message label
        self.message_label = QLabel()
        self.message_label.setStyleSheet("padding: 2px 5px;")
        layout.addWidget(self.message_label)
        
        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Add stretch to push to left
        layout.addStretch(1)
    
    def show_message(self, message: str, style: str, timeout: int = 5000):
        """
        Show a status message with specified style
        
        Args:
            message: Message text to display
            style: CSS style to apply (success, error, warning)
            timeout: Auto-clear timeout in milliseconds
        """
        # Stop any pending timer
        if self.clear_timer.isActive():
            self.clear_timer.stop()
        
        # Set message and style
        self.message_label.setText(message)
        self.message_label.setStyleSheet(f"padding: 2px 5px; {style}")
        
        # Start auto-clear timer if timeout > 0
        if timeout > 0:
            self.clear_timer.start(timeout)
    
    def show_success(self, message: str, timeout: int = 5000):
        """Show success message"""
        self.show_message(
            message, 
            "background-color: #4CAF50; color: white; border-radius: 3px;",
            timeout
        )
    
    def show_error(self, message: str, timeout: int = 5000):
        """Show error message"""
        self.show_message(
            message, 
            "background-color: #F44336; color: white; border-radius: 3px;",
            timeout
        )
    
    def show_warning(self, message: str, timeout: int = 5000):
        """Show warning message"""
        self.show_message(
            message, 
            "background-color: #FF9800; color: white; border-radius: 3px;",
            timeout
        )
    
    def show_running(self, message: str = "Processing..."):
        """Show processing indicator"""
        self.show_message(
            message,
            "background-color: #2196F3; color: white; border-radius: 3px;",
            0  # No timeout
        )
        self.progress_bar.show()
    
    def clear(self):
        """Clear message and hide progress bar"""
        self.message_label.clear()
        self.message_label.setStyleSheet("padding: 2px 5px;")
        self.progress_bar.hide()