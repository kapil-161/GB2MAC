"""
SingleInstanceApp implementation using a file lock mechanism similar to the Java example.
"""
import os
import sys
import logging
import atexit
from PyQt6.QtWidgets import QApplication, QMessageBox

logger = logging.getLogger(__name__)

class SingleInstanceApp(QApplication):
    """Application class that ensures only one instance runs at a time using file locking."""
    
    def __init__(self, argv, app_id="com.dssat.viewer.app"):
        super().__init__(argv)
        self.app_id = app_id
        self.lock_file = None
        self.lock_file_path = None
        self.file_lock = None
        
        # Check if we're the first instance
        self.is_first_instance = not self.lock_instance()
        
    def lock_instance(self):
        """
        Attempts to create and lock a file to ensure single instance.
        
        Returns:
            bool: True if another instance is already running, False if this is the first instance
        """
        # Create lock file in current directory, similar to Java example
        lock_file_name = ".lock.instance.GB2"
        self.lock_file_path = os.path.join(os.getcwd(), lock_file_name)
        
        try:
            # Create or open the lock file
            self.lock_file = open(self.lock_file_path, 'w')
            
            # Try to obtain a lock on the file
            if sys.platform == 'win32':
                # Windows implementation
                import msvcrt
                try:
                    # Try to lock the file
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    
                    # If we get here, we got the lock
                    # Register a shutdown hook to clean up
                    atexit.register(self._cleanup_lock)
                    
                    # Write PID to lock file
                    self.lock_file.write(str(os.getpid()))
                    self.lock_file.flush()
                    
                    logger.info(f"Successfully locked file: {self.lock_file_path}")
                    return False  # No other instance is running
                    
                except IOError:
                    # Could not lock the file, another instance is running
                    self.lock_file.close()
                    logger.info(f"Could not lock file, another instance is running")
                    return True  # Another instance is running
            else:
                # Unix/Linux/Mac implementation
                import fcntl
                try:
                    # Try to acquire an exclusive lock (non-blocking)
                    fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # If we get here, we got the lock
                    # Register a shutdown hook to clean up
                    atexit.register(self._cleanup_lock)
                    
                    # Write PID to lock file
                    self.lock_file.write(str(os.getpid()))
                    self.lock_file.flush()
                    
                    logger.info(f"Successfully locked file: {self.lock_file_path}")
                    return False  # No other instance is running
                    
                except IOError:
                    # Could not lock the file, another instance is running
                    self.lock_file.close()
                    logger.info(f"Could not lock file, another instance is running")
                    return True  # Another instance is running
                
        except Exception as e:
            logger.error(f"Error creating/locking file: {self.lock_file_path}, {str(e)}")
            # In case of error, allow the application to run
            return False
    
    def _cleanup_lock(self):
        """Clean up the lock file when the application exits."""
        try:
            if self.lock_file:
                # Release the lock by closing the file
                self.lock_file.close()
                
            # Delete the lock file
            if self.lock_file_path and os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
                logger.info(f"Removed lock file: {self.lock_file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up lock file: {e}")
            
    def show_already_running_message(self):
        """Show a message that the application is already running."""
        QMessageBox.critical(
            None, 
            "ERROR",
            "GB2 is already opened.",
            QMessageBox.StandardButton.Ok
        )