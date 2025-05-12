import os
import sys
import logging
from PyQt6.QtCore import Qt, QCoreApplication, QSettings
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication, QStyleFactory

# Configure logging
logger = logging.getLogger(__name__)

def optimize_qt_settings():
    """Apply performance optimizations for PyQt"""
    # Set application name and organization for settings
    QCoreApplication.setApplicationName("DSSAT Viewer")
    QCoreApplication.setOrganizationName("DSSAT")
    
    # Set environment variables for performance
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    
    # For OpenGL acceleration
    os.environ["QSG_RENDER_LOOP"] = "basic"  # Use basic render loop for better performance
    os.environ["QT_QUICK_BACKEND"] = "software"  # Reduce GPU usage
    os.environ["QT_OPENGL"] = "desktop"  # Force desktop OpenGL
    
    logger.info("Applied Qt performance optimizations")
    return True

def optimize_qtgraph_settings():
    """Apply performance optimizations for PyQtGraph"""
    try:
        import pyqtgraph as pg
        
        # Use OpenGL for hardware acceleration if available
        pg.setConfigOption('useOpenGL', True)
        
        # Disable antialiasing for faster rendering
        pg.setConfigOption('antialias', False)
        
        # Enable downsampling for large datasets
        pg.setConfigOption('downsample', True)
        
        # Set background to solid color (faster than transparent)
        pg.setConfigOption('background', 'w')  # White
        
        # Optimize other settings
        pg.setConfigOption('exitCleanup', False)  # Faster exit
        
        logger.info("Applied PyQtGraph performance optimizations")
        return True
    except ImportError:
        logger.warning("PyQtGraph not found, skipping optimizations")
        return False

def set_memory_optimizations():
    """Apply memory usage optimizations"""
    # Limit cache sizes
    import pandas as pd
    pd.options.mode.chained_assignment = None  # Disable chained assignment warning
    pd.options.display.max_columns = 50  # Limit display columns
    
    # Set numpy performance options
    try:
        import numpy as np
        np.seterr(all='ignore')  # Ignore numpy warnings for performance
    except ImportError:
        pass
    
    # Clean up memory
    import gc
    gc.enable()  # Ensure garbage collection is enabled
    
    logger.info("Applied memory optimizations")
    return True

def optimize_application(app):
    """Apply optimizations to a QApplication instance"""
    # Set style to Fusion which is generally more performant
    app.setStyle(QStyleFactory.create('Fusion'))
    
    # Set desktop settings for performance
    settings = QSettings()
    settings.setValue("GUI/ComputeHighDPI", False)
    settings.setValue("GUI/ShowToolTips", False)
    settings.setValue("GUI/AnimateMenu", False)
    
    # Set palette for better performance (less translucency)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    # Set a reasonable default font
    from PyQt6.QtGui import QFont
    app.setFont(QFont("Segoe UI", 9))
    
    logger.info("Optimized QApplication instance")
    return app

"""
Optimized startup configuration for PyQtGraph plotting
"""
import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import QThread
import config

def configure_pyqtgraph():
    """Configure PyQtGraph settings for optimal performance"""
    import pyqtgraph as pg
    pg.setConfigOptions(
        useOpenGL=True,
        antialias=True,
        foreground='k',
        background='w',
        imageAxisOrder='row-major',
        exitCleanup=True  # Clean up on exit
    )

def optimize_thread_priority():
    """Optimize thread priorities for better performance"""
    try:
        # Set current thread priority to high
        current_thread = QThread.currentThread()
        current_thread.setPriority(QThread.Priority.HighestPriority)
    except Exception as e:
        print(f"Error setting thread priority: {e}")
        
def configure_numpy():
    """Configure NumPy settings for optimal performance"""
    try:
        import numpy as np
        # Use threadpool_limits instead of set_num_threads
        from threadpoolctl import threadpool_limits
        
        # Limit number of threads globally
        threadpool_limits(limits=4, user_api='blas')
        
        # Additional NumPy optimizations
        np.seterr(all='ignore')  # Ignore numeric warnings
        np.random.seed(42)  # Set random seed for reproducibility
        
    except ImportError as e:
        logger.warning(f"Could not fully configure NumPy: {e}")
    except Exception as e:
        logger.warning(f"Error during NumPy configuration: {e}")

def optimize_plot_settings():
    """Configure optimal plot settings"""
    return {
        'batch_size': config.PLOT_BATCH_SIZE,
        'downsampling': True,
        'downsampling_threshold': config.DOWNSAMPLING_THRESHOLD,
        'antialiasing': False,
        'clip_to_view': True,
        'auto_range': False,
        'mouse_enabled': True,
        'background': 'w',
        'line_width': 1,
        'marker_size': 6
    }
    
def initialize_optimized_plotting():
    """Initialize all plotting optimizations"""
    configure_pyqtgraph()
    optimize_thread_priority()
    configure_numpy()
    return optimize_plot_settings()