# Set Qt attributes before ANY Qt imports or initialization
from PyQt6.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)

"""
DSSAT Viewer - Main entry point
Updated with pure PyQt6 implementation (no Dash)
Optimized for performance and fast tab switching
"""
import sys
import os
import warnings
import logging
import gc
from pathlib import Path
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMessageBox, QStyleFactory
from utils.performance_monitor import PerformanceMonitor, function_timer
from ui.main_window import MainWindow  # Import MainWindow at the top level

# Import SingleInstanceApp
from single_instance import SingleInstanceApp

# Initialize performance monitor
perf_monitor = PerformanceMonitor()

# Configure logging first - use INFO level to avoid excessive logs
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Setup performance monitoring
import time
start_time = time.time()
logger = logging.getLogger(__name__)
logger.info("Starting DSSAT Viewer application...")

# Add constants for window dimensions and positioning
WINDOW_CONFIG = {
    'width': 1200,
    'height': 600,
    'min_width': 1000,
    'min_height': 700
}

# Apply startup optimizations before creating any Qt objects
try:
    from optimized_startup import (
        optimize_qt_settings, 
        optimize_qtgraph_settings, 
        set_memory_optimizations,
        optimize_application
    )
    # Apply Qt optimizations before creating QApplication
    logger.info("Applying Qt optimizations...")
    optimize_qt_settings()
    
    # Configure memory optimizations
    logger.info("Applying memory optimizations...")
    set_memory_optimizations()
    
except ImportError:
    logging.warning("Optimization module not found, running without optimizations")

# Suppress warnings for performance
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root directory to the Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

def center_window(window):
    """Center window on screen"""
    screen = window.screen().availableGeometry()
    window_size = window.geometry()
    x = (screen.width() - window_size.width()) // 2
    y = (screen.height() - window_size.height()) // 2
    window.move(x, y)

# Import splash screen after optimizations
from splash_screen import show_splash

@function_timer("startup")
def initialize_app():
    """Initialize Qt application with monitoring"""
    timer_id = perf_monitor.start_timer("startup", "qt_init")
    app = SingleInstanceApp(sys.argv)
    perf_monitor.stop_timer(timer_id)
    return app

@function_timer("startup")
def create_main_window():
    """Create and set up main window with monitoring"""
    timer_id = perf_monitor.start_timer("startup", "window_creation")
    window = MainWindow()
    window.show()
    center_window(window)
    perf_monitor.stop_timer(timer_id)
    return window

def setup_application_icon(app):
    """
    Set up application icon for both window and taskview
    
    This function configures the application icon for all platforms
    and handles both the application icon and the window icon.
    It also sets platform-specific taskbar/taskview identifiers
    when needed (Windows AppUserModelID, etc.)
    
    Args:
        app: The QApplication instance
        
    Returns:
        QIcon: The application icon object for reuse
    """
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import QSize
    import os
    import sys
    from pathlib import Path
    
    # Get path to resources folder
    project_dir = Path(__file__).parent
    resources_dir = os.path.join(project_dir, 'resources')
    
    # Create icon object
    app_icon = QIcon()
    
    # Check platform and add appropriate icons
    if sys.platform == "win32":
        # Try to load ICO file first if available
        ico_path = os.path.join(resources_dir, 'gbuild2.ico')
        if os.path.exists(ico_path):
            app_icon.addFile(ico_path)
            logger.info(f"Loaded Windows ICO file: {ico_path}")
        else:
            # Fall back to PNG if ICO not available
            app_icon.addFile(os.path.join(resources_dir, 'final.ico'))
            logger.info(f"Loaded PNG icon file for Windows: {os.path.join(resources_dir, 'final.ico')}")
            
        # Set Windows-specific application ID for taskbar grouping
        try:
            import ctypes
            app_id = "DSSAT.GBuild2.v2"  # This should be unique for your app
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            logger.info(f"Set Windows application ID: {app_id}")
        except Exception as e:
            logger.warning(f"Failed to set Windows AppUserModelID: {e}")
            
    elif sys.platform == "darwin":
        # For macOS, load icons in standard sizes
        # The ICNS file is typically handled by the app bundle on macOS
        app_icon.addFile(os.path.join(resources_dir, 'icon1.png'))
        logger.info(f"Loaded icon file for macOS: {os.path.join(resources_dir, 'icon1.png')}")
        
    else:
        # Linux and other platforms
        # Check for size-specific PNG files
        for size in [16, 32, 48, 64, 128, 256]:
            size_file = os.path.join(resources_dir, f'icon{size}.png')
            if os.path.exists(size_file):
                app_icon.addFile(size_file, QSize(size, size))
                logger.info(f"Loaded {size}x{size} icon: {size_file}")
        
        # Always add the default icon as fallback
        app_icon.addFile(os.path.join(resources_dir, 'icon1.png'))
        logger.info(f"Loaded default icon file: {os.path.join(resources_dir, 'icon1.png')}")
    
    # Set the application icon (for taskview/taskbar)
    app.setWindowIcon(app_icon)
    logger.info("Application icon set successfully")
    
    # Return the icon for use in the main window
    return app_icon

@function_timer("startup")
def main():
    """Main application entry point with error handling and optimizations."""
    startup_timer = perf_monitor.start_timer("application", "total_startup")
    
    try:
        app = initialize_app()
        
        # Check if this is the first instance - matches your Java implementation
        if not app.is_first_instance:
            logger.info("Another instance is already running. Exiting...")
            app.show_already_running_message()
            sys.exit(0)  # Exit gracefully - just like in your Java code
        
        # Apply Qt optimizations and settings
        try:
            from optimized_startup import optimize_qt_settings
            optimize_qt_settings()
        except ImportError:
            pass
            
        # Apply application optimizations
        try:
            from optimized_startup import optimize_application
            app = optimize_application(app)
        except Exception as e:
            logger.warning(f"Could not apply application optimizations: {e}")
            app.setStyle(QStyleFactory.create('Fusion'))
        
        # Set application icon (new addition)
        app_icon = setup_application_icon(app)
            
        # Show splash screen
        splash = show_splash(app)
        app.processEvents()  # Ensure splash is displayed
        
        # Apply PyQtGraph optimizations
        try:
            from optimized_startup import optimize_qtgraph_settings
            optimize_qtgraph_settings()
        except Exception as e:
            logger.warning(f"Could not apply PyQtGraph optimizations: {e}")
        
        # Import and initialize main application
        try:
            logger.info("Initializing main window...")
            
            # Start timing main window creation
            window_timer = perf_monitor.start_timer("ui", "main_window_creation")
            main_window = create_main_window()
            
            # Set the window icon (new addition)
            main_window.setWindowIcon(app_icon)
            
            perf_monitor.stop_timer(window_timer)
            
            # Configure window
            main_window.resize(WINDOW_CONFIG['width'], WINDOW_CONFIG['height'])
            main_window.setMinimumSize(
                QSize(WINDOW_CONFIG['min_width'], WINDOW_CONFIG['min_height'])
            )
            
            # Center window
            center_window(main_window)
            
            # Stop total initialization timer
            perf_monitor.stop_timer(startup_timer, "Application startup completed")
            
            # Log startup time
            init_time = time.time() - start_time
            logger.info(f"Application initialized in {init_time:.2f} seconds")
            perf_monitor.print_report()
            
            # Run garbage collection before showing window
            gc.collect()
            
            # Show main window and close splash
            main_window.show()
            splash.finish(main_window)
            
            # Start event loop
            return app.exec()
            
        except Exception as e:
            splash.close()
            raise
            
    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)
        perf_monitor.stop_timer(startup_timer, f"Error during startup: {str(e)}")
        
        # Use SingleInstanceApp.instance() instead of QApplication.instance()
        if SingleInstanceApp.instance():
            QMessageBox.critical(
                None,
                "Startup Error",
                f"Failed to start GB2:\n{str(e)}"
            )
        return 1

if __name__ == "__main__":
    # Use return code from main function
    exit_code = main()
    
    # Force cleanup before exit
    gc.collect()
    sys.exit(exit_code)