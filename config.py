"""
Configuration settings for DSSAT Viewer
"""
import logging
import os
import platform
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#vironment and initialization
if platform.system() == 'Windows':
    DSSAT_BASE = r"C:\DSSAT48"  # Windows DSSAT installation path
    DSSAT_EXE = "DSCSM048.EXE"
else:
    DSSAT_BASE = "/Applications/DSSAT48"  # macOS DSSAT installation path
    DSSAT_EXE = "DSCSM048"  # macOS executable name

# Performance optimization settings
PLOT_BATCH_SIZE = 5000  # Number of points to render in each batch
ENABLE_GPU_ACCELERATION = True
DOWNSAMPLING_THRESHOLD = 1000  # Points before downsampling is applied
MAX_CACHE_ENTRIES = 1000
ENABLE_DATA_COMPRESSION = True
ENABLE_ANTIALIASING = False  # Disable antialiasing for better performance
ENABLE_OPENGL = True  # Use OpenGL for hardware acceleration
CACHE_SIZE_LIMIT = 1000  # Maximum number of cached items

# Default values
DEFAULT_ENCODING = 'utf-8'
FALLBACK_ENCODING = 'latin-1'

# Missing values for DSSAT files
MISSING_VALUES = {-99, -99.0, -99.9, -99.99, -99., '-99', '-99.0', '-99.9'}

# Plot styling optimized for performance
LINE_STYLES = ["solid", "dash", "dot"]
MARKER_SYMBOLS = ['o', 's', 'd', 't', '+', 'x', 'p', 'h', 'star']  # Reduced number of symbols for better performance
PLOT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',  # Optimized color set
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
