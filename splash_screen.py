import sys
import os
import time
import logging
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QBrush

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 450
GRAPH_MARGIN_LEFT = 80
GRAPH_MARGIN_RIGHT = 70
GRAPH_MARGIN_TOP = 100
GRAPH_MARGIN_BOTTOM = 80

# Colors and styles
BACKGROUND_COLOR = QColor(255, 255, 255)
GRID_COLOR = QColor(230, 230, 230)
AXIS_COLOR = QColor(100, 100, 100)
LINE_COLOR = QColor(255, 140, 0)  # Orange for simulated data
POINT_COLOR = QColor(255, 140, 0)  # Orange for observed data
Y_AXIS_LABEL_COLOR = QColor(0, 0, 255)  # Blue for y-axis label

# Sample crop growth data (date, simulated value, observed value)
CROP_DATA = [
    ("Mar 24 1991", 1000, 1000),
    ("Apr 7", 1100, 1050),
    ("Apr 21", 1400, 1350),
    ("May 5", 3000, 2900),
    ("May 19", 5000, 4800),
    ("Jun 2", 7500, 6700),
    ("Jun 16", 10000, 7900)
]

class DSSATSplashScreen(QSplashScreen):
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
        """Initialize splash screen with given dimensions"""
        logging.debug("Initializing DSSATSplashScreen with width=%d and height=%d", width, height)
        self.width = width
        self.height = height
        
        # Create a pixmap with white background
        pixmap = QPixmap(self.width, self.height)
        pixmap.fill(BACKGROUND_COLOR)
        
        # Initialize with the blank pixmap
        super().__init__(pixmap)
        
        # Now draw on the pixmap
        self.update_pixmap()
        
        logging.debug("DSSATSplashScreen initialized successfully")

    def update_pixmap(self):
        """Update the pixmap with all drawings"""
        pixmap = self.pixmap()
        if pixmap.isNull():
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(BACKGROUND_COLOR)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Do all drawing
        self._draw_background_grid(painter)
        self._draw_axes(painter)
        self._draw_data(painter)
        self._draw_labels(painter)
        self._draw_legend(painter)
        
        painter.end()
        
        # Update the pixmap
        self.setPixmap(pixmap)

    def _draw_background_grid(self, painter):
        """Draw grid lines"""
        painter.setPen(QPen(GRID_COLOR, 1, Qt.PenStyle.SolidLine))
        
        # Graph area dimensions
        graph_width = self.width - GRAPH_MARGIN_LEFT - GRAPH_MARGIN_RIGHT
        graph_height = self.height - GRAPH_MARGIN_TOP - GRAPH_MARGIN_BOTTOM
        
        # Horizontal grid lines (y-axis)
        y_steps = 5  # Number of horizontal grid lines
        for i in range(y_steps + 1):
            y = GRAPH_MARGIN_TOP + (graph_height / y_steps) * i
            painter.drawLine(
                GRAPH_MARGIN_LEFT, int(y),
                self.width - GRAPH_MARGIN_RIGHT, int(y)
            )
        
        # Vertical grid lines (x-axis)
        for i, (date, _, _) in enumerate(CROP_DATA):
            x_pos = GRAPH_MARGIN_LEFT + (graph_width / (len(CROP_DATA) - 1)) * i
            painter.drawLine(
                int(x_pos), GRAPH_MARGIN_TOP,
                int(x_pos), self.height - GRAPH_MARGIN_BOTTOM
            )

    def _draw_axes(self, painter):
        """Draw axes with labels"""
        # Set pen for axes
        painter.setPen(QPen(AXIS_COLOR, 2))
        
        # Graph area dimensions
        graph_height = self.height - GRAPH_MARGIN_TOP - GRAPH_MARGIN_BOTTOM
        
        # X-axis (horizontal)
        painter.drawLine(
            GRAPH_MARGIN_LEFT, self.height - GRAPH_MARGIN_BOTTOM,
            self.width - GRAPH_MARGIN_RIGHT, self.height - GRAPH_MARGIN_BOTTOM
        )
        
        # Y-axis (vertical)
        painter.drawLine(
            GRAPH_MARGIN_LEFT, GRAPH_MARGIN_TOP,
            GRAPH_MARGIN_LEFT, self.height - GRAPH_MARGIN_BOTTOM
        )
        
        # Y-axis labels (values)
        painter.setPen(AXIS_COLOR)
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        y_max = 10000
        y_steps = 5
        
        for i in range(y_steps + 1):
            y_value = y_max - (y_max / y_steps) * i
            y_pos = GRAPH_MARGIN_TOP + (graph_height / y_steps) * i
            
            if i == 0:
                label = "10k"
            else:
                label = f"{int(y_value/1000)}k"
                
            painter.drawText(
                GRAPH_MARGIN_LEFT - 45, int(y_pos) + 5, 
                40, 20, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label
            )
        
        # X-axis labels (dates)
        graph_width = self.width - GRAPH_MARGIN_LEFT - GRAPH_MARGIN_RIGHT
        for i, (date, _, _) in enumerate(CROP_DATA):
            x_pos = GRAPH_MARGIN_LEFT + (graph_width / (len(CROP_DATA) - 1)) * i
            painter.drawText(
                int(x_pos) - 30, self.height - GRAPH_MARGIN_BOTTOM + 15,
                60, 20, Qt.AlignmentFlag.AlignCenter, date
            )
            
        # Bold "DATE" label centered under x-axis
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 255))  # Blue color
        painter.drawText(
            GRAPH_MARGIN_LEFT, self.height - 40,
            graph_width, 30, Qt.AlignmentFlag.AlignCenter, "DATE"
        )

    def _draw_data(self, painter):
        """Draw crop growth data (simulated line and observed points)"""
        graph_width = self.width - GRAPH_MARGIN_LEFT - GRAPH_MARGIN_RIGHT
        graph_height = self.height - GRAPH_MARGIN_TOP - GRAPH_MARGIN_BOTTOM
        
        y_max = 10000  # Maximum y value
        
        # Prepare points for simulated data (line)
        sim_points = []
        obs_points = []
        
        for i, (_, sim_value, obs_value) in enumerate(CROP_DATA):
            x_pos = GRAPH_MARGIN_LEFT + (graph_width / (len(CROP_DATA) - 1)) * i
            sim_y_pos = GRAPH_MARGIN_TOP + graph_height - (sim_value / y_max * graph_height)
            obs_y_pos = GRAPH_MARGIN_TOP + graph_height - (obs_value / y_max * graph_height)
            
            sim_points.append(QPointF(x_pos, sim_y_pos))
            obs_points.append(QPointF(x_pos, obs_y_pos))
        
        # Draw simulated data line
        pen = QPen(LINE_COLOR, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        for i in range(1, len(sim_points)):
            painter.drawLine(sim_points[i-1], sim_points[i])
        
        # Draw observed data points as squares
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(POINT_COLOR))
        for point in obs_points:
            painter.drawRect(int(point.x() - 4), int(point.y() - 4), 8, 8)

    def _draw_labels(self, painter):
        """Draw title and y-axis label"""
        # Draw title
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(
            0, 20, self.width, 30, 
            Qt.AlignmentFlag.AlignCenter, "DSSAT Visualization"
        )
        
        # Draw y-axis label text (rotated)
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(Y_AXIS_LABEL_COLOR)  # Blue for y-axis label
        
        painter.save()
        painter.translate(25, self.height / 2)
        painter.rotate(-90)
        painter.drawText(
            -100, 0, 200, 20, 
            Qt.AlignmentFlag.AlignCenter, "Tops wt kg/ha"
        )
        painter.restore()
        
        # Draw loading text
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(
            0, self.height - 30, self.width, 30,
            Qt.AlignmentFlag.AlignCenter, "Loading application..."
        )

    def _draw_legend(self, painter):
        """Draw legend showing simulated and observed data"""
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        # Legend background
        legend_x = self.width - GRAPH_MARGIN_RIGHT - 120
        legend_y = GRAPH_MARGIN_TOP - 40
        legend_width = 110
        legend_height = 40  # Reduced height
        
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        
        # Line heights
        line1_y = legend_y + 15
        line2_y = legend_y + 30
        
        # Simulated Data - line
        painter.setPen(QPen(LINE_COLOR, 2))
        painter.drawLine(
            legend_x + 10, line1_y, 
            legend_x + 35, line1_y
        )
        
        # Simulated Data - text
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(
            legend_x + 40, line1_y - 5, 
            legend_width - 45, 20, 
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Simulated"
        )
        
        # Observed Data - point
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(POINT_COLOR))
        painter.drawRect(legend_x + 21, line2_y - 4, 8, 8)
        
        # Observed Data - text
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(
            legend_x + 40, line2_y - 5, 
            legend_width - 45, 20, 
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Observed"
        )

def show_splash(app=None):
    """
    Display the splash screen.
    
    Args:
        app: Optional QApplication instance. If None, will use existing or create new.
    
    Returns:
        Tuple of (QApplication, QSplashScreen)
    """
    logging.debug("Displaying splash screen")
    
    # Use provided app, existing app, or create new app
    app = app or QApplication.instance() or QApplication([])
    
    splash = DSSATSplashScreen()
    splash.show()
    
    # Process events to ensure splash is rendered
    app.processEvents()
    
    logging.debug("Splash screen displayed")
    return splash

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = show_splash(app)
    
    # Simulate loading time
    time.sleep(3)
    print("Application loaded")
    
    sys.exit(app.exec())  # Note: In PyQt6, exec() doesn't have parentheses