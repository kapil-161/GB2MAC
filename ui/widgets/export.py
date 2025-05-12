import os
import logging
import math
from PyQt6.QtGui import QImage, QPainter, QFont, QColor, QPen, QPolygon, QBrush
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtWidgets import QFileDialog
import pyqtgraph as pg
# Import exporters explicitly
from pyqtgraph.exporters import ImageExporter
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def export_plot(plot_widget, plot_items_metadata, scaling_label, get_variable_info_func):
    """
    Export the plot with title, legend (with three-column layout), and scaling labels.
    
    Args:
        plot_widget: The pyqtgraph PlotWidget object
        plot_items_metadata: List of tuples with (plot_item, metadata_dict)
        scaling_label: The QLabel containing scaling information
        get_variable_info_func: Function to get variable info
    
    Returns:
        bool: Whether the export was successful
    """
    file_path, _ = QFileDialog.getSaveFileName(
        plot_widget,
        "Export Plot",
        "",
        "Images (*.png *.jpg *.jpeg *.tiff);;All Files (*)"
    )
    
    if not file_path:
        return False
        
    # Determine format from extension or default to PNG
    file_format = file_path.split('.')[-1].lower() if '.' in file_path else 'png'
    if not file_format or file_format not in ['png', 'jpg', 'jpeg', 'tiff']:
        file_format = 'png'
        file_path += '.png'
    
    try:
        # First make sure the plot is updated and autoranged
        plot_widget.autoRange()
        
        # Create an exporter for the plot - using the explicit import
        exporter = ImageExporter(plot_widget.plotItem)
        
        # Set fixed dimensions
        width = 1200
        height = 900
        
        exporter.parameters()['width'] = width
        exporter.parameters()['height'] = height
        
        # Export to temporary file
        temp_plot_path = f"{file_path}_temp.png"
        exporter.export(temp_plot_path)
        
        # Load the exported plot image
        plot_image = QImage(temp_plot_path)
        
        # Get dimensions
        plot_width = plot_image.width()
        plot_height = plot_image.height()
        
        # Create a title image
        title_height = 60
        
        # Try to get variables from the plot for a meaningful title
        y_vars = []
        for item, metadata in plot_items_metadata:
            if 'variable' in metadata and metadata['variable'] not in y_vars:
                y_vars.append(metadata['variable'])
        
        title_text = ""  # Empty title as in original code
        
        # Create an image for the title
        title_image = QImage(plot_width, title_height, QImage.Format.Format_ARGB32)
        title_image.fill(Qt.GlobalColor.white)
        
        # Render the title text
        title_painter = QPainter(title_image)
        
        # Set up font for the title
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_painter.setFont(title_font)
        
        # Draw the title centered in the image
        title_painter.drawText(title_image.rect(), Qt.AlignmentFlag.AlignCenter, title_text)
        title_painter.end()
        
        # Create a separate image for the scaling label
        scaling_text = scaling_label.text()
        if scaling_text:
            scale_height = 50
            
            # Create an image for the scaling label
            scale_image = QImage(plot_width, scale_height, QImage.Format.Format_ARGB32)
            scale_image.fill(Qt.GlobalColor.white)
            
            # Render the scaling text centered in the image
            scale_painter = QPainter(scale_image)
            font = scale_painter.font()
            font.setPointSize(12)
            scale_painter.setFont(font)
            scale_painter.drawText(scale_image.rect(), Qt.AlignmentFlag.AlignCenter, scaling_text)
            scale_painter.end()
        else:
            scale_image = None
            scale_height = 0
        
        # Create a new image with space for everything
        final_width = plot_width + 300  # Increased width for legend with three columns
        final_height = title_height + plot_height + scale_height
        
        # Create the final image
        final_image = QImage(final_width, final_height, QImage.Format.Format_ARGB32)
        final_image.fill(Qt.GlobalColor.white)
        
        # Create painter for the final image
        painter = QPainter(final_image)
        
        # Draw the title at the top
        painter.drawImage(0, 0, title_image)
        
        # Draw the plot below the title
        painter.drawImage(0, title_height, plot_image)
        
        # Draw the scaling label below the plot if it exists
        if scale_image:
            painter.drawImage(0, title_height + plot_height, scale_image)
        
        # Create custom legend image with three-column layout
        legend_width = 300
        legend_image = QImage(legend_width, final_height, QImage.Format.Format_ARGB32)
        legend_image.fill(Qt.GlobalColor.white)
        legend_painter = QPainter(legend_image)
        
        # Set font for legend
        legend_font = QFont()
        legend_font.setPointSize(10)
        legend_painter.setFont(legend_font)
        
        # Draw a border around the legend
        legend_painter.setPen(QPen(QColor(200, 200, 200), 1))
        legend_painter.drawRect(0, 0, legend_width-1, final_height-1)
        
        # Draw column headers
        legend_y = 25
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        legend_painter.setFont(header_font)
        
        # Draw header columns
        obs_col_width = 80
        sim_col_width = 80
        trt_col_width = legend_width - obs_col_width - sim_col_width
        
        # Draw header texts
        legend_painter.drawText(
            QRect(5, 5, obs_col_width-5, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Observed"
        )
        legend_painter.drawText(
            QRect(obs_col_width+5, 5, sim_col_width-5, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Simulated"
        )
        legend_painter.drawText(
            QRect(obs_col_width+sim_col_width+5, 5, trt_col_width-10, 20),
            Qt.AlignmentFlag.AlignLeft,
            "Treatment"
        )
        
        # Draw separator line below headers
        legend_painter.setPen(QPen(QColor(100, 100, 100), 1))
        legend_painter.drawLine(5, legend_y, legend_width-5, legend_y)
        legend_y += 15
        
        # Organize items by variable first
        variables = set()
        for item, metadata in plot_items_metadata:
            if 'variable' in metadata:
                variables.add(metadata['variable'])
        
        # Define the mapping of variables to line styles for simulated data
        line_styles = [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, 
                       Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine]
        
        var_line_styles = {}
        var_idx = 0
        
        # First pass to identify variables and assign line styles
        for var in variables:
            var_line_styles[var] = line_styles[var_idx % len(line_styles)]
            var_idx += 1
        
        # Create a mapping of treatments by variable
        var_treatments = {}
        for var in variables:
            var_treatments[var] = {}
            
            # First pass to gather metadata
            for item, metadata in plot_items_metadata:
                if metadata.get('variable') == var:
                    source = metadata.get('source', '')
                    trt_id = metadata.get('treatment', '')
                    
                    if trt_id not in var_treatments[var]:
                        var_treatments[var][trt_id] = {
                            "name": metadata.get('treatment_name', trt_id),
                            "sim_item": None,
                            "sim_meta": None,
                            "obs_item": None,
                            "obs_meta": None
                        }
                    
                    if source == 'sim':
                        var_treatments[var][trt_id]["sim_item"] = item
                        var_treatments[var][trt_id]["sim_meta"] = metadata
                    elif source == 'obs':
                        var_treatments[var][trt_id]["obs_item"] = item
                        var_treatments[var][trt_id]["obs_meta"] = metadata
        
        # Now draw each variable and its treatments
        normal_font = QFont()
        normal_font.setPointSize(10)
        bold_font = QFont()
        bold_font.setPointSize(10)
        bold_font.setBold(True)
        
        for var in sorted(variables):
            # Draw variable name
            legend_painter.setFont(bold_font)
            var_label, _ = get_variable_info_func(var)
            display_name = var_label or var
            
            legend_painter.setPen(QPen(QColor(0, 0, 0)))
            legend_painter.drawText(10, legend_y, display_name)
            legend_y += 20
            
            # Draw treatments for this variable
            legend_painter.setFont(normal_font)
            
            for trt_id, trt_data in sorted(var_treatments[var].items()):
                row_height = 20
                row_y = legend_y
                
                # Draw observed column
                if trt_data["obs_item"] is not None and isinstance(trt_data["obs_item"], pg.ScatterPlotItem):
                    item = trt_data["obs_item"]
                    symbol = item.opts.get('symbol', 'o')
                    
                    # Extract the actual color
                    brush = item.opts.get('brush')
                    if brush is not None:
                        color = QColor(brush) if not isinstance(brush, QBrush) else brush.color()
                    else:
                        pen = item.opts.get('pen')
                        if pen is not None:
                            color = QColor(pen) if not isinstance(pen, QPen) else pen.color()
                        else:
                            color = QColor(0, 0, 0)
                    
                    # Set pen and brush for drawing
                    legend_painter.setPen(QPen(color, 1.5))
                    legend_painter.setBrush(QBrush(color))
                    
                    # Draw the appropriate symbol
                    symbol_size = 4
                    center_x = obs_col_width / 2
                    center_y = row_y - row_height / 2
                    
                    if symbol == 'o':
                        # Circle
                        legend_painter.drawEllipse(QPoint(int(center_x), int(center_y)), 
                                                 symbol_size, symbol_size)
                    elif symbol == 's':
                        # Square
                        legend_painter.drawRect(int(center_x - symbol_size), int(center_y - symbol_size), 
                                              symbol_size*2, symbol_size*2)
                    elif symbol == 't':
                        # Triangle
                        points = [
                            QPoint(int(center_x), int(center_y - symbol_size)),
                            QPoint(int(center_x - symbol_size), int(center_y + symbol_size)),
                            QPoint(int(center_x + symbol_size), int(center_y + symbol_size))
                        ]
                        legend_painter.drawPolygon(QPolygon(points))
                    elif symbol == 'd':
                        # Diamond
                        points = [
                            QPoint(int(center_x), int(center_y - symbol_size)),
                            QPoint(int(center_x + symbol_size), int(center_y)),
                            QPoint(int(center_x), int(center_y + symbol_size)),
                            QPoint(int(center_x - symbol_size), int(center_y))
                        ]
                        legend_painter.drawPolygon(QPolygon(points))
                    elif symbol == '+':
                        # Plus
                        legend_painter.drawLine(int(center_x - symbol_size), int(center_y), 
                                              int(center_x + symbol_size), int(center_y))
                        legend_painter.drawLine(int(center_x), int(center_y - symbol_size), 
                                              int(center_x), int(center_y + symbol_size))
                    elif symbol == 'x':
                        # X
                        legend_painter.drawLine(int(center_x - symbol_size), int(center_y - symbol_size), 
                                              int(center_x + symbol_size), int(center_y + symbol_size))
                        legend_painter.drawLine(int(center_x - symbol_size), int(center_y + symbol_size), 
                                              int(center_x + symbol_size), int(center_y - symbol_size))
                    else:
                        # Default to circle
                        legend_painter.drawEllipse(QPoint(int(center_x), int(center_y)), 
                                                 symbol_size, symbol_size)
                else:
                    # Draw placeholder dash for missing observed data
                    legend_painter.setPen(QPen(QColor(180, 180, 180)))
                    legend_painter.drawText(
                        QRect(5, row_y-row_height, obs_col_width-5, row_height),
                        Qt.AlignmentFlag.AlignCenter,
                        "-"
                    )
                
                # Draw simulated column
                if trt_data["sim_item"] is not None and isinstance(trt_data["sim_item"], pg.PlotDataItem):
                    item = trt_data["sim_item"]
                    pen = item.opts.get('pen')
                    if pen is not None:
                        if isinstance(pen, QPen):
                            color = pen.color()
                        else:
                            color = QColor(pen)
                    else:
                        color = QColor(0, 0, 0)
                    
                    # Use the assigned line style for this variable
                    pen_style = var_line_styles.get(var, Qt.PenStyle.SolidLine)
                    
                    # Draw the line with correct style
                    legend_painter.setPen(QPen(color, 2, pen_style))
                    center_y = row_y - row_height / 2
                    legend_painter.drawLine(
                        int(obs_col_width + 15), int(center_y),
                        int(obs_col_width + sim_col_width - 15), int(center_y)
                    )
                else:
                    # Draw placeholder dash for missing simulated data
                    legend_painter.setPen(QPen(QColor(180, 180, 180)))
                    legend_painter.drawText(
                        QRect(obs_col_width+5, row_y-row_height, sim_col_width-5, row_height),
                        Qt.AlignmentFlag.AlignCenter,
                        "-"
                    )
                
                # Draw treatment name
                legend_painter.setPen(QPen(QColor(0, 0, 0)))
                legend_painter.drawText(
                    QRect(obs_col_width+sim_col_width+5, row_y-row_height, trt_col_width-10, row_height),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    trt_data["name"]
                )
                
                legend_y += row_height
            
            # Add separator between variables except after the last one
            if var != sorted(variables)[-1]:
                legend_painter.setPen(QPen(QColor(220, 220, 220), 1))
                legend_painter.drawLine(10, legend_y, legend_width-10, legend_y)
                legend_y += 10
        
        # Draw the custom legend to the right of the plot
        painter.drawImage(plot_width, 0, legend_image)
        
        # End painting
        legend_painter.end()
        painter.end()
        
        # Save the final image
        success = final_image.save(file_path, format=file_format, quality=200)
        
        # Remove temporary file
        if os.path.exists(temp_plot_path):
            os.remove(temp_plot_path)
        
        if success:
            logger.info(f"Plot exported successfully to {file_path}")
            return True
        else:
            logger.error(f"Failed to save image to {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False