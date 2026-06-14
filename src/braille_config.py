"""
Braille printer configuration and constants.
"""

# Braille cell dot positions (standard 6-dot braille)
# Layout:
#   1 4
#   2 5
#   3 6
# Coordinates relative to cell origin (X=0, Y=0)

BRAILLE_DOT_POSITIONS = {
    1: (0.0, 0.0),      # top-left
    2: (0.0, 2.5),      # middle-left
    3: (0.0, 5.0),      # bottom-left
    4: (2.5, 0.0),      # top-right
    5: (2.5, 2.5),      # middle-right
    6: (2.5, 5.0),      # bottom-right
}

# Standard braille character mappings (dot sets)
BRAILLE_PATTERNS = {
    'A': {1},
    'B': {1, 2},
    'C': {1, 4},
    'D': {1, 4, 5},
    'E': {1, 5},
    'F': {1, 2, 4},
    'G': {1, 2, 4, 5},
    'H': {1, 2, 5},
    'I': {2, 4},
    'J': {2, 4, 5},
    'K': {1, 3},
    'L': {1, 2, 3},
    'M': {1, 3, 4},
    'N': {1, 3, 4, 5},
    'O': {1, 3, 5},
    'P': {1, 2, 3, 4},
    'Q': {1, 2, 3, 4, 5},
    'R': {1, 2, 3, 5},
    'S': {2, 3, 4},
    'T': {2, 3, 4, 5},
    'U': {1, 3, 6},
    'V': {1, 2, 3, 6},
    'W': {2, 4, 5, 6},
    'X': {1, 3, 4, 6},
    'Y': {1, 3, 4, 5, 6},
    'Z': {1, 3, 5, 6},
    ' ': set(),          # space = no dots
    'DELETE': set(),     # delete = no dots (handled separately)
}

# G-code generator parameters (from br.txt analysis)
GCODE_CONFIG = {
    'feed_rate': 500,           # F500
    'spindle_speed': 90,        # Unused for solenoid (kept for compatibility)
    'dwell_time': 0.6,          # P0.6 (seconds)
    'edge_dwell_multiplier': 1.0,  # Same dwell for all dots
    'dot_spacing_y': 2.5,       # Y spacing between rows
    'dot_spacing_x': 2.5,       # X spacing between columns
    'cell_spacing_x': 6.0,      # X spacing between cells (letters)
    'start_x': 0.0,             # Starting X coordinate (relative to primed position)
    'start_y': 0.0,             # Starting Y coordinate (relative to primed position)
    'pre_start_offset_x': 6.0,  # Move right one cell before printing
    'pre_start_offset_y': 6,  # Move down before printing
    'cell_width': 2.5,          # Width of single cell
    'cell_height': 5.0,         # Height of single cell
}

# Serial communication
SERIAL_CONFIG = {
    'baud_rate': 115200,
    'timeout': 1.0,
    'preferred_ports': ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1'],
}

# Canvas visualization
CANVAS_CONFIG = {
    'scale': 5,                 # pixels per mm
    'dot_radius': 4,            # pixels
    'cell_spacing_visual': 50,  # pixels between cells
}
