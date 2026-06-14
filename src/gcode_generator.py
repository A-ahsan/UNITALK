"""
Generate G-code from braille dot coordinates.
"""

from typing import List, Tuple
from braille_config import GCODE_CONFIG


class GcodeGenerator:
    """Generate G-code commands from braille dot coordinates."""

    def __init__(self, config: dict = None):
        """
        Initialize with G-code configuration.
        
        Args:
            config: Override default GCODE_CONFIG
        """
        self.config = GCODE_CONFIG.copy()
        if config:
            self.config.update(config)

    def generate_header(self) -> List[str]:
        """Generate G-code header (setup commands)."""
        offset_x = float(self.config.get("pre_start_offset_x", 0.0))
        offset_y = float(self.config.get("pre_start_offset_y", 0.0))
        move_to_start = abs(offset_x) > 1e-6 or abs(offset_y) > 1e-6
        return [
            "G21",  # Millimeters
            "G90",  # Absolute positioning
            f"F{self.config['feed_rate']}",  # Feed rate
            "",  # Spacer
            "; Prime solenoid (M9=down, M8=up)",
            "M8",
            f"G4 P{self.config['dwell_time']}",
            "M9",
            f"G4 P{self.config['dwell_time']}",
            "M8",
            f"G4 P{self.config['dwell_time']}",
            "" if not move_to_start else "; Shift to next cell from prime dot",
            "" if not move_to_start else "G91",
            "" if not move_to_start else f"G0 X{offset_x:.1f} Y{offset_y:.1f}",
            "" if not move_to_start else "G92 X0 Y0",
            "" if not move_to_start else "G90",
            "",
        ]

    def generate_footer(self) -> List[str]:
        """Generate G-code footer (shutdown commands)."""
        return [
            "",
            "G0 X0 Y0",  # Return to origin
        ]

    def generate_dot_gcode(self, x: float, y: float, dwell_time: float) -> List[str]:
        """
        Generate G-code for a single dot tap.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of G-code lines
        """
        commands = [
            f"G0 X{x:.1f} Y{y:.1f}",           # Move to position
            "M9",                               # Pen down
            f"G4 P{dwell_time}",               # Dwell
            "M8",                               # Pen up
            f"G4 P{dwell_time}",               # Dwell
        ]
        return commands

    def coordinates_to_gcode(self, dot_coordinates: List[List[Tuple[float, float]]]) -> List[str]:
        """
        Convert braille dot coordinates to complete G-code.
        
        Args:
            dot_coordinates: List of dot lists per character
                            e.g., [[(20.0, 20.0), (20.0, 22.5)], ...]
            
        Returns:
            List of G-code commands
        """
        gcode = self.generate_header()
        
        # Group dots by character (with comments)
        all_dots = []
        for dots in dot_coordinates:
            for x, y in dots:
                all_dots.append((x, y))

        edge_multiplier = float(self.config.get("edge_dwell_multiplier", 1.0))
        base_dwell = float(self.config["dwell_time"])

        for char_idx, dots in enumerate(dot_coordinates):
            if dots:  # Only if character has dots
                gcode.append("")
                gcode.append(f"; Character {char_idx + 1}")
                
                for x, y in dots:
                    dwell_time = base_dwell
                    if all_dots and ((x, y) == all_dots[0] or (x, y) == all_dots[-1]):
                        dwell_time = round(base_dwell * edge_multiplier, 3)
                    gcode.extend(self.generate_dot_gcode(x, y, dwell_time))
        
        gcode.extend(self.generate_footer())
        return gcode

    def gcode_to_string(self, gcode: List[str]) -> str:
        """Convert G-code list to single string."""
        return "\n".join(gcode)

    def validate_coordinates(self, dot_coordinates: List[List[Tuple[float, float]]]) -> Tuple[bool, str]:
        """
        Validate G-code coordinates.
        
        Args:
            dot_coordinates: Coordinates to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not dot_coordinates:
            return False, "No coordinates provided"
        
        total_dots = sum(len(dots) for dots in dot_coordinates)
        if total_dots == 0:
            return False, "No dots to print"
        
        # Check for reasonable coordinate ranges
        for char_dots in dot_coordinates:
            for x, y in char_dots:
                if x < 0 or x > 500 or y < 0 or y > 500:
                    return False, f"Coordinate out of range: ({x}, {y})"
        
        return True, "OK"


if __name__ == "__main__":
    gen = GcodeGenerator()
    
    # Test coordinates (single "A" = dot 1 only)
    test_coords = [
        [(20.0, 20.0)],  # Character 1: single dot at (20, 20)
    ]
    
    is_valid, msg = gen.validate_coordinates(test_coords)
    print(f"Validate: {is_valid} - {msg}")
    
    gcode = gen.coordinates_to_gcode(test_coords)
    gcode_str = gen.gcode_to_string(gcode)
    print("Generated G-code:")
    print(gcode_str)
