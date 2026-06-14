"""
Convert text to braille dot patterns.
"""

from typing import List, Set, Dict, Tuple, Optional
from braille_config import BRAILLE_PATTERNS, BRAILLE_DOT_POSITIONS, GCODE_CONFIG


class BrailleConverter:
    """Convert text strings to braille dot patterns."""

    def __init__(self):
        self.patterns = BRAILLE_PATTERNS
        self.dot_positions = BRAILLE_DOT_POSITIONS

    def text_to_braille(self, text: str) -> List[Set[int]]:
        """
        Convert text to list of braille dot sets.
        
        Args:
            text: Input text (uppercase letters, spaces)
            
        Returns:
            List of dot sets, one per character
        """
        text = text.upper().strip()
        braille_chars = []
        
        for char in text:
            if char in self.patterns:
                braille_chars.append(self.patterns[char])
            else:
                # Unknown character: treat as space
                braille_chars.append(set())
        
        return braille_chars

    def get_dot_coordinates(self, dots: Set[int], cell_x: float, cell_y: float) -> List[Tuple[float, float]]:
        """
        Convert dot set to absolute coordinates.
        
        Args:
            dots: Set of dot numbers (1-6)
            cell_x: X coordinate of cell origin
            cell_y: Y coordinate of cell origin
            
        Returns:
            List of (x, y) coordinates for each dot
        """
        coords = []
        for dot in sorted(dots):
            if dot in self.dot_positions:
                dx, dy = self.dot_positions[dot]
                coords.append((cell_x + dx, cell_y + dy))
        return coords

    def validate_text(self, text: str) -> Tuple[bool, str]:
        """
        Validate if text can be converted to braille.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not text:
            return False, "Text is empty"
        
        if len(text) > 10:
            return False, "Text must be 10 characters or fewer"
        
        unsupported = set()
        for char in text.upper():
            if char not in self.patterns and char != ' ':
                unsupported.add(char)
        
        if unsupported:
            return False, f"Unsupported characters: {', '.join(sorted(unsupported))}"
        
        return True, "OK"

    def text_to_gcode_coordinates(
        self,
        text: str,
        start_x: Optional[float] = None,
        start_y: Optional[float] = None,
        cell_spacing: Optional[float] = None,
    ) -> List[List[Tuple[float, float]]]:
        """
        Convert text to absolute braille dot coordinates (gcode positions).
        
        Args:
            text: Input text
            start_x: Starting X coordinate (defaults to config)
            start_y: Starting Y coordinate (defaults to config)
            cell_spacing: Horizontal spacing between cells (defaults to config)
            
        Returns:
            List of dot coordinate lists, one list per character
        """
        if start_x is None:
            start_x = float(GCODE_CONFIG.get("start_x", 20.0))
        if start_y is None:
            start_y = float(GCODE_CONFIG.get("start_y", 20.0))
        if cell_spacing is None:
            cell_spacing = float(GCODE_CONFIG.get("cell_spacing_x", 10.0))

        braille_chars = self.text_to_braille(text)
        coordinates = []
        
        for idx, dots in enumerate(braille_chars):
            cell_x = start_x + (idx * cell_spacing)
            cell_y = start_y
            coords = self.get_dot_coordinates(dots, cell_x, cell_y)
            coordinates.append(coords)
        
        return coordinates


if __name__ == "__main__":
    converter = BrailleConverter()
    
    # Test
    test_text = "HELLO"
    is_valid, msg = converter.validate_text(test_text)
    print(f"Validate '{test_text}': {is_valid} - {msg}")
    
    braille = converter.text_to_braille(test_text)
    print(f"Braille patterns: {braille}")
    
    coords = converter.text_to_gcode_coordinates(test_text)
    print(f"Coordinates: {coords}")
