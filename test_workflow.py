"""
End-to-end test of braille printer workflow.
Tests: text → braille → G-code → preview
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from braille_converter import BrailleConverter
from gcode_generator import GcodeGenerator
from serial_sender import SerialSender


def test_full_workflow():
    """Test complete braille printing workflow."""
    
    print("=" * 60)
    print("BRAILLE PRINTER WORKFLOW TEST")
    print("=" * 60)
    
    # Initialize modules
    converter = BrailleConverter()
    gcode_gen = GcodeGenerator()
    serial = SerialSender()
    
    # Test 1: Text validation and conversion
    print("\n1. TEXT VALIDATION & CONVERSION")
    print("-" * 60)
    
    test_texts = ["HELLO", "AI LOVE YOU", "ALIF LOVES URMI", "INVALID_LONG_TEXT_OVER_10_CHARS"]
    
    for text in test_texts:
        is_valid, msg = converter.validate_text(text)
        if is_valid:
            braille = converter.text_to_braille(text)
            print(f"✓ '{text}' → {len(braille)} characters")
        else:
            print(f"✗ '{text}' → {msg}")
    
    # Test 2: G-code generation
    print("\n2. G-CODE GENERATION")
    print("-" * 60)
    
    text = "HELLO"
    coords = converter.text_to_gcode_coordinates(text)
    is_valid, msg = gcode_gen.validate_coordinates(coords)
    
    if is_valid:
        gcode = gcode_gen.coordinates_to_gcode(coords)
        print(f"✓ Generated {len(gcode)} G-code lines for '{text}'")
        
        # Show first few lines
        print("\nFirst 15 lines of G-code:")
        for i, line in enumerate(gcode[:15]):
            print(f"  {i+1}: {line}")
    else:
        print(f"✗ G-code validation failed: {msg}")
    
    # Test 3: Serial port detection
    print("\n3. SERIAL PORT DETECTION")
    print("-" * 60)
    
    ports = serial.list_ports()
    if ports:
        print(f"✓ Found {len(ports)} serial port(s): {ports}")
        
        auto_port = serial.auto_detect_port()
        if auto_port:
            print(f"✓ Auto-detected port: {auto_port}")
        else:
            print("✗ Could not auto-detect port")
    else:
        print("✗ No serial ports found")
    
    # Test 4: Complete pipeline
    print("\n4. COMPLETE PIPELINE TEST")
    print("-" * 60)
    
    test_text = "BRAILLE"
    is_valid, msg = converter.validate_text(test_text)
    
    if is_valid:
        coords = converter.text_to_gcode_coordinates(test_text)
        is_valid_coord, msg_coord = gcode_gen.validate_coordinates(coords)
        
        if is_valid_coord:
            gcode = gcode_gen.coordinates_to_gcode(coords)
            gcode_str = gcode_gen.gcode_to_string(gcode)
            
            print(f"✓ Complete pipeline successful for '{test_text}'")
            print(f"  - {len(coords)} characters")
            print(f"  - {sum(len(c) for c in coords)} total dots")
            print(f"  - {len(gcode)} G-code lines")
            print(f"\nGenerated G-code sample ({len(gcode_str)} bytes):")
            print(gcode_str[:300] + "...")
        else:
            print(f"✗ Coordinate validation failed: {msg_coord}")
    else:
        print(f"✗ Text validation failed: {msg}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()
