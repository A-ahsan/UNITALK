# Braille Printer Control System

A complete system to convert text and sign language to braille and print it using an Arduino-controlled plotter.

## Features

- **Text-to-Braille**: Type text (up to 10 characters) and convert to braille for printing
- **Sign-to-Braille**: Capture live sign language using your IP camera, buffer recognized signs, and convert to braille
- **G-code Generation**: Automatically generates precision G-code matching your plotter specifications
- **2D Preview**: Visualize braille dot placement before printing
- **Serial Communication**: Auto-detects Arduino/plotter and sends G-code reliably
- **PyQt5 GUI**: User-friendly interface with tabbed controls

## Hardware Requirements

- Raspberry Pi 5 (or Linux/Mac with Python 3.12+)
- Arduino with G-code firmware (tested with Universal G-code Sender)
- USB plotter/engraver (connected to Arduino)
- IP camera or USB webcam for sign detection
- USB serial cable for Arduino connection

## Installation

### 1. Install Dependencies

```bash
cd /path/to/sg
source .venv312/bin/activate
pip install -r requirements.txt
```

### 2. Verify Hardware

Check serial ports and camera connection:

```bash
# List serial ports
ls /dev/ttyACM* /dev/ttyUSB*

# Test IP camera (if using IP camera)
curl -v https://192.168.0.100:8080
```

## Usage

### Run the GUI Application

```bash
cd /home/hopeless/sg
source .venv312/bin/activate
DISPLAY=${DISPLAY:-:0} python src/braille_gui.py
```

The GUI will open with three tabs:

#### Tab 1: Text → Braille
1. Type text (up to 10 characters) in the input field
2. Click **Preview G-code** to see the braille dot layout
3. Click **Print** to send to the plotter

#### Tab 2: Sign → Braille
1. Configure camera URL in Settings tab
2. Click **Start Capture** to begin sign detection
3. Show signs to camera (1-10 signs maximum)
4. Detected signs appear in real-time with confidence scores
5. Click **Preview G-code** to visualize
6. Click **Print** to send to plotter

#### Tab 3: Settings
- **Serial Connection**: Auto-detect or manually select Arduino port
- **Connection Test**: Verify Arduino is responding
- **Camera URL**: IP camera stream URL (default: `https://192.168.0.100:8080`)
- **Min Confidence**: Threshold for sign detection (default: 0.60)

## Module Reference

### `braille_config.py`
Configuration constants:
- Braille dot positions (standard 6-dot layout)
- G-code parameters (from `br.txt` reference)
- Serial communication settings
- Canvas visualization settings

### `braille_converter.py`
Convert text to braille:
```python
from braille_converter import BrailleConverter

converter = BrailleConverter()
text = "HELLO"
braille_patterns = converter.text_to_braille(text)
coordinates = converter.text_to_gcode_coordinates(text)
```

### `gcode_generator.py`
Generate G-code from coordinates:
```python
from gcode_generator import GcodeGenerator

gen = GcodeGenerator()
gcode_lines = gen.coordinates_to_gcode(coordinates)
gcode_string = gen.gcode_to_string(gcode_lines)
```

### `serial_sender.py`
Send G-code to Arduino:
```python
from serial_sender import SerialSender

sender = SerialSender()
success, msg = sender.connect()  # Auto-detect or specify port
success, msg, lines_sent = sender.send_gcode(gcode_lines)
```

### `sign_to_braille_bridge.py`
Capture signs in real-time:
```python
from sign_to_braille_bridge import SignToBrailleCapture

capture = SignToBrailleCapture(camera_url="https://192.168.0.100:8080")

def on_sign(sign, confidence):
    print(f"Detected: {sign} ({confidence:.2f})")

capture.on_sign_detected = on_sign
capture.start()
```

### `braille_gui.py`
PyQt5 GUI application with integrated controls.

## Testing

Run the end-to-end workflow test:

```bash
cd /home/hopeless/sg
source .venv312/bin/activate
python test_workflow.py
```

This verifies:
- Text validation and braille conversion
- G-code generation
- Serial port detection
- Complete pipeline

## Braille Reference

Braille uses 6 dots arranged in 2 columns × 3 rows:

```
1 4
2 5
3 6
```

Character mappings follow international unified English braille standards.

### Supported Characters

- Letters: A–Z (all 26)
- Space (blank cell)
- DELETE (special, no dots)
- Numbers and punctuation: Currently letters + space only

## G-code Specifications

Generated G-code matches your plotter requirements:
- Units: Millimeters (G21)
- Positioning: Absolute (G90)
- Feed Rate: 300 mm/min
- Spindle Speed: 90 (adjustable)
- Dwell Time: 0.4 seconds per dot

Each dot is created by:
1. Move to position (G0)
2. Stop spindle (M5)
3. Dwell (G4 P0.4)
4. Start spindle (M3 S90)
5. Dwell (G4 P0.4)

## Configuration Files

### `br.txt`
Sample G-code reference for plotter sizing and specifications.

### `models/` directory
- `sign_model_tasks_v6_combo.joblib` - Primary sign detection model
- `hand_landmarker.task` - MediaPipe hand landmark model (auto-downloaded)

### `data/` directory
Braille landmark datasets for training/reference.

## Troubleshooting

### Serial Connection Issues
```bash
# Check if Arduino appears
ls -la /dev/ttyACM* /dev/ttyUSB*

# Set permissions (if needed)
sudo usermod -a -G dialout $USER
```

### Camera Connection Issues
```bash
# Test IP camera reachability
curl -v https://192.168.0.100:8080

# Verify camera is on same network
ping 192.168.0.100
```

### Sign Detection Not Working
- Increase `--min-confidence` threshold in Settings
- Ensure good lighting and clear hand visibility
- Check that trained model exists in `models/`

### G-code Not Sending
- Verify Arduino is connected and powered
- Check baud rate (should be 115200)
- Use Universal G-code Sender to test connection separately
- Check Arduino firmware supports standard G-code

## Advanced Usage

### Custom G-code Parameters

Modify `braille_config.py`:

```python
GCODE_CONFIG = {
    'feed_rate': 300,        # Adjust speed
    'spindle_speed': 90,     # Adjust pressure
    'dwell_time': 0.4,       # Adjust dot duration
    'cell_spacing_x': 10.0,  # Letter spacing
    # ... other parameters
}
```

### Custom Braille Patterns

Add new characters in `braille_config.py`:

```python
BRAILLE_PATTERNS = {
    'A': {1},
    'B': {1, 2, 4},
    # Add more: 'YOUR_CHAR': {dot_numbers}
}
```

### Batch Processing

```python
from braille_converter import BrailleConverter
from gcode_generator import GcodeGenerator
from serial_sender import SerialSender

converter = BrailleConverter()
gen = GcodeGenerator()
sender = SerialSender()

texts = ["HELLO", "BRAILLE", "PRINTER"]

sender.connect()
for text in texts:
    coords = converter.text_to_gcode_coordinates(text)
    gcode = gen.coordinates_to_gcode(coords)
    sender.send_gcode(gcode)
    print(f"Printed: {text}")
```

## File Structure

```
/home/hopeless/sg/
├── src/
│   ├── braille_config.py         # Configuration constants
│   ├── braille_converter.py       # Text → Braille
│   ├── gcode_generator.py         # Braille → G-code
│   ├── serial_sender.py           # Serial communication
│   ├── sign_to_braille_bridge.py  # Real-time sign capture
│   ├── braille_gui.py             # PyQt5 GUI (main app)
│   ├── common.py                  # Existing sign recognition utilities
│   ├── realtime_inference.py      # Existing sign detection
│   └── ...
├── models/
│   ├── sign_model_tasks_v6_combo.joblib  # Trained model
│   ├── hand_landmarker.task               # MediaPipe model
│   └── ...
├── data/                          # Landmark datasets
├── br.txt                         # G-code reference sample
├── requirements.txt               # Python dependencies
├── test_workflow.py               # End-to-end test
└── README.md                      # This file
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review generated G-code with Universal G-code Sender
3. Test serial connection separately
4. Verify camera URL and credentials

## License

This project integrates with your existing sign language recognition system.

## Changelog

### v1.0 (Initial Release)
- Full text-to-braille conversion
- Real-time sign-to-braille capture
- G-code generation and preview
- PyQt5 GUI with dual modes
- Serial communication with auto-detect
- Support for A–Z, SPACE, DELETE
- 2D braille preview canvas
