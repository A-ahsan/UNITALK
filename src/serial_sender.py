"""
Serial communication with Arduino/plotter.
"""

import time
import serial
from pathlib import Path
from typing import List, Optional, Tuple
from braille_config import SERIAL_CONFIG


class SerialSender:
    """Send G-code commands to Arduino via serial."""

    def __init__(self, port: Optional[str] = None, baud: int = 115200, timeout: float = 1.0):
        """
        Initialize serial sender.
        
        Args:
            port: Serial port (auto-detect if None)
            baud: Baud rate
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.is_connected = False
        self.first_job = True

    def auto_detect_port(self) -> Optional[str]:
        """Auto-detect Arduino/plotter serial port."""
        try:
            import serial.tools.list_ports
        except ImportError:
            return None
        
        # Try preferred ports first
        for port in SERIAL_CONFIG['preferred_ports']:
            if Path(port).exists():
                return port
        
        # Scan all ports
        for port, desc, hwid in serial.tools.list_ports.comports():
            # Look for Arduino/USB device indicators
            if any(x in desc.lower() for x in ['arduino', 'usb', 'ch340', 'cp210x']):
                return port
        
        # Fallback: return first available port
        ports = serial.tools.list_ports.comports()
        if ports:
            return ports[0][0]
        
        return None

    def connect(self, port: Optional[str] = None) -> Tuple[bool, str]:
        """
        Connect to serial port.
        
        Args:
            port: Port to connect to (uses self.port if None)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if port:
                self.port = port
            elif not self.port:
                self.port = self.auto_detect_port()
            
            if not self.port:
                return False, "No serial port found"
            
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2.0)  # Give GRBL time to reset
            self._wake_grbl()
            warm_ok, warm_msg = self.warmup()
            self.is_connected = True
            self.first_job = True
            if warm_ok:
                return True, f"Connected to {self.port} @ {self.baud} baud"
            return True, f"Connected to {self.port} @ {self.baud} baud (Warmup: {warm_msg})"
        
        except serial.SerialException as e:
            self.is_connected = False
            return False, f"Serial error: {str(e)}"
        except Exception as e:
            self.is_connected = False
            return False, f"Error: {str(e)}"

    def disconnect(self) -> None:
        """Disconnect from serial port."""
        if self.ser and self.is_connected:
            try:
                self.ser.close()
            except Exception:
                pass
        self.is_connected = False

    def send_gcode(self, gcode_lines: List[str], verbose: bool = False) -> Tuple[bool, str, int]:
        """
        Send G-code lines to plotter.
        
        Args:
            gcode_lines: List of G-code commands
            verbose: Print each line sent
            
        Returns:
            Tuple of (success, message, lines_sent)
        """
        if not self.is_connected:
            return False, "Not connected", 0
        
        if not gcode_lines:
            return False, "No G-code to send", 0
        
        try:
            ready_ok, ready_msg = self._ensure_ready()
            if not ready_ok:
                return False, ready_msg, 0
            if self.first_job:
                prime_ok, prime_msg = self._prime_first_job()
                if not prime_ok:
                    return False, prime_msg, 0
                self.first_job = False
            lines_sent = 0
            for line in gcode_lines:
                if line.strip() and not line.strip().startswith(';'):
                    line = line.rstrip()
                    ok = self._send_line_wait_ok(line, verbose=verbose)
                    if not ok:
                        return False, f"GRBL did not acknowledge: {line}", lines_sent
                    lines_sent += 1
            
            # Wait a bit for plotter to finish
            time.sleep(0.5)
            return True, f"Sent {lines_sent} commands", lines_sent
        
        except serial.SerialException as e:
            self.is_connected = False
            return False, f"Serial error: {str(e)}", 0
        except Exception as e:
            return False, f"Error: {str(e)}", 0

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection with a simple command."""
        try:
            ok = self._send_line_wait_ok("$G")
            return (True, "Connection test OK") if ok else (False, "No response from GRBL")
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def pen_up(self) -> Tuple[bool, str]:
        """Lift the pen (M8)."""
        if not self.is_connected:
            return False, "Not connected"
        ok = self._send_line_wait_ok("M8")
        return (True, "Pen up") if ok else (False, "Pen up failed")

    def pen_down(self) -> Tuple[bool, str]:
        """Press the pen down (M9)."""
        if not self.is_connected:
            return False, "Not connected"
        ok = self._send_line_wait_ok("M9")
        return (True, "Pen down") if ok else (False, "Pen down failed")

    def unlock(self) -> Tuple[bool, str]:
        """Unlock GRBL ($X)."""
        if not self.is_connected:
            return False, "Not connected"
        ok = self._send_line_wait_ok("$X")
        return (True, "Unlocked") if ok else (False, "Unlock failed")

    def list_ports(self) -> List[str]:
        """List available serial ports."""
        try:
            import serial.tools.list_ports
            return [port for port, _, _ in serial.tools.list_ports.comports()]
        except ImportError:
            return []

    def _wake_grbl(self) -> None:
        """Wake GRBL and drain startup lines (if present)."""
        if not self.ser:
            return
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.write(b"\r\n\r\n")
            time.sleep(2.0)
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def _send_line_wait_ok(self, line: str, verbose: bool = False, timeout_s: Optional[float] = None) -> bool:
        """Send a line and wait for GRBL ok/error."""
        if not self.ser:
            return False
        self.ser.write((line + "\n").encode("utf-8"))
        if verbose:
            print(f">> {line}")
        timeout = max(1.0, (timeout_s if timeout_s is not None else self.timeout * 3))
        start = time.time()
        while time.time() - start < timeout:
            resp = self.ser.readline().decode("utf-8", errors="ignore").strip().lower()
            if not resp:
                continue
            if resp.startswith("grbl"):
                continue
            if resp.startswith("<"):
                continue
            if resp.startswith("["):
                continue
            if resp.startswith("ok"):
                return True
            if resp.startswith("error"):
                return False
        return False

    def _ensure_ready(self) -> Tuple[bool, str]:
        """Check GRBL state before sending G-code."""
        if not self.ser:
            return False, "Not connected"
        try:
            self._wake_grbl()
            self.ser.write(b"?\n")
            time.sleep(0.2)
            status = self._read_status_line(timeout_s=1.5)
            if not status:
                return False, "No status from GRBL"
            if status.startswith("<alarm"):
                return False, "GRBL in ALARM. Run Home ($H) first."
            return True, "OK"
        except Exception as e:
            return False, f"GRBL status error: {e}"

    def _read_status_line(self, timeout_s: float = 1.0) -> Optional[str]:
        start = time.time()
        while time.time() - start < timeout_s:
            resp = self.ser.readline().decode("utf-8", errors="ignore").strip().lower()
            if not resp:
                continue
            if resp.startswith("<"):
                return resp
        return None

    def _prime_first_job(self) -> Tuple[bool, str]:
        """Extra sync before the first print after connect."""
        if not self.ser:
            return False, "Not connected"
        if not self._send_line_wait_ok("G4 P0"):
            return False, "Prime failed (G4 P0)"
        time.sleep(0.2)
        if not self._send_line_wait_ok("G4 P0"):
            return False, "Prime failed (G4 P0)"
        self.ser.write(b"?\n")
        time.sleep(0.2)
        status = self._read_status_line(timeout_s=1.5)
        if not status:
            return False, "No status after prime"
        if status.startswith("<alarm"):
            return False, "GRBL in ALARM. Run Home ($H) first."
        return True, "Primed"

    def warmup(self) -> Tuple[bool, str]:
        """Send harmless commands to avoid losing the first real command."""
        if not self.ser:
            return False, "Not connected"
        try:
            # Ensure we have a clean state before printing the first job.
            if not self._send_line_wait_ok("G4 P0"):
                return False, "Warmup failed (G4 P0) - run Home"
            if not self._send_line_wait_ok("G90"):
                return False, "Warmup failed (G90) - run Home"
            if not self._send_line_wait_ok("G21"):
                return False, "Warmup failed (G21) - run Home"
            return True, "Warmup OK"
        except Exception as exc:
            return False, f"Warmup error: {exc}"

    def home(self) -> Tuple[bool, str]:
        """Run GRBL homing cycle."""
        if not self.is_connected:
            return False, "Not connected"
        ok = self._send_line_wait_ok("$H", timeout_s=10.0)
        return (True, "Homing complete") if ok else (False, "Homing failed or timed out")


if __name__ == "__main__":
    sender = SerialSender()
    
    # Test auto-detect
    port = sender.auto_detect_port()
    print(f"Auto-detected port: {port}")
    
    # List all ports
    ports = sender.list_ports()
    print(f"Available ports: {ports}")
