"""
Tkinter GUI for braille printer control.
Supports text-to-braille and sign-to-braille modes.
"""

import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Set

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox

def _load_gpiozero_button():
    for factory in ("lgpio", "rpigpio"):
        os.environ["GPIOZERO_PIN_FACTORY"] = factory
        try:
            from gpiozero import Button as GpioButton

            return GpioButton, factory
        except Exception:
            continue
    return None, None


Button, GPIO_FACTORY = _load_gpiozero_button()

# Import braille modules
sys.path.insert(0, str(Path(__file__).parent))
from braille_converter import BrailleConverter
from gcode_generator import GcodeGenerator
from serial_sender import SerialSender
from sign_to_braille_bridge import SignToBrailleCapture
from braille_config import CANVAS_CONFIG, BRAILLE_PATTERNS
from lcd_display import LcdDisplay


class BraillePreviewCanvas(tk.Canvas):
    """Canvas to visualize braille dots."""

    def __init__(self, master):
        super().__init__(master, height=200, bg="white", highlightthickness=1, highlightbackground="black")
        self.dots = []
        self.scale = CANVAS_CONFIG["scale"]
        self.dot_radius = CANVAS_CONFIG["dot_radius"]

    def set_dots(self, dot_coordinates: list):
        """Set dots to display (list of lists of (x, y) tuples)."""
        self.dots = []
        for char_dots in dot_coordinates:
            self.dots.extend(char_dots)
        self.redraw()

    def redraw(self):
        """Redraw canvas."""
        self.delete("all")
        width = self.winfo_width() or 600
        height = self.winfo_height() or 200

        # Grid
        base_scale = float(self.scale)
        preview_scale = base_scale
        if self.dots:
            xs_mm = [x for x, _ in self.dots]
            ys_mm = [y for _, y in self.dots]
            min_x_mm, max_x_mm = min(xs_mm), max(xs_mm)
            min_y_mm, max_y_mm = min(ys_mm), max(ys_mm)
            span_x_mm = max(1.0, max_x_mm - min_x_mm)
            span_y_mm = max(1.0, max_y_mm - min_y_mm)
            # Fit dots inside the canvas with a small margin.
            margin = 20
            scale_x = (width - margin) / span_x_mm
            scale_y = (height - margin) / span_y_mm
            preview_scale = max(1.0, min(base_scale, scale_x, scale_y))

        for x in range(0, width, int(preview_scale) * 10):
            self.create_line(x, 0, x, height, fill="#e6e6e6")
        for y in range(0, height, int(preview_scale) * 10):
            self.create_line(0, y, width, y, fill="#e6e6e6")

        # Dots
        if not self.dots:
            return

        xs = [x * preview_scale for x, _ in self.dots]
        ys = [y * preview_scale for _, y in self.dots]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dots_w = max(1.0, max_x - min_x)
        dots_h = max(1.0, max_y - min_y)
        offset_x = (width - dots_w) / 2.0 - min_x
        offset_y = (height - dots_h) / 2.0 - min_y

        for x_mm, y_mm in self.dots:
            x_px = int(x_mm * preview_scale + offset_x)
            y_px = int(y_mm * preview_scale + offset_y)
            r = self.dot_radius
            self.create_oval(x_px - r, y_px - r, x_px + r, y_px + r, fill="black")


class BraillePrinterGUI(tk.Tk):
    """Main Tkinter GUI for braille printer."""

    def __init__(self):
        super().__init__()
        self.title("Braille Printer Control")
        self.geometry("1100x750")

        # Modules
        self.converter = BrailleConverter()
        self.gcode_gen = GcodeGenerator()
        self.serial = SerialSender()
        self.sign_capture = None
        self.sign_buffer = ""
        self.sign_queue: queue.Queue = queue.Queue()
        self.last_added_label = None
        self.last_added_time = 0.0
        self.preview_window = None
        self.preview_label = None
        self.preview_image = None
        self.lcd = LcdDisplay(address=0x27, port=1, cols=16, rows=2)
        self.lcd_line1 = ""
        self.lcd_line2 = ""
        self.lcd_last1 = None
        self.lcd_last2 = None
        self.lcd_anim_running = False
        self.lcd_anim_index = 0
        self.asl_image_refs = []
        self.braille_buttons = {}
        self.braille_enter_button = None
        self.braille_gpio_ready = False
        self.braille_dot_canvas = None
        self.braille_dot_ids = {}
        self.braille_output = None
        self.braille_status = None
        self.braille_text = ""
        self.braille_rev = self._build_braille_reverse_map()
        self.braille_asl_refs = []
        self.braille_prev_pressed = set()
        self.braille_last_single_dot = None
        self.braille_last_single_time = 0.0
        self.braille_double_tap_window = 0.6
        self.gpio_factory = GPIO_FACTORY

        self._build_ui()
        self._init_braille_gpio()
        self._poll_braille_buttons()
        self._poll_sign_queue()

        if self.lcd.is_ready:
            self._init_lcd_animation()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)

        self.text_tab = ttk.Frame(tabs)
        self.sign_tab = ttk.Frame(tabs)
        self.settings_tab = ttk.Frame(tabs)
        self.asl_tab = ttk.Frame(tabs)
        self.braille_tab = ttk.Frame(tabs)

        tabs.add(self.text_tab, text="Text -> Braille")
        tabs.add(self.sign_tab, text="Sign -> Braille")
        tabs.add(self.asl_tab, text="Text -> ASL")
        tabs.add(self.braille_tab, text="Braille -> Text")
        tabs.add(self.settings_tab, text="Settings")

        self._build_text_tab()
        self._build_sign_tab()
        self._build_asl_tab()
        self._build_braille_tab()
        self._build_settings_tab()

    def _build_text_tab(self):
        frm = self.text_tab
        ttk.Label(frm, text="Enter text (max 10 chars):").pack(anchor="w", padx=10, pady=5)

        self.text_input = tk.Text(frm, height=2)
        self.text_input.pack(fill="x", padx=10)
        self.text_input.bind("<KeyRelease>", lambda _e: self.on_text_changed())

        self.text_count_label = ttk.Label(frm, text="0 / 10 characters")
        self.text_count_label.pack(anchor="w", padx=10, pady=5)

        ttk.Label(frm, text="Braille Preview:").pack(anchor="w", padx=10, pady=5)
        self.text_preview = BraillePreviewCanvas(frm)
        self.text_preview.pack(fill="both", expand=False, padx=10, pady=5)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", padx=10, pady=10)

        ttk.Button(btn_row, text="Preview G-code", command=self.on_text_preview).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Print", command=self.on_text_print).pack(side="left", padx=5)

    def _build_sign_tab(self):
        frm = self.sign_tab

        controls = ttk.Frame(frm)
        controls.pack(fill="x", padx=10, pady=10)

        self.sign_start_btn = ttk.Button(controls, text="Start Capture", command=self.on_sign_start)
        self.sign_start_btn.pack(side="left", padx=5)

        self.sign_stop_btn = ttk.Button(controls, text="Stop Capture", command=self.on_sign_stop, state="disabled")
        self.sign_stop_btn.pack(side="left", padx=5)

        ttk.Button(controls, text="Clear Buffer", command=self.on_sign_clear).pack(side="left", padx=5)

        self.sign_status_label = ttk.Label(frm, text="Status: Idle")
        self.sign_status_label.pack(anchor="w", padx=10)

        self.sign_current_label = ttk.Label(frm, text="Current: —")
        self.sign_current_label.pack(anchor="w", padx=10)

        ttk.Label(frm, text="Detected Signs:").pack(anchor="w", padx=10, pady=5)
        self.sign_buffer_display = tk.Text(frm, height=2)
        self.sign_buffer_display.pack(fill="x", padx=10)
        self.sign_buffer_display.configure(state="disabled")

        ttk.Label(frm, text="Braille Preview:").pack(anchor="w", padx=10, pady=5)
        self.sign_preview = BraillePreviewCanvas(frm)
        self.sign_preview.pack(fill="both", expand=False, padx=10, pady=5)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", padx=10, pady=10)

        ttk.Button(btn_row, text="Preview G-code", command=self.on_sign_preview).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Print", command=self.on_sign_print).pack(side="left", padx=5)

    def _build_asl_tab(self):
        frm = self.asl_tab

        ttk.Label(frm, text="Enter text (A-Z, max 15 chars):").pack(anchor="w", padx=10, pady=5)

        self.asl_input = tk.Entry(frm)
        self.asl_input.pack(fill="x", padx=10)

        ttk.Button(frm, text="Convert", command=self.on_asl_convert).pack(anchor="w", padx=10, pady=8)

        self.asl_canvas = tk.Canvas(frm, height=240, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.asl_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        self.asl_scroll = ttk.Scrollbar(frm, orient="horizontal", command=self.asl_canvas.xview)
        self.asl_scroll.pack(fill="x", padx=10)
        self.asl_canvas.configure(xscrollcommand=self.asl_scroll.set)

        self.asl_inner = ttk.Frame(self.asl_canvas)
        self.asl_canvas_window = self.asl_canvas.create_window((0, 0), window=self.asl_inner, anchor="nw")

        def _on_configure(_event):
            self.asl_canvas.configure(scrollregion=self.asl_canvas.bbox("all"))

        self.asl_inner.bind("<Configure>", _on_configure)

    def _build_braille_tab(self):
        frm = self.braille_tab

        ttk.Label(frm, text="Braille Input (hold dots, press Enter):").pack(anchor="w", padx=10, pady=5)

        self.braille_output = tk.Text(frm, height=2)
        self.braille_output.pack(fill="x", padx=10)
        self.braille_output.configure(state="disabled")

        self.braille_status = ttk.Label(frm, text="GPIO: not initialized")
        self.braille_status.pack(anchor="w", padx=10, pady=5)

        self.braille_dot_canvas = tk.Canvas(frm, width=120, height=170, bg="white", highlightthickness=1)
        self.braille_dot_canvas.pack(anchor="w", padx=10, pady=5)

        positions = {
            1: (30, 30),
            2: (30, 75),
            3: (30, 120),
            4: (80, 30),
            5: (80, 75),
            6: (80, 120),
        }
        r = 9
        for dot, (x, y) in positions.items():
            item = self.braille_dot_canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                outline="#333333",
                fill="white",
            )
            self.braille_dot_ids[dot] = item

        controls = ttk.Frame(frm)
        controls.pack(fill="x", padx=10, pady=5)
        ttk.Button(controls, text="Clear", command=self._clear_braille_output).pack(side="left", padx=5)
        ttk.Button(controls, text="Show ASL", command=self._show_braille_asl).pack(side="left", padx=5)

        self.braille_asl_canvas = tk.Canvas(frm, height=240, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.braille_asl_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        self.braille_asl_scroll = ttk.Scrollbar(frm, orient="vertical", command=self.braille_asl_canvas.yview)
        self.braille_asl_scroll.pack(side="right", fill="y", padx=5, pady=5)
        self.braille_asl_canvas.configure(yscrollcommand=self.braille_asl_scroll.set)

        self.braille_asl_inner = ttk.Frame(self.braille_asl_canvas)
        self.braille_asl_canvas_window = self.braille_asl_canvas.create_window((0, 0), window=self.braille_asl_inner, anchor="nw")

        def _on_braille_asl_configure(_event):
            self.braille_asl_canvas.configure(scrollregion=self.braille_asl_canvas.bbox("all"))

        self.braille_asl_inner.bind("<Configure>", _on_braille_asl_configure)

    def _build_settings_tab(self):
        frm = self.settings_tab

        ttk.Label(frm, text="Serial Connection:").pack(anchor="w", padx=10, pady=5)

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=10)

        self.port_combo = ttk.Combobox(row, values=["Auto-detect"] + self.serial.list_ports(), state="readonly")
        self.port_combo.current(0)
        self.port_combo.pack(side="left", padx=5)

        ttk.Button(row, text="Connect", command=self.connect_serial).pack(side="left", padx=5)
        ttk.Button(row, text="Test", command=self.test_serial).pack(side="left", padx=5)
        ttk.Button(row, text="Unlock", command=self.unlock_serial).pack(side="left", padx=5)
        ttk.Button(row, text="Home", command=self.home_serial).pack(side="left", padx=5)

        pen_row = ttk.Frame(frm)
        pen_row.pack(fill="x", padx=10, pady=5)
        ttk.Button(pen_row, text="Pen Up", command=self.pen_up).pack(side="left", padx=5)
        ttk.Button(pen_row, text="Pen Down", command=self.pen_down).pack(side="left", padx=5)

        self.serial_status = ttk.Label(frm, text="Status: Disconnected")
        self.serial_status.pack(anchor="w", padx=10, pady=5)

        ttk.Label(frm, text="Camera URL:").pack(anchor="w", padx=10, pady=5)
        self.camera_url_input = tk.Entry(frm)
        self.camera_url_input.insert(0, "http://<phone-ip>:8080/shot.jpg")
        self.camera_url_input.pack(fill="x", padx=10)

        ttk.Label(frm, text="Min Confidence:").pack(anchor="w", padx=10, pady=5)
        self.min_conf_var = tk.DoubleVar(value=0.60)
        ttk.Spinbox(frm, from_=0.1, to=1.0, increment=0.05, textvariable=self.min_conf_var, width=10).pack(
            anchor="w", padx=10
        )

    def on_text_changed(self):
        text = self.text_input.get("1.0", "end").strip()
        self.text_count_label.config(text=f"{len(text)} / 10 characters")
        if self.lcd.is_ready:
            self._set_lcd_lines("Text: " + text[:10], "")

    def connect_serial(self):
        port = None if self.port_combo.get() == "Auto-detect" else self.port_combo.get()
        success, msg = self.serial.connect(port)
        if success:
            self.serial_status.config(text=f"Status: {msg}")
        else:
            messagebox.showwarning("Connection Failed", msg)

    def test_serial(self):
        if self.serial.is_connected:
            success, msg = self.serial.test_connection()
            messagebox.showinfo("Connection Test", msg)
        else:
            messagebox.showwarning("Not Connected", "Serial connection not established")

    def home_serial(self):
        if self.serial.is_connected:
            success, msg = self.serial.home()
            if success:
                messagebox.showinfo("Homing", msg)
            else:
                messagebox.showwarning("Homing", msg)
        else:
            messagebox.showwarning("Not Connected", "Serial connection not established")

    def unlock_serial(self):
        if self.serial.is_connected:
            success, msg = self.serial.unlock()
            if success:
                messagebox.showinfo("Unlock", msg)
            else:
                messagebox.showwarning("Unlock", msg)
        else:
            messagebox.showwarning("Not Connected", "Serial connection not established")

    def pen_up(self):
        if self.serial.is_connected:
            success, msg = self.serial.pen_up()
            if success:
                messagebox.showinfo("Pen Up", msg)
            else:
                messagebox.showwarning("Pen Up", msg)
        else:
            messagebox.showwarning("Not Connected", "Serial connection not established")

    def pen_down(self):
        if self.serial.is_connected:
            success, msg = self.serial.pen_down()
            if success:
                messagebox.showinfo("Pen Down", msg)
            else:
                messagebox.showwarning("Pen Down", msg)
        else:
            messagebox.showwarning("Not Connected", "Serial connection not established")

    def on_text_preview(self):
        text = self.text_input.get("1.0", "end").strip()
        is_valid, msg = self.converter.validate_text(text)
        if not is_valid:
            messagebox.showwarning("Validation Error", msg)
            return

        coords = self.converter.text_to_gcode_coordinates(text)
        is_valid, msg = self.gcode_gen.validate_coordinates(coords)
        if not is_valid:
            messagebox.showwarning("Validation Error", msg)
            return

        self.text_preview.set_dots(coords)
        gcode = self.gcode_gen.coordinates_to_gcode(coords)
        gcode_str = self.gcode_gen.gcode_to_string(gcode)
        messagebox.showinfo("G-code Preview", f"Generated {len(gcode)} lines\n\n{gcode_str[:500]}...")

    def on_asl_convert(self):
        text = self.asl_input.get().strip().upper()
        if not text:
            messagebox.showwarning("Error", "Please enter text")
            return

        if len(text) > 15:
            messagebox.showwarning("Error", "Text must be 15 characters or fewer")
            return

        for child in self.asl_inner.winfo_children():
            child.destroy()
        self.asl_image_refs.clear()

        base_dir = Path(__file__).parent.parent / "asl_alphabet_test" / "asl_alphabet_test"
        unsupported = []

        for idx, char in enumerate(text):
            row = idx // 4
            col = idx % 4
            if char == " ":
                spacer = ttk.Frame(self.asl_inner, width=30)
                spacer.grid(row=row, column=col, padx=5, pady=5)
                continue

            if char < "A" or char > "Z":
                unsupported.append(char)
                continue

            img_path = base_dir / f"{char}_test.jpg"
            if not img_path.exists():
                unsupported.append(char)
                continue

            img = Image.open(img_path)
            img = img.resize((200, 200))
            photo = ImageTk.PhotoImage(img)
            lbl = ttk.Label(self.asl_inner, image=photo)
            lbl.grid(row=row, column=col, padx=5, pady=5)
            self.asl_image_refs.append(photo)

        if unsupported:
            messagebox.showwarning("Unsupported", f"Missing images for: {', '.join(unsupported)}")

    def _build_braille_reverse_map(self):
        mapping = {}
        for key, dots in BRAILLE_PATTERNS.items():
            if key == " ":
                mapping[frozenset(dots)] = " "
            elif len(key) == 1 and key.isalpha():
                mapping[frozenset(dots)] = key
        return mapping

    def _init_braille_gpio(self):
        if Button is None:
            if self.braille_status:
                self.braille_status.config(text="GPIO: no pin factory (install lgpio or RPi.GPIO)")
            return

        dot_pins = {
            1: 22,
            2: 6,
            3: 27,
            4: 26,
            5: 25,
            6: 16,
        }
        enter_pin = 5

        try:
            for dot, pin in dot_pins.items():
                btn = Button(pin, pull_up=True, bounce_time=0.1)
                btn.when_pressed = lambda d=dot: self.after(0, lambda: self._on_braille_dot_pressed(d))
                self.braille_buttons[dot] = btn
            self.braille_enter_button = Button(enter_pin, pull_up=True, bounce_time=0.1)
            self.braille_enter_button.when_pressed = lambda: self.after(0, self._handle_braille_enter)
            self.braille_gpio_ready = True
            if self.braille_status:
                factory = self.gpio_factory or "unknown"
                self.braille_status.config(text=f"GPIO: ready ({factory})")
        except Exception as exc:
            self.braille_gpio_ready = False
            messagebox.showwarning("GPIO Error", f"Braille GPIO init failed: {exc}")
            if self.braille_status:
                self.braille_status.config(text="GPIO: init failed")

    def _poll_braille_buttons(self):
        if self.braille_gpio_ready and self.braille_dot_canvas:
            pressed = self._get_pressed_dots()
            for dot, oval in self.braille_dot_ids.items():
                fill = "black" if dot in pressed else "white"
                self.braille_dot_canvas.itemconfig(oval, fill=fill)
            if self.braille_status is not None:
                self.braille_status.config(text=f"GPIO: ready (pressed: {sorted(pressed)})")
        self.after(100, self._poll_braille_buttons)

    def _on_braille_dot_pressed(self, dot: int):
        if not self.braille_gpio_ready:
            return
        # Backspace on double-tap of any single dot without Enter.
        now = time.monotonic()
        if self.braille_last_single_dot == dot and (now - self.braille_last_single_time) <= self.braille_double_tap_window:
            self._backspace_braille()
            self.braille_last_single_dot = None
            self.braille_last_single_time = 0.0
            return

        # Only arm double-tap when this is the only pressed dot.
        pressed = self._get_pressed_dots()
        if pressed == {dot}:
            self.braille_last_single_dot = dot
            self.braille_last_single_time = now

    def _get_pressed_dots(self) -> Set[int]:
        pressed = set()
        for dot, btn in self.braille_buttons.items():
            try:
                if btn.is_pressed:
                    pressed.add(dot)
            except Exception:
                continue
        return pressed

    def _handle_braille_enter(self):
        if not self.braille_gpio_ready:
            return
        dots = frozenset(self._get_pressed_dots())
        if len(dots) == 0:
            self._append_braille_char(" ")
            return
        char = self.braille_rev.get(dots)
        if char is None:
            messagebox.showwarning("Braille", "Unknown dot pattern")
            return

        self._append_braille_char(char)

    def _append_braille_char(self, char: str):
        if len(self.braille_text) >= 15:
            messagebox.showwarning("Braille", "Output is limited to 15 characters")
            return

        self.braille_text += char
        self._set_braille_output(self.braille_text)

    def _backspace_braille(self):
        if not self.braille_text:
            return
        self.braille_text = self.braille_text[:-1]
        self._set_braille_output(self.braille_text)

    def _set_braille_output(self, text: str):
        if not self.braille_output:
            return
        self.braille_output.configure(state="normal")
        self.braille_output.delete("1.0", "end")
        self.braille_output.insert("1.0", text)
        self.braille_output.configure(state="disabled")

    def _clear_braille_output(self):
        self.braille_text = ""
        self._set_braille_output("")
        self._clear_braille_asl()

    def _clear_braille_asl(self):
        for child in self.braille_asl_inner.winfo_children():
            child.destroy()
        self.braille_asl_refs.clear()

    def _show_braille_asl(self):
        text = self.braille_text.strip().upper()
        if not text:
            messagebox.showwarning("ASL", "No text to show")
            return

        self._clear_braille_asl()
        base_dir = Path(__file__).parent.parent / "asl_alphabet_test" / "asl_alphabet_test"
        unsupported = []

        for idx, char in enumerate(text):
            row = idx // 4
            col = idx % 4
            if char == " ":
                spacer = ttk.Frame(self.braille_asl_inner, width=30)
                spacer.grid(row=row, column=col, padx=5, pady=5)
                continue

            if char < "A" or char > "Z":
                unsupported.append(char)
                continue

            img_path = base_dir / f"{char}_test.jpg"
            if not img_path.exists():
                unsupported.append(char)
                continue

            img = Image.open(img_path)
            img = img.resize((200, 200))
            photo = ImageTk.PhotoImage(img)
            lbl = ttk.Label(self.braille_asl_inner, image=photo)
            lbl.grid(row=row, column=col, padx=5, pady=5)
            self.braille_asl_refs.append(photo)

        if unsupported:
            messagebox.showwarning("ASL", f"Missing images for: {', '.join(unsupported)}")

    def on_text_print(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Error", "Please enter text")
            return

        is_valid, msg = self.converter.validate_text(text)
        if not is_valid:
            messagebox.showwarning("Validation Error", msg)
            return

        if not self.serial.is_connected:
            messagebox.showwarning("Error", "Serial not connected")
            return

        if self.lcd.is_ready:
            self._set_lcd_lines("Braille", "printing...")

        coords = self.converter.text_to_gcode_coordinates(text)
        gcode = self.gcode_gen.coordinates_to_gcode(coords)
        success, msg, _lines = self.serial.send_gcode(gcode, verbose=False)

        if success:
            messagebox.showinfo("Print Complete", msg)
            if self.lcd.is_ready:
                self._set_lcd_lines("Print done", "")
        else:
            messagebox.showerror("Print Error", msg)
            if self.lcd.is_ready:
                self._set_lcd_lines("Print failed", "")

    def on_sign_start(self):
        if self.sign_capture:
            return

        camera_url = self.camera_url_input.get().strip()
        min_conf = float(self.min_conf_var.get())

        # Reset capture state for new session
        self.sign_buffer = ""
        self.last_added_label = None
        self.last_added_time = 0.0
        self._set_sign_buffer_text("")
        self.sign_preview.set_dots([])
        if self.lcd.is_ready:
            self._set_lcd_lines("Sign: ", "")

        self.sign_capture = SignToBrailleCapture(
            model_path="models/sign_model_tasks_v6_combo.joblib",
            camera_url=camera_url,
            min_confidence=min_conf,
            smoothing_window=11,
            use_roi=True,
            roi_scale=0.5,
            backend="mediapipe",
            show_preview=True,
            display_width=800,
        )

        def on_sign(sign: str, conf: float):
            self.sign_queue.put((sign, conf))

        self.sign_capture.on_sign_detected = on_sign
        if self.sign_capture.start():
            self.sign_status_label.config(text="Status: Capturing...")
            self.sign_start_btn.config(state="disabled")
            self.sign_stop_btn.config(state="normal")
        else:
            self.sign_capture = None
            messagebox.showwarning("Capture Error", "Failed to start capture")
            if self.lcd.is_ready:
                self._set_lcd_lines("Capture", "failed")

    def on_sign_stop(self):
        if self.sign_capture:
            self.sign_capture.stop()
            self.sign_capture = None
        self.sign_status_label.config(text="Status: Stopped")
        self.sign_start_btn.config(state="normal")
        self.sign_stop_btn.config(state="disabled")
        self._close_preview_window()

    def on_sign_clear(self):
        if self.sign_capture:
            self.sign_capture.clear_buffer()
        self.sign_buffer = ""
        self._set_sign_buffer_text("")
        self.sign_preview.set_dots([])

    def on_sign_preview(self):
        buffer_text = self.sign_buffer
        if not buffer_text:
            messagebox.showwarning("Error", "No signs captured")
            return

        coords = self.converter.text_to_gcode_coordinates(buffer_text)
        gcode = self.gcode_gen.coordinates_to_gcode(coords)
        gcode_str = self.gcode_gen.gcode_to_string(gcode)
        messagebox.showinfo("G-code Preview", f"Generated {len(gcode)} lines\n\n{gcode_str[:500]}...")

    def on_sign_print(self):
        buffer_text = self.sign_buffer
        if not buffer_text:
            messagebox.showwarning("Error", "No signs captured")
            return

        if not self.serial.is_connected:
            messagebox.showwarning("Error", "Serial not connected")
            return

        if self.lcd.is_ready:
            self._set_lcd_lines("Braille", "printing...")

        coords = self.converter.text_to_gcode_coordinates(buffer_text)
        gcode = self.gcode_gen.coordinates_to_gcode(coords)
        success, msg, _lines = self.serial.send_gcode(gcode, verbose=False)

        if success:
            messagebox.showinfo("Print Complete", msg)
            if self.lcd.is_ready:
                self._set_lcd_lines("Print done", "")
        else:
            messagebox.showerror("Print Error", msg)
            if self.lcd.is_ready:
                self._set_lcd_lines("Print failed", "")

    def _set_sign_buffer_text(self, text: str):
        self.sign_buffer_display.configure(state="normal")
        self.sign_buffer_display.delete("1.0", "end")
        self.sign_buffer_display.insert("1.0", text)
        self.sign_buffer_display.configure(state="disabled")

    def _poll_sign_queue(self):
        # Always reflect current prediction in the UI
        if self.sign_capture:
            current = self.sign_capture.current_prediction
            conf = self.sign_capture.confidence
            if current:
                self.sign_current_label.config(text=f"Current: {current} ({conf:.2f})")

            frame = self.sign_capture.get_latest_frame()
            if frame is not None:
                self._show_preview_frame(frame)

        while True:
            try:
                sign, confidence = self.sign_queue.get_nowait()
            except queue.Empty:
                break

            if self.sign_capture:
                now = time.time()
                if len(self.sign_buffer) < 10:
                    if sign and sign not in ["UNSURE", "NOTHING"] and confidence >= float(self.min_conf_var.get()):
                        # Only add when label changes and enforce a cooldown
                        if sign != self.last_added_label and (now - self.last_added_time) > 1.0:
                            self.sign_buffer += sign
                            self.last_added_label = sign
                            self.last_added_time = now
                buffer_text = self.sign_buffer
                self.sign_current_label.config(text=f"Current: {sign} ({confidence:.2f})")
                self._set_sign_buffer_text(buffer_text)
                if self.lcd.is_ready:
                    self._set_lcd_lines("Sign: " + buffer_text[:10], "")
                if buffer_text:
                    coords = self.converter.text_to_gcode_coordinates(buffer_text)
                    self.sign_preview.set_dots(coords)

        self.after(100, self._poll_sign_queue)

    def on_close(self):
        if self.sign_capture:
            self.sign_capture.stop()
        self._close_preview_window()
        if self.serial.is_connected:
            self.serial.disconnect()
        for btn in self.braille_buttons.values():
            try:
                btn.close()
            except Exception:
                pass
        if self.braille_enter_button:
            try:
                self.braille_enter_button.close()
            except Exception:
                pass
        self.destroy()

    def _show_preview_frame(self, frame):
        if self.preview_window is None:
            self.preview_window = tk.Toplevel(self)
            self.preview_window.title("Sign Capture")
            self.preview_window.geometry("820x620")
            self.preview_label = ttk.Label(self.preview_window)
            self.preview_label.pack(fill="both", expand=True)
            self.preview_window.protocol("WM_DELETE_WINDOW", self._close_preview_window)

        # Convert BGR to RGB for Tkinter
        rgb = frame[:, :, ::-1]
        img = Image.fromarray(rgb)
        img.thumbnail((800, 600))
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_image)

    def _close_preview_window(self):
        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
            self.preview_window = None
            self.preview_label = None
            self.preview_image = None

    def _set_lcd_lines(self, line1: str, line2: str = ""):
        self.lcd_line1 = line1
        self.lcd_line2 = line2
        if self.lcd.is_ready:
            if self.lcd_last1 != self.lcd_line1 or self.lcd_last2 != self.lcd_line2:
                self.lcd.show_lines(self.lcd_line1, self.lcd_line2)
                self.lcd_last1 = self.lcd_line1
                self.lcd_last2 = self.lcd_line2
        if (line1, line2) != ("UNITALK By", "Team HoPeLeSS"):
            self.lcd_anim_running = False

    def _refresh_lcd(self):
        # Intentionally unused to avoid LCD flicker.
        pass

    def _init_lcd_animation(self):
        # Custom cyber badge icon (5x8) at index 0.
        icon = [
            0b00100,
            0b01110,
            0b10101,
            0b11111,
            0b10001,
            0b10101,
            0b11111,
            0b01010,
        ]
        self.lcd.create_char(0, icon)
        self.lcd_anim_running = True
        self.lcd_anim_index = 0
        self.lcd_anim_stage = 0
        self._animate_lcd()

    def _animate_lcd(self):
        if not self.lcd_anim_running or not self.lcd.is_ready:
            return

        icon = chr(0)
        msg1 = "UNITALK By"
        msg2 = "Team HoPeLeSS " + icon

        if self.lcd_anim_stage == 0:
            # Type in line 1.
            count = min(self.lcd_anim_index + 1, len(msg1))
            line1 = msg1[:count]
            line2 = ""
            if count >= len(msg1):
                self.lcd_anim_stage = 1
                self.lcd_anim_index = 0
        elif self.lcd_anim_stage == 1:
            # Type in line 2.
            count = min(self.lcd_anim_index + 1, len(msg2))
            line1 = msg1
            line2 = msg2[:count]
            if count >= len(msg2):
                self.lcd_anim_stage = 2
                self.lcd_anim_index = 0
        else:
            # Hold final message for a moment, then restart.
            line1 = msg1
            line2 = msg2
            if self.lcd_anim_index >= 8:
                self.lcd_anim_stage = 0
                self.lcd_anim_index = 0

        self.lcd.show_lines(line1, line2)
        self.lcd_last1 = line1
        self.lcd_last2 = line2
        self.lcd_anim_index += 1
        self.after(250, self._animate_lcd)


def main():
    app = BraillePrinterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
