"""
I2C 16x2 LCD display helper (PCF8574 backpack).
"""

from __future__ import annotations

from typing import Optional


class LcdDisplay:
    """Simple wrapper for a 16x2 I2C LCD."""

    def __init__(self, address: int = 0x27, port: int = 1, cols: int = 16, rows: int = 2):
        self.address = address
        self.port = port
        self.cols = cols
        self.rows = rows
        self._lcd = None

        try:
            from RPLCD.i2c import CharLCD

            self._lcd = CharLCD(
                "PCF8574",
                address=address,
                port=port,
                cols=cols,
                rows=rows,
                charmap="A00",
                auto_linebreaks=False,
            )
            self._lcd.backlight_enabled = True
            self._lcd.clear()
        except Exception:
            self._lcd = None

    @property
    def is_ready(self) -> bool:
        return self._lcd is not None

    def show_lines(self, line1: str, line2: str = "") -> None:
        if not self._lcd:
            return
        line1 = self._format_line(line1)
        line2 = self._format_line(line2)
        try:
            self._lcd.clear()
            self._lcd.cursor_pos = (0, 0)
            self._lcd.write_string(line1)
            self._lcd.cursor_pos = (1, 0)
            self._lcd.write_string(line2)
        except Exception:
            pass

    def create_char(self, index: int, bitmap: list[int]) -> None:
        if not self._lcd:
            return
        try:
            self._lcd.create_char(index, bitmap)
        except Exception:
            pass

    def clear(self) -> None:
        if not self._lcd:
            return
        try:
            self._lcd.clear()
        except Exception:
            pass

    def _format_line(self, text: str) -> str:
        text = text or ""
        if len(text) > self.cols:
            return text[: self.cols]
        return text.ljust(self.cols)
