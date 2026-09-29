"""Texto con acentos y ñ sobre fotogramas de OpenCV.

cv2.putText solo dibuja ASCII: "Muñeca" o "índice" salen con signos de interrogación.
Aquí se acumulan los textos de un fotograma y se dibujan de una vez con Pillow.
"""
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=None)
def _font(size):
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class TextLayer:
    """Uso: layer.add(...) varias veces por fotograma y frame = layer.draw(frame)."""

    def __init__(self):
        self.items = []

    def add(self, text, xy, color_bgr, size=20):
        self.items.append((text, xy, color_bgr, size))

    def draw(self, frame):
        if not self.items:
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for text, xy, (b, g, r), size in self.items:
            draw.text(xy, text, font=_font(size), fill=(r, g, b),
                      stroke_width=2, stroke_fill=(0, 0, 0))
        self.items = []
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
