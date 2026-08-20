"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QThread,
    QTimer,
    QVariantAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame import eye_detail, hair_transfer
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

# Raised from 820/640: the extra option switches plus the roomier glass
# padding no longer fit the old frame.  Width is set from the widest card's
# minimum (measured ~801px) plus the scrollbar gutter; height scrolls.
ROOT_HEIGHT = 900
ROOT_WIDTH = 830

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── stylesheet ───────────────────────────────────────────────────────────

# Instrument panel, not a web page. The reference points are an aircraft-grade
# control surface and a vehicle cluster: a near-black ground, surfaces that
# separate by a single hairline rather than by a border or a glow, uppercase
# micro-labels with wide tracking, and every number set in tabular figures so
# readouts never jitter as they count. One accent — platinum white — spent
# sparingly, so the eye lands on state, not on decoration.

QSS = """
QMainWindow, QDialog { background: #08080a; color: #f2f3f5; }
QWidget {
    color: #f2f3f5;
    font-family: "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", sans-serif;
    font-size: 10.5pt;
}

/* Panels are surfaces, not cards: no gradient, no glow, one hairline. */
QGroupBox {
    background: #0f0f11;
    border: 1px solid #1c1c20;
    border-radius: 4px;
    margin-top: 26px;
    padding: 20px 20px 18px 20px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 2px;
    padding: 0;
    color: #6b6f78;
    font-family: "Segoe UI Variable Small", "Segoe UI", sans-serif;
    font-size: 8pt;
    font-weight: 600;
    letter-spacing: 2.4px;
    text-transform: uppercase;
}

/* Primary action reads as the only lit control in the panel. */
QPushButton {
    background: #f2f3f5;
    color: #08080a;
    border: 1px solid #f2f3f5;
    border-radius: 3px;
    padding: 10px 22px;
    font-size: 9.5pt;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
QPushButton:hover   { background: #ffffff; border-color: #ffffff; }
QPushButton:pressed { background: #c8cbd2; border-color: #c8cbd2; }
QPushButton:disabled {
    background: transparent;
    color: #3d4048;
    border: 1px solid #1c1c20;
}
QPushButton#secondary {
    background: transparent;
    color: #b4b8c0;
    border: 1px solid #2a2a30;
}
QPushButton#secondary:hover { color: #f2f3f5; border-color: #4a4d56; }
QPushButton#danger {
    background: transparent;
    color: #e5563a;
    border: 1px solid #4a2018;
}
QPushButton#danger:hover { background: #e5563a; color: #08080a; border-color: #e5563a; }

QComboBox {
    background: #131316;
    border: 1px solid #26262c;
    border-radius: 3px;
    padding: 9px 12px;
    min-height: 22px;
    color: #e4e6ea;
}
QComboBox:hover { border-color: #3f424b; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView {
    background: #131316;
    color: #e4e6ea;
    selection-background-color: #26262c;
    border: 1px solid #26262c;
    outline: none;
    padding: 3px;
}

QCheckBox { spacing: 12px; padding: 3px 0; color: #b4b8c0; }
QCheckBox:hover { color: #f2f3f5; }
QCheckBox::indicator {
    width: 34px; height: 18px;
    border-radius: 2px;
    background: #17171b;
    border: 1px solid #2a2a30;
}
QCheckBox::indicator:hover { border-color: #4a4d56; }
QCheckBox::indicator:checked { background: #f2f3f5; border-color: #f2f3f5; }

/* Thin travel, square handle — a fader on a console, not a toy. */
QSlider::groove:horizontal {
    height: 2px;
    background: #26262c;
}
QSlider::handle:horizontal {
    background: #f2f3f5;
    width: 4px; height: 18px;
    margin: -8px 0;
    border-radius: 1px;
}
QSlider::handle:horizontal:hover { background: #ffffff; width: 6px; margin: -8px -1px; }
QSlider::sub-page:horizontal { background: #f2f3f5; }

QLabel#imageDrop {
    background: #0c0c0e;
    border: 1px solid #1c1c20;
    border-radius: 3px;
    color: #4a4d56;
    font-size: 8.5pt;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}
QLabel#statusLabel {
    color: #5c6068;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 8.5pt;
}
QLabel#linkLabel { color: #b4b8c0; }

/* Wordmark: tight tracking, optical weight — the one piece of real display type. */
QLabel#brandLabel {
    font-family: "Segoe UI Variable Display", "Segoe UI", sans-serif;
    font-size: 22pt;
    font-weight: 700;
    font-style: italic;
    letter-spacing: -0.5px;
    color: #f2f3f5;
    padding: 0;
}
QLabel#brandSub {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 7.5pt;
    font-weight: 400;
    letter-spacing: 3.4px;
    text-transform: uppercase;
    color: #4a4d56;
}

/* Field labels sit quiet; the value beside them is what you read. */
QLabel#fieldLabel {
    color: #8b8f98;
    font-size: 9pt;
    letter-spacing: 0.8px;
}
/* Tabular figures: the column will not shuffle as digits change. */
QLabel#valueLabel {
    color: #f2f3f5;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 9pt;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
QLabel#valueLabel[muted="true"] { color: #4a4d56; }

QFrame#rule { background: #1c1c20; border: none; max-height: 1px; min-height: 1px; }

/* The scroll viewport and its content widget are separate widgets from the
   window, and would otherwise paint their own default (light) background. */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QWidget#scrollContent { background: transparent; }
QScrollBar:vertical { background: transparent; width: 3px; margin: 0; }
QScrollBar::handle:vertical { background: #2a2a30; border-radius: 1px; min-height: 40px; }
QScrollBar::handle:vertical:hover { background: #4a4d56; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QToolTip {
    background: #131316;
    color: #b4b8c0;
    border: 1px solid #26262c;
    padding: 7px 10px;
    font-size: 9pt;
}

QFrame#card { background: #0f0f11; border: 1px solid #1c1c20; border-radius: 4px; }
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "enable_flow_tracking": modules.globals.enable_flow_tracking,
        "show_ai_badge": modules.globals.show_ai_badge,
        "hair_transfer": modules.globals.hair_transfer,
        "hair_transfer_strength": modules.globals.hair_transfer_strength,
        "eye_color_lock": modules.globals.eye_color_lock,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.enable_flow_tracking = state.get("enable_flow_tracking", True)
        modules.globals.show_ai_badge = state.get("show_ai_badge", False)
        modules.globals.hair_transfer = state.get("hair_transfer", False)
        modules.globals.hair_transfer_strength = state.get("hair_transfer_strength", 100.0)
        modules.globals.eye_color_lock = state.get("eye_color_lock", 0.0)
        # Eye detail, like mouth mask, always starts off so a saved
        # session never silently re-enables a face-altering overlay.
        modules.globals.eyes_mask_size = 0.0
        modules.globals.eyes_mask = False
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class SplashScreen(QWidget):
    """Cold-open title card: a hairline splits from centre, the wordmark
    resolves out of it, then the whole thing steps aside.

    Everything is drawn by hand rather than assembled from widgets — the
    wordmark's letter-spacing has to animate, and Qt only exposes that through
    QFont, not through QSS.
    """

    finished = Signal()

    # Beat structure, in ms from t=0.
    _RULE_MS = 620      # hairline draws out from the centre
    _MARK_MS = 900      # wordmark fades up and tightens
    _HOLD_MS = 640      # let it sit
    _EXIT_MS = 420      # fade away

    def __init__(self, text: str, subtitle: str = ""):
        super().__init__(None)
        self._text = text
        self._subtitle = subtitle
        self._t = 0.0            # 0..1 across the whole sequence
        self._exit = 0.0         # 0..1 exit fade

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.SplashScreen
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(720, 400)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            self.move(geo.center().x() - 360, geo.center().y() - 200)

        total = self._RULE_MS + self._MARK_MS + self._HOLD_MS
        self._intro = QVariantAnimation(self)
        self._intro.setDuration(total)
        self._intro.setStartValue(0.0)
        self._intro.setEndValue(1.0)
        self._intro.setEasingCurve(QEasingCurve.Type.Linear)
        self._intro.valueChanged.connect(self._on_tick)
        self._intro.finished.connect(self._start_exit)

        self._outro = QVariantAnimation(self)
        self._outro.setDuration(self._EXIT_MS)
        self._outro.setStartValue(0.0)
        self._outro.setEndValue(1.0)
        self._outro.setEasingCurve(QEasingCurve.Type.InCubic)
        self._outro.valueChanged.connect(self._on_exit_tick)
        self._outro.finished.connect(self._done)

    # ── lifecycle ────────────────────────────────────────────────────────

    def run(self) -> None:
        self.show()
        self.raise_()
        self._intro.start()

    def _on_tick(self, value) -> None:
        self._t = float(value)
        self.update()

    def _start_exit(self) -> None:
        self._outro.start()

    def _on_exit_tick(self, value) -> None:
        self._exit = float(value)
        self.update()

    def _done(self) -> None:
        self.close()
        self.finished.emit()

    def mousePressEvent(self, _event) -> None:
        # Never make someone sit through a splash twice.
        self._intro.stop()
        self._outro.stop()
        self._done()

    # ── painting ─────────────────────────────────────────────────────────

    @staticmethod
    def _ease_out(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 1.0 - pow(1.0 - x, 3)

    def _phase(self) -> Tuple[float, float]:
        """(rule progress, wordmark progress), each eased 0..1."""
        total = self._RULE_MS + self._MARK_MS + self._HOLD_MS
        now = self._t * total
        rule = self._ease_out(now / self._RULE_MS)
        mark = self._ease_out((now - self._RULE_MS * 0.55) / self._MARK_MS)
        return rule, mark

    def paintEvent(self, _event) -> None:
        from PySide6.QtGui import QFont, QPainter, QPen

        rule_p, mark_p = self._phase()
        alpha = 1.0 - self._exit

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#08080a"))

        cx, cy = w / 2, h / 2

        # 1. Hairline splitting outward from dead centre.
        rule_half = rule_p * (w * 0.34)
        if rule_half > 0.5:
            pen = QPen(QColor(242, 243, 245, int(255 * min(1.0, rule_p) * alpha)))
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawLine(int(cx - rule_half), int(cy + 34),
                             int(cx + rule_half), int(cy + 34))

        if mark_p <= 0.0:
            painter.end()
            return

        # 2. Wordmark resolving: tracking closes in as opacity comes up, so it
        #    reads as the letters gathering rather than a plain cross-fade.
        font = QFont("Segoe UI Variable Display", 40)
        font.setItalic(True)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing,
                              18.0 * (1.0 - mark_p))
        painter.setFont(font)
        painter.setPen(QColor(242, 243, 245, int(255 * mark_p * alpha)))

        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(self._text)
        painter.drawText(int(cx - tw / 2),
                         int(cy + metrics.capHeight() / 2 - 6),
                         self._text)

        # 3. Subtitle trails the wordmark by a beat.
        if self._subtitle and mark_p > 0.45:
            sub_p = self._ease_out((mark_p - 0.45) / 0.55)
            sub = QFont("Cascadia Mono", 8)
            sub.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 5.0)
            painter.setFont(sub)
            painter.setPen(QColor(90, 95, 104, int(255 * sub_p * alpha)))
            sm = painter.fontMetrics()
            sw = sm.horizontalAdvance(self._subtitle)
            painter.drawText(int(cx - sw / 2), int(cy + 62), self._subtitle)

        painter.end()


class _SmoothScrollArea(QScrollArea):
    """Scroll area that eases to its target instead of stepping to it.

    Wheel deltas accumulate into a single running animation, so spinning the
    wheel keeps extending one glide rather than restarting it each notch.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._anim = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._anim.setDuration(380)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def wheelEvent(self, event) -> None:
        bar = self.verticalScrollBar()
        if bar.maximum() == 0:
            event.ignore()
            return

        # Extend the in-flight glide when one is running, so fast scrolling
        # accelerates rather than stuttering back to the current position.
        running = self._anim.state() == QPropertyAnimation.State.Running
        base = self._anim.endValue() if running else bar.value()
        target = int(base) - int(event.angleDelta().y() * 0.9)
        target = max(bar.minimum(), min(bar.maximum(), target))

        self._anim.stop()
        self._anim.setStartValue(bar.value())
        self._anim.setEndValue(target)
        self._anim.start()
        event.accept()


def _rule() -> QFrame:
    """A 1px separator — the only divider this design uses."""
    line = QFrame()
    line.setObjectName("rule")
    line.setFrameShape(QFrame.Shape.NoFrame)
    line.setFixedHeight(1)
    return line


class _AnimatedSlider(QSlider):
    """Slider whose handle eases to clicked positions instead of jumping.

    Dragging stays 1:1 — animating under the cursor would feel like lag.  Only
    track clicks and wheel/key steps are animated.
    """

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._anim = QPropertyAnimation(self, b"value", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._dragging = False

    def _animate_to(self, target: int) -> None:
        target = max(self.minimum(), min(self.maximum(), target))
        self._anim.stop()
        self._anim.setStartValue(self.value())
        self._anim.setEndValue(target)
        self._anim.start()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            span = self.width() - 12
            if span > 0:
                ratio = (event.position().x() - 6) / span
                ratio = max(0.0, min(1.0, ratio))
                target = self.minimum() + round(
                    ratio * (self.maximum() - self.minimum())
                )
                # Near the handle this is a grab, not a jump — hand it to Qt so
                # the drag starts without the animation fighting the cursor.
                if abs(target - self.value()) > (self.maximum() - self.minimum()) * 0.02:
                    self._animate_to(target)
                    event.accept()
                    return
        self._dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        step = max(1, (self.maximum() - self.minimum()) // 20)
        delta = step if event.angleDelta().y() > 0 else -step
        self._animate_to(self.value() + delta)
        event.accept()


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        # Content is taller than the window on purpose — it scrolls.  The
        # minimum only has to stay usable on a short screen.
        self.setMinimumSize(ROOT_WIDTH, 520)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        scroll = _SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll)

        root = QWidget()
        root.setObjectName("scrollContent")
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(12)

        # Wordmark block, centred, with a hairline closing it off.
        brand = QLabel(modules.metadata.name)
        brand.setObjectName("brandLabel")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(brand)

        brand_sub = QLabel(modules.metadata.edition)
        brand_sub.setObjectName("brandSub")
        brand_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(brand_sub)

        layout.addSpacing(4)
        layout.addWidget(_rule())
        layout.addSpacing(4)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())

        # Action buttons
        layout.addLayout(self._build_action_row())

        # Camera selection
        layout.addWidget(self._build_camera_card())

        # Status & footer
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        footer = QLabel(
            f"{modules.metadata.name} {modules.metadata.edition}"
        )
        footer.setObjectName("statusLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options").upper())
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        # The switch pills are 20px tall inside a padded QCheckBox; 6px of row
        # spacing let neighbouring rows clip each other once the list grew.
        grid.setVerticalSpacing(12)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")
        self.sw_flow_tracking = make("enable_flow_tracking", "Motion Tracking",
                                "Track the face between detections with optical flow "
                                "instead of holding a stale position (reduces lag on fast head movement)")
        self.sw_ai_badge = make("show_ai_badge", "AI Generated Badge",
                                "Overlay an \"AI GENERATED\" label on the live preview "
                                "(recommended before streaming/recording for disclosure compliance)")
        self.sw_hair = make("hair_transfer", "Hair Transfer (beta)",
                            "Also transfer the source image's hair, which the swap model "
                            "cannot do on its own. 2D approximation — looks best near-frontal "
                            "and degrades as the head turns")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
            self.sw_flow_tracking, self.sw_ai_badge,
            self.sw_hair,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown — round up so an odd switch count leaves the
        # last (half-empty) switch row intact instead of overlapping it.
        enhancer_row = (len(items) + 1) // 2
        enhancer_label = QLabel(_("Face Enhancer"))
        enhancer_label.setObjectName("fieldLabel")
        grid.addWidget(enhancer_label, enhancer_row, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, enhancer_row, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement").upper())
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(1, 1)

        def row(index, label, min_v, max_v, default, denom, on_change,
                fmt, tooltip):
            """Label | fader | live readout, on one baseline."""
            name = QLabel(_(label))
            name.setObjectName("fieldLabel")
            name.setMinimumWidth(104)
            grid.addWidget(name, index, 0)

            s = _AnimatedSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.setToolTip(_(tooltip))
            grid.addWidget(s, index, 1)

            readout = QLabel()
            readout.setObjectName("valueLabel")
            readout.setMinimumWidth(56)
            readout.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(readout, index, 2)

            def _sync(raw: int) -> None:
                value = raw / denom
                readout.setText(fmt(value))
                # Dim a readout that is doing nothing, so the lit numbers are
                # the ones actually affecting the output.
                readout.setProperty("muted", value <= 0)
                readout.style().unpolish(readout)
                readout.style().polish(readout)
                on_change(value)

            s.valueChanged.connect(_sync)
            _sync(s.value())
            return s

        pct = lambda v: f"{v * 100:.0f}%"

        self.s_transparency = row(
            0, "Transparency", 0.0, 1.0, 1.0, 100, self._on_transparency_change,
            pct,
            "Blend between original and swapped face (0% = original, 100% = fully swapped)",
        )
        self.s_sharpness = row(
            1, "Sharpness", 0.0, 5.0, 0.0, 10, self._on_sharpness_change,
            lambda v: f"{v:.1f}",
            "Sharpen the enhanced face output",
        )
        self.s_mouth = row(
            2, "Mouth Mask", 0.0, 100.0, 0.0, 1, self._on_mouth_mask_change,
            lambda v: f"{v:.0f}%",
            "0 = use swapped mouth, 100 = expose original mouth to chin area",
        )
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)

        self.s_hair = row(
            3, "Hair Blend", 0.0, 100.0, modules.globals.hair_transfer_strength, 1,
            self._on_hair_strength_change,
            lambda v: f"{v:.0f}%",
            "Opacity of the transferred hair layer (needs Hair Transfer enabled)",
        )
        self.s_eyes = row(
            4, "Eye Detail", 0.0, 100.0, 0.0, 1, self._on_eyes_mask_change,
            lambda v: f"{v:.0f}%",
            "Restore the real eyes over the swap — recovers true gaze, eye colour "
            "and full sharpness that the 128px swap loses",
        )
        self.s_eye_color = row(
            5, "Iris Lock", 0.0, 100.0, modules.globals.eye_color_lock, 1,
            self._on_eye_color_change,
            lambda v: f"{v:.0f}%",
            "Re-tint the restored iris toward the source face's eye colour, "
            "keeping the source identity's eye colour with live gaze (needs Eye Detail)",
        )
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera").upper())
        layout = QHBoxLayout(card)

        layout.addWidget(QLabel(_("Select Camera:")))
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        layout.addWidget(self.cb_camera, 1)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        layout.addWidget(self.btn_live)

        return card

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(path, (200, 200)))
            self.source_label.setText("")
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            "Media (*.png *.jpg *.jpeg *.gif *.bmp *.mp4 *.mkv)",
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(render_image_preview(temp_path, (200, 200)))
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (200, 200)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_hair_strength_change(self, value: float) -> None:
        modules.globals.hair_transfer_strength = value
        update_status(f"Hair blend set to {value:.0f}%")

    def _on_eyes_mask_change(self, value: float) -> None:
        modules.globals.eyes_mask_size = value
        modules.globals.eyes_mask = value > 0

    def _on_eye_color_change(self, value: float) -> None:
        modules.globals.eye_color_lock = value

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                "Videos (*.mp4 *.mkv)",
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_one_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


def _draw_ai_badge(frame: np.ndarray) -> None:
    """Overlay a small, legible "AI GENERATED" disclosure label in-place.

    Intended for streamers who want automatic on-screen disclosure that a
    live face swap is active, matching platform synthetic-media policies
    (e.g. TikTok requires labeling realistic AI-altered video).
    """
    h, w = frame.shape[:2]
    text = "AI GENERATED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.0, w / 960))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad = 10
    x0, y0 = 10, h - th - baseline - 2 * pad - 10
    x1, y1 = x0 + tw + 2 * pad, h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(
        frame, text, (x0 + pad, y1 - pad - baseline // 2),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        det_interval = max(1, round(self._fps * 0.08))

        # Optical-flow tracking state (single-face mode only). Between full
        # re-detections we nudge the cached bbox/kps toward where the face
        # actually moved instead of holding it frozen for det_interval
        # frames — cuts perceived swap lag on fast head motion.
        flow_prev_gray = None
        flow_pts = None
        hair_source_ready = False
        iris_source_ready = False
        lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    last_source_path = modules.globals.source_path
                    source_image = get_one_face(imread_unicode(modules.globals.source_path))
                    hair_source_ready = False
                    iris_source_ready = False

                # Sample the source identity's iris colour once per source, so
                # Iris Lock has something to tint toward.
                if (
                    modules.globals.eye_color_lock > 0
                    and modules.globals.eyes_mask
                    and not iris_source_ready
                    and source_image is not None
                ):
                    iris_source_ready = eye_detail.prepare_source_iris(
                        modules.globals.source_path, source_image
                    )

                # Segment the source's hair once per source image, off the
                # per-frame path. Retried while disabled→enabled mid-session.
                if modules.globals.hair_transfer and not hair_source_ready and source_image is not None:
                    hair_source_ready = hair_transfer.prepare_source(
                        modules.globals.source_path, source_image
                    )
                    if not hair_source_ready:
                        update_status(
                            "Hair transfer: no usable hair found in the source image."
                        )

                det_count += 1
                is_redetect_frame = det_count % det_interval == 0
                use_flow = (
                    modules.globals.enable_flow_tracking
                    and not modules.globals.many_faces
                )
                flow_gray = None
                if use_flow:
                    flow_gray = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2GRAY)

                if is_redetect_frame:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None
                    # Fresh detection — reset tracking to the true kps.
                    if use_flow and cached_target_face is not None and cached_target_face.kps is not None:
                        flow_pts = cached_target_face.kps.reshape(-1, 1, 2).astype(np.float32)
                        flow_prev_gray = flow_gray
                    else:
                        flow_pts = None
                        flow_prev_gray = None
                elif (
                    use_flow
                    and cached_target_face is not None
                    and flow_pts is not None
                    and flow_prev_gray is not None
                ):
                    # Skip-detection frame — nudge the cached face toward its
                    # actual position via sparse Lucas-Kanade optical flow
                    # instead of leaving the swap frozen at the last detection.
                    try:
                        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                            flow_prev_gray, flow_gray, flow_pts, None, **lk_params
                        )
                    except cv2.error:
                        new_pts, status = None, None

                    if new_pts is not None and status is not None and int(status.sum()) >= 3:
                        valid = status.reshape(-1).astype(bool)
                        old_pts = flow_pts.reshape(-1, 2)
                        upd_pts = new_pts.reshape(-1, 2).copy()
                        shift = np.median(upd_pts[valid] - old_pts[valid], axis=0)
                        # Points optical flow lost individually still move
                        # with the group's median shift, so kps stays a
                        # coherent 5-point set for alignment.
                        upd_pts[~valid] = old_pts[~valid] + shift
                        cached_target_face.kps = upd_pts.astype(np.float32)
                        cached_target_face.bbox = (
                            cached_target_face.bbox
                            + np.array([shift[0], shift[1], shift[0], shift[1]])
                        ).astype(cached_target_face.bbox.dtype)
                        flow_pts = upd_pts.reshape(-1, 1, 2).astype(np.float32)
                        flow_prev_gray = flow_gray
                    else:
                        # Lost track — stop nudging until the next real
                        # detection resyncs us; avoids drifting off-face.
                        flow_pts = None
                        flow_prev_gray = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if (modules.globals.mouth_mask or modules.globals.eyes_mask) and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))

                        # Hair rides on top of the finished swap: inswapper
                        # rebuilds only the face oval, so without this the
                        # target's own hair always survives the swap.
                        if (
                            modules.globals.hair_transfer
                            and not modules.globals.many_faces
                            and cached_target_face is not None
                            and hair_source_ready
                        ):
                            temp_frame = hair_transfer.apply_hair(
                                temp_frame, cached_target_face
                            )

                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            if modules.globals.show_ai_badge:
                _draw_ai_badge(temp_frame)

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        if not self._cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        self._stop_event.set()
        try:
            self._timer.stop()
        except Exception:
            pass
        for worker in (self._capture_worker, self._processing_worker):
            try:
                worker.wait(2000)
            except Exception:
                pass
        try:
            self._cap.release()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        # Title card first; the main window is revealed when it steps aside.
        splash = SplashScreen(modules.metadata.name, modules.metadata.edition)
        splash.finished.connect(self._main.show)
        splash.run()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
