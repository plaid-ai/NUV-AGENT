from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from nuvion_app.inference.motor import MotorCommand

try:
    import cv2
except Exception as exc:  # pragma: no cover - depends on runtime extras
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(frozen=True)
class TrackingFaceBox:
    x: int
    y: int
    width: int
    height: int
    primary: bool = False
    stale: bool = False

    @classmethod
    def from_face(cls, face: FaceBox, *, primary: bool = False, stale: bool = False) -> "TrackingFaceBox":
        return cls(face.x, face.y, face.width, face.height, primary=primary, stale=stale)


@dataclass(frozen=True)
class TrackingOverlaySnapshot:
    enabled: bool = False
    show_bbox: bool = True
    faces: tuple[TrackingFaceBox, ...] = ()
    deadzone: tuple[int, int, int, int] | None = None
    primary_center: tuple[float, float] | None = None
    status_text: str = ""
    updated_at: float = 0.0


@dataclass(frozen=True)
class TrackingDecision:
    status_text: str
    faces: tuple[FaceBox, ...] = ()
    primary_face: FaceBox | None = None
    stale_face: FaceBox | None = None
    pan_command: MotorCommand | None = None
    tilt_command: MotorCommand | None = None
    centered: bool = False
    deadzone: tuple[int, int, int, int] | None = None


class TrackingOverlayState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = TrackingOverlaySnapshot()

    def update(self, snapshot: TrackingOverlaySnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> TrackingOverlaySnapshot:
        with self._lock:
            return self._snapshot


class FaceDetector:
    def __init__(self) -> None:
        self.ready = False
        self.error = ""
        self._classifier = None
        if cv2 is None:
            self.error = f"opencv unavailable: {_CV2_IMPORT_ERROR}"
            return
        cascade_path = getattr(getattr(cv2, "data", None), "haarcascades", "") + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            self.error = f"failed to load cascade: {cascade_path}"
            return
        self._classifier = classifier
        self.ready = True

    def detect(self, frame_rgb) -> list[FaceBox]:
        if not self.ready or cv2 is None:
            return []
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        results = self._classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in results]
        return faces


class FaceTrackingController:
    def __init__(
        self,
        *,
        detector: FaceDetector,
        deadzone_pct: float,
        lost_timeout_sec: float,
    ) -> None:
        self.detector = detector
        self.deadzone_pct = max(0.01, min(deadzone_pct, 0.45))
        self.lost_timeout_sec = max(0.1, lost_timeout_sec)
        self._last_face: FaceBox | None = None
        self._last_seen_at = 0.0

    def _select_primary(self, faces: list[FaceBox], frame_width: int, frame_height: int) -> FaceBox | None:
        if not faces:
            return None
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        return min(
            faces,
            key=lambda face: (face.center_x - center_x) ** 2 + (face.center_y - center_y) ** 2,
        )

    def _deadzone(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        zone_width = int(frame_width * self.deadzone_pct)
        zone_height = int(frame_height * self.deadzone_pct)
        center_x = frame_width // 2
        center_y = frame_height // 2
        left = center_x - zone_width // 2
        top = center_y - zone_height // 2
        return (left, top, zone_width, zone_height)

    def _commands_for_face(
        self,
        face: FaceBox,
        deadzone: tuple[int, int, int, int],
    ) -> tuple[MotorCommand | None, MotorCommand | None, bool]:
        left, top, zone_width, zone_height = deadzone
        right = left + zone_width
        bottom = top + zone_height

        pan_command: MotorCommand | None = None
        tilt_command: MotorCommand | None = None

        if face.center_x < left:
            pan_command = MotorCommand.LEFT
        elif face.center_x > right:
            pan_command = MotorCommand.RIGHT

        if face.center_y < top:
            tilt_command = MotorCommand.UP
        elif face.center_y > bottom:
            tilt_command = MotorCommand.DOWN

        centered = pan_command is None and tilt_command is None
        return pan_command, tilt_command, centered

    def process_frame(self, frame_rgb) -> TrackingDecision:
        frame_height, frame_width = frame_rgb.shape[:2]
        faces = self.detector.detect(frame_rgb)
        deadzone = self._deadzone(frame_width, frame_height)
        now = time.time()

        primary_face = self._select_primary(faces, frame_width, frame_height)
        if primary_face is not None:
            self._last_face = primary_face
            self._last_seen_at = now
            pan_command, tilt_command, centered = self._commands_for_face(primary_face, deadzone)
            status = "TRACK face centered" if centered else "TRACK face active"
            return TrackingDecision(
                status_text=status,
                faces=tuple(faces),
                primary_face=primary_face,
                pan_command=pan_command,
                tilt_command=tilt_command,
                centered=centered,
                deadzone=deadzone,
            )

        if self._last_face is not None and now - self._last_seen_at <= self.lost_timeout_sec:
            return TrackingDecision(
                status_text="TRACK face lost",
                faces=(),
                stale_face=self._last_face,
                deadzone=deadzone,
            )

        self._last_face = None
        return TrackingDecision(status_text="TRACK face idle", faces=(), deadzone=deadzone)


def build_overlay_snapshot(
    decision: TrackingDecision,
    *,
    enabled: bool,
    show_bbox: bool,
) -> TrackingOverlaySnapshot:
    faces: list[TrackingFaceBox] = []
    primary_center: tuple[float, float] | None = None

    if decision.primary_face is not None:
        primary_center = (decision.primary_face.center_x, decision.primary_face.center_y)

    for face in decision.faces:
        faces.append(
            TrackingFaceBox.from_face(face, primary=decision.primary_face == face)
        )

    if decision.stale_face is not None:
        faces.append(TrackingFaceBox.from_face(decision.stale_face, primary=True, stale=True))
        primary_center = (decision.stale_face.center_x, decision.stale_face.center_y)

    return TrackingOverlaySnapshot(
        enabled=enabled,
        show_bbox=show_bbox,
        faces=tuple(faces),
        deadzone=decision.deadzone,
        primary_center=primary_center,
        status_text=decision.status_text,
        updated_at=time.time(),
    )


def draw_tracking_overlay(context, snapshot: TrackingOverlaySnapshot) -> None:
    if not snapshot.enabled or not snapshot.show_bbox:
        return

    for face in snapshot.faces:
        if face.stale:
            context.set_source_rgba(1.0, 0.85, 0.1, 0.95)
            context.set_line_width(2.0)
        elif face.primary:
            context.set_source_rgba(0.1, 0.9, 0.2, 0.95)
            context.set_line_width(3.0)
        else:
            context.set_source_rgba(1.0, 1.0, 1.0, 0.9)
            context.set_line_width(1.5)
        context.rectangle(face.x, face.y, face.width, face.height)
        context.stroke()

    if snapshot.deadzone is not None:
        left, top, width, height = snapshot.deadzone
        context.set_source_rgba(0.1, 0.8, 1.0, 0.45)
        context.set_line_width(1.0)
        context.rectangle(left, top, width, height)
        context.stroke()

    if snapshot.primary_center is not None:
        context.set_source_rgba(0.1, 0.9, 0.2, 0.95)
        context.arc(snapshot.primary_center[0], snapshot.primary_center[1], 4.0, 0.0, 2.0 * math.pi)
        context.fill()
