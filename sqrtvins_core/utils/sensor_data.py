"""
Sensor data structs — Python port of ov_core/src/utils/sensor_data.h.

`ImuData` is a `(timestamp, wm, am)` triple; `CameraData` bundles every image
in one frame together with its timestamp and the camera id of each sensor.

Both structs are plain dataclasses. The C++ `operator<` gives STL containers an
ordering — Python's `functools.total_ordering` does the same, and the
`CameraData` tiebreak on the *minimum* sensor id is preserved verbatim because
`VioManager` relies on it when an IMU and a camera frame arrive at the same
timestamp.

`images` and `masks` hold `cv2` arrays (numpy `ndarray`s); the front-end in
`python_frontend/` produces them, so this file never imports `cv2` itself —
numpy arrays are exactly what `cv2.imread` / `cv2.cvtColor` return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering

import numpy as np


@total_ordering
@dataclass
class ImuData:
    """Single IMU reading: `wm` = gyroscope (rad/s), `am` = accelerometer (m/s^2)."""

    timestamp: float
    wm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    am: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.wm = np.asarray(self.wm, dtype=np.float64).reshape(3)
        self.am = np.asarray(self.am, dtype=np.float64).reshape(3)

    def __lt__(self, other: "ImuData") -> bool:
        return self.timestamp < other.timestamp


@total_ordering
@dataclass
class CameraData:
    """One camera frame, possibly from several stereo sensors at once.

    `sensor_ids[i]` identifies the camera that produced `images[i]`;
    `masks[i]` is the optional per-camera tracking mask.
    """

    timestamp: float
    sensor_ids: list[int] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)

    def __lt__(self, other: "CameraData") -> bool:
        if self.timestamp == other.timestamp:
            # Same timestamp: order by the lowest sensor id, as in C++.
            id_min = min(self.sensor_ids) if self.sensor_ids else 0
            other_id_min = min(other.sensor_ids) if other.sensor_ids else 0
            return id_min < other_id_min
        return self.timestamp < other.timestamp
