"""
CameraPoseBuffer — JAX port of ov_srvins/src/utils/CameraPoseBuffer.h/.cpp.

A ring buffer of camera poses indexed by `(camera_id, timestamp)`. The C++ uses
`std::map<double, int>` for the timestamp -> clone_id map and a flat
`std::vector<PoseData>` of size `max_clone_size * max_camera_size` for the
underlying storage. Slot layout: `max_clone_size_ * camera_id + clone_id`.

Because `clone_id` is a rolling counter (mod `max_clone_size_`), each timestamp
occupies a fresh slot when it is added and reuses an older slot once the ring
wraps around — the older entry is invalidated by calling `remove_timestamp` on
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PoseData:
    """Camera pose in the global frame.

    `R_GtoC`  — 3x3 rotation from G to C.
    `p_CinG`  — 3-vector, position of C in G.
    """

    R_GtoC: np.ndarray = None
    p_CinG: np.ndarray = None

    def __post_init__(self) -> None:
        if self.R_GtoC is None:
            self.R_GtoC = np.eye(3, dtype=np.float64)
        if self.p_CinG is None:
            self.p_CinG = np.zeros(3, dtype=np.float64)

    def copy(self) -> "PoseData":
        return PoseData(self.R_GtoC.copy(), self.p_CinG.copy())


class CameraPoseBuffer:
    def __init__(self, max_clone_size: int, max_camera_size: int) -> None:
        self.max_clone_size_ = max_clone_size
        self.buffer_ = [PoseData() for _ in range(max_clone_size * max_camera_size)]
        self.timestamps_id_: Dict[float, int] = {}
        self.curr_clone_id_ = 0

    # -------------------------------------------------------------- access
    def get_buffer_unsafe(self, camera_id: int, timestamp: float) -> PoseData:
        idx = self.get_buffer_index(camera_id, timestamp)
        if idx < 0 or idx >= len(self.buffer_):
            raise IndexError(
                f"CameraPoseBuffer.get_buffer_unsafe: bad index {idx} "
                f"(cam {camera_id}, ts {timestamp})"
            )
        return self.buffer_[idx]

    def add_timestamp(self, timestamp: float) -> None:
        if timestamp not in self.timestamps_id_:
            self.timestamps_id_[timestamp] = self.curr_clone_id_
            self.curr_clone_id_ += 1
            self.curr_clone_id_ %= self.max_clone_size_

    def remove_timestamp(self, timestamp: float) -> None:
        self.timestamps_id_.pop(timestamp, None)

    # ------------------------------------------------------------- internals
    def get_buffer_index(self, camera_id: int, timestamp: float) -> int:
        if timestamp not in self.timestamps_id_:
            return -1
        clone_id = self.timestamps_id_[timestamp]
        return self.max_clone_size_ * camera_id + clone_id
