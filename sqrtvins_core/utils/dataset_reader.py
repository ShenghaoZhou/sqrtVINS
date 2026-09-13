"""
Dataset readers — Python port of ov_core/src/utils/dataset_reader.h.

Three loaders, all module-level functions (the C++ class is static-only with a
private constructor, so it is not instantiable — nothing to port there):

* :func:`load_gt_file` — ASL/EuRoC-MAV groundtruth CSV → `{timestamp: 17-vector}`
* :func:`get_gt_state` — nearest-neighbor lookup with the 17-vector remap
* :func:`load_simulated_trajectory` — space-separated (t, p, q) trajectory

The EuRoC groundtruth CSV is 17 comma-separated columns::

    timestamp(us), X, Y, Z, q0, q1, q2, q3, vx, vy, vz,
    bgx, bgy, bgz, bax, bay, baz

Note the timestamp is **micro**seconds, not nanoseconds, and the quaternion is
**scalar-first** `[w, x, y, z]` (standard/OpenCV convention) — neither matches
OpenVINS' JPL `[x, y, z, w]`. Both conversions happen in the ports:
``load_gt_file`` divides by ``1e-6``, and ``get_gt_state`` maps column 4 last
into index 1..4, i.e. the returned 17-vector is already JPL-ordered:

    [time, qx, qy, qz, qw, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]

The 17-vector index layout below is therefore exactly the C++ layout and can be
fed straight to the estimator-facing code without a second shuffle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .print import print_debug, print_error, print_warning, RED, RESET, YELLOW


# ---------------------------------------------------------------------------
# Groundtruth (EuRoC / ASL)
# ---------------------------------------------------------------------------

GT_COLUMNS = 17


def load_gt_file(path: str | Path,
                 gt_states: dict[float, np.ndarray] | None = None) -> dict[float, np.ndarray]:
    """Load an ASL-format groundtruth CSV into ``{timestamp_s: 17-vector}``.

    Mirrors `DatasetReader::load_gt_file`. The header line is skipped and the
    timestamp column is converted from microseconds to seconds.
    """
    states: dict[float, np.ndarray] = {} if gt_states is None else gt_states
    states.clear()

    p = Path(path)
    if not p.is_file():
        print_error(f"{RED}ERROR: Unable to open groundtruth file...{RESET}")
        print_error(f"{RED}ERROR: {path}{RESET}")
        raise FileNotFoundError(path)

    # `open(..., encoding=...)` avoids the platform default on Windows; the C++
    # reader just does `getline` + `atof`, which tolerates blank fields.
    with p.open("r", encoding="utf-8") as fh:
        next(fh, None)  # header

        for line in fh:
            fields = line.strip().split(",")
            temp = np.zeros(GT_COLUMNS, dtype=np.float64)
            for i, field in enumerate(fields):
                if i >= GT_COLUMNS:
                    print_error(f"{RED}ERROR: Invalid groudtruth line, too long!{RESET}")
                    print_error(f"{RED}ERROR: {line.strip()}{RESET}")
                    raise ValueError(f"gt line too long: {line!r}")
                # `atof("")` is 0.0 — leave the slot zeroed rather than crashing.
                temp[i] = float(field) if field.strip() else 0.0
            states[1e-6 * temp[0]] = temp

    return states


def get_gt_state(timestep: float, gt_states: dict[float, np.ndarray]) -> np.ndarray | None:
    """Return the 17-vector groundtruth state nearest to ``timestep``.

    Mirrors `DatasetReader::get_gt_state`. Returns ``None`` when no groundtruth
    is within 0.10 s (the C++ snaps the timestamp first, then checks membership).

    ``gt_states`` must have been populated by :func:`load_gt_file`.
    """
    if not gt_states:
        print_error("Groundtruth data loaded is empty, make sure you call load "
                    "before asking for a state.")
        return None

    # Nearest neighbour: the C++ walks the whole map, so do the same rather than
    # bisecting — a `sortedcontainers`-free port is the intent.
    closest_time = float("inf")
    for t in gt_states:
        if abs(t - timestep) < abs(closest_time - timestep):
            closest_time = t

    # Snap the request to the groundtruth timestamp if we are within 0.1 s.
    if abs(closest_time - timestep) < 0.10:
        timestep = closest_time

    state = gt_states.get(timestep)
    if state is None:
        print_warning(f"{YELLOW}Unable to find {timestep:.6f} timestamp in GT "
                      f"file, wrong GT file loaded???:{RESET}")
        return None

    imustate = np.zeros(GT_COLUMNS, dtype=np.float64)
    imustate[0] = timestep          # time
    imustate[1:4] = state[5:8]      # quat: JPL [qx, qy, qz] (scalar-first → moved)
    imustate[4] = state[4]          #        ...and qw goes last
    imustate[5:8] = state[1:4]      # pos
    imustate[8:11] = state[8:11]    # vel
    imustate[11:14] = state[11:14]  # bg
    imustate[14:17] = state[14:17]  # ba
    return imustate


def gt_position(timestep: float, gt_states: dict[float, np.ndarray]) -> np.ndarray | None:
    """Convenience: the groundtruth position at ``timestep``, for ATE tooling."""
    state = get_gt_state(timestep, gt_states)
    return None if state is None else state[5:8].copy()


def gt_quat_jpl(timestep: float, gt_states: dict[float, np.ndarray]) -> np.ndarray | None:
    """Convenience: the groundtruth quaternion at ``timestep``, JPL `[x,y,z,w]`."""
    state = get_gt_state(timestep, gt_states)
    return None if state is None else state[1:5].copy()


# ---------------------------------------------------------------------------
# Simulated trajectory
# ---------------------------------------------------------------------------

def load_simulated_trajectory(path: str | Path) -> list[np.ndarray]:
    """Load a space-separated simulation trajectory.

    Mirrors `DatasetReader::load_simulated_trajectory`. Lines are
    ``(timestamp(s) tx ty tz qx qy qz qw)``, and a line is accepted only if all
    eight fields parsed.

    Two quirks are preserved exactly:

    * Comment lines are skipped only when ``#`` is the **first** character — the
      C++ test is ``!current_line.find("#")``, and ``std::string::find`` returns
      the index, which is falsy only at position 0. A trailing comment is NOT
      skipped in the reference and will be silently truncated here too.
    * The C++ gate is ``i > 7`` on a 1-indexed counter, i.e. ``>= 8`` fields.
    """
    traj_data: list[np.ndarray] = []

    p = Path(path)
    if not p.is_file():
        print_error(f"{RED}ERROR: Unable to open simulation trajectory "
                    f"file...{RESET}")
        print_error(f"{RED}ERROR: {path}{RESET}")
        raise FileNotFoundError(path)

    print_debug(f"loaded trajectory {p.name}")

    with p.open("r", encoding="utf-8") as fh:
        for current_line in fh:
            line = current_line.rstrip("\n")
            if line.startswith("#"):
                continue

            data = np.zeros(8, dtype=np.float64)
            fields = line.split()
            i = 0
            for field in fields:
                if not field or i >= 8:
                    continue
                try:
                    data[i] = float(field)
                except ValueError:
                    continue
                i += 1

            # 1-indexed `i > 7` in C++ == all eight fields parsed.
            if i >= 8:
                traj_data.append(data)

    if not traj_data:
        print_error(f"{RED}ERROR: Could not parse any data from the "
                    f"file!!{RESET}")
        print_error(f"{RED}ERROR: {path}{RESET}")

    return traj_data
