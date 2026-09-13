"""
YamlParser — ↔ ov_core/src/utils/opencv_yaml_parse.h (778 LOC).

`opencv_yaml_parse.h` is a thin wrapper over `cv::FileStorage` that lets
`VioManagerOptions::print_and_load` read a config with `parser->parse_config("key",
val)` where `val` is pre-filled with its default. This port replaces
`cv::FileStorage` with `yaml.safe_load` and keeps the same three-level structure:

    parse_config(name, default)                 main config, top level
    parse_external(external_node, sensor_node,  main config holds `external_node`
                   name, default)               as a relative path to a second
                                                file, which holds `sensor_node`

Type dispatch is by the *type of the default value*, mirroring the C++ which
dispatches on the template parameter `T`:

    default is bool           custom truthy-string parser (0/false/False/FALSE, ...)
    default is int or float   scalar
    default is str            string
    default is a list of      vector (REPLACES the default, does not append --
      scalars                                  cv::FileStorage resizes, so the
                                               pre-filled {1,1,0,0} is a size hint)
    default is a 2-D array    matrix; 4x4 gains the T_cam_imu / T_imu_cam flip

Two deliberate deviations from the C++:
  * the return value is the parsed value (Python has no in/out ref parameter),
    so the default is passed in and returned out;
  * matrices are returned as `numpy.ndarray` rather than `Eigen::MatrixXd`, and
    `Inv_se3` is inlined in numpy so config loading never touches JAX.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

from sqrtvins_core.utils.print import (
    print_debug,
    print_error,
    print_info,
    print_warning,
)

Scalar = Union[int, float, str]
Vector = list[Scalar]
Matrix = np.ndarray
Value = Union[Scalar, bool, Vector, Matrix]

# Every OpenVINS config starts with `%YAML:1.0` -- the colon form that
# cv::FileStorage accepts. PyYAML refuses to parse it in two independent ways,
# so the adapter has to normalize both:
#   * the colon: `%YAML 1.0` is the spec's space form; the colon trips
#     `scan_directive_name` with a ScannerError;
#   * the missing document-start marker: once a directive is present PyYAML
#     demands `---` before the first mapping, else it is a ParserError.
#     OpenVINS' configs never carry one -- cv::FileStorage never needed it.
#
# Only the first line can hold a directive, so a single prefix check suffices,
# and a trailing `# comment` on that line is accepted by PyYAML as-is.
_YAML_DIRECTIVE_COLON = "%YAML:"
_DOC_START = "---"


def _load_yaml(path: Path) -> Any:
    """
    `yaml.safe_load` with OpenVINS' `%YAML:1.0` directive adapted to PyYAML.

    Returns `None` for an empty file, exactly as `yaml.safe_load` does -- callers
    use that to tell "absent" from "present but empty". Raises `yaml.YAMLError`
    on malformed input, which the callers turn into the C++'s hard failure.
    """
    with open(path, "r") as f:
        text = f.read()
    if text.startswith(_YAML_DIRECTIVE_COLON):
        text = text.replace(_YAML_DIRECTIVE_COLON, "%YAML ", 1)
    if text.startswith("%YAML"):
        # splice `---` onto its own line right after the directive line, unless
        # the file already opens with one
        nl = text.find("\n")
        if nl >= 0:
            rest = text[nl + 1:]
            if rest.split("\n", 1)[0].strip() != _DOC_START:
                text = text[: nl + 1] + _DOC_START + "\n" + rest
    return yaml.safe_load(text)


def _inv_se3(T: np.ndarray) -> np.ndarray:
    """
    Analytical SE(3) inverse, numpy flavour: [[R, t], [0 1]] -> [[R^T, -R^T t], [0 1]].

    Inlined rather than importing `Inv_se3` from `quat_ops` so the config path
    stays free of JAX (that one uses `jnp` and `.at[].set`). Mirrors
    `ov_core::Inv_se3` (quat_ops.h:498-504).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Tinverse = np.eye(4, dtype=T.dtype)
    Tinverse[:3, :3] = R.T
    Tinverse[:3, 3] = -R.T @ t
    return Tinverse


def _is_bool_default(default: Any) -> bool:
    # bool is a subclass of int, so it must be tested first.
    return isinstance(default, (bool, np.bool_))


def _is_scalar_default(default: Any) -> bool:
    return isinstance(default, (int, float, np.integer, np.floating))


def _is_vector_default(default: Any) -> bool:
    if not isinstance(default, (list, tuple, np.ndarray)):
        return False
    arr = np.asarray(default)
    return arr.ndim == 1


def _is_matrix_default(default: Any) -> bool:
    if not isinstance(default, np.ndarray):
        return False
    return default.ndim == 2 and default.shape[0] == default.shape[1]


# Truthy strings, copied verbatim from the bool parser
# (opencv_yaml_parse.h:529-534).
_TRUE_STRINGS = {"1", "true", "True", "TRUE"}
_FALSE_STRINGS = {"0", "false", "False", "FALSE"}


class YamlParser:
    """
    Config reader mirroring `ov_core::YamlParser`.

    Args:
        config_path: Path to the main YAML config file.
        fail_if_not_found: If the file cannot be opened, `True` exits the
            process (as `std::exit(EXIT_FAILURE)` does in the C++), `False`
            leaves the parser empty and every later call is a no-op.
        exit_on_error: `False` raises instead of calling `sys.exit` for the two
            hard-failure paths (missing external node, missing external file).
            The C++ has no such flag; it exists so the test suite can assert
            the failure without killing the pytest worker.
    """

    def __init__(
        self,
        config_path: str | Path,
        fail_if_not_found: bool = True,
        exit_on_error: bool = True,
    ) -> None:
        self.config_path_: str = str(config_path)
        self.all_params_found_successfully: bool = True
        self.exit_on_error: bool = exit_on_error
        self.config: dict[str, Any] | None = None

        if not self.config_path_:
            self.config = None
            return

        path = Path(self.config_path_)
        if not path.exists():
            if not fail_if_not_found:
                self.config = None
                return
            self._hard_fail(
                "unable to open the configuration file!\n" + self.config_path_,
            )

        try:
            data = _load_yaml(path)
        except yaml.YAMLError as e:
            # cv::FileStorage fails to open a malformed file; mirror that.
            self._hard_fail(
                "unable to open the configuration file!\n" + self.config_path_
                + " (" + str(e) + ")\n",
            )
            return
        if not fail_if_not_found and data is None:
            self.config = None
            return
        self.config = data if data is not None else {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_config_folder(self) -> str:
        """`config_path_` truncated at the last '/', with a trailing '/'.

        Mirrors `get_config_folder` (opencv_yaml_parse.h:124-126)."""
        return self.config_path_[: self.config_path_.rfind("/")] + "/"

    def successful(self) -> bool:
        """`True` if every required parameter was found and parsed."""
        return self.all_params_found_successfully

    def parse_config(
        self,
        node_name: str,
        default: Value,
        required: bool = True,
    ) -> Value:
        """
        Read a top-level key of the main config file, defaulting to `default`.

        Mirrors `parse_config` (opencv_yaml_parse.h:146-170). The ROS override
        branches are dropped -- they are `#if ROS_AVAILABLE` and this port is
        the ROS-free build, exactly like `NO_ROS=1`.
        """
        if self.config is None:
            return default
        return self._parse_node(self.config, node_name, default, required)

    def parse_external(
        self,
        external_node_name: str,
        sensor_name: str,
        node_name: str,
        default: Value,
        required: bool = True,
    ) -> Value:
        """
        Read `node_name` under `sensor_name` in an external config file.

        The main config holds `external_node_name: <relative path>`; that file
        is opened relative to `get_config_folder()` and `sensor_name` is looked
        up inside it. Mirrors `parse_external` (191-220) and
        `parse_external_yaml` (725-773). The hard failures (missing external
        node, missing file) mirror `std::exit(EXIT_FAILURE)`.
        """
        if self.config is None:
            return default

        if external_node_name not in self.config:
            self._hard_fail(
                "the external node " + external_node_name + " could not be found!",
            )

        relative = str(self.config[external_node_name])
        folder = self.get_config_folder()
        external_path = folder + relative

        ext = Path(external_path)
        if not ext.exists():
            self._hard_fail(
                "unable to open the configuration file!\n" + external_path,
            )
        try:
            external_config = _load_yaml(ext)
        except yaml.YAMLError:
            self._hard_fail(
                "unable to open the configuration file!\n" + external_path,
            )
        if not isinstance(external_config, dict):
            external_config = {}

        if not isinstance(external_config.get(sensor_name), dict):
            print_warning(
                "the sensor " + sensor_name + " was not found...\n"
            )
            self.all_params_found_successfully = False
            return default

        return self._parse_node(
            external_config[sensor_name], node_name, default, required
        )

    # ------------------------------------------------------------------
    # type dispatch
    # ------------------------------------------------------------------

    def _parse_node(
        self, file_node: dict[str, Any], node_name: str, default: Value,
        required: bool,
    ) -> Value:
        """Dispatch on `type(default)`, mirroring the C++ template overloads."""
        if _is_bool_default(default):
            return self._parse_bool(file_node, node_name, default, required)
        if _is_scalar_default(default):
            return self._parse_scalar(file_node, node_name, default, required)
        if _is_vector_default(default):
            return self._parse_vector(file_node, node_name, default, required)
        if _is_matrix_default(default):
            return self._parse_matrix(file_node, node_name, default, required)
        if isinstance(default, str):
            return self._parse_string(file_node, node_name, default, required)
        raise TypeError(
            "YamlParser: unsupported default type "
            + type(default).__name__ + " for node " + node_name
        )

    def _not_found(self, node_name: str, required: bool, default: Value) -> bool:
        """Missing-key bookkeeping; `True` means bail out and keep the default."""
        if required:
            print_warning(
                "the node " + node_name + " of type ["
                + type(default).__name__ + "] was not found...\n"
            )
            self.all_params_found_successfully = False
        else:
            print_debug(
                "the node " + node_name + " of type ["
                + type(default).__name__
                + "] was not found (not required)...\n"
            )
        return True

    def _parse_scalar(
        self, node: dict[str, Any], name: str, default: Value, required: bool,
    ) -> Value:
        if name not in node:
            self._not_found(name, required, default)
            return default
        try:
            return node[name]
        except Exception:
            self._flag_parse_failure(name, required, default)
            return default

    def _parse_string(
        self, node: dict[str, Any], name: str, default: Value, required: bool,
    ) -> Value:
        if name not in node:
            self._not_found(name, required, default)
            return default
        try:
            return str(node[name])
        except Exception:
            self._flag_parse_failure(name, required, default)
            return default

    def _parse_bool(
        self, node: dict[str, Any], name: str, default: Value, required: bool,
    ) -> Value:
        if name not in node:
            self._not_found(name, required, default)
            return default

        try:
            raw = node[name]
            # yaml.safe_load already resolves `true`/`false` to Python bools;
            # handle those plus the integer and string spellings the C++ allows.
            if isinstance(raw, (bool, np.bool_)):
                return bool(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                if int(raw) == 1:
                    return True
                if int(raw) == 0:
                    return False
            # The C++ strips a trailing `# comment` and everything after the
            # first space, since `key: true # comment here` is common:
            #     value = value.substr(0, value.find_first_of('#'))
            # `split(...)[0]` is the faithful form -- `value[:value.find("#")]`
            # is NOT: `find` returns -1 on a miss and Python's `[:-1]` drops the
            # last character, so 'true' would become 'tru'. C++'s substr treats
            # `npos` as the whole string. Unquoted YAML booleans never reach
            # here (safe_load resolves them to real bools), which is why the
            # defect only shows up in the quoted/spelling cases.
            value = str(raw).split("#", 1)[0].split(" ", 1)[0]
            if value in _TRUE_STRINGS:
                return True
            if value in _FALSE_STRINGS:
                return False
            print_warning(
                "the node " + name + " has an invalid boolean type of ["
                + value + "]\n"
            )
            self.all_params_found_successfully = False
            return default
        except Exception:
            self._flag_parse_failure(name, required, default)
            return default

    def _parse_vector(
        self, node: dict[str, Any], name: str, default: Value, required: bool,
    ) -> Value:
        if name not in node:
            self._not_found(name, required, default)
            return default
        try:
            raw = node[name]
            if not isinstance(raw, list):
                raise ValueError("expected a list, got " + type(raw).__name__)
            result = [float(x) if isinstance(default[0], float) else x
                      for x in raw]
            return result
        except Exception:
            self._flag_parse_failure(name, required, default)
            return default

    def _parse_matrix(
        self, node: dict[str, Any], name: str, default: Value, required: bool,
    ) -> Value:
        # The T_cam_imu / T_imu_cam fallback (opencv_yaml_parse.h:616-626).
        name_local = name
        if name == "T_cam_imu" and name not in node:
            if "T_imu_cam" in node:
                print_info(
                    "parameter T_cam_imu not found, trying T_imu_cam instead "
                    "(will return T_cam_imu still)!\n"
                )
                name_local = "T_imu_cam"
        elif name == "T_imu_cam" and name not in node:
            if "T_cam_imu" in node:
                print_info(
                    "parameter T_imu_cam not found, trying T_cam_imu instead "
                    "(will return T_imu_cam still)!\n"
                )
                name_local = "T_cam_imu"

        if name_local not in node:
            self._not_found(name_local, required, default)
            return default

        try:
            raw = node[name_local]
            result = np.array(default, dtype=float, copy=True)
            dims = min(len(raw), result.shape[0])
            for r in range(dims):
                row = raw[r]
                cols = min(len(row), result.shape[1])
                for c in range(cols):
                    result[r, c] = float(row[c])
        except Exception:
            self._flag_parse_failure(name, required, default)
            return default

        # If we flipped the transform, invert it (opencv_yaml_parse.h:667-671).
        if name_local != name and name_local.startswith("T_"):
            result = _inv_se3(result)
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _flag_parse_failure(
        self, name: str, required: bool, default: Value,
    ) -> None:
        if required:
            print_warning(
                "unable to parse " + name + " node of type ["
                + type(default).__name__ + "] in the config file!\n"
            )
        else:
            print_debug(
                "unable to parse " + name + " node of type ["
                + type(default).__name__
                + "] in the config file (not required)\n"
            )
        self.all_params_found_successfully = False

    def _hard_fail(self, message: str) -> None:
        """Mirror `PRINT_ERROR(...) + std::exit(EXIT_FAILURE)`."""
        print_error(message)
        if self.exit_on_error:
            sys.exit(1)
        raise FileNotFoundError(message)
