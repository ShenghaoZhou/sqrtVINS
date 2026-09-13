"""
Gate 1b — YamlParser tests.

Validates `sqrtvins_core/utils/yaml_parse.py` against:
  * the real `config/euroc_mav/` files, including the two-level
    `relative_config_imucam` -> `cam0` indirection the C++ uses for
    intrinsics / extrinsics / noise;
  * the T_imu_cam / T_cam_imu fallback and its `Inv_se3` inversion;
  * every truthy/falsey boolean spelling plus trailing `# comment` stripping;
  * required vs. optional missing-key bookkeeping (`successful()`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sqrtvins_core.utils import YamlParser as YamlParserFromPackage
from sqrtvins_core.utils.print import set_print_level
from sqrtvins_core.utils.yaml_parse import YamlParser, _inv_se3


def test_package_reexport():
    """`YamlParser` is reachable through `sqrtvins_core.utils`."""
    assert YamlParserFromPackage is YamlParser

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "euroc_mav"
MAIN_CONFIG = CONFIG_DIR / "estimator_config_srvins.yaml"


@pytest.fixture(autouse=True)
def _quiet():
    # Keep the intentional missing-key warnings out of the test output.
    set_print_level("ERROR")
    yield
    set_print_level("ALL")


@pytest.fixture
def parser() -> YamlParser:
    assert MAIN_CONFIG.exists(), f"missing config: {MAIN_CONFIG}"
    return YamlParser(MAIN_CONFIG)


# ---------------------------------------------------------------------------
# scalar / string / vector / bool over the real config
# ---------------------------------------------------------------------------

def test_parse_config_scalars(parser: YamlParser):
    assert parser.parse_config("gravity_mag", 9.81) == 9.81
    assert parser.parse_config("max_cameras", 1) == 2
    assert parser.parse_config("max_clones", 0) == 11
    assert parser.parse_config("max_slam", 0) == 50
    assert parser.parse_config("track_frequency", 0.0) == pytest.approx(21.0)
    assert parser.parse_config("init_ba_huber_th", 0.0) == pytest.approx(999.0)
    assert parser.parse_config("dt_slam_delay", 0.0) == 0.0
    # an unset key keeps its default
    assert parser.parse_config("up_slam_sigma_px", 3.0) == 1


def test_parse_config_strings(parser: YamlParser):
    assert parser.parse_config("integration", "rk4") == "rk4"
    assert parser.parse_config("histogram_method", "NONE") == "HISTOGRAM"
    assert parser.parse_config("feat_rep_msckf", "") == "GLOBAL_3D"
    # quoted and unquoted strings both resolve to str
    assert parser.parse_config("record_timing_filepath", "") == \
        "/tmp/traj_timing.txt"


def test_parse_config_vectors(parser: YamlParser):
    # The C++ pre-fills {1,1,0,0} as a size hint and cv::FileStorage REPLACES
    # the vector, so the parsed length comes from the YAML, not the default.
    assert parser.parse_config(
        "init_dyn_bias_g", [0.0, 0.0, 0.0]
    ) == pytest.approx([-0.0022, 0.0216, 0.0783])
    assert parser.parse_config(
        "init_dyn_bias_a", [0.0, 0.0, 0.0]
    ) == pytest.approx([-0.0358, -0.0005, 0.0145])


def test_parse_config_booleans(parser: YamlParser):
    assert parser.parse_config("use_klt", False) is True
    assert parser.parse_config("use_aruco", True) is False
    assert parser.parse_config("try_zupt", True) is False
    assert parser.parse_config("use_fej", False) is True
    assert parser.parse_config("zupt_only_at_beginning", True) is False
    assert parser.parse_config("downsample_cameras", True) is False
    assert parser.parse_config("downsize_aruco", False) is True


# ---------------------------------------------------------------------------
# two-level external config: intrinsics / extrinsics / noise
# ---------------------------------------------------------------------------

def test_get_config_folder(parser: YamlParser):
    assert parser.get_config_folder().endswith("euroc_mav/")


def test_parse_external_intrinsics(parser: YamlParser):
    intrinsics = parser.parse_external(
        "relative_config_imucam", "cam0", "intrinsics", [1.0, 1.0, 0.0, 0.0]
    )
    assert intrinsics == pytest.approx([458.654, 457.296, 367.215, 248.375])

    dist = parser.parse_external(
        "relative_config_imucam", "cam0", "distortion_coeffs",
        [0.0, 0.0, 0.0, 0.0],
    )
    assert dist == pytest.approx([-0.283408, 0.0739591, 0.00019359, 1.76187e-05])

    model = parser.parse_external(
        "relative_config_imucam", "cam0", "distortion_model", "radtan"
    )
    assert model == "radtan"

    # `timeshift_cam_imu` is absent from the euroc chain -> optional default.
    assert parser.parse_external(
        "relative_config_imu", "imu0", "timeshift_cam_imu", 0.0, required=False
    ) == 0.0


def test_parse_external_int_vector(parser: YamlParser):
    resolution = parser.parse_external(
        "relative_config_imucam", "cam0", "resolution", [1, 1]
    )
    assert resolution == [752, 480]
    assert all(isinstance(v, int) for v in resolution)


def test_parse_external_matrix(parser: YamlParser):
    T_CtoI = parser.parse_external(
        "relative_config_imucam", "cam0", "T_imu_cam", np.eye(4)
    )
    assert T_CtoI.shape == (4, 4)
    np.testing.assert_allclose(
        T_CtoI[0, :],
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    )
    np.testing.assert_allclose(T_CtoI[3, :], [0.0, 0.0, 0.0, 1.0])
    # it must be a proper SE(3) matrix: orthonormal block, unit last row/col
    R = T_CtoI[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_parse_external_matrix_flip_to_cam_imu(parser: YamlParser):
    """Asking for `T_cam_imu` when only `T_imu_cam` exists returns the inverse."""
    T_CtoI = parser.parse_external(
        "relative_config_imucam", "cam0", "T_imu_cam", np.eye(4)
    )
    T_CtoI_parsed = parser.parse_external(
        "relative_config_imucam", "cam0", "T_cam_imu", np.eye(4)
    )
    # T_cam_imu = Inv_se3(T_imu_cam)
    np.testing.assert_allclose(T_CtoI_parsed, _inv_se3(T_CtoI), atol=1e-14)
    np.testing.assert_allclose(
        T_CtoI @ T_CtoI_parsed, np.eye(4), atol=1e-13
    )


def test_parse_external_second_camera(parser: YamlParser):
    T1 = parser.parse_external(
        "relative_config_imucam", "cam1", "T_imu_cam", np.eye(4)
    )
    T0 = parser.parse_external(
        "relative_config_imucam", "cam0", "T_imu_cam", np.eye(4)
    )
    assert not np.allclose(T1, T0)


def test_parse_external_missing_sensor_keeps_default(parser: YamlParser):
    before = parser.successful()
    assert parser.parse_external(
        "relative_config_imucam", "cam9", "intrinsics", [1.0] * 4, required=False
    ) == [1.0] * 4
    assert parser.successful() is False
    assert before is True


def test_parse_external_missing_node_exits(parser: YamlParser):
    """A missing external node is a hard failure in the C++ (`std::exit`)."""
    p = YamlParser(MAIN_CONFIG, exit_on_error=False)
    with pytest.raises(FileNotFoundError, match="could not be found"):
        p.parse_external(
            "relative_config_nonexistent", "imu0", "sigma_w", 0.0,
            required=False,
        )


def test_missing_file_without_fail_flag_is_noop(tmp_path: Path):
    missing = tmp_path / "no_such.yaml"
    p = YamlParser(missing, fail_if_not_found=False)
    assert p.config is None
    # every call is a no-op returning the default
    assert p.parse_config("gravity_mag", 9.81) == 9.81
    assert p.parse_config("max_cameras", 1) == 1
    assert p.successful() is True


def test_missing_file_default_exits(tmp_path: Path):
    """The default flags mirror `std::exit(EXIT_FAILURE)`."""
    with pytest.raises(SystemExit):
        YamlParser(tmp_path / "no_such.yaml")


def test_missing_file_raise_instead_of_exit(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="unable to open"):
        YamlParser(tmp_path / "no_such.yaml", exit_on_error=False)


# ---------------------------------------------------------------------------
# required vs. optional missing-key bookkeeping
# ---------------------------------------------------------------------------

def test_optional_missing_key_keeps_default(parser: YamlParser):
    assert parser.successful() is True
    assert parser.parse_config("totally_absent_key", 42.0, required=False) == 42.0
    # optional misses do NOT trip successful()
    assert parser.successful() is True


def test_required_missing_key_flags_failure(parser: YamlParser):
    assert parser.parse_config("totally_absent_key", 42.0) == 42.0
    assert parser.successful() is False


def test_ransac_th_is_optional_in_callee(parser: YamlParser):
    """`VioManagerOptions.cpp:297` passes `required=false` for ransac_th."""
    assert parser.parse_config("ransac_th", 1.0, required=False) == 2.0


# ---------------------------------------------------------------------------
# synthetic: every boolean spelling + comment stripping
# ---------------------------------------------------------------------------

def test_bool_spellings_and_comments(tmp_path: Path):
    cfg = tmp_path / "bools.yaml"
    cfg.write_text(
        "one_int: 1\n"
        "zero_int: 0\n"
        "true_lower: true\n"
        "true_title: True\n"
        "true_upper: TRUE\n"
        "false_lower: false\n"
        "false_title: False\n"
        "false_upper: FALSE\n"
        "one_true_spelling: '1'\n"
        "zero_false_spelling: '0'\n"
        "true_with_comment: true # this is a comment\n"
        "false_with_comment: false # another one\n"
        "one_with_comment: 1 # trailing\n"
        "true_quoted: 'true'\n"
        "invalid: banana\n"
    )
    p = YamlParser(cfg)

    for key in ("one_int", "one_true_spelling", "true_lower", "true_title",
                "true_upper", "one_with_comment", "true_with_comment",
                "true_quoted"):
        assert p.parse_config(key, False) is True, key

    for key in ("zero_int", "zero_false_spelling", "false_lower", "false_title",
                "false_upper", "false_with_comment"):
        assert p.parse_config(key, True) is False, key

    # an invalid spelling warns and returns the default (as the C++ does),
    # but trips successful().
    assert p.parse_config("invalid", True) is True
    assert p.successful() is False


def test_missing_config_is_noop(tmp_path: Path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    p = YamlParser(cfg)
    assert p.config == {}
    assert p.parse_config("anything", 1.5) == 1.5


def test_yaml_directive_colon_form_is_adapted(tmp_path: Path):
    """
    OpenVINS' `%YAML:1.0` idiom: cv::FileStorage accepts the colon, PyYAML
    rejects it -- the port must adapt it, since every config in this repo
    (all 60+ files) carries it.
    """
    cfg = tmp_path / "directive.yaml"
    cfg.write_text(
        "%YAML:1.0 # need to specify the file type at the top!\n"
        "\n"
        "verbosity: \"INFO\" # ALL, DEBUG, INFO, WARNING, ERROR, SILENT\n"
        "\n"
        "use_klt: true\n"
        "intrinsics: [100.0, 100.0, 50.0, 50.0]\n"
    )
    p = YamlParser(cfg)
    assert p.parse_config("use_klt", False) is True
    assert p.parse_config("intrinsics", [0.0] * 4) == pytest.approx(
        [100.0, 100.0, 50.0, 50.0]
    )
    assert p.parse_config("verbosity", "") == "INFO"
    assert p.successful() is True


def test_yaml_directive_spec_form_still_works(tmp_path: Path):
    """A spec-compliant config (space form + `---`) must parse unchanged."""
    cfg = tmp_path / "spec.yaml"
    cfg.write_text("%YAML 1.0\n---\na: 1\nb: [2.0, 3.0]\n")
    p = YamlParser(cfg)
    assert p.parse_config("a", 0) == 1
    assert p.parse_config("b", [0.0, 0.0]) == pytest.approx([2.0, 3.0])


def test_no_directive_parses(tmp_path: Path):
    cfg = tmp_path / "plain.yaml"
    cfg.write_text("a: 1\n")
    p = YamlParser(cfg)
    assert p.parse_config("a", 0) == 1


def test_malformed_config_exits(tmp_path: Path):
    """
    A syntactically broken config is a failed `isOpened()` in the C++, which
    is `PRINT_ERROR + std::exit`.
    """
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("a: [unclosed\n  broken: {{{{\n")
    with pytest.raises(FileNotFoundError, match="unable to open"):
        YamlParser(cfg, exit_on_error=False)


# ---------------------------------------------------------------------------
# the SE(3) inverse used by the matrix flip
# ---------------------------------------------------------------------------

def test_inv_se3():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([1.0, 2.0, 3.0])

    Tinverse = _inv_se3(T)
    np.testing.assert_allclose(T @ Tinverse, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(Tinverse @ T, np.eye(4), atol=1e-14)
    # translation: -R^T t
    np.testing.assert_allclose(Tinverse[:3, 3], -R.T @ T[:3, 3], atol=1e-14)


def test_matrix_parse_overwrites_identity(parser: YamlParser):
    """A matrix default is Identity, and only the parsed entries replace it."""
    T = parser.parse_external(
        "relative_config_imucam", "cam0", "T_imu_cam", np.eye(4)
    )
    np.testing.assert_allclose(T[3, 3], 1.0)
    np.testing.assert_allclose(T[0, 0], 0.0148655429818, atol=1e-12)


def test_all_required_params_found_on_euroc_config():
    """
    End-to-end: parse every key VioManagerOptions::print_and_load reads with
    required=true and confirm the euroc config satisfies all of them.
    """
    p = YamlParser(MAIN_CONFIG)
    required_keys = [
        ("dt_slam_delay", 2.0),
        ("try_zupt", False),
        ("zupt_max_velocity", 1.0),
        ("zupt_noise_multiplier", 1.0),
        ("zupt_max_disparity", 1.0),
        ("zupt_only_at_beginning", False),
        ("gravity_mag", 9.81),
        ("max_cameras", 2),
        ("downsample_cameras", False),
        ("use_mask", False),
        ("use_stereo", True),
        ("use_klt", True),
        ("use_aruco", True),
        ("downsize_aruco", True),
        ("num_opencv_threads", 4),
        ("num_pts", 150),
        ("fast_threshold", 20),
        ("grid_x", 5),
        ("grid_y", 5),
        ("min_px_dist", 10),
        ("knn_ratio", 0.85),
        ("track_frequency", 20.0),
        ("up_msckf_sigma_px", 1.0),
        ("up_msckf_chi2_multipler", 1.0),
        ("up_slam_sigma_px", 1.0),
        ("up_slam_chi2_multipler", 1.0),
        ("zupt_chi2_multipler", 1.0),
    ]
    for key, default in required_keys:
        p.parse_config(key, default)
    for cam in ("cam0", "cam1"):
        p.parse_external(
            "relative_config_imucam", cam, "distortion_model", "radtan"
        )
        p.parse_external(
            "relative_config_imucam", cam, "intrinsics", [1.0] * 4
        )
        p.parse_external(
            "relative_config_imucam", cam, "distortion_coeffs", [0.0] * 4
        )
        p.parse_external(
            "relative_config_imucam", cam, "resolution", [1, 1]
        )
        p.parse_external(
            "relative_config_imucam", cam, "T_imu_cam", np.eye(4)
        )
    assert p.successful() is True
