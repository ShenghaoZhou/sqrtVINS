# ${\sqrt{\rm VINS}}$

[![ROS 1 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml)
[![ROS 2 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml)
[![ROS Free Workflow](https://github.com/rpng/open_vins/actions/workflows/build.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build.yml)

<p align="center">
  <img src="ov_srvins/demo/Peng2025TRO.gif" alt="VINS Motion Tracking Demo" width="480"/>
</p>

Welcome to the ${\sqrt{\rm VINS}}$ project! This repository provides the first open-source implementation of a robust and efficient square-root filter (SRF)-based visual-inertial navigation system (VINS), see [arXiv](https://arxiv.org/abs/2510.10346). 
Extended from [OpenVINS](https://github.com/rpng/open_vins), unlike conventional EKF- or MSCKF-based systems, ${\sqrt{\rm VINS}}$ leverages a novel LLT-based SRF update to preserve the triangular structure of the covariance matrix, ensuring numerical stability, guaranteed positive semi-definiteness, and efficient memory usage. This makes it especially well-suited for embedded robotic platforms and systems constrained by limited precision.

Key highlights of ${\sqrt{\rm VINS}}$ include:

⚡ Ultra-fast and stable filtering – runs efficiently wih 32-bit single-precision on embedded hardware (e.g., Jetson Nano in 5W mode with single thread (<1 GHz CPU)).

🧮 Cholesky (LLT)-based SRF update – fully exploits system structure to achieve significant speedups over canonical SRFs.

🚀 Dynamic initialization – recovers minimal states without triangulating 3D features, enabling reliable startup even in extreme conditions (as small as a 100 ms time window).


## Dual-Formulation Support (SqrtVINS / Original OpenVINS)

This consolidated codebase contains BOTH filter formulations in a single build, sharing one `ov_core`:

| Module | Formulation |
|---|---|
| `ov_srvins` | SqrtVINS square-root EKF (this project) |
| `ov_msckf` + `ov_init` | Original OpenVINS full-covariance EKF (ported non-ROS core) |

Both are selected with a flag on the unified EuRoC runner:

```shell
# Build (pixi environment; Ceres/Boost come from the env configured via CMAKE_PREFIX_PATH)
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PIXI_ENV;$CERES_ENV"
ninja -C build run_euroc ov_srvins_py

# SqrtVINS formulation (default)
build/ov_srvins/run_euroc <dataset> config/euroc_mav/estimator_config.yaml out_sqrt.csv

# Original OpenVINS full-covariance formulation
build/ov_srvins/run_euroc <dataset> config/euroc_mav/estimator_config_full.yaml out_full.csv --estimator full
```

Notes:
- `--estimator sqrt|full` (aliases `--sqrt` / `--full`, or `--estimator=<value>`) selects the formulation; the default is `sqrt`.
- The two formulations are tuned differently: `estimator_config.yaml` carries the sqrt tuning, while `estimator_config_full.yaml` carries the upstream OpenVINS init/update tuning (e.g. `init_window_time`, `dt_slam_delay`).
- In Python, the same module exposes both: `SqrtEstimator` / `Frontend` (sqrt, orchestrated manually) and `FullVioManager` (self-contained, like upstream `VioManager`).
- The formulation-specific sources live in separate translation units (`run_euroc_sqrt.cpp` / `run_euroc_full.cpp`, `pybind_full.cpp`) because the two modules expose colliding header paths.

## Dependences
* ROS
* Eigen
* OpenCV

## Getting Started
### Run in Native System with EurocMav Dataset (Ubuntu 20.04 + ROS1 as Example)
```shell
# Step 1: Create the workspace
mkdir -p sqrt_vins_ws/src && cd sqrt_vins_ws/src
git clone https://github.com/rpng/sqrtVINS.git

# Step 2: Build 
catkin build

# Step 3: Download eurocmav rosbag(https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) in $HOME/datasets/euroc_mav or change the path in /ov_srvins/launch/serial.launch accordingly

# Step 4: Run the launch file
source devel/setup.bash
roslaunch ov_srvins serial.launch
```

### Docker Support
As the code base is extended from [OpenVINS](https://github.com/rpng/open_vins), the dockcer support can refer to OpenVINS's [doc](https://docs.openvins.com/dev-docker.html).

### Generate Results with Scripts
Please refer to ``src/srf_init/ov_srvins/scripts/run_abtest.sh``

## Credit / Licensing

This code was written by the [Robot Perception and Navigation Group (RPNG)](https://sites.udel.edu/robot/) at the
University of Delaware. If you have any issues with the code please open an issue on our github page with relevant
implementation details and references. For researchers that have leveraged or compared to this work, please cite the
following:

```txt
@InProceedings{Peng2024ICRA,
  title     = {Ultrafast Square-Root Filter-based VINS}, 
  author    = {Peng, Yuxiang and Chen, Chuchu and Huang, Guoquan},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2024},
  address   = {Yokohama, Japan},
  month     = {may}
}

@Article{Peng2025TRO,
  title     = {sqrt-VINS: Robust and Ultrafast Square-Root Filter-based 3D Motion Tracking},
  author    = {Peng, Yuxiang and Chen, Chuchu and Wu, Kejian and Huang, Guoquan},
  journal   = {IEEE Transactions on Robotics (TRO)},
  year      = {2025},
  month     = {oct},
  url       = {https://ieeexplore.ieee.org/document/11219345}
}
```

The codebase is licensed under the [GNU Lesser General Public License v3 (LGPL-3.0)](https://www.gnu.org/licenses/lgpl-3.0.txt). You must preserve the copyright and license notices in your derivative work. If you modify the library itself, you must make those modifications available under the same license. You may link this library into your own programs (including proprietary software), provided that users are able to replace or relink with a modified version of the library. ([See this](https://choosealicense.com/licenses/lgpl-3.0/); this is not legal advice.)


