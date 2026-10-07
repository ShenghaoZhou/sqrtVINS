/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program. If not, see
 * <https://www.gnu.org/licenses/>.
 */

/**
 * @file bindings.cpp
 * @brief Python bindings for the Sqrt-VINS visual frontend.
 *
 * Exposes the camera models, the KLT feature tracker and the feature
 * database, plus the vision::CVBackend abstraction so that the frontend can
 * be extended from Python (e.g. replacing FAST detection with a learned
 * detector such as XFeat) through pysqrtvins.vision.Backend.
 */

#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/imgproc.hpp>

#include "PyBackend.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackKLT.h"

namespace py = pybind11;
using namespace ov_core;

namespace {

/// Copy a grayscale/BGR numpy image into an owned cv::Mat (color images are
/// converted to grayscale). The frontend keeps references to the images
/// beyond the feed call, so we always copy.
cv::Mat numpyToGrayMat(const py::array &arr, const char *what) {
  py::array_t<uint8_t, py::array::forcecast> u8(arr);
  if (u8.ndim() != 2 && u8.ndim() != 3)
    throw std::runtime_error(std::string(what) +
                             ": expected a 2D (grayscale) or 3D (color) "
                             "uint8 image");
  int height = (int)u8.shape(0);
  int width = (int)u8.shape(1);
  int channels = u8.ndim() == 3 ? (int)u8.shape(2) : 1;
  if (channels != 1 && channels != 3 && channels != 4)
    throw std::runtime_error(std::string(what) +
                             ": expected 1, 3 or 4 channels");
  auto buf = u8.unchecked();
  cv::Mat img(height, width, CV_MAKETYPE(CV_8U, channels));
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      for (int c = 0; c < channels; c++)
        img.at<uint8_t>(y, x * channels + c) =
            u8.ndim() == 3 ? buf(y, x, c) : buf(y, x);
  if (channels == 3) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
  } else if (channels == 4) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    return gray;
  }
  return img;
}

/// Convert a cv::Mat (8-bit) into a freshly allocated numpy array.
py::array_t<uint8_t> matToNumpy(const cv::Mat &mat) {
  std::vector<py::ssize_t> shape =
      mat.channels() == 1
          ? std::vector<py::ssize_t>{mat.rows, mat.cols}
          : std::vector<py::ssize_t>{mat.rows, mat.cols, mat.channels()};
  py::array_t<uint8_t> out(shape);
  size_t row_bytes = (size_t)mat.cols * (size_t)mat.channels();
  for (int y = 0; y < mat.rows; y++)
    std::memcpy(out.mutable_data() + (size_t)y * row_bytes,
                mat.ptr<uint8_t>(y), row_bytes);
  return out;
}

/// Convert keypoints into an (N,2) float32 numpy array.
py::array_t<float> keypointsToNumpy(const std::vector<cv::KeyPoint> &kpts) {
  py::array_t<float> out({(py::ssize_t)kpts.size(), (py::ssize_t)2});
  auto buf = out.mutable_unchecked<2>();
  for (size_t i = 0; i < kpts.size(); i++) {
    buf(i, 0) = kpts.at(i).pt.x;
    buf(i, 1) = kpts.at(i).pt.y;
  }
  return out;
}

/// Convert a vector of Vec2 into an (N,2) float32 numpy array.
py::array_t<float> vec2sToNumpy(const std::vector<Vec2> &pts) {
  py::array_t<float> out({(py::ssize_t)pts.size(), (py::ssize_t)2});
  auto buf = out.mutable_unchecked<2>();
  for (size_t i = 0; i < pts.size(); i++) {
    buf(i, 0) = pts.at(i)(0);
    buf(i, 1) = pts.at(i)(1);
  }
  return out;
}

/// Convert a vector of timestamps into a float64 numpy array.
py::array_t<double> doublesToNumpy(const std::vector<double> &vals) {
  py::array_t<double> out((py::ssize_t)vals.size());
  auto buf = out.mutable_unchecked<1>();
  for (size_t i = 0; i < vals.size(); i++)
    buf(i) = vals.at(i);
  return out;
}

/// Convert a vector of ids into a uint64 numpy array.
py::array_t<uint64_t> idsToNumpy(const std::vector<size_t> &ids) {
  py::array_t<uint64_t> out((py::ssize_t)ids.size());
  auto buf = out.mutable_unchecked<1>();
  for (size_t i = 0; i < ids.size(); i++)
    buf(i) = (uint64_t)ids.at(i);
  return out;
}

/// Per-camera map of measurement arrays (cam_id -> numpy).
template <typename F, typename C>
py::dict perCamMap(const std::unordered_map<size_t, F> &map, C &&convert) {
  py::dict out;
  for (const auto &kv : map)
    out[py::int_(kv.first)] = convert(kv.second);
  return out;
}

} // namespace

PYBIND11_MODULE(pysqrtvins, m) {
  m.doc() = "Python API for the Sqrt-VINS visual frontend";

  //=========================================================================
  // vision: backend abstraction and Python extension point
  //=========================================================================
  auto vision = m.def_submodule(
      "vision", "Computer vision backend abstraction (extension point)");

  py::class_<vision::Keypoint>(vision, "Keypoint",
                               "A single detected feature location")
      .def(py::init<>())
      .def_readwrite("x", &vision::Keypoint::x)
      .def_readwrite("y", &vision::Keypoint::y)
      .def_readwrite("response", &vision::Keypoint::response)
      .def_readwrite("size", &vision::Keypoint::size)
      .def_readwrite("octave", &vision::Keypoint::octave)
      .def("__repr__", [](const vision::Keypoint &kp) {
        return "Keypoint(x=" + std::to_string(kp.x) +
               ", y=" + std::to_string(kp.y) +
               ", response=" + std::to_string(kp.response) + ")";
      });

  py::enum_<vision::Op>(vision, "Op", "Individual vision operations")
      .value("HistogramEqualization", vision::Op::HistogramEqualization)
      .value("CLAHE", vision::Op::CLAHE)
      .value("ImagePyramid", vision::Op::ImagePyramid)
      .value("FeatureDetection", vision::Op::FeatureDetection)
      .value("SubpixRefinement", vision::Op::SubpixRefinement)
      .value("SparsePointTracking", vision::Op::SparsePointTracking)
      .value("FundamentalRansac", vision::Op::FundamentalRansac);

  py::class_<vision::CVBackend, std::shared_ptr<vision::CVBackend>>(
      vision, "CVBackend", "A complete computer vision backend")
      .def("name", &vision::CVBackend::name)
      .def("supports", &vision::CVBackend::supports, py::arg("op"));

  vision.def("create_backend", &vision::CVBackend::create, py::arg("name"),
             "Create a built-in backend by name ('opencv', 'ocean')");

  py::class_<PyBackend, vision::CVBackend, std::shared_ptr<PyBackend>>(
      vision, "Backend",
      "Backend dispatching the vision operations to a duck-typed Python "
      "object; operations it does not implement are delegated to the "
      "fallback backend")
      .def(py::init<py::object, std::string, std::string>(),
           py::arg("impl"), py::arg("name") = "python",
           py::arg("fallback") = "opencv");

  //=========================================================================
  // cam: camera models
  //=========================================================================
  auto cam = m.def_submodule("cam", "Camera models");

  py::class_<CamBase, std::shared_ptr<CamBase>>(cam, "CamBase",
                                                "Base pinhole camera model")
      .def("set_value",
           [](CamBase &self, py::array_t<DataType, py::array::forcecast> v) {
             if (v.ndim() != 1 || v.shape(0) != 8)
               throw std::runtime_error(
                   "set_value: expected 8 calibration values (fx, fy, cx, "
                   "cy, k1, k2, p1/k3, p2/k4)");
             VecX calib(8);
             auto buf = v.unchecked<1>();
             for (int i = 0; i < 8; i++)
               calib(i) = buf(i);
             self.set_value(calib);
           },
           py::arg("calib"),
           "Set the 8 camera calibration values (fx, fy, cx, cy, d1..d4)")
      .def("undistort",
           [](CamBase &self, std::pair<float, float> uv) {
             Vec2 out = self.undistort(Vec2(uv.first, uv.second));
             return std::make_pair(out(0), out(1));
           },
           py::arg("uv"),
           "Undistort/normalize a raw uv point, returns (x_norm, y_norm)")
      .def("distort",
           [](CamBase &self, std::pair<float, float> uv) {
             Vec2 out = self.distort(Vec2(uv.first, uv.second));
             return std::make_pair(out(0), out(1));
           },
           py::arg("uv"), "Distort a normalized uv point to raw pixels")
      .def("get_K",
           [](CamBase &self) {
             cv::Matx33d K = self.get_K();
             py::array_t<double> out({(py::ssize_t)3, (py::ssize_t)3});
             auto buf = out.mutable_unchecked<2>();
             for (int r = 0; r < 3; r++)
               for (int c = 0; c < 3; c++)
                 buf(r, c) = K(r, c);
             return out;
           })
      .def("w", &CamBase::w, "Width of the camera images")
      .def("h", &CamBase::h, "Height of the camera images");

  py::class_<CamRadtan, CamBase, std::shared_ptr<CamRadtan>>(
      cam, "CamRadtan", "Radtan (Brown) pinhole camera model")
      .def(py::init<int, int>(), py::arg("width"), py::arg("height"));

  py::class_<CamEqui, CamBase, std::shared_ptr<CamEqui>>(
      cam, "CamEqui", "Equidistant (fisheye) pinhole camera model")
      .def(py::init<int, int>(), py::arg("width"), py::arg("height"));

  //=========================================================================
  // frontend: feature tracking
  //=========================================================================
  auto frontend =
      m.def_submodule("frontend", "Visual frontend (feature tracking)");

  py::enum_<TrackBase::HistogramMethod>(frontend, "HistogramMethod",
                                        "Image pre-processing method")
      .value("NONE", TrackBase::HistogramMethod::NONE)
      .value("HISTOGRAM", TrackBase::HistogramMethod::HISTOGRAM)
      .value("CLAHE", TrackBase::HistogramMethod::CLAHE);

  // NOTE: Feature / FeatureDatabase are also bound by the ov_srvins_py
  // estimator bindings; keep our registrations module-local so both modules
  // can be imported into the same process.
  py::class_<Feature, std::shared_ptr<Feature>>(frontend, "Feature",
                                                py::module_local(),
                                                "A single feature track")
      .def_readonly("featid", &Feature::featid, "Unique feature id")
      .def("timestamps",
           [](const Feature &self) {
             return perCamMap(self.timestamps, &doublesToNumpy);
           })
      .def("uvs",
           [](const Feature &self) { return perCamMap(self.uvs, &vec2sToNumpy); })
      .def("uvs_norm",
           [](const Feature &self) {
             return perCamMap(self.uvs_norm, &vec2sToNumpy);
           })
      .def("num_measurements", [](const Feature &self) {
        size_t n = 0;
        for (const auto &kv : self.timestamps)
          n += kv.second.size();
        return n;
      });

  py::class_<FeatureDatabase, std::shared_ptr<FeatureDatabase>>(
      frontend, "FeatureDatabase", py::module_local(),
      "Database with all feature tracks")
      .def("size", &FeatureDatabase::size)
      .def("features_containing", &FeatureDatabase::features_containing,
           py::arg("timestamp"), py::arg("remove") = false,
           py::arg("skip_deleted") = false,
           "Features with a measurement at the given timestamp")
      .def("features_containing_older",
           &FeatureDatabase::features_containing_older, py::arg("timestamp"),
           py::arg("remove") = false, py::arg("skip_deleted") = false,
           "Features with measurements older than the given timestamp")
      .def("features_not_containing_newer",
           &FeatureDatabase::features_not_containing_newer,
           py::arg("timestamp"), py::arg("remove") = false,
           py::arg("skip_deleted") = false,
           "Features without a measurement newer than the given timestamp");

  py::class_<TrackKLT, std::shared_ptr<TrackKLT>>(
      frontend, "TrackKLT",
      "KLT feature tracker (monocular / stereo visual frontend)")
      .def(py::init([](std::unordered_map<size_t, std::shared_ptr<CamBase>>
                           cameras,
                       int numfeats, int numaruco, bool stereo,
                       TrackBase::HistogramMethod histmethod, int fast_threshold,
                       int gridx, int gridy, int minpxdist, DataType ransacth,
                       std::shared_ptr<vision::CVBackend> backend) {
             return std::make_shared<TrackKLT>(
                 std::move(cameras), numfeats, numaruco, stereo, histmethod,
                 fast_threshold, gridx, gridy, minpxdist, ransacth,
                 std::move(backend));
           }),
           py::arg("cameras"), py::arg("numfeats") = 200,
           py::arg("numaruco") = 1024, py::arg("stereo") = false,
           py::arg("histmethod") = TrackBase::HistogramMethod::HISTOGRAM,
           py::arg("fast_threshold") = 20, py::arg("gridx") = 5,
           py::arg("gridy") = 5, py::arg("minpxdist") = 10,
           py::arg("ransacth") = 1.0, py::arg("backend") = py::none())
      .def(
          "feed",
          [](TrackKLT &self, double timestamp,
             const std::vector<py::array> &images,
             const std::vector<int> &sensor_ids,
             py::object masks_obj) {
            if (images.empty() || images.size() != sensor_ids.size())
              throw std::runtime_error(
                  "feed: images and sensor_ids must be non-empty and of "
                  "equal size");
            std::vector<cv::Mat> masks;
            if (!masks_obj.is_none()) {
              std::vector<py::array> mask_arrs =
                  masks_obj.cast<std::vector<py::array>>();
              if (mask_arrs.size() != images.size())
                throw std::runtime_error(
                    "feed: masks must be None or one mask per image");
              for (size_t i = 0; i < mask_arrs.size(); i++) {
                cv::Mat mask = numpyToGrayMat(mask_arrs.at(i), "feed(masks)");
                if (mask.rows != (int)images.at(i).request().shape.at(0) ||
                    mask.cols != (int)images.at(i).request().shape.at(1))
                  throw std::runtime_error(
                      "feed: mask and image sizes do not match");
                masks.push_back(mask);
              }
            }
            CameraData message;
            message.timestamp = timestamp;
            message.sensor_ids = sensor_ids;
            for (size_t i = 0; i < images.size(); i++) {
              cv::Mat img = numpyToGrayMat(images.at(i), "feed(images)");
              message.images.push_back(img);
              if (masks_obj.is_none())
                message.masks.push_back(
                    cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
              else
                message.masks.push_back(masks.at(i));
            }
            {
              py::gil_scoped_release release;
              self.feed_new_camera(message);
            }
          },
          py::arg("timestamp"), py::arg("images"), py::arg("sensor_ids"),
          py::arg("masks") = py::none(),
          "Process a new camera reading. images is a list of uint8 numpy "
          "arrays (grayscale HxW, or color HxWx3/HxWx4 which is converted "
          "to grayscale); sensor_ids the camera id of each image; masks "
          "optionally one uint8 mask per image (values > 127 are ignored "
          "regions). For stereo tracking pass both images with stereo=True.")
      .def("get_last_obs",
           [](TrackKLT &self) {
             py::dict out;
             for (const auto &kv : self.get_last_obs())
               out[py::int_(kv.first)] = keypointsToNumpy(kv.second);
             return out;
           })
      .def("get_last_ids",
           [](TrackKLT &self) {
             py::dict out;
             for (const auto &kv : self.get_last_ids())
               out[py::int_(kv.first)] = idsToNumpy(kv.second);
             return out;
           })
      .def("get_num_features", &TrackKLT::get_num_features)
      .def("set_num_features", &TrackKLT::set_num_features)
      .def("get_feature_database", &TrackKLT::get_feature_database)
      .def(
          "display_active",
          [](TrackKLT &self, int r1, int g1, int b1, int r2, int g2, int b2,
             const std::string &overlay) {
            cv::Mat img_out;
            self.display_active(img_out, r1, g1, b1, r2, g2, b2, overlay);
            return matToNumpy(img_out);
          },
          py::arg("r1") = 0, py::arg("g1") = 255, py::arg("b1") = 0,
          py::arg("r2") = 255, py::arg("g2") = 0, py::arg("b2") = 0,
          py::arg("overlay") = "",
          "Render the active tracks into a BGR numpy image")
      .def(
          "display_history",
          [](TrackKLT &self, int r1, int g1, int b1, int r2, int g2, int b2,
             const std::vector<size_t> &highlighted,
             const std::string &overlay) {
            cv::Mat img_out;
            self.display_history(img_out, r1, g1, b1, r2, g2, b2, highlighted,
                                 overlay);
            return matToNumpy(img_out);
          },
          py::arg("r1") = 0, py::arg("g1") = 255, py::arg("b1") = 0,
          py::arg("r2") = 255, py::arg("g2") = 0, py::arg("b2") = 0,
          py::arg("highlighted") = std::vector<size_t>(),
          py::arg("overlay") = "",
          "Render the track history into a BGR numpy image");
}
