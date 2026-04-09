#ifndef NDARRAY_CONVERTER_H
#define NDARRAY_CONVERTER_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

/**
 * @brief This is a simple type caster to allow for conversion between cv::Mat and numpy arrays.
 * This is based on the following: https://github.com/edmarriner/pybind11-opencv-numpy
 */
namespace pybind11 {
namespace detail {

template <> struct type_caster<cv::Mat> {
public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  /**
   * @brief Conversion part 1 (Python -> C++)
   */
  bool load(handle src, bool) {
    if (!isinstance<array>(src)) {
      return false;
    }

    auto buf = array::ensure(src);
    if (!buf) {
      return false;
    }

    auto dims = buf.ndim();
    if (dims < 2 || dims > 3) {
      return false;
    }

    int rows = buf.shape(0);
    int cols = buf.shape(1);
    int type = CV_8UC1;

    if (dims == 3) {
      int channels = buf.shape(2);
      if (channels == 3) {
        type = CV_8UC3;
      } else if (channels == 1) {
        type = CV_8UC1;
      } else {
        return false;
      }
    }

    // NOTE: This does NOT copy the data, so the numpy array must stay alive!
    value = cv::Mat(rows, cols, type, (void *)buf.data());
    return true;
  }

  /**
   * @brief Conversion part 2 (C++ -> Python)
   */
  static handle cast(const cv::Mat &m, return_value_policy, handle) {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t elemsize = 1;
    std::string format = "B";

    if (m.depth() == CV_8U) {
      format = "B";
      elemsize = 1;
    } else if (m.depth() == CV_32F) {
      format = "f";
      elemsize = 4;
    } else if (m.depth() == CV_64F) {
      format = "d";
      elemsize = 8;
    } else {
      throw std::runtime_error("Unsupported cv::Mat depth for numpy conversion");
    }

    if (m.dims == 2) {
      shape = {(size_t)m.rows, (size_t)m.cols};
      strides = {(size_t)m.step[0], (size_t)m.step[1]};
      if (m.channels() > 1) {
        shape.push_back((size_t)m.channels());
        strides.push_back((size_t)elemsize);
      }
    } else {
      for (int i = 0; i < m.dims; i++) {
        shape.push_back((size_t)m.size[i]);
        strides.push_back((size_t)m.step[i]);
      }
    }

    return array(buffer_info((void *)m.data, elemsize, format, shape.size(), shape, strides)).release();
  }
};

} // namespace detail
} // namespace pybind11

#endif // NDARRAY_CONVERTER_H
