/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#ifndef OV_CORE_GRIDER_GRID_H
#define OV_CORE_GRIDER_GRID_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/DataType.h"
#include "utils/opencv_lambda_body.h"
#include "vision/CVBackend.h"
#include "vision/OpenCvBackend.h"
#include "vision/Types.h"

namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in
 * each. We then pick enough top points in each grid so that we have the total
 * number of desired points.
 *
 * The actual corner detection (and optional sub-pixel refinement) is
 * delegated to a vision::FeatureDetector of the active CV backend, so this
 * helper is backend-agnostic.
 */
class Grider_GRID {

public:
  /**
   * @brief Compare keypoints based on their response value.
   * @param first First keypoint
   * @param second Second keypoint
   *
   * We want to have the keypoints with the highest values!
   * See: https://stackoverflow.com/a/10910921
   */
  static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) {
    return first.response > second.response;
  }

  /**
   * @brief This function will perform grid extraction using the feature
   * detector of the given backend.
   * @param img Image we will do the extraction on (level 0 of the pyramid)
   * @param mask Region of the image we do not want to extract features in (255
   * = do not detect features)
   * @param valid_locs grid cells (x,y grid indices) we should extract in
   * @param pts vector of extracted points we will return
   * @param num_features max number of features we want to extract
   * @param grid_x size of grid in the x-direction / u-direction
   * @param grid_y size of grid in the y-direction / v-direction
   * @param threshold FAST threshold paramter (10 is a good value normally)
   * @param nonmaxSuppression if the detector should perform non-max
   * suppression (true normally; note that the Ocean backend always applies
   * non-max suppression)
   * @param detector feature detector operation of the active CV backend
   *
   * Given a specified grid size, this will try to extract fast features from
   * each grid. It will then return the best from each grid in the return
   * vector.
   */
  static void
  perform_griding(const vision::Image &img, const cv::Mat &mask,
                  const std::vector<std::pair<int, int>> &valid_locs,
                  std::vector<cv::KeyPoint> &pts, int num_features, int grid_x,
                  int grid_y, int threshold, bool nonmaxSuppression,
                  vision::FeatureDetector &detector) {

    // Return if there is nothing to extract
    if (valid_locs.empty())
      return;

    // We want to have equally distributed features
    // NOTE: If we have more grids than number of total points, we calc the
    // biggest grid we can do NOTE: Thus if we extract 1 point per grid we have
    // NOTE:    -> 1 = num_features / (grid_x * grid_y)
    // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
    // NOTE:    -> grid_y = sqrt(num_features / ratio)
    if (num_features < grid_x * grid_y) {
      DataType ratio = (DataType)grid_x / (DataType)grid_y;
      grid_y = std::ceil(std::sqrt(num_features / ratio));
      grid_x = std::ceil(grid_y * ratio);
    }
    int num_features_grid =
        (int)((DataType)num_features / (DataType)(grid_x * grid_y)) + 1;
    assert(grid_x > 0);
    assert(grid_y > 0);
    assert(num_features_grid > 0);

    // Calculate the size our extraction boxes should be
    int size_x = img.width / grid_x;
    int size_y = img.height / grid_y;

    // Make sure our sizes are not zero
    assert(size_x > 0);
    assert(size_y > 0);

    // Fast path: backends whose detector is expensive to invoke per cell
    // (e.g. learned detectors wrapped through the Python bindings) can detect
    // on the whole image in a single call; we then distribute the returned
    // keypoints over the grid cells ourselves.
    std::vector<vision::Keypoint> whole_pts;
    if (detector.detectWholeImage(img, mask, threshold, nonmaxSuppression,
                                  whole_pts)) {

      // Bucket the detections into their grid cells
      std::vector<std::vector<vision::Keypoint>> cell_pts(grid_x * grid_y);
      for (const auto &kp : whole_pts) {
        int gx = (int)(kp.x / (float)size_x);
        int gy = (int)(kp.y / (float)size_y);
        if (gx < 0 || gx >= grid_x || gy < 0 || gy >= grid_y)
          continue;
        cell_pts.at(gy * grid_x + gx).push_back(kp);
      }

      // Keep the best num_features_grid detections of each requested cell
      for (const auto &grid : valid_locs) {

        // Skip if we are out of bounds (same check as the per-cell path)
        int x = grid.first * size_x;
        int y = grid.second * size_y;
        if (x + size_x > img.width || y + size_y > img.height)
          continue;

        // Now lets get the top number from this cell
        auto &pts_new = cell_pts.at(grid.second * grid_x + grid.first);
        std::sort(pts_new.begin(), pts_new.end(),
                  [](const vision::Keypoint &first,
                     const vision::Keypoint &second) {
                    return first.response > second.response;
                  });

        // Append the "best" ones to our vector (coordinates are already in
        // the full image frame since we detected on the whole image)
        for (size_t i = 0;
             i < (size_t)num_features_grid && i < pts_new.size(); i++) {

          // Create keypoint
          cv::KeyPoint pt_cor;
          pt_cor.pt.x = pts_new.at(i).x;
          pt_cor.pt.y = pts_new.at(i).y;
          pt_cor.response = pts_new.at(i).response;
          pt_cor.size = pts_new.at(i).size;
          pt_cor.octave = pts_new.at(i).octave;

          // Reject if out of bounds (shouldn't be possible...)
          if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x > img.width ||
              (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y > img.height)
            continue;

          // Check if it is in the mask region
          // NOTE: mask has max value of 255 (white) if it should be removed
          if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127)
            continue;
          pts.push_back(pt_cor);
        }
      }
    } else {

      // Parallelize our 2d grid extraction!!
      std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());
      parallel_for_(
          cv::Range(0, (int)valid_locs.size()),
          LambdaBody([&](const cv::Range &range) {
            for (int r = range.start; r < range.end; r++) {

              // Calculate what cell xy value we are in
              auto grid = valid_locs.at(r);
              int x = grid.first * size_x;
              int y = grid.second * size_y;

              // Skip if we are out of bounds
              if (x + size_x > img.width || y + size_y > img.height)
                continue;

              // Calculate where we should be extracting from
              vision::Image cell = img;
              cell.data = img.data + (size_t)y * img.stride + (size_t)x;
              cell.width = size_x;
              cell.height = size_y;

              // Extract features for this part of the image
              std::vector<vision::Keypoint> pts_new =
                  detector.detect(cell, vision::Image(), threshold,
                                  nonmaxSuppression);

              // Now lets get the top number from this
              std::sort(pts_new.begin(), pts_new.end(),
                        [](const vision::Keypoint &first,
                           const vision::Keypoint &second) {
                          return first.response > second.response;
                        });

              // Append the "best" ones to our vector
              // Note that we need to "correct" the point u,v since we
              // extracted it in a ROI So we should append the location of that
              // ROI in the image
              for (size_t i = 0;
                   i < (size_t)num_features_grid && i < pts_new.size(); i++) {

                // Create keypoint
                cv::KeyPoint pt_cor;
                pt_cor.pt.x = pts_new.at(i).x + (float)x;
                pt_cor.pt.y = pts_new.at(i).y + (float)y;
                pt_cor.response = pts_new.at(i).response;
                pt_cor.size = pts_new.at(i).size;
                pt_cor.octave = pts_new.at(i).octave;

                // Reject if out of bounds (shouldn't be possible...)
                if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x > img.width ||
                    (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y > img.height)
                  continue;

                // Check if it is in the mask region
                // NOTE: mask has max value of 255 (white) if it should be
                // removed
                if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) >
                    127)
                  continue;
                collection.at(r).push_back(pt_cor);
              }
            }
          }));

      // Combine all the collections into our single vector
      for (size_t r = 0; r < collection.size(); r++) {
        pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
      }
    }

    // Return if no points
    if (pts.empty())
      return;

    // Get vector of points
    std::vector<cv::Point2f> pts_refined;
    for (size_t i = 0; i < pts.size(); i++) {
      pts_refined.push_back(pts.at(i).pt);
    }

    // Finally get sub-pixel for all extracted features (if the backend
    // supports it; otherwise the integer positions are kept)
    if (detector.refineSubpix(img, pts_refined, 5, 20, 0.001)) {
      // Save the refined points!
      for (size_t i = 0; i < pts.size(); i++) {
        pts.at(i).pt = pts_refined.at(i);
      }
    }
  }

  /**
   * @brief Legacy OpenCV convenience overload (uses the OpenCV backend).
   */
  static void
  perform_griding(const cv::Mat &img, const cv::Mat &mask,
                  const std::vector<std::pair<int, int>> &valid_locs,
                  std::vector<cv::KeyPoint> &pts, int num_features, int grid_x,
                  int grid_y, int threshold, bool nonmaxSuppression) {
    static std::shared_ptr<vision::CVBackend> backend =
        vision::CVBackend::create("opencv");
    perform_griding(vision::Image::fromCv(img), mask, valid_locs, pts,
                    num_features, grid_x, grid_y, threshold, nonmaxSuppression,
                    backend->featureDetector());
  }
};

} // namespace ov_core

#endif /* OV_CORE_GRIDER_GRID_H */
