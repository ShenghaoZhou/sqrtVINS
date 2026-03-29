#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_srvins;

// Global instance of VioManager
static std::unique_ptr<VioManager> g_vio_manager = nullptr;
static std::mutex g_vio_mutex;

extern "C" JNIEXPORT jboolean JNICALL
Java_edu_udel_rpng_sqrtvins_SqrtVinsNative_init(JNIEnv* env, jobject thiz, jstring config_path) {
    std::lock_guard<std::mutex> lock(g_vio_mutex);
    
    const char* path = env->GetStringUTFChars(config_path, nullptr);
    std::string config_str(path);
    env->ReleaseStringUTFChars(config_path, path);
    
    try {
        auto parser = std::make_shared<ov_core::YamlParser>(config_str);
        if (!parser->successful()) {
            return JNI_FALSE;
        }

        VioManagerOptions options;
        options.print_and_load(parser);
        
        g_vio_manager = std::make_unique<VioManager>(options);
    } catch (...) {
        return JNI_FALSE;
    }
    
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_edu_udel_rpng_sqrtvins_SqrtVinsNative_feedImu(JNIEnv* env, jobject thiz, jdouble timestamp, 
                                                   jdouble gx, jdouble gy, jdouble gz, 
                                                   jdouble ax, jdouble ay, jdouble az) {
    std::lock_guard<std::mutex> lock(g_vio_mutex);
    if (!g_vio_manager) return;

    ImuData data;
    data.timestamp = timestamp;
    data.wm << (DataType)gx, (DataType)gy, (DataType)gz;
    data.am << (DataType)ax, (DataType)ay, (DataType)az;
    
    g_vio_manager->feed_measurement_imu(data);
}

extern "C" JNIEXPORT void JNICALL
Java_edu_udel_rpng_sqrtvins_SqrtVinsNative_feedImage(JNIEnv* env, jobject thiz, jdouble timestamp, 
                                                     jlong mat_addr) {
    std::lock_guard<std::mutex> lock(g_vio_mutex);
    if (!g_vio_manager) return;

    cv::Mat& img = *(cv::Mat*)mat_addr;
    
    CameraData data;
    data.timestamp = timestamp;
    data.sensor_ids.push_back(0); // Assuming single camera for now
    data.images.push_back(img.clone());
    
    g_vio_manager->feed_measurement_camera(data);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_edu_udel_rpng_sqrtvins_SqrtVinsNative_getPose(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_vio_mutex);
    if (!g_vio_manager || !g_vio_manager->initialized()) {
        return nullptr;
    }

    auto state = g_vio_manager->get_state();
    if (!state || !state->imu) {
        return nullptr;
    }
    
    Vec4 q = state->imu->quat();
    Vec3 p = state->imu->pos();
    
    jdoubleArray result = env->NewDoubleArray(7);
    jdouble fill[7];
    fill[0] = (double)q(0); // qx
    fill[1] = (double)q(1); // qy
    fill[2] = (double)q(2); // qz
    fill[3] = (double)q(3); // qw
    fill[4] = (double)p(0); // tx
    fill[5] = (double)p(1); // ty
    fill[6] = (double)p(2); // tz
    
    env->SetDoubleArrayRegion(result, 0, 7, fill);
    return result;
}
